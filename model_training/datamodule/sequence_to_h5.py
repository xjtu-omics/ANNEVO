import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from BCBio import GFF
from Bio import SeqIO

from model_training.datamodule import genome_track


def reverse_complement(sequence):
    table = str.maketrans("ATGCatgcNXnx", "TACGtacgNXnx")
    return sequence.translate(table)[::-1]


def _annotation_records(genome, annotation, max_parallel_length=1_000_000_000):
    regular = []
    large = []
    with open(annotation) as handle:
        for record in GFF.parse(handle):
            item = (record.id, genome[record.id], list(record.features))
            (large if len(item[1]) > max_parallel_length else regular).append(item)
    return regular, large


def _parse_annotations(genome, annotation, threads):
    regular, large = _annotation_records(genome, annotation)
    parsed = [genome_track.parse_files(item) for item in large]
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(genome_track.parse_files, item) for item in regular]
        for future in as_completed(futures):
            parsed.append(future.result())
    return {
        seq_id: (sequence, forward, reverse)
        for seq_id, sequence, forward, reverse in parsed
    }


def _pad(array, front, back):
    return np.pad(array, (front, back), mode="constant")


def _split_windows(item, window_size, flank_length, intergenic_ratio):
    sequence, annotation_forward, annotation_reverse = item
    sequence_length = len(sequence)
    window_count = math.ceil(sequence_length / window_size)
    padded_length = window_count * window_size + 2 * flank_length
    pad_back = padded_length - sequence_length - flank_length
    padded_sequence = "X" * flank_length + sequence + "X" * pad_back
    annotation_forward = _pad(annotation_forward, flank_length, pad_back)
    annotation_reverse = _pad(annotation_reverse, flank_length, pad_back)

    coding = []
    intergenic = []
    for start in range(flank_length, window_count * window_size + flank_length, window_size):
        end = start + window_size
        forward_sequence = padded_sequence[start - flank_length:end + flank_length]
        forward_label = annotation_forward[start:end]
        reverse_label = annotation_reverse[start:end][::-1]
        candidates = (
            (forward_sequence, forward_label),
            (reverse_complement(forward_sequence), reverse_label),
        )
        for sample_sequence, label in candidates:
            if np.any(label != 0):
                coding.append((sample_sequence, label))
            else:
                intergenic.append((sample_sequence, label))

    if intergenic_ratio == -1:
        selected_intergenic = intergenic
    else:
        count = min(len(intergenic), round(len(intergenic) * intergenic_ratio))
        selected_intergenic = random.sample(intergenic, count)
    return coding, selected_intergenic


def _group_name(h5_file, chromosome):
    if chromosome not in h5_file:
        return chromosome
    index = 1
    while f"{chromosome}__{index}" in h5_file:
        index += 1
    return f"{chromosome}__{index}"


def _append_samples(path, chromosome, samples):
    if not samples:
        return
    with h5py.File(path, "a") as h5_file:
        group = h5_file.create_group(_group_name(h5_file, chromosome))
        string_type = h5py.special_dtype(vlen=str)
        group.create_dataset("sequence", data=[sample[0] for sample in samples], dtype=string_type)
        group.create_dataset(
            "annotation",
            data=np.asarray([sample[1] for sample in samples], dtype=np.uint8),
            chunks=True,
            compression="gzip",
        )


def sequence_to_h5(genome_path, annotation_path, output_prefix, threads,
                   window_size, flank_length, intergenic_ratio):
    if intergenic_ratio != -1 and not 0 <= intergenic_ratio <= 1:
        raise ValueError("intergenic_ratio must be -1 or between 0 and 1.")
    output_prefix = output_prefix[:-3] if output_prefix.lower().endswith(".h5") else output_prefix
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_files = {
        "coding": f"{output_prefix}_coding.h5",
        "all": f"{output_prefix}_all.h5",
    }
    records = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
    genome = {
        seq_id: re.sub(r"[^ATCG]", "N", str(record.seq).upper())
        for seq_id, record in records.items()
    }
    annotations = _parse_annotations(genome, annotation_path, max(1, threads))
    for chromosome, item in annotations.items():
        coding, intergenic = _split_windows(item, window_size, flank_length, intergenic_ratio)
        _append_samples(output_files["coding"], chromosome, coding)
        _append_samples(output_files["all"], chromosome, coding + intergenic)
    return output_files
