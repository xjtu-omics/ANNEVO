import os
import numpy as np
from Bio import SeqIO
from BCBio import GFF
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed


def error_checking_forward(max_mRNA, mask_forward_rec, sequence):
    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
    CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start))
    first_CDS = CDS_features[0]
    last_CDS = CDS_features[-1]

    # calculate whether intron length < 30 bp
    if len(CDS_features) > 1:
        for i in range(len(CDS_features) - 1):
            if int(CDS_features[i + 1].location.start) - int(CDS_features[i].location.end) <= 30:
                mask_forward_rec[int(CDS_features[i].location.end):int(CDS_features[i + 1].location.start)] = 0

    # CDS_seq = ''
    # for CDS_feature in CDS_features:
    #     CDS_seq += sequence[int(CDS_feature.location.start):int(CDS_feature.location.end)]
    # # Missing or incorrect start codon
    # if CDS_seq[0:3] != 'ATG':
    #     mask_forward_rec[int(max_mRNA.location.start):int(first_CDS.location.start)] = 0
    # # Missing or incorrect stop codon
    # if CDS_seq[-3:] not in ['TAA', 'TAG', 'TGA']:
    #     mask_forward_rec[int(last_CDS.location.end):int(max_mRNA.location.end)] = 0

    return mask_forward_rec


def error_checking_reverse(max_mRNA, mask_reverse_rec, sequence):
    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
    CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start), reverse=True)
    first_CDS = CDS_features[0]
    last_CDS = CDS_features[-1]

    # calculate whether intron length < 30 bp
    if len(CDS_features) > 1:
        for i in range(len(CDS_features) - 1):
            if int(CDS_features[i].location.start) - int(CDS_features[i + 1].location.end) <= 30:
                mask_reverse_rec[int(CDS_features[i + 1].location.end):int(CDS_features[i].location.start)] = 0

    # CDS_seq = ''
    # for CDS_feature in CDS_features:
    #     CDS_seq = sequence[int(CDS_feature.location.start):int(CDS_feature.location.end)] + CDS_seq
    # # Missing or incorrect start codon
    # if CDS_seq[-3:] != 'CAT':
    #     mask_reverse_rec[int(first_CDS.location.end):int(max_mRNA.location.end)] = 0
    # # Missing or incorrect stop codon
    # if CDS_seq[0:3] not in ['TTA', 'CTA', 'TCA']:
    #     mask_reverse_rec[int(max_mRNA.location.start):int(last_CDS.location.start)] = 0

    return mask_reverse_rec


def define_coding_exon(arr, start, end, phase, strand):
    pattern = [0, 2, 1]

    phase_index = pattern.index(phase)
    adjusted_pattern = pattern[phase_index:] + pattern[:phase_index]

    length = end - start
    if strand == 1:
        for i in range(length):
            arr[start + i] = adjusted_pattern[i % 3] + 1
    else:
        for i in range(length):
            arr[end - i - 1] = adjusted_pattern[i % 3] + 1
    return arr


def parse_files(seq_information, window_size, flank_length, keep_intergenic_sample):
    """
    The label of each position represents the class of every position. The definition of label are as follows:
    0: intergenic
    1: Coding_exon 0
    2: Coding_exon 1
    3: Coding_exon 2
    4: intron 0
    5: intron 1
    6: intron 2
    """
    seq_id, sequence, features = seq_information
    category_annotation_forward_rec = np.zeros(len(sequence), dtype=np.uint8)
    category_annotation_reverse_rec = np.zeros(len(sequence), dtype=np.uint8)
    mask_forward_rec = np.ones(len(sequence), dtype=np.uint8)  # 0 represents the mask mark, which is convenient for calculation with loss
    mask_reverse_rec = np.ones(len(sequence), dtype=np.uint8)
    record_location_forward = 0
    record_location_reverse = 0
    features = sorted(features, key=lambda x: x.location.start)
    for feature in features:
        if feature.type == 'gene':
            if not feature.sub_features:
                if feature.location.strand == 1:
                    mask_forward_rec[feature.location.start:feature.location.end] = 0
                elif feature.location.strand == -1:
                    mask_reverse_rec[feature.location.start:feature.location.end] = 0
            else:
                exist_mRNA = False
                max_mRNA_len = 0
                max_mRNA = None
                for sub_feature in feature.sub_features:
                    if sub_feature.type in ['mRNA']:
                        exist_mRNA = True
                        mRNA_len = len(sub_feature)
                        if mRNA_len > max_mRNA_len:
                            max_mRNA_len = mRNA_len
                            max_mRNA = sub_feature
                if exist_mRNA:
                    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
                    CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start))
                    if feature.location.strand == 1:
                        if not CDS_features:
                            # In rare case, there is no CDS region within the mRNA in the GFF file.
                            mask_forward_rec[feature.location.start:feature.location.end] = 0
                            continue
                        CDS_start = CDS_features[0].location.start
                        CDS_end = CDS_features[-1].location.end
                        if CDS_start < record_location_forward:
                            mask_forward_rec[CDS_start:CDS_end] = 0
                            continue
                        record_location_forward = CDS_end

                        first_CDS = CDS_features[0]
                        mask_forward_rec[first_CDS.location.start:first_CDS.location.start + 3] = 100
                        category_annotation_forward_rec[CDS_start:CDS_end] = 4
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'CDS':
                                phase = int(sub_feature.qualifiers.get('phase')[0])
                                category_annotation_forward_rec = define_coding_exon(category_annotation_forward_rec, sub_feature.location.start, sub_feature.location.end, phase, strand=1)
                        mask_forward_rec = error_checking_forward(max_mRNA, mask_forward_rec, sequence)

                    elif feature.location.strand == -1:
                        if not CDS_features:
                            mask_reverse_rec[feature.location.start:feature.location.end] = 0
                            continue
                        CDS_start = CDS_features[0].location.start
                        CDS_end = CDS_features[-1].location.end
                        if CDS_start < record_location_reverse:
                            mask_reverse_rec[CDS_start:CDS_end] = 0
                            continue
                        record_location_reverse = CDS_end

                        first_CDS = CDS_features[-1]
                        mask_reverse_rec[first_CDS.location.end - 3:first_CDS.location.end] = 100
                        category_annotation_reverse_rec[CDS_start:CDS_end] = 4
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'CDS':
                                phase = int(sub_feature.qualifiers.get('phase')[0])
                                category_annotation_reverse_rec = define_coding_exon(category_annotation_reverse_rec, sub_feature.location.start, sub_feature.location.end, phase, strand=-1)
                        mask_reverse_rec = error_checking_reverse(max_mRNA, mask_reverse_rec, sequence)

    windows, windows_intergenic = split_sequence(sequence, category_annotation_forward_rec, category_annotation_reverse_rec, mask_forward_rec, mask_reverse_rec, window_size, flank_length, keep_intergenic_sample)

    return seq_id, windows, windows_intergenic


def reverse_complement(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'X': 'X',
                  'Y': 'Y', 'R': 'R', 'M': 'M', 'W': 'W', 'K': 'K', 'B': 'B', 'S': 'S', 'D': 'D', 'H': 'H', 'V': 'V'}
    return ''.join(complement[nucleotide] for nucleotide in reversed(dna_sequence))


def split_sequence(sequence, category_annotation_forward_rec, category_annotation_reverse_rec, mask_forward_rec, mask_reverse_rec, window_size, flank_length, keep_intergenic_sample):
    length = len(sequence)
    sequence_forward = sequence
    windows = []
    windows_with_intergenic = []

    for start in range(0, length, window_size):
        end = start + window_size
        if start - flank_length < 0:
            if end + flank_length <= length:
                pad_before = 'X' * (flank_length - start)
                window_seq_forward = pad_before + sequence_forward[0:end + flank_length]
            else:
                pad_before = 'X' * (flank_length - start)
                pad_after = 'X' * (end + flank_length - length)
                window_seq_forward = pad_before + sequence_forward[0:length] + pad_after
        elif end + flank_length > length:
            pad_after = 'X' * (end + flank_length - length)
            window_seq_forward = sequence_forward[start - flank_length:length] + pad_after
        else:
            window_seq_forward = sequence_forward[start - flank_length:end + flank_length]
        window_seq_reverse = reverse_complement(window_seq_forward)

        if end > length:
            window_ann_forward = category_annotation_forward_rec[start:length]
            window_ann_forward = np.pad(window_ann_forward, (0, window_size - len(window_ann_forward)), 'constant')
            window_ann_reverse = category_annotation_reverse_rec[start:length]
            window_ann_reverse = np.pad(window_ann_reverse, (0, window_size - len(window_ann_reverse)), 'constant')
            window_ann_reverse = window_ann_reverse[::-1]

            window_weights_forward = mask_forward_rec[start:length]
            window_weights_forward = np.pad(window_weights_forward, (0, window_size - len(window_weights_forward)), 'constant')
            window_weights_reverse = mask_reverse_rec[start:length]
            window_weights_reverse = np.pad(window_weights_reverse, (0, window_size - len(window_weights_reverse)), 'constant')
            window_weights_reverse = window_weights_reverse[::-1]
        else:
            window_ann_forward = category_annotation_forward_rec[start:end]
            window_ann_reverse = category_annotation_reverse_rec[start:end][::-1]

            window_weights_forward = mask_forward_rec[start:end]
            window_weights_reverse = mask_reverse_rec[start:end][::-1]

        if not np.all(mask_forward_rec[start:end] == 0):
            if np.all(category_annotation_forward_rec[start:end] == 0):
                windows_with_intergenic.append((window_seq_forward, window_ann_forward, window_weights_forward))
            else:
                windows_with_intergenic.append((window_seq_forward, window_ann_forward, window_weights_forward))
                windows.append((window_seq_forward, window_ann_forward, window_weights_forward))

        if not np.all(mask_reverse_rec[start:end] == 0):
            if np.all(category_annotation_reverse_rec[start:end] == 0):
                windows_with_intergenic.append((window_seq_reverse, window_ann_reverse, window_weights_reverse))
            else:
                windows_with_intergenic.append((window_seq_reverse, window_ann_reverse, window_weights_reverse))
                windows.append((window_seq_reverse, window_ann_reverse, window_weights_reverse))

        # if not np.all(mask_forward_rec[start:end] == 0):
        #     if np.all(category_annotation_forward_rec[start:end] == 0):
        #         windows_intergenic.append((window_seq_forward, window_ann_forward, window_weights_forward))
        #     else:
        #         windows.append((window_seq_forward, window_ann_forward, window_weights_forward))
        #
        # if not np.all(mask_reverse_rec[start:end] == 0):
        #     if np.all(category_annotation_reverse_rec[start:end] == 0):
        #         windows_intergenic.append((window_seq_reverse, window_ann_reverse, window_weights_reverse))
        #     else:
        #         windows.append((window_seq_reverse, window_ann_reverse, window_weights_reverse))

    return windows, windows_with_intergenic


def write_h5(output_file, split_data):
    with h5py.File(output_file, "a") as f:
        for chromosome, windows in split_data.items():
            if chromosome in f:
                raise Exception(f"Seq-ID {chromosome} already exists, please check input data.")
            grp = f.create_group(str(chromosome))
            dt = h5py.special_dtype(vlen=str)
            sequences = [w[0] for w in windows]
            annotations = np.array([w[1] for w in windows])
            masks = np.array([w[2] for w in windows])

            grp.create_dataset("sequences", data=sequences, dtype=dt, chunks=True, compression="gzip")
            grp.create_dataset("annotations", data=annotations, chunks=True, compression="gzip")
            grp.create_dataset("masks", data=masks, chunks=True, compression="gzip")
            # grp.create_dataset("sequences", data=sequences, dtype=dt)
            # grp.create_dataset("annotations", data=annotations)
            # grp.create_dataset("masks", data=masks)


def generate_h5_file(genome, annotation, output_file, threads, window_size, flank_length, keep_intergenic_sample):
    path_name = os.path.dirname(output_file)
    if path_name:
        os.makedirs(path_name, exist_ok=True)

    seq_information_total = []
    seq_information_total_exceed_max = []
    max_overflow_length = 300000000
    with open(genome) as genome_data:
        genome_seq = SeqIO.to_dict(SeqIO.parse(genome_data, "fasta"))
    with open(annotation) as gff_handle:
        for rec in GFF.parse(gff_handle):
            features = []
            sequence = str(genome_seq[rec.id].seq).upper()
            for feature in rec.features:
                features.append(feature)
            if len(sequence) > max_overflow_length:
                seq_information_total_exceed_max.append((rec.id, sequence, features))
            else:
                seq_information_total.append((rec.id, sequence, features))

    results = [parse_files(seq_information, window_size, flank_length, keep_intergenic_sample)
               for seq_information in seq_information_total_exceed_max]

    with ProcessPoolExecutor(max_workers=threads) as executor:
        future_to_segment = {executor.submit(parse_files, seq_information, window_size, flank_length, keep_intergenic_sample): seq_information for seq_information in seq_information_total}
        for future in as_completed(future_to_segment):
            result = future.result()
            results.append(result)

    split_genome = {}
    split_genome_with_intergenic = {}
    for result in results:
        seq_id, windows, windows_intergenic = result
        split_genome[seq_id] = windows
        split_genome_with_intergenic[seq_id] = windows_intergenic

    write_h5(f"{output_file}.h5", split_genome)
    write_h5(f"{output_file}_with_intergenic.h5", split_genome_with_intergenic)

