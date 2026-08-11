import torch
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
from datamodule.data_load import sequence_encode
import numpy as np
import time
from src.utils import model_construction_for_pred, model_load_weights
from Bio import SeqIO
import re


class GenomeDataset(Dataset):
    def __init__(self, genome_data):
        self.data = genome_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window_seq = self.data[idx]
        one_hot_seq = sequence_encode(window_seq)
        one_hot_seq = torch.tensor(one_hot_seq, dtype=torch.float)
        return one_hot_seq


def rev_complement(dna_sequence):
    complement_map = str.maketrans('ATGCatgcNXnx', 'TACGtacgNXnx')
    return dna_sequence.translate(complement_map)[::-1]


def build_window_starts(length, step_size):
    if length <= 0:
        return [0]
    return list(range(0, length, step_size))


def windows_split(length, sequence_fwd, window_size, flank_length, step_size, count):
    window_starts = build_window_starts(length, step_size)
    pad_front = flank_length
    total_len_needed = window_starts[-1] + window_size + 2 * flank_length
    pad_behind = total_len_needed - (length + flank_length)

    sequence_fwd = 'X' * pad_front + sequence_fwd + 'X' * pad_behind

    windows_reverse_disorder = []
    windows_forward_rec = []

    for output_start in window_starts:
        start = output_start + flank_length
        end = start + window_size

        window_seq_fwd = sequence_fwd[start - flank_length:end + flank_length].upper()
        window_seq_rev = rev_complement(window_seq_fwd)

        windows_forward_rec.append(window_seq_fwd)
        windows_reverse_disorder.append(window_seq_rev)
        count += 1
    return count, windows_forward_rec, windows_reverse_disorder, window_starts, total_len_needed


def predict_probability(model, windows, device, num_classes, batch_size, num_workers):
    dataset = GenomeDataset(windows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers)
    all_outputs = None
    write_start = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            seqs = batch_data
            seqs = seqs.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            outputs = model(seqs)[0]

            outputs = F.softmax(outputs, dim=-1)
            outputs_np = outputs.to(torch.float16).cpu().numpy()
            if all_outputs is None:
                all_outputs = np.empty((len(dataset), outputs_np.shape[1], outputs_np.shape[2]), dtype=np.float16)
            batch_size_actual = outputs_np.shape[0]
            all_outputs[write_start:write_start + batch_size_actual] = outputs_np
            write_start += batch_size_actual
    return all_outputs


def merge_window_predictions(window_predictions, window_starts, length, overlap_pred=False, from_tail=False):
    if len(window_starts) == 0:
        return np.zeros((length, window_predictions.shape[-1]), dtype=np.float16)

    window_size = window_predictions.shape[1]
    num_classes = window_predictions.shape[2]
    if not overlap_pred:
        flattened = window_predictions.reshape(-1, num_classes)
        if from_tail:
            return flattened[-length:]
        return flattened[:length]

    if len(window_predictions) == 1:
        flattened = window_predictions.reshape(-1, num_classes)
        if from_tail:
            return flattened[-length:]
        return flattened[:length]

    step_size = window_size // 2
    left = window_predictions[:, :step_size]
    right = window_predictions[:, step_size:]
    merged = np.empty(((len(window_predictions) + 1) * step_size, num_classes), dtype=np.float16)
    merged[:step_size] = left[0]
    merged_middle = merged[step_size:-step_size].reshape(len(window_predictions) - 1, step_size, num_classes)
    np.add(right[:-1], left[1:], out=merged_middle)
    merged_middle *= np.float16(0.5)
    merged[-step_size:] = right[-1]
    if from_tail:
        return merged[-length:]
    return merged[:length]


def pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
              seq_id_chunk, seq_length_chunk, offset, window_starts_chunk, step_size, overlap_pred):
    predictions_forward = predict_probability(model, windows_forward, device, num_classes, batch_size, num_workers)
    predictions_reverse = predict_probability(model, windows_reverse, device, num_classes, batch_size, num_workers)
    if overlap_pred:
        print("Merging window predictions")
    genome_predictions = {}
    for i, seq_id in enumerate(seq_id_chunk):
        length = seq_length_chunk[i]
        range_start = offset[i]
        range_end = offset[i + 1]
        forward_window_starts = window_starts_chunk[i]
        reverse_window_starts = [j * step_size for j in range(range_end - range_start)]
        predictions_forward_rec = merge_window_predictions(
            predictions_forward[range_start:range_end],
            forward_window_starts,
            length,
            overlap_pred=overlap_pred,
            from_tail=False,
        )
        predictions_reverse_rec = merge_window_predictions(
            predictions_reverse[range_start:range_end],
            reverse_window_starts,
            length,
            overlap_pred=overlap_pred,
            from_tail=True,
        )
        genome_predictions[seq_id] = [predictions_forward_rec, predictions_reverse_rec]

    return genome_predictions


def save_prediction_result(genome_predictions, prediction_path, comp=False):
    start_time = time.time()
    dataset_options = {"chunks": True, "compression": "lzf"} if comp else {}
    with h5py.File(f'{prediction_path}', "a") as f:
        for seq_id, data in genome_predictions.items():
            grp = f.create_group(seq_id)
            pred_fwd_rec = data[0]
            pred_rev_rec = data[1]
            grp.create_dataset("predictions_forward", data=pred_fwd_rec, **dataset_options)
            grp.create_dataset("predictions_reverse", data=pred_rev_rec, **dataset_options)
    end_time = time.time()
    return end_time - start_time


def process_input(genome):
    print('---------------------------------------Processing genome information---------------------------------------')
    start_time = time.time()
    genome_seq = {}
    with open(genome) as genome_data:
        genome_seqIO = SeqIO.to_dict(SeqIO.parse(genome_data, "fasta"))
    for seq_id in genome_seqIO:
        sequence = str(genome_seqIO[seq_id].seq).upper()
        sequence = re.sub(r'[^ATCG]', 'N', sequence)
        genome_seq[seq_id] = sequence

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing genome information took {elapsed_time:.1f} seconds")

    return genome_seq


def base_pred(genome, model_path, genome_size_threshold, prediction_path, num_workers, batch_size,
              window_size, flank_length, local_pattern_size, lineage, num_classes=15, overlap_pred=False,
              comp=False):
    print('Model loading')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_construction_for_pred(
        device,
        window_size,
        flank_length,
        local_pattern_size,
        num_classes=num_classes,
        lineage=lineage,
    )
    model = model_load_weights(model_path, model, device)
    model.eval()
    print('Model loading complete')
    if overlap_pred and window_size % 2 != 0:
        raise ValueError(f"overlap_pred requires an even window_size, got window_size={window_size}")
    step_size = window_size // 2 if overlap_pred else window_size
    if step_size <= 0:
        raise ValueError(f"Invalid prediction step_size={step_size}")
    print(f"Prediction step_size={step_size}, overlap_pred={overlap_pred}")
    file_saving_time = 0
    windows_forward = []
    windows_reverse = []
    offset = [0]
    window_starts_chunk = []
    count = 0
    cumulative_size = 0
    seq_id_chunk = []
    seq_length_chunk = []
    chunk_num = 1
    genome_seq = process_input(genome)
    for seq_id in genome_seq:
        sequence_forward = genome_seq[seq_id]

        length = len(sequence_forward)
        seq_id_chunk.append(seq_id)
        seq_length_chunk.append(length)

        count, windows_forward_rec, windows_reverse_disorder, window_starts, padded_length = windows_split(
            length,
            sequence_forward,
            window_size,
            flank_length,
            step_size,
            count,
        )
        cumulative_size = cumulative_size + padded_length
        for window in windows_forward_rec:
            windows_forward.append(window)
        windows_reverse += windows_reverse_disorder[::-1].copy()
        window_starts_chunk.append(window_starts)
        offset.append(count)

        if cumulative_size > genome_size_threshold * 1e6:
            print(f'---------------------------------------Prediction on chunk {chunk_num}---------------------------------------')
            chunk_num += 1
            genome_predictions = pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
                                           seq_id_chunk, seq_length_chunk, offset, window_starts_chunk, step_size, overlap_pred)
            runtime = save_prediction_result(genome_predictions, prediction_path, comp=comp)
            file_saving_time += runtime

            # Reinitialization
            windows_forward = []
            windows_reverse = []
            offset = [0]
            window_starts_chunk = []
            count = 0
            cumulative_size = 0
            seq_id_chunk = []
            seq_length_chunk = []
    if seq_id_chunk:
        print(f'---------------------------------------Prediction on chunk {chunk_num}---------------------------------------')
        chunk_num += 1
        genome_predictions = pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
                                       seq_id_chunk, seq_length_chunk, offset, window_starts_chunk, step_size, overlap_pred)
        runtime = save_prediction_result(genome_predictions, prediction_path, comp=comp)
        file_saving_time += runtime

    print(f"file saving cost {file_saving_time:.1f} seconds")

    del windows_forward
    del windows_reverse
    del window_starts_chunk
    del seq_id_chunk
    del seq_length_chunk
    del genome_predictions
    del model
    torch.cuda.empty_cache()
    gc.collect()
