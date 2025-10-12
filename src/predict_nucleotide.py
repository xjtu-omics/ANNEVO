from src.utils import model_construction, model_load_weights
from Bio import SeqIO
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import gc
import time
import torch.nn as nn


def reverse_complement(dna_sequence):
    complement_map = str.maketrans('ATGCRMYWKBSHDVNXatgcrmywkbshdvnx', 'TACGRMYWKBSHDVNXtacgrmywkbshdvnx')
    return dna_sequence.translate(complement_map)[::-1]


class GenomeDataset(Dataset):
    def __init__(self, genome_data):
        self.data = genome_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window_seq = self.data[idx]
        one_hot_seq = sequence_encode(window_seq)
        return torch.tensor(one_hot_seq, dtype=torch.float)


def sequence_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0.25, 0.25, 0.25, 0.25],
               'M': [0.25, 0.25, 0.25, 0.25],
               'W': [0.25, 0.25, 0.25, 0.25],
               'R': [0.25, 0.25, 0.25, 0.25],
               'Y': [0.25, 0.25, 0.25, 0.25],
               'K': [0.25, 0.25, 0.25, 0.25],
               'B': [0.25, 0.25, 0.25, 0.25],
               'S': [0.25, 0.25, 0.25, 0.25],
               'D': [0.25, 0.25, 0.25, 0.25],
               'H': [0.25, 0.25, 0.25, 0.25],
               'V': [0.25, 0.25, 0.25, 0.25],
               'X': [0, 0, 0, 0]}
    return [mapping[s] for s in seq]


def predict_probability(model, windows, device, num_classes, batch_size, num_workers):
    data = GenomeDataset(windows)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accumulated_outputs_base = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            seqs = data.to(device)
            outputs, _, _ = model(seqs)
            outputs = outputs.reshape(-1, num_classes)
            # if device.type == 'cpu':
            #     outputs = outputs.reshape(-1, num_classes)
            # else:
            #     outputs = outputs.view(-1, num_classes)

            accumulated_outputs_base.append(outputs.cpu())
    all_outputs = torch.cat(accumulated_outputs_base, dim=0)
    all_outputs = F.softmax(all_outputs, dim=-1).numpy().astype('float16')

    return all_outputs


def pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
              seq_id_chunk, seq_length_chunk, offset, window_size):
    predictions_forward = predict_probability(model, windows_forward, device, num_classes, batch_size, num_workers)
    predictions_reverse = predict_probability(model, windows_reverse, device, num_classes, batch_size, num_workers)
    genome_predictions = {}
    for i, seq_id in enumerate(seq_id_chunk):
        length = seq_length_chunk[i]
        range_start = offset[i] * window_size
        range_end = offset[i + 1] * window_size
        predictions_forward_rec = predictions_forward[range_start:range_end][:length]
        predictions_reverse_rec = predictions_reverse[range_start:range_end][-length:]
        genome_predictions[seq_id] = [predictions_forward_rec, predictions_reverse_rec]

    return genome_predictions


def save_prediction_result(genome_predictions, prediction_path):
    start_time = time.time()
    with h5py.File(f'{prediction_path}', "a") as f:
        for seq_id, data in genome_predictions.items():
            grp = f.create_group(seq_id)
            pred_fwd_rec = data[0]
            pred_rev_rec = data[1]
            grp.create_dataset("predictions_forward", data=pred_fwd_rec)
            grp.create_dataset("predictions_reverse", data=pred_rev_rec)
    end_time = time.time()
    return end_time - start_time


def nucleotide_prediction(genome, model_path, genome_size_threshold, num_workers, prediction_path, batch_size, window_size, flank_length, channels, dim_feedforward,
                          num_encoder_layers, num_heads, num_blocks, num_branches, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(genome) as fna:
        genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))

    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches, num_classes, top_k=2)
    model = model_load_weights(model_path, model, device)
    model.eval()

    file_saving_time = 0

    windows_forward = []
    windows_reverse = []
    offset = [0]
    count = 0
    cumulative_size = 0
    seq_id_chunk = []
    seq_length_chunk = []
    for chromosome in genome_seq:
        chromosome_seq_record = genome_seq[chromosome]
        sequence_forward = str(chromosome_seq_record.seq).upper()
        length = len(sequence_forward)
        cumulative_size = cumulative_size + length
        windows_reverse_disorder = []
        seq_id_chunk.append(chromosome)
        seq_length_chunk.append(length)
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
            windows_forward.append(window_seq_forward)
            windows_reverse_disorder.append(reverse_complement(window_seq_forward))
            count += 1
        windows_reverse += windows_reverse_disorder[::-1]
        offset.append(count)

        if cumulative_size > genome_size_threshold:
            genome_predictions = pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
                                           seq_id_chunk, seq_length_chunk, offset, window_size)
            runtime = save_prediction_result(genome_predictions, prediction_path)
            file_saving_time += runtime

            # Reinitialization
            windows_forward = []
            windows_reverse = []
            offset = [0]
            count = 0
            cumulative_size = 0
            seq_id_chunk = []
            seq_length_chunk = []
    if seq_id_chunk:
        genome_predictions = pred_only(model, windows_forward, windows_reverse, device, num_classes, batch_size, num_workers,
                                       seq_id_chunk, seq_length_chunk, offset, window_size)
        runtime = save_prediction_result(genome_predictions, prediction_path)
        file_saving_time += runtime

    print(f"file saving cost {file_saving_time:.1f} seconds")

    del windows_forward
    del windows_reverse
    del seq_id_chunk
    del seq_length_chunk
    del genome_predictions
    del model
    del genome_seq
    torch.cuda.empty_cache()
    gc.collect()
