import os
from ANNEVO.utils.utils import model_construction, model_load_weights
from Bio import SeqIO
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import gc
import subprocess


def reverse_complement(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'X': 'X',
                  'Y': 'Y', 'R': 'R', 'M': 'M', 'W': 'W', 'K': 'K', 'B': 'B', 'S': 'S', 'D': 'D', 'H': 'H', 'V': 'V'}
    return ''.join(complement[nucleotide] for nucleotide in reversed(dna_sequence))


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


def predict_probability(model, windows, device, num_classes_base, num_classes_transition, num_classes_phases, batch_size, num_workers):
    data = GenomeDataset(windows)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accumulated_outputs_base = []
    accumulated_outputs_transition = []
    accumulated_outputs_phase = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            seqs = data
            seqs = seqs.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            outputs_base, outputs_transition, outputs_phase = model(seqs)
            if device.type == 'cpu':
                outputs_base = outputs_base.reshape(-1, num_classes_base)
                outputs_transition = outputs_transition.reshape(-1, num_classes_transition)
                outputs_phase = outputs_phase.reshape(-1, num_classes_phases)
            else:
                outputs_base = outputs_base.reshape(-1, num_classes_base)
                outputs_transition = outputs_transition.reshape(-1, num_classes_transition)
                outputs_phase = outputs_phase.reshape(-1, num_classes_phases)
            accumulated_outputs_base.append(outputs_base.cpu())
            accumulated_outputs_transition.append(outputs_transition.cpu())
            accumulated_outputs_phase.append(outputs_phase.cpu())
    all_outputs_base = torch.cat(accumulated_outputs_base, dim=0)
    all_outputs_transition = torch.cat(accumulated_outputs_transition, dim=0)
    all_outputs_phase = torch.cat(accumulated_outputs_phase, dim=0)
    all_outputs_base = F.softmax(all_outputs_base, dim=-1).numpy().astype('float16')
    all_outputs_transition = F.softmax(all_outputs_transition, dim=-1).numpy().astype('float16')
    all_outputs_phase = F.softmax(all_outputs_phase, dim=-1).numpy().astype('float16')
    return all_outputs_base, all_outputs_transition, all_outputs_phase


def predict_proba_of_bases(genome, lineage, chunk_num, num_workers, prediction_path, batch_size, window_size, flank_length, channels, dim_feedforward,
                           num_encoder_layers, num_heads, num_blocks, num_branches, num_classes_base, num_classes_transition, num_classes_phases):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks,
                               num_branches, num_classes_base, num_classes_transition, num_classes_phases)
    model = model_load_weights(lineage, model, device)
    model.eval()

    seq_num = sum(1 for _ in SeqIO.parse(genome, "fasta"))
    chunk_num = min(chunk_num, seq_num)
    print(f'The prediction file will be saved in {chunk_num} blocks.')

    cmd = [
        "seqkit", "split2",
        "-p", str(chunk_num),
        "-f",
        "-O", f'{prediction_path}/temp_genome_split',
        genome
    ]
    subprocess.run(cmd)

    file_name = os.path.basename(genome)
    file_name_without_ext, ext = os.path.splitext(file_name)

    for chunk_order in range(chunk_num):
        print(f'predicting chunk {chunk_order + 1} / {chunk_num}')
        chunk_str = str(chunk_order+1)
        if len(chunk_str) == 1:  # 1 位数
            part_name = f".part_00{chunk_str}"
        elif len(chunk_str) == 2:
            part_name = f".part_0{chunk_str}"
        elif len(chunk_str) == 3:
            part_name = f".part_{chunk_str}"
        else:
            raise Exception("chunk_num should be less than 1000.")

        with open(f'{prediction_path}/temp_genome_split/{file_name_without_ext}{part_name}{ext}') as fna:
            genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
        seq_id_chunk = []
        seq_length_chunk = []
        for chromosome in genome_seq:
            seq_id_chunk.append(chromosome)
            chromosome_seq_record = genome_seq[chromosome]
            sequence = str(chromosome_seq_record.seq).upper()
            length = len(sequence)
            seq_length_chunk.append(length)
        windows_forward = []
        windows_reverse = []
        genome_predictions = {}
        offset = [0]
        count = 0
        for chromosome in seq_id_chunk:
            chromosome_seq_record = genome_seq[chromosome]
            sequence_forward = str(chromosome_seq_record.seq).upper()
            windows_reverse_disorder = []
            length = len(sequence_forward)
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

        base_predictions_forward, transition_predictions_forward, phase_predictions_forward = predict_probability(model, windows_forward, device, num_classes_base,
                                                                                                                  num_classes_transition, num_classes_phases, batch_size, num_workers)
        base_predictions_reverse, transition_predictions_reverse, phase_predictions_reverse = predict_probability(model, windows_reverse, device, num_classes_base,
                                                                                                                  num_classes_transition, num_classes_phases, batch_size, num_workers)

        for i, chromosome in enumerate(seq_id_chunk):
            length = seq_length_chunk[i]
            range_start = offset[i] * window_size
            range_end = offset[i + 1] * window_size
            base_predictions_forward_rec = base_predictions_forward[range_start:range_end][:length]
            transition_predictions_forward_rec = transition_predictions_forward[range_start:range_end][:length]
            phase_predictions_forward_rec = phase_predictions_forward[range_start:range_end][:length]
            base_predictions_reverse_rec = base_predictions_reverse[range_start:range_end][-length:]
            transition_predictions_reverse_rec = transition_predictions_reverse[range_start:range_end][-length:]
            phase_predictions_reverse_rec = phase_predictions_reverse[range_start:range_end][-length:]
            genome_predictions[chromosome] = [base_predictions_forward_rec, transition_predictions_forward_rec, phase_predictions_forward_rec,
                                              base_predictions_reverse_rec, transition_predictions_reverse_rec, phase_predictions_reverse_rec]
        with h5py.File(f'{prediction_path}/model_predictions_{chunk_order}.h5', "w") as f:
            for chromosome, data in genome_predictions.items():
                chr_group = f.create_group(chromosome)
                labels = ['base_predictions_forward', 'transition_predictions_forward', 'phase_predictions_forward',
                          'base_predictions_reverse', 'transition_predictions_reverse', 'phase_predictions_reverse']
                for i, dataset in enumerate(data):
                    chr_group.create_dataset(labels[i], data=dataset)

        windows_forward.clear()
        windows_reverse.clear()
        genome_predictions.clear()

        del base_predictions_forward, transition_predictions_forward, phase_predictions_forward
        del base_predictions_reverse, transition_predictions_reverse, phase_predictions_reverse
        del base_predictions_forward_rec, transition_predictions_forward_rec, phase_predictions_forward_rec
        del base_predictions_reverse_rec, transition_predictions_reverse_rec, phase_predictions_reverse_rec

        torch.cuda.empty_cache()
        gc.collect()
