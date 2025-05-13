from src.utils import model_construction, model_load_weights
import glob
from Bio import SeqIO
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import gc


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


def predict_probability(model, windows, device, num_classes, batch_size, num_workers):
    data = GenomeDataset(windows)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accumulated_outputs_base = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            seqs = data
            seqs = seqs.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            outputs, _, _ = model(seqs)
            if device.type == 'cpu':
                outputs = outputs.reshape(-1, num_classes)
            else:
                outputs = outputs.view(-1, num_classes)

            accumulated_outputs_base.append(outputs.cpu())
    all_outputs = torch.cat(accumulated_outputs_base, dim=0)
    all_outputs = F.softmax(all_outputs, dim=-1).numpy().astype('float16')

    return all_outputs


def nucleotide_prediction(genome, lineage, chunk_num, num_workers, prediction_path, batch_size, window_size, flank_length, channels, dim_feedforward,
                          num_encoder_layers, num_heads, num_blocks, num_branches, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(genome) as fna:
        genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))

    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches, num_classes, top_k=2)
    model = model_load_weights(lineage, model, device)
    model.eval()

    chromosome_name = []
    chromosome_length = []
    for chromosome in genome_seq:
        chromosome_name.append(chromosome)
        chromosome_seq_record = genome_seq[chromosome]
        sequence = str(chromosome_seq_record.seq).upper()
        length = len(sequence)
        chromosome_length.append(length)
    print(f'The number of sequence in this species is {len(chromosome_name)}')
    print(f'The prediction file will be saved in {min(chunk_num, len(chromosome_name))} blocks.')
    chunk_num = min(chunk_num, len(chromosome_name))

    chromosomes = list(zip(chromosome_name, chromosome_length))
    chromosomes.sort(key=lambda x: x[1], reverse=True)
    chromosomes_groups = [([], 0) for _ in range(chunk_num)]
    chromosomes_groups = [[list(chromosomes_group[0]), chromosomes_group[1]] for chromosomes_group in chromosomes_groups]
    for name, length in chromosomes:
        min_group = min(chromosomes_groups, key=lambda x: x[1])
        min_group[0].append(name)
        min_group[1] += length

    for chunk_order, (chromosome_name_list, _) in enumerate(chromosomes_groups):
        # for chunk_order, (chunk_start, chunk_end) in enumerate(chunk_index):
        print(f'Processing chunk {chunk_order}')
        seq_id_chunk = chromosome_name_list
        seq_length_chunk = []
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

        predictions_forward = predict_probability(model, windows_forward, device, num_classes, batch_size, num_workers)
        predictions_reverse = predict_probability(model, windows_reverse, device, num_classes, batch_size, num_workers)

        for i, chromosome in enumerate(seq_id_chunk):
            length = seq_length_chunk[i]
            range_start = offset[i] * window_size
            range_end = offset[i + 1] * window_size
            predictions_forward_rec = predictions_forward[range_start:range_end][:length]
            predictions_reverse_rec = predictions_reverse[range_start:range_end][-length:]
            genome_predictions[chromosome] = [predictions_forward_rec, predictions_reverse_rec]

        with h5py.File(f'{prediction_path}/model_predictions_{chunk_order}.h5', "w") as f:
            for chromosome, data in genome_predictions.items():
                chr_group = f.create_group(chromosome)
                labels = ['predictions_forward', 'predictions_reverse']
                for i, dataset in enumerate(data):
                    chr_group.create_dataset(labels[i], data=dataset)

        windows_forward.clear()
        windows_reverse.clear()
        genome_predictions.clear()

        torch.cuda.empty_cache()
        gc.collect()
