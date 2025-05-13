from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader, random_split
import torch
import h5py
import os
import time


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


class GenomeDataset(Dataset):
    def __init__(self, genome_data):
        self.data = genome_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window_seq, window_ann, window_weights, window_tran_ann, window_tran_mask, window_phase_ann = self.data[idx]
        one_hot_seq = sequence_encode(window_seq)
        return (torch.tensor(one_hot_seq, dtype=torch.float), torch.tensor(window_ann, dtype=torch.long), torch.tensor(window_weights, dtype=torch.float),
                torch.tensor(window_tran_ann, dtype=torch.long), torch.tensor(window_tran_mask, dtype=torch.float), torch.tensor(window_phase_ann, dtype=torch.long))


def read_h5_data(h5_path, species_list):
    datasets = []

    for species_num, species_name in enumerate(species_list):
        hdf5_path = f"{h5_path}/{species_name}.h5"
        if os.path.exists(hdf5_path):
            print(f'Loading process data of species {species_num+1}: {species_name}......')
            with h5py.File(hdf5_path, 'r') as f:
                genome_data = []
                for chromosome in f.keys():
                    grp = f[chromosome]
                    sequences_dataset = grp['sequences']
                    sequences = sequences_dataset[:].astype(str)
                    annotations = grp['annotations'][:]
                    weights = grp['weights'][:]
                    transition_annotation = grp['transition_annotation'][:]
                    transition_mask = grp['transition_mask'][:]
                    phases = grp['phases'][:]
                    for seq, ann, wgt, tran_ann, tran_mask, phase in zip(sequences, annotations, weights, transition_annotation, transition_mask, phases):
                        genome_data.append((seq, ann, wgt, tran_ann, tran_mask, phase))
            genome_dataset = GenomeDataset(genome_data)
            datasets.append(genome_dataset)
        else:
            raise Exception(f'The processed data file of {species_name} does not exist, please check.')

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def data_load(h5_path, species_list, batch_size, sampled_ratio):
    start_time = time.time()
    data = read_h5_data(h5_path, species_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The datasets are loaded in {elapsed_time} seconds.")
    total_length = len(data)
    if sampled_ratio == 1:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"The number of samples is {total_length}")
    else:
        length_1 = int(sampled_ratio * total_length)
        length_2 = total_length - length_1
        subset, _ = random_split(data, [length_1, length_2])
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"The number of samples is {length_1}")
    return dataloader
