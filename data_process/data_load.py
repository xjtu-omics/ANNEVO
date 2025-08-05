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
        window_seq, window_ann, window_mask = self.data[idx]
        one_hot_seq = sequence_encode(window_seq)
        return (torch.tensor(one_hot_seq, dtype=torch.float), torch.tensor(window_ann, dtype=torch.float), torch.tensor(window_mask, dtype=torch.float))


def read_h5_data(h5_path, species_list, training_phase):
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
                    mask = grp['masks'][:]

                    for seq, ann, msk in zip(sequences, annotations, mask):
                        genome_data.append((seq, ann, msk))
            genome_dataset = GenomeDataset(genome_data)
            datasets.append(genome_dataset)
        else:
            raise Exception(f'The processed data file of {species_name} does not exist, please check.')
    if training_phase == 2:
        for species_num, species_name in enumerate(species_list):
            hdf5_path = f"{h5_path}/{species_name}_intergenic.h5"
            if os.path.exists(hdf5_path):
                print(f'Loading process data of species {species_num + 1}: {species_name}......')
                with h5py.File(hdf5_path, 'r') as f:
                    genome_data = []
                    for chromosome in f.keys():
                        grp = f[chromosome]
                        sequences_dataset = grp['sequences']
                        sequences = sequences_dataset[:].astype(str)
                        annotations = grp['annotations'][:]
                        mask = grp['masks'][:]

                        for seq, ann, msk in zip(sequences, annotations, mask):
                            genome_data.append((seq, ann, msk))
                genome_dataset = GenomeDataset(genome_data)
                datasets.append(genome_dataset)
            else:
                raise Exception(f'The processed data file of {species_name} does not exist, please check.')

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def get_dataloader(h5_path, species_list, batch_size, num_workers, training_phase):
    start_time = time.time()
    data = read_h5_data(h5_path, species_list, training_phase)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The datasets are loaded in {elapsed_time} seconds.")
    total_length = len(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"The number of samples is {total_length}")

    return dataloader
