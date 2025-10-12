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


class H5GenomeDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index_map = []  # (chrom, idx)

        with h5py.File(h5_path, "r") as f:
            for chrom in f.keys():
                n = len(f[chrom]["sequences"])
                self.index_map.extend([(chrom, i) for i in range(n)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chrom, i = self.index_map[idx]
        with h5py.File(self.h5_path, "r") as f:
            # seq = f[chrom]["sequences"][i].astype(str)
            seq = f[chrom]["sequences"][i].decode("utf-8")
            ann = f[chrom]["annotations"][i]
            msk = f[chrom]["masks"][i]
        one_hot = sequence_encode(seq)
        return (
            torch.tensor(one_hot, dtype=torch.float),
            torch.tensor(ann, dtype=torch.float),
            torch.tensor(msk, dtype=torch.float),
        )


def get_dataloader(h5_path, batch_size, num_workers=8):
    dataset = H5GenomeDataset(h5_path)
    print(f"The number of samples is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


