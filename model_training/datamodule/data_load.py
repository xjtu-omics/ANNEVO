import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


_SEQ_LUT = np.zeros((256, 4), dtype=np.float32)
_SEQ_LUT[ord("A")] = [1.0, 0.0, 0.0, 0.0]
_SEQ_LUT[ord("C")] = [0.0, 1.0, 0.0, 0.0]
_SEQ_LUT[ord("G")] = [0.0, 0.0, 1.0, 0.0]
_SEQ_LUT[ord("T")] = [0.0, 0.0, 0.0, 1.0]
_SEQ_LUT[ord("a")] = _SEQ_LUT[ord("A")]
_SEQ_LUT[ord("c")] = _SEQ_LUT[ord("C")]
_SEQ_LUT[ord("g")] = _SEQ_LUT[ord("G")]
_SEQ_LUT[ord("t")] = _SEQ_LUT[ord("T")]


def sequence_encode(sequence):
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _SEQ_LUT[sequence_bytes]


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index_map = []
        self._h5 = None
        with h5py.File(h5_path, "r") as h5_file:
            for chromosome in h5_file:
                sample_count = int(h5_file[chromosome]["sequence"].shape[0])
                self.index_map.extend((chromosome, index) for index in range(sample_count))

    def __len__(self):
        return len(self.index_map)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        h5_file = getattr(self, "_h5", None)
        if h5_file is not None:
            h5_file.close()

    def __getitem__(self, index):
        chromosome, sample_index = self.index_map[index]
        group = self._file()[chromosome]
        sequence = group["sequence"][sample_index]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        annotation = np.asarray(group["annotation"][sample_index], dtype=np.int64)
        return torch.from_numpy(sequence_encode(sequence)), torch.from_numpy(annotation)


def get_dataloader(h5_path, batch_size, num_workers, shuffle, distributed=False, seq_step=True):
    if not seq_step:
        raise ValueError("This package supports sequence-only training.")
    dataset = H5SequenceDataset(h5_path)
    is_main = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if is_main:
        print(f"The number of samples is {len(dataset)}")
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 4
    return DataLoader(**options)
