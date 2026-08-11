import random
import torch
import os
import numpy as np
import torch.distributed as dist
from model import ANNEVO_seq
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Silence verbose C++ warnings from DDP reducer (e.g., find_unused_parameters warning).
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def update_metrics(accuracy_metric, f1_metric_none, confusion_matrix, outputs, labels):
    predictions = outputs.argmax(1)
    if labels.numel() > 0:
        accuracy_metric.update(predictions, labels)
        f1_metric_none.update(predictions, labels)
        confusion_matrix.update(predictions, labels)
    return accuracy_metric, f1_metric_none, confusion_matrix


def init_dist():
    if not dist.is_available() or dist.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def is_main_process():
    return get_rank() == 0


def log_batches_per_rank(dataloader, device=None, tag="loader"):
    if dataloader is None:
        return

    gather_device = torch.device("cpu")
    if is_dist_initialized() and get_world_size() > 1 and dist.get_backend() == "nccl":
        if device is not None and device.type == "cuda":
            gather_device = device
        elif torch.cuda.is_available():
            gather_device = torch.device(f"cuda:{get_local_rank()}")

    local_batches = torch.tensor([len(dataloader)], dtype=torch.int32, device=gather_device)

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "num_samples"):
        local_samples_count = int(sampler.num_samples)
    else:
        local_samples_count = len(dataloader.dataset)

    local_samples = torch.tensor([local_samples_count], dtype=torch.int64, device=gather_device)

    if is_dist_initialized() and get_world_size() > 1:
        gathered = [torch.zeros_like(local_batches) for _ in range(get_world_size())]
        gathered_samples = [torch.zeros_like(local_samples) for _ in range(get_world_size())]
        dist.all_gather(gathered, local_batches)
        dist.all_gather(gathered_samples, local_samples)
        if is_main_process():
            print(f"[{tag}] batches per rank: {[int(x.item()) for x in gathered]}")
            samples_per_rank = [int(x.item()) for x in gathered_samples]
            print(f"[{tag}] samples per rank: {samples_per_rank}")
            batch_size = dataloader.batch_size
            if sampler is not None and hasattr(sampler, "total_size") and hasattr(sampler, "num_samples"):
                global_samples = len(dataloader.dataset)
            else:
                global_samples = sum(samples_per_rank)
            single_gpu_batches = (global_samples + batch_size - 1) // batch_size
            print(f"[{tag}] single_gpu_batches(no DDP, global): {single_gpu_batches}")
            if sampler is not None and hasattr(sampler, "total_size") and hasattr(sampler, "num_samples"):
                padding = int(sampler.total_size - global_samples)
                print(
                    f"[{tag}] sampler num_samples_per_rank={int(sampler.num_samples)}, "
                    f"total_size={int(sampler.total_size)}, padding={padding}"
                )
            else:
                print(f"[{tag}] sampler has no total_size/num_samples -> no sampler padding info (subset sharding).")
    else:
        print(f"[{tag}] batches per rank: {[int(local_batches.item())]}")
        print(f"[{tag}] samples per rank: {[int(local_samples.item())]}")


def wrap_model_for_parallel(model, device):
    use_ddp = is_dist_initialized() and get_world_size() > 1
    if use_ddp and device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    if use_ddp:
        if device.type == "cuda":
            local_rank = get_local_rank()
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            model = DDP(model, find_unused_parameters=True)
    return model


def compute_transformer_seq_len(window_size, flank_length, local_pattern_size):
    total_len = window_size + 2 * flank_length
    if total_len % local_pattern_size != 0:
        raise ValueError(
            f"(window_size + 2 * flank_length) must be divisible by local_pattern_size, "
            f"got total_len={total_len}, local_pattern_size={local_pattern_size}"
        )
    return total_len // local_pattern_size


def model_construction_seq(device, window_size, flank_length, local_pattern_size, num_classes, wrap_for_parallel=True):
    CHANNELS = 64
    NUM_ENCODER_LAYERS = 6
    NUM_HEADS = 8
    NUM_EXPERTS = 8
    model = ANNEVO_seq.ANNEVO(channels=CHANNELS, num_classes=num_classes, num_heads=NUM_HEADS,
                              window_size=window_size, flank_length=flank_length, num_encoder_layers=NUM_ENCODER_LAYERS,
                              n_experts=NUM_EXPERTS, local_pattern_size=local_pattern_size, bal_loss_coef=1e-3)
    if wrap_for_parallel:
        model = wrap_model_for_parallel(model, device)
    else:
        model.to(device)
    return model
