import torch
from model import ANNEVO_seq, ANNEVO_plant
import torch.nn as nn
import math


def model_construction_for_pred(device, window_size, flank_length, local_pattern_size, num_classes, lineage):
    CHANNELS = 64
    NUM_ENCODER_LAYERS = 6
    NUM_HEADS = 8
    NUM_EXPERTS = 8
    PLANT_LINEAGES = {"Magnoliopsida"}

    if local_pattern_size <= 0 or (local_pattern_size & (local_pattern_size - 1)) != 0:
        raise ValueError("local_pattern_size must be a power of 2")

    if lineage in PLANT_LINEAGES:
        num_blocks = int(math.log2(local_pattern_size))
        dim_feedforward = 2 * CHANNELS * (num_blocks + 1)
        model = ANNEVO_plant.ANNEVO(
            channels=CHANNELS,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            window_size=window_size,
            flank_length=flank_length,
            num_blocks=num_blocks,
            num_branches=NUM_EXPERTS,
            top_k=2,
        )
    else:
        model = ANNEVO_seq.ANNEVO(channels=CHANNELS, num_classes=num_classes, num_heads=NUM_HEADS,
                                  window_size=window_size, flank_length=flank_length,
                                  num_encoder_layers=NUM_ENCODER_LAYERS, n_experts=NUM_EXPERTS,
                                  local_pattern_size=local_pattern_size, bal_loss_coef=1e-3)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)
    model.to(device)
    return model


def model_load_weights(path, model, device):
    state_dict = torch.load(path, map_location='cpu')
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in state_dict and isinstance(state_dict[key], dict) and len(state_dict[key]) > 0:
                sample_val = next(iter(state_dict[key].values()))
                if isinstance(sample_val, torch.Tensor):
                    state_dict = state_dict[key]
                    break

    model_is_parallel = hasattr(model, 'module')
    ckpt_has_module_prefix = list(state_dict.keys())[0].startswith('module.')

    if ckpt_has_module_prefix and not model_is_parallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
    elif (not ckpt_has_module_prefix) and model_is_parallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[f'module.{k}'] = v
        state_dict = new_state_dict

    # PositionalEncoding buffer length can change when using longer inference windows.
    # Skip loading this buffer if shape differs.
    state_dict.pop("FE.PositionalEncodingLayer.pe", None)
    state_dict.pop("module.FE.PositionalEncodingLayer.pe", None)

    model.load_state_dict(state_dict, strict=False)
    return model
