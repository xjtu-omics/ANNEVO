import random
import os
import numpy as np
import torch.distributed as dist
from model import model_architecture
import torch
import torch.nn as nn


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_metrics(accuracy_metric, f1_metric_none, confusion_matrix, outputs, labels, position_weights):
    position_weights = position_weights.reshape(-1)
    predictions = outputs.argmax(1)
    mask = position_weights != 0
    filtered_labels = labels[mask]
    filtered_predictions = predictions[mask]
    if filtered_labels.numel() > 0:
        accuracy_metric.update(filtered_predictions, filtered_labels)
        f1_metric_none.update(filtered_predictions, filtered_labels)
        confusion_matrix.update(filtered_predictions, filtered_labels)
    return accuracy_metric, f1_metric_none, confusion_matrix


def init_dist():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12342'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    rank = 0
    world_size = 1
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def model_construction(device, window_size, flank_length, num_classes):
    model = model_architecture.ANNEVO(channels=64, dim_feedforward=768, num_classes=num_classes, num_heads=8, num_encoder_layers=6,
                                      window_size=window_size, flank_length=flank_length, num_blocks=5, num_branches=8, top_k=2)
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)
    model.to(device)
    return model


def model_load_weights(path, model, device):
    state_dict = torch.load(path, map_location='cpu') if device.type == 'cpu' else torch.load(path)

    if list(state_dict.keys())[0].startswith('module.'):
        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            '''
            When processing a PyTorch model state dictionary saved using DataParallel, a 'module.' prefix is automatically added to each parameter key.
            This is because DataParallel encapsulates the model into a property called 'module' in order to distribute different parts of the model across multiple GPUs.
            In the code, the purpose of the line "name = k[7:]" is to remove the 'module.' prefix from each key name.
            '''
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            '''
            When processing a PyTorch model state dictionary saved using DataParallel, a 'module.' prefix is automatically added to each parameter key.
            This is because DataParallel encapsulates the model into a property called 'module' in order to distribute different parts of the model across multiple GPUs.
            In the code, the purpose of the line "name = k[7:]" is to remove the 'module.' prefix from each key name.
            '''
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
    return model
