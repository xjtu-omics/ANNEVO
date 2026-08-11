import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from model_training.datamodule.data_load import get_dataloader
from tqdm import tqdm
from model_training.src.early_stop import EarlyStopping
from model_training.src.loss_function import DiceLoss, FocalLoss
from model_training.src.evaluate import model_evaluate_seq
from model_training.src.utils import (
    model_construction_seq,
    get_local_rank,
    get_world_size,
    is_dist_initialized,
    is_main_process,
    log_batches_per_rank,
)
import gc


def _broadcast_stop_flag(should_stop, device):
    if not is_dist_initialized() or get_world_size() <= 1:
        return should_stop
    flag = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _save_model_state(model, save_path):
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(model_state, save_path)


def training_loop(model, train_dataloader, val_dataloader, optimizer, device, scheduler, loss_fn_FCE, loss_fn_dice,
                  early_stopping, num_classes, epoch, step_scheduler, save_model_each_epoch=False,
                  save_model_path=None):
    log_batches_per_rank(train_dataloader, device, tag="train")
    for i in range(epoch):
        sampler = getattr(train_dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(i)

        if is_main_process():
            print(f'Epoch {i + 1}/{epoch}')
        model.train()
        total_loss_CE = 0
        total_loss_dice = 0
        total_loss_balance = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {i + 1}/{epoch}") if is_main_process() else enumerate(train_dataloader)
        for inx, data in pbar:
            optimizer.zero_grad(set_to_none=True)
            seq, labels = data
            seq = seq.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            labels = labels.to(device).long()
            outputs, loss_balance = model(seq)

            loss_dice = loss_fn_dice(outputs, labels)
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)
            loss_CE = loss_fn_FCE(outputs, labels)

            loss = loss_CE.mean() + loss_dice.mean() + loss_balance.float().mean()

            total_loss_CE += loss_CE.mean().item()
            total_loss_dice += loss_dice.mean().item()
            total_loss_balance += loss_balance.mean().item()
            if is_main_process():
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step_scheduler and scheduler is not None:
                scheduler.step()

        if is_main_process():
            print('--------------------------------The performance of training set-----------------------------------')
            print(f'total_loss_CE: {total_loss_CE / len(train_dataloader):.4f}')
            print(f'total_loss_dice: {total_loss_dice / len(train_dataloader):.4f}')
            print(f'total_loss_balance: {total_loss_balance / len(train_dataloader):.4f}')

        if val_dataloader is None:
            raise ValueError("Validation dataloader is required on all ranks.")
        checkpoint_metrics = model_evaluate_seq(model, loss_fn_FCE, loss_fn_dice, device, val_dataloader, num_classes=num_classes)
        should_stop = False
        if early_stopping is not None and is_main_process():
            early_stopping(checkpoint_metrics, model)
            should_stop = early_stopping.early_stop
            if should_stop:
                print("Early stopping")

        should_stop = _broadcast_stop_flag(should_stop, device)
        if should_stop:
            break

        if save_model_each_epoch and save_model_path is not None and is_main_process():
            _save_model_state(model, save_model_path)

        torch.cuda.empty_cache()
        gc.collect()
        if is_main_process():
            print(f'\n\n\n')


def model_train(model_save_path, train_h5_path, val_h5_path, train_h5_path_2, val_h5_path_2,
                learning_rate, epoch, batch_size, patience, warmup_steps, window_size, flank_length, local_pattern_size, num_classes=15):
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = is_dist_initialized() and get_world_size() > 1

    model = model_construction_seq(device, window_size, flank_length, local_pattern_size, num_classes=num_classes)
    if is_main_process():
        print(model)
    trainable_para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Number of trainable parameters: {trainable_para_count}")

    # loss_fn_CE = nn.CrossEntropyLoss(weight=weights, reduction='none').to(device)
    loss_fn_FCE = FocalLoss(gamma=2.0, reduction='none')
    loss_fn_dice = DiceLoss(classes_list=list(range(1, 15)), classes_weights=[1] * 14)

    early_stopping = EarlyStopping(patience, verbose=True, path=model_save_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # ---------------------------------Step 2---------------------------------
    # Train model on all region.
    # from src.utils import model_load_weights
    # step1_path = model_save_path[:-3] + '_step1.pt'
    # model = model_load_weights(step1_path, model, device)
    train_all_dataloader = get_dataloader(train_h5_path_2, batch_size, num_workers=4, shuffle=True,
                                          distributed=distributed, seq_step=True)
    val_all_dataloader = get_dataloader(val_h5_path_2, batch_size, num_workers=4, shuffle=False,
                                        distributed=distributed, seq_step=True)
    total_steps = epoch * len(train_all_dataloader)
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be greater than 0 for OneCycleLR warmup.")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
        div_factor=10.0,
    )
    training_loop(model, train_all_dataloader, val_all_dataloader, optimizer, device, scheduler,
                  loss_fn_FCE, loss_fn_dice, early_stopping, num_classes, epoch=epoch,
                  step_scheduler=True)

    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    # # ---------------------------------Step 1---------------------------------
    # # Train model on gene region.
    #
    # train_cd_dataloader = get_dataloader(train_h5_path, batch_size, num_workers=4, shuffle=True,
    #                                      distributed=distributed, seq_step=True)
    # val_cd_dataloader = get_dataloader(val_h5_path, batch_size, num_workers=4, shuffle=False,
    #                                    distributed=distributed, seq_step=True)
    #
    # step1_path = model_save_path[:-3] + '_step1.pt'
    # training_loop(model, train_cd_dataloader, val_cd_dataloader, optimizer, device, scheduler,
    #               loss_fn_FCE, loss_fn_dice, None, num_classes, epoch=15, step_scheduler=True,
    #               save_model_each_epoch=True, save_model_path=step1_path)
    #
    # # ---------------------------------Step 2---------------------------------
    # # Train model on all region.
    # # from src.utils import model_load_weights
    # # step1_path = model_save_path[:-3] + '_step1.pt'
    # # model = model_load_weights(step1_path, model, device)
    # train_all_dataloader = get_dataloader(train_h5_path_2, batch_size, num_workers=4, shuffle=True,
    #                                       distributed=distributed, seq_step=True)
    # val_all_dataloader = get_dataloader(val_h5_path_2, batch_size, num_workers=4, shuffle=False,
    #                                     distributed=distributed, seq_step=True)
    # training_loop(model, train_all_dataloader, val_all_dataloader, optimizer, device, scheduler,
    #               loss_fn_FCE, loss_fn_dice, early_stopping, num_classes, epoch=epoch,
    #               step_scheduler=False)
