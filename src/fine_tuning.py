import torch
import torch.nn as nn
from data_process.data_load import get_dataloader
from tqdm import tqdm
from model.loss_function import DiceLoss, multi_loss, FocalLoss
from src.utils import model_construction, model_load_weights
import gc


def training_loop(model, train_dataloader, optimizer, device, num_branches, loss_fn_CE, loss_fn_dice_CDS0, loss_fn_dice_CDS1, loss_fn_dice_CDS2,
                  loss_fn_dice_intron, coefficient, epoch, model_save_path, training_phase):
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10000)

    for i in range(epoch):
        print(f'Epoch {i + 1}/{epoch}')
        model.train()
        total_loss_CE = 0
        total_loss_dice_CDS = 0
        total_loss_dice_intron = 0
        total_loss_balance = 0
        for inx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {i + 1}/{epoch}"):
            optimizer.zero_grad()
            seqs, labels, position_mask = data
            seqs = seqs.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            labels = labels.to(device).long()
            position_mask = position_mask.to(device)
            outputs, topk_indices, topk_vals = model(seqs)

            loss, loss_CE, loss_dice_CDS0, loss_dice_CDS1, loss_dice_CDS2, loss_dice_intron, loss_balance = multi_loss(outputs, topk_indices, topk_vals, num_branches, labels, position_mask, loss_fn_CE,
                                                                                                                       loss_fn_dice_CDS0, loss_fn_dice_CDS1, loss_fn_dice_CDS2,
                                                                                                                       loss_fn_dice_intron, coefficient)

            total_loss_CE += loss_CE.mean().item()
            total_loss_dice_CDS += (loss_dice_CDS0.mean().item() + loss_dice_CDS1.mean().item() + loss_dice_CDS2.mean().item()) * coefficient
            total_loss_dice_intron += loss_dice_intron.mean().item() * coefficient
            total_loss_balance += loss_balance.item() * 0.01

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if training_phase == 1:
                scheduler.step()

        print('--------------------------------The performance of training set-----------------------------------')
        print(f'\n\n\n')
        print(f'total_loss_CE: {total_loss_CE / len(train_dataloader)}')
        print(f'total_loss_dice_CDS: {total_loss_dice_CDS / len(train_dataloader)}')
        print(f'total_loss_dice_intron: {total_loss_dice_intron / len(train_dataloader)}')
        print(f'total_loss_balance: {total_loss_balance / len(train_dataloader)}')
        print(f'\n\n\n')
        torch.cuda.empty_cache()
        gc.collect()
    torch.save(model.state_dict(), model_save_path)


def tuning(model_path, model_save_path, h5_path, learning_rate, epoch, batch_size,
           window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches, num_classes=5, top_k=2)
    core_model = model.module if hasattr(model, 'module') else model
    print(model)
    # freeze FE and MoE
    for param in core_model.FE.parameters():
        param.requires_grad = False
    for param in core_model.MoE.parameters():
        param.requires_grad = False

    for m in core_model.modules():
        if isinstance(m, torch.nn.SyncBatchNorm):
            m.requires_grad_(True)

    model = model_load_weights(model_path, model, device)
    trainable_para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_para_count = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_para_count}")
    print(f"Number of all parameters: {all_para_count}")

    loss_fn_CE = FocalLoss(gamma=2.0, reduction='none')
    loss_fn_dice_CDS0 = DiceLoss(positive_classes=[1])
    loss_fn_dice_CDS1 = DiceLoss(positive_classes=[2])
    loss_fn_dice_CDS2 = DiceLoss(positive_classes=[3])
    loss_fn_dice_intron = DiceLoss(positive_classes=[4])
    coefficient = 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    '''
    ---------------------------------Train phase 2---------------------------------
    Train model on all region.
    '''
    train_dataloader = get_dataloader(f'{h5_path}/fine_tune_with_intergenic.h5', batch_size, num_workers=8)
    training_loop(model, train_dataloader, optimizer, device, num_branches, loss_fn_CE, loss_fn_dice_CDS0, loss_fn_dice_CDS1, loss_fn_dice_CDS2,
                  loss_fn_dice_intron, coefficient, epoch, model_save_path, training_phase=2)


