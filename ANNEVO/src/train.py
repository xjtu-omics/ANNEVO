import torch
import torch.nn as nn
from ANNEVO.data_process.dataset import data_load
from tqdm import tqdm
from ANNEVO.src.early_stop import EarlyStopping
from ANNEVO.src.evaluate import model_evaluate
from ANNEVO.utils.utils import model_construction
import gc


def model_train(train_species_list, val_species_list, model_save_path, h5_path, learning_rate, epoch, batch_size, patience, class_weights_base, class_weights_transition,
                class_weights_phases, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks,
                num_branches, num_classes_base, num_classes_transition, num_classes_phases):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks,
                               num_branches, num_classes_base, num_classes_transition, num_classes_phases)

    class_weights_base = torch.tensor(class_weights_base, dtype=torch.float)
    loss_fn_base = nn.CrossEntropyLoss(weight=class_weights_base, reduction='none').to(device)  # set reduction='none' to return the loss of every base rather than average loss
    class_weights_transition = torch.tensor(class_weights_transition, dtype=torch.float)
    loss_fn_transition = nn.CrossEntropyLoss(weight=class_weights_transition, reduction='none').to(device)  # set reduction='none' to return the loss of every base rather than average loss
    class_weights_phases = torch.tensor(class_weights_phases, dtype=torch.float)
    loss_fn_phases = nn.CrossEntropyLoss(weight=class_weights_phases, reduction='none').to(device)  # set reduction='none' to return the loss of every base rather than average loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    early_stopping = EarlyStopping(patience, verbose=True, path=model_save_path)

    print('---------------------------------The details of training sets---------------------------------')
    train_dataloader = data_load(h5_path, train_species_list, batch_size, sampled_ratio=1)
    print('---------------------------------The details of validation sets---------------------------------')
    val_dataloader = data_load(h5_path, val_species_list, batch_size, sampled_ratio=1)

    for i in range(epoch):
        print(f'Epoch {i + 1}/{epoch}')
        model.train()
        total_train_loss_base = 0
        total_train_loss_transition = 0
        total_train_loss_phase = 0
        for inx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {i + 1}/{epoch}"):
        # for inx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            seqs, labels_base, position_weights_base, labels_transition, position_weights_transition, labels_phases = data
            seqs = seqs.to(device).float()  # Shape of [batch_size, sequence_length, num_classes]
            labels_base = labels_base.to(device).long().view(-1)
            position_weights_base = position_weights_base.to(device).view(-1)
            labels_transition = labels_transition.to(device).long().view(-1)
            position_weights_transition = position_weights_transition.to(device).view(-1)
            labels_phases = labels_phases.to(device).long().view(-1)

            # outputs_base, outputs_transition, gate_weights = model(seqs)
            outputs_base, outputs_transition, outputs_phase = model(seqs)
            outputs_base = outputs_base.reshape(-1, num_classes_base)
            outputs_transition = outputs_transition.reshape(-1, num_classes_transition)
            outputs_phase = outputs_phase.reshape(-1, num_classes_phases)
            loss_base = loss_fn_base(outputs_base, labels_base) * position_weights_base
            loss_transition = loss_fn_transition(outputs_transition, labels_transition) * position_weights_transition
            loss_phases = loss_fn_phases(outputs_phase, labels_phases) * position_weights_base
            total_train_loss_base = total_train_loss_base + loss_base.mean().item()
            total_train_loss_transition = total_train_loss_transition + loss_transition.mean().item()
            total_train_loss_phase = total_train_loss_phase + loss_phases.mean().item()

            loss = loss_base.mean() + 10 * loss_transition.mean() + 10 * loss_phases.mean()
            # loss = 10 * loss_phases.mean()

            # losses = torch.stack([loss_base.mean(), loss_transition.mean()])
            # mtl_loss = MultiTaskLossWrapper(2).to(device)
            # loss = mtl_loss(losses)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # 显式删除不再需要的变量
            del outputs_base, outputs_transition, outputs_phase, loss_base, loss_transition, loss_phases, loss
            gc.collect()  # 强制垃圾回收
        scheduler.step()

        print('--------------------------------The performance of training set-----------------------------------')
        print(f'The loss of base classification on training set: {total_train_loss_base / len(train_dataloader)}')
        print(f'The loss of transition classification on training set: {total_train_loss_transition / len(train_dataloader)}')
        print(f'The loss of phase classification on training set: {total_train_loss_phase / len(train_dataloader)}')

        checkpoint_metrics = model_evaluate(model, loss_fn_base, loss_fn_transition, loss_fn_phases, num_classes_base, num_classes_transition, num_classes_phases, device, val_dataloader)
        early_stopping(checkpoint_metrics, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
