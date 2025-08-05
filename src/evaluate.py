import torch
from tqdm import tqdm
from model.loss_function import multi_loss
from src import utils
import torchmetrics
import torch.nn.functional as F
import gc


def model_evaluate(model, loss_fn_CE, loss_fn_dice_CDS0, loss_fn_dice_CDS1, loss_fn_dice_CDS2, loss_fn_dice_intron, coefficient,
                   num_classes, device, val_dataloader, num_branches):

    model.eval()
    total_loss_CE = 0
    total_loss_dice_CDS = 0
    total_loss_dice_intron = 0
    total_loss_balance = 0
    acc = torchmetrics.Accuracy().to(device)
    f1_score = torchmetrics.F1Score(num_classes=num_classes, average='none').to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes).to(device)

    with torch.no_grad():
        for data in tqdm(val_dataloader, desc="Evaluation in the validation set:"):
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
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)
            acc_base, f1_base, confusion_matrix_base = utils.update_metrics(acc, f1_score, confusion_matrix, outputs, labels, position_mask)

    final_acc_base = acc_base.compute()
    final_f1_base = f1_base.compute()
    final_confusion_matrix_base = confusion_matrix_base.compute()

    print('--------------------------------The performance of evaluation set-----------------------------------')
    print(f'\n\n\n')
    print(f'total_loss_CE: {total_loss_CE / len(val_dataloader)}')
    print(f'total_loss_dice_CDS: {total_loss_dice_CDS / len(val_dataloader)}')
    print(f'total_loss_dice_intron: {total_loss_dice_intron / len(val_dataloader)}')
    print(f'total_loss_balance: {total_loss_balance / len(val_dataloader)}')
    print('------------------The metrics of category classification--------------------')
    print(f"Validation Accuracy: {final_acc_base}")
    print(f"Validation F1 Score: {final_f1_base}")
    print(f"Average validation F1 Score: {torch.mean(final_f1_base)}")
    print('The confusion matrix of base prediction\n')
    print(final_confusion_matrix_base)

    checkpoint_metrics = 1 - final_f1_base[1:].mean()
    acc.reset()
    f1_score.reset()
    confusion_matrix.reset()
    gc.collect()
    return checkpoint_metrics
