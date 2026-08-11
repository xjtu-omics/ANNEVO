import gc

import torch
import torch.distributed as dist
import torchmetrics
from tqdm import tqdm

from model_training.src import utils


CLASS_NAMES = [
    "Intergenic",
    "Coding_exon_0",
    "Coding_exon_1",
    "Coding_exon_2",
    "Intron_0",
    "Intron_1",
    "Intron_2",
    "DSS_0",
    "DSS_1",
    "DSS_2",
    "ASS_0",
    "ASS_1",
    "ASS_2",
    "start",
    "end",
]


def _ddp_weighted_mean(local_sum, local_count, device):
    sum_tensor = torch.tensor(local_sum, dtype=torch.float32, device=device)
    count_tensor = torch.tensor(local_count, dtype=torch.float32, device=device)
    if utils.is_dist_initialized() and utils.get_world_size() > 1:
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    denom = max(count_tensor.item(), 1.0)
    return sum_tensor.item() / denom


def _build_metric_bundle(num_classes, device):
    return {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="none").to(device),
        "cm": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device),
    }


def _update_metric_bundle(bundle, outputs_flat, labels_flat):
    acc, f1, cm = utils.update_metrics(bundle["acc"], bundle["f1"], bundle["cm"], outputs_flat, labels_flat)
    bundle["acc"] = acc
    bundle["f1"] = f1
    bundle["cm"] = cm


def _finalize_metric_bundle(bundle):
    final_acc = bundle["acc"].compute()
    final_f1 = bundle["f1"].compute()
    final_cm = bundle["cm"].compute()
    correct_counts = torch.diag(final_cm)
    return final_acc, final_f1, final_cm, correct_counts


def _reset_metric_bundle(bundle):
    bundle["acc"].reset()
    bundle["f1"].reset()
    bundle["cm"].reset()


def _print_correct_counts(correct_counts):
    for idx, count in enumerate(correct_counts):
        class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        print(f"{class_name}: {int(count.item())}")


def _checkpoint_from_f1(final_f1):
    f1_mean = final_f1[1:].mean()
    return 1 - f1_mean


def model_evaluate_seq(model, loss_fn_fce, loss_fn_dice, device, val_dataloader, num_classes):
    model.eval()
    total_loss_fce = 0.0
    total_loss_dice = 0.0
    total_loss_balance = 0.0
    metrics = _build_metric_bundle(num_classes, device)

    val_iterator = tqdm(val_dataloader, desc="Evaluation in the validation set:") if utils.is_main_process() else val_dataloader
    with torch.no_grad():
        for data in val_iterator:
            seq, labels = data
            seq = seq.to(device).float()
            labels = labels.to(device).long()

            outputs, loss_balance = model(seq)
            loss_dice = loss_fn_dice(outputs, labels)

            outputs_flat = outputs.reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            loss_ce = loss_fn_fce(outputs_flat, labels_flat)

            total_loss_fce += loss_ce.mean().item()
            total_loss_dice += loss_dice.mean().item()
            total_loss_balance += loss_balance.mean().item()
            _update_metric_bundle(metrics, outputs_flat, labels_flat)

    final_acc, final_f1, final_confusion_matrix, correct_counts = _finalize_metric_bundle(metrics)

    local_batches = len(val_dataloader)
    avg_loss_ce = _ddp_weighted_mean(total_loss_fce, local_batches, device)
    avg_loss_dice = _ddp_weighted_mean(total_loss_dice, local_batches, device)
    avg_loss_balance = _ddp_weighted_mean(total_loss_balance, local_batches, device)

    if utils.is_main_process():
        print('--------------------------------The performance of evaluation set-----------------------------------')
        print(f'total_loss_CE: {avg_loss_ce:.4f}')
        print(f'total_loss_dice_CDS: {avg_loss_dice:.4f}')
        print(f'total_loss_balance: {avg_loss_balance:.4f}')
        print('------------------The metrics of category classification--------------------')
        print(f"Validation Accuracy: {final_acc:.4f}")
        print(f"Validation F1 Score: {final_f1}")
        print("Correct predictions per class:")
        _print_correct_counts(correct_counts)

    checkpoint_metrics = _checkpoint_from_f1(final_f1)
    _reset_metric_bundle(metrics)
    gc.collect()
    return checkpoint_metrics
