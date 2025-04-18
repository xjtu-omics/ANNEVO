import torch
from tqdm import tqdm
from ANNEVO.utils import utils
import torchmetrics
import gc


def model_evaluate(model, loss_fn_base, loss_fn_transition, loss_fn_phases, num_classes_base, num_classes_transition, num_classes_phases, device, dataloader):
    model.eval()
    total_val_loss_base = 0
    total_val_loss_transition = 0
    total_val_loss_phase = 0
    acc_base = torchmetrics.Accuracy().to(device)
    f1_base = torchmetrics.F1Score(num_classes=num_classes_base, average='none').to(device)
    confusion_matrix_base = torchmetrics.ConfusionMatrix(num_classes=num_classes_base).to(device)
    acc_transition = torchmetrics.Accuracy().to(device)
    f1_transition = torchmetrics.F1Score(num_classes=num_classes_transition, average='none').to(device)
    confusion_matrix_transition = torchmetrics.ConfusionMatrix(num_classes=num_classes_transition).to(device)
    acc_phase = torchmetrics.Accuracy().to(device)
    f1_phase = torchmetrics.F1Score(num_classes=num_classes_phases, average='none').to(device)
    confusion_matrix_phase = torchmetrics.ConfusionMatrix(num_classes=num_classes_phases).to(device)

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluation in the validation set:"):
        # for data in dataloader:
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

            total_val_loss_base = total_val_loss_base + loss_base.mean().item()
            total_val_loss_transition = total_val_loss_transition + loss_transition.mean().item()
            total_val_loss_phase = total_val_loss_phase + loss_phases.mean().item()

            acc_base, f1_base, confusion_matrix_base = utils.update_metrics(acc_base, f1_base, confusion_matrix_base, outputs_base, labels_base, position_weights_base)
            acc_transition, f1_transition, confusion_matrix_transition = utils.update_metrics(acc_transition, f1_transition, confusion_matrix_transition, outputs_transition, labels_transition, position_weights_transition)
            acc_phase, f1_phase, confusion_matrix_phase = utils.update_metrics(acc_phase, f1_phase, confusion_matrix_phase, outputs_phase, labels_phases, position_weights_base)

            del outputs_base, outputs_transition, outputs_phase, loss_base, loss_transition, loss_phases
            gc.collect()

    final_acc_base = acc_base.compute()
    final_f1_base = f1_base.compute()
    final_confusion_matrix_base = confusion_matrix_base.compute()
    final_acc_transition = acc_transition.compute()
    final_f1_transition = f1_transition.compute()
    final_confusion_matrix_transition = confusion_matrix_transition.compute()
    final_acc_phase = acc_phase.compute()
    final_f1_phase = f1_phase.compute()
    final_confusion_matrix_phase = confusion_matrix_phase.compute()
    print('----------------------------------------------------------------The performance of evaluation dataset-------------------------------------------------------------------')
    print('------------------The metrics of bases classification--------------------')
    print(f'The loss of base classification on evaluation dataset: {total_val_loss_base/len(dataloader)}')
    print(f"Validation Accuracy: {final_acc_base}")
    print(f"Validation F1 Score: {final_f1_base}")
    print(f"Average validation F1 Score: {torch.mean(final_f1_base)}")
    print('The confusion matrix of base prediction')
    print(final_confusion_matrix_base)

    print('--------------------------------------------------The metrics of transition classification----------------------------------------------------')
    print(f'The loss of transition classification on evaluation dataset: {total_val_loss_transition/len(dataloader)}')
    print(f"Validation Accuracy: {final_acc_transition}")
    print(f"Validation F1 Score: {final_f1_transition}")
    print(f"Average validation F1 Score: {torch.mean(final_f1_transition)}")
    print('The confusion matrix of transition prediction')
    print(final_confusion_matrix_transition)

    print('--------------------------------------------------The metrics of transition classification----------------------------------------------------')
    print(f'The loss of phase classification on evaluation dataset: {total_val_loss_phase / len(dataloader)}')
    print(f"Validation Accuracy: {final_acc_phase}")
    print(f"Validation F1 Score: {final_f1_phase}")
    print(f"Average validation F1 Score: {torch.mean(final_f1_phase)}")
    print('The confusion matrix of transition prediction')
    print(final_confusion_matrix_phase)
    checkpoint_metrics = (torch.mean(final_f1_base) + torch.mean(final_f1_transition) + torch.mean(final_f1_phase)) / 3
    # checkpoint_metrics = torch.mean(final_f1_phase)
    return checkpoint_metrics

