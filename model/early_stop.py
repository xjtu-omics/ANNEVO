import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.best_loss = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score == np.Inf:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss
