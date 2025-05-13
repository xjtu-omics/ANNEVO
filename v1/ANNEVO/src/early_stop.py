import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoints/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_f1 = -np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, f1_score, model):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def save_checkpoint(self, f1_score, model):
        if self.verbose:
            print(f'F1 score increased ({self.best_f1:.6f} --> {f1_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_f1 = f1_score
