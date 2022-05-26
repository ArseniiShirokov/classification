from omegaconf import DictConfig
import torch.nn as nn
import torch
import sys


class LabelSmoothingCE(nn.Module):
    def __init__(self, ignore_index=-1, weights=None, ls=0.1):
        super(LabelSmoothingCE, self).__init__()
        self.ignore_index = ignore_index
        if weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=ls)
        else:
            self.ce = nn.CrossEntropyLoss(label_smoothing=ls)

    def forward(self, logits, labels):
        mask = labels == self.ignore_index
        return self.ce(logits[~mask], labels[~mask])


def get_loss(params: DictConfig, weights: torch.tensor):
    if weights is not None:
        if params['name'] == 'CrossEntropy':
            if not params['label smoothing']:
                loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
            else:
                loss = LabelSmoothingCE(weights=weights, ignore_index=-1, ls=params['smoothing params'])
        else:
            sys.exit('Unsupported loss type: %s !' % params['type'])
    else:
        if params['name'] == 'CrossEntropy':
            loss = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            sys.exit('Unsupported loss type: %s !' % params['name'])
    return loss
