from typing import Tuple, List
from torch import nn
import torch
import torch.nn.functional as F


def simple_forward(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, criterion: List[nn.Module]) \
        -> Tuple[list, list]:
    logits = model(data)
    loss = []
    for i, loss_function in enumerate(criterion):
        loss.append(loss_function(logits[i], labels[:, i].to(logits[i].device)))
    return logits, loss


def jsd_forward(model: nn.Module, data: torch.Tensor, labels: torch.Tensor,
                criterion: List[nn.Module], with_jsd_att: List[int]) \
        -> Tuple[list, list]:
    data_all = torch.chunk(data, data.size(1), dim=1)
    data_all = torch.squeeze(torch.cat(data_all, dim=0))
    logits_all = model(data_all)
    # CE for no aug images
    loss = []
    for i, loss_function in enumerate(criterion):
        if i in with_jsd_att:
            logits, _, _ = torch.chunk(
                logits_all[i], data.size(1)
            )
        else:
            _, logits, _ = torch.chunk(
                logits_all[i], data.size(1)
            )
        loss_val = loss_function(logits, labels[:, i].to(logits.device))
        loss.append(loss_val)
    # add JSD loss
    list_logits = []
    for i, loss_function in enumerate(criterion):
        logits, logits_aug1, logits_aug2 = torch.chunk(
            logits_all[i], data.size(1)
        )
        list_logits.append(logits)
        if i not in with_jsd_att:
            continue
        # Filter mask for samples with label -1
        mask = labels[:, i] == -1
        # continue if all labeles = -1
        if labels[~mask].size(0) == 0:
            continue

        # Compute probability
        p = F.softmax(logits, dim=1)
        p_aug1 = F.softmax(logits_aug1, dim=1)
        p_aug2 = F.softmax(logits_aug2, dim=1)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        jsd_loss_val = (F.kl_div(p_mixture[~mask], p[~mask], reduction='batchmean') +
                        F.kl_div(p_mixture[~mask], p_aug1[~mask], reduction='batchmean') +
                        F.kl_div(p_mixture[~mask], p_aug2[~mask], reduction='batchmean')) / 3.
        loss[i] += 12 * jsd_loss_val

    logits = list_logits
    return logits, loss
