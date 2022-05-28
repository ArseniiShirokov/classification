import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.to(pred.device)) + (1 - lam) * criterion(pred, y_b.to(pred.device))


class CutMix:
    def __init__(self, betta=1.0):
        self.betta = betta

    def apply(self, model, images, labels, criterion):
        lam = np.random.beta(self.betta, self.betta)
        rand_index = torch.randperm(images.size()[0]).cuda()
        labels_a = labels
        labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        # compute output
        logits = model(images)
        losses = []
        for i, loss_function in enumerate(criterion):
            losses.append(mixed_criterion(loss_function, logits[i], labels_a[:, i], labels_b[:, i], lam))
        return logits, losses


class MixUp:
    def __init__(self, betta=1.0):
        self.betta = betta

    def apply(self, model, images, labels, criterion):
        lam = np.random.beta(self.betta, self.betta)
        rand_index = torch.randperm(images.size()[0]).cuda()
        labels_a = labels
        labels_b = labels[rand_index]
        images = lam * images + (1 - lam) * images[rand_index, :]
        # compute output
        logits = model(images)
        losses = []
        for i, loss_function in enumerate(criterion):
            losses.append(mixed_criterion(loss_function, logits[i], labels_a[:, i], labels_b[:, i], lam))
        return logits, losses
