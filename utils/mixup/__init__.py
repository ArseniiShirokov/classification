from utils.mixup.mixup import CutMix
from utils.mixup.mixup import MixUp


def get_transform(tType, model, criterion):
    if tType == 'CutMix':
        return CutMix(model, criterion)
    elif tType == 'MixUp':
        return MixUp(model, criterion)
    else:
        return None
