from utils.mixup.mixup import CutMix
from utils.mixup.mixup import MixUp


def get_transform(tType):
    if tType == 'CutMix':
        return CutMix()
    elif tType == 'MixUp':
        return MixUp()
    else:
        return None
