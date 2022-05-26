import albumentations as A
import sys


def get_aug(params: dict, transform_type: str):
    means = params['means']
    stds = params['stds']
    ch, h, w = params['input shape']

    if transform_type != 'train':
        return A.Compose([
                   A.Resize(width=w, height=h),
                   A.Normalize(mean=means, std=stds)
            ])

    if params['name'] == 'baseline':
        return A.Compose([
            A.SmallestMaxSize(max_size=h + 10),
            A.RandomCrop(width=w, height=h),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.5),
            A.Blur(),
            A.CLAHE(),
            A.Normalize(mean=means, std=stds)
        ])
    elif params['name'] == 'fixed':
        return A.Compose([
            A.Resize(width=w + 20, height=h + 20),
            A.RandomCrop(width=w, height=h),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=4, p=0.5),
            A.Normalize(mean=means, std=stds)
        ])
    else:
        sys.exit('Unsupported aug type: %s !' % params['name'])
