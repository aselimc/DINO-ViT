import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.segmentation import statistics

class ImgRotation:
    """Rotate by one of the given angles."""
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.angles = [0, 90, 180, 270]
        self.n_angles = len(self.angles)

    def __call__(self, x):
        labels = random.sample(range(self.n_angles), k=self.n_samples)
        rotated = [TF.rotate(x, self.angles[l]) for l in labels]
        return rotated, labels


class ApplyAfterRotations:
    """ Apply a transformation to a list of images (e.g. after applying ImgRotation)"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        images, labels = x
        images = [self.transform(i) for i in images]
        return images, labels


class ToTensorAfterRotations:
    """ Transform a list of images to a pytorch tensor (e.g. after applying ImgRotation)"""
    def __call__(self, x):
        images, labels = x
        return [TF.to_tensor(i) for i in images], [torch.tensor(l) for l in labels]


def get_transforms_pretraining(args):
    size = [args.size]*2
    train_transform = Compose([
        Resize(size=(args.size,args.size)),
        RandomCrop(size, pad_if_needed=True),
        ImgRotation(args.num_rotations),
        ApplyAfterRotations(RandomApply([ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5)) if args.jitter=="small" else
        ApplyAfterRotations(RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)),
        ToTensorAfterRotations(),
        ApplyAfterRotations(Normalize(statistics['mean'], statistics['std']))
    ])
    val_transform = Compose([Resize(size=(args.size,args.size)), RandomCrop(size, pad_if_needed=True),
                             ImgRotation(4), ToTensorAfterRotations(),
                             ApplyAfterRotations(Normalize(statistics['mean'], statistics['std']))])
    return train_transform, val_transform


def get_transforms_binary_segmentation(args):
    from PIL import Image
    size = [args.size]*2
    train_transform = Compose([
        Resize(size=(args.size,args.size)),
        # RandomCrop([args.size]*2, pad_if_needed=True),
        # RandomHorizontalFlip(),
        RandomApply([ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
        ToTensor(),
        Normalize(statistics['mean'], statistics['std'])])
    train_transform_mask = Compose([Resize(size=(args.size,args.size), interpolation=Image.NEAREST),
                                    # RandomCrop(size, pad_if_needed=True),
                                    # RandomHorizontalFlip(),
                                    ToTensor()])
    val_transform = Compose([Resize(size=(args.size,args.size)), ToTensor(),
                             Normalize(statistics['mean'], statistics['std'])])
    val_transform_mask = Compose([Resize(size=(args.size,args.size), interpolation=Image.NEAREST), ToTensor()])
    return train_transform, val_transform, train_transform_mask, val_transform_mask

