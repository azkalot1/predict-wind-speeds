import albumentations as A
import numpy as np
import torch


def get_valid_transforms(resize_size):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size)
        ],
        p=1.0)


def light_training_transforms(resize_size):
    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
    ])


def medium_training_transforms(resize_size):
    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
    ])


def heavy_training_transforms(resize_size):
    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
                A.NoOp(),
                A.ShiftScaleRotate(),
            ], p=1.0),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
    ])


def get_training_trasnforms(transforms_type, resize_size):
    if transforms_type == 'light':
        return(light_training_transforms(resize_size))
    elif transforms_type == 'medium':
        return(medium_training_transforms(resize_size))
    elif transforms_type == 'heavy':
        return(heavy_training_transforms(resize_size))
    elif transforms_type == 'valid':
        return(get_valid_transforms(resize_size))
    else:
        raise NotImplementedError("Not implemented transformation configuration")


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
