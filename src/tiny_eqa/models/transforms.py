from torchvision import transforms


def transform_imagenet(resize: tuple[int, int]=None, interpolation=transforms.InterpolationMode.BICUBIC):
    """
    Returns transform that normalizes an image with ImageNet mean and std and optionally resizes it.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=interpolation) if resize else lambda x: x,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ]
    )