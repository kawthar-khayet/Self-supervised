"""Augmentation pipelines for self-supervised learning."""

import torch
import torchvision.transforms as transforms
from typing import Tuple


class SimCLRAugmentation:
    """
    Augmentation pipeline for SimCLR.
    Creates two different augmented views of the same image.
    """

    def __init__(self, image_size: int = 32, s: float = 0.5):
        """
        Initialize SimCLR augmentation pipeline.

        Args:
            image_size: Size of input images (32 for CIFAR-10, 96 for STL-10)
            s: Strength of color jittering (0.5 for stronger augmentation)
        """
        self.image_size = image_size
        self.s = s

        # Color jittering
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        )

        # Augmentation pipeline
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size, scale=(0.2, 1.0), interpolation=2
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

    def __call__(self, x):
        """
        Apply two different augmentations to the same image.

        Args:
            x: Input image

        Returns:
            Tuple of two augmented images
        """
        x_i = self.train_transform(x)
        x_j = self.train_transform(x)
        return x_i, x_j


class BYOLAugmentation:
    """
    Augmentation pipeline for BYOL.
    Creates two different augmented views with asymmetric augmentation strength.
    """

    def __init__(self, image_size: int = 32):
        """
        Initialize BYOL augmentation pipeline.

        Args:
            image_size: Size of input images (32 for CIFAR-10, 96 for STL-10)
        """
        self.image_size = image_size

        # Strong augmentation for view 1
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size, scale=(0.2, 1.0), interpolation=2
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * image_size) * 2 + 1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

        # Weak augmentation for view 2
        self.weak_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size, scale=(0.2, 1.0), interpolation=2
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

    def __call__(self, x):
        """
        Apply asymmetric augmentations to the same image.

        Args:
            x: Input image

        Returns:
            Tuple of two augmented images (strong, weak)
        """
        x_1 = self.strong_transform(x)
        x_2 = self.weak_transform(x)
        return x_1, x_2


class EvalAugmentation:
    """Standard augmentation for evaluation (no random crops/flips)."""

    def __init__(self, image_size: int = 32):
        """
        Initialize evaluation augmentation pipeline.

        Args:
            image_size: Size of input images
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

    def __call__(self, x):
        """Apply evaluation augmentation."""
        return self.transform(x)
