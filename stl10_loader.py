"""STL-10 dataset loader."""

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from .augmentations import SimCLRAugmentation, BYOLAugmentation, EvalAugmentation
import torchvision.transforms as transforms


class STL10DataLoader:
    """
    DataLoader for STL-10 dataset.
    Supports both supervised and self-supervised training.
    """

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 256,
        num_workers: int = 4,
        method: str = "simclr",
    ):
        """
        Initialize STL-10 DataLoader.

        Args:
            root: Root directory for dataset
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            method: "simclr", "byol", or "supervised"
        """
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.method = method

        # Create data directory if it doesn't exist
        os.makedirs(root, exist_ok=True)

    def get_train_loader(self):
        """
        Get training DataLoader with appropriate augmentations.
        For SSL, uses unlabeled data. For supervised, uses labeled training split.

        Returns:
            DataLoader for training
        """
        if self.method in ["simclr", "byol"]:
            # Use unlabeled data for self-supervised learning
            if self.method == "simclr":
                train_transform = SimCLRAugmentation(image_size=96)
            else:  # byol
                train_transform = BYOLAugmentation(image_size=96)

            train_dataset = STL10SSL(
                root=self.root,
                split="unlabeled",
                transform=train_transform,
                download=True,
            )
        else:  # supervised
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ]
            )
            train_dataset = STL10Supervised(
                root=self.root,
                split="train",
                transform=train_transform,
                download=True,
            )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_test_loader(self):
        """
        Get test DataLoader with standard augmentations.

        Returns:
            DataLoader for testing
        """
        test_transform = EvalAugmentation(image_size=96)
        test_dataset = STL10Supervised(
            root=self.root,
            split="test",
            transform=test_transform,
            download=True,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_train_eval_loader(self):
        """
        Get labeled training set with evaluation augmentations (for linear probing).

        Returns:
            DataLoader for evaluation on training set
        """
        eval_transform = EvalAugmentation(image_size=96)
        train_dataset = STL10Supervised(
            root=self.root,
            split="train",
            transform=eval_transform,
            download=True,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class STL10SSL(Dataset):
    """STL-10 dataset wrapper for self-supervised learning."""

    def __init__(
        self,
        root: str,
        split: str = "unlabeled",
        transform=None,
        download: bool = False,
    ):
        """
        Initialize STL10SSL dataset.

        Args:
            root: Root directory
            split: "unlabeled", "train", or "test"
            transform: Augmentation transform
            download: Download dataset if not present
        """
        self.dataset = datasets.STL10(
            root=root, split=split, download=download, transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset.split == "unlabeled":
            image = self.dataset[idx]
        else:
            image, _ = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image


class STL10Supervised(Dataset):
    """STL-10 dataset wrapper for supervised learning."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        download: bool = False,
    ):
        """
        Initialize STL10Supervised dataset.

        Args:
            root: Root directory
            split: "train" or "test"
            transform: Augmentation transform
            download: Download dataset if not present
        """
        self.dataset = datasets.STL10(
            root=root, split=split, download=download, transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
