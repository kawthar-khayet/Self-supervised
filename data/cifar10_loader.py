"""CIFAR-10 dataset loader."""

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from .augmentations import SimCLRAugmentation, BYOLAugmentation, EvalAugmentation


class CIFAR10DataLoader:
    """
    DataLoader for CIFAR-10 dataset.
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
        Initialize CIFAR-10 DataLoader.

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

        Returns:
            DataLoader for training
        """
        if self.method == "simclr":
            train_transform = SimCLRAugmentation(image_size=32)
            train_dataset = CIFAR10SSL(
                root=self.root,
                train=True,
                transform=train_transform,
                download=True,
            )
        elif self.method == "byol":
            train_transform = BYOLAugmentation(image_size=32)
            train_dataset = CIFAR10SSL(
                root=self.root,
                train=True,
                transform=train_transform,
                download=True,
            )
        else:  # supervised
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ]
            )
            train_dataset = datasets.CIFAR10(
                root=self.root,
                train=True,
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
        test_transform = EvalAugmentation(image_size=32)
        test_dataset = CIFAR10Supervised(
            root=self.root,
            train=False,
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
        Get training set with evaluation augmentations (for linear probing).

        Returns:
            DataLoader for evaluation on training set
        """
        eval_transform = EvalAugmentation(image_size=32)
        train_dataset = CIFAR10Supervised(
            root=self.root,
            train=True,
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


class CIFAR10SSL(Dataset):
    """CIFAR-10 dataset wrapper for self-supervised learning."""

    def __init__(self, root: str, train: bool = True, transform=None, download: bool = False):
        """
        Initialize CIFAR10SSL dataset.

        Args:
            root: Root directory
            train: Use training set
            transform: Augmentation transform
            download: Download dataset if not present
        """
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image


class CIFAR10Supervised(Dataset):
    """CIFAR-10 dataset wrapper for supervised learning."""

    def __init__(self, root: str, train: bool = True, transform=None, download: bool = False):
        """
        Initialize CIFAR10Supervised dataset.

        Args:
            root: Root directory
            train: Use training set
            transform: Augmentation transform
            download: Download dataset if not present
        """
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
