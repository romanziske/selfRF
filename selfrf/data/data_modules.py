from typing import Callable, Optional, Dict
from pathlib import Path
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from torchsig.transforms import Transform, Identity
from torchsig.utils.dataset import collate_fn as collate_fn_default


from selfrf.data.datasets import RFCOCODataset


class RFCOCODataModule(pl.LightningDataModule):
    """Base DataModule for RFCOCO datasets"""

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        transform: Optional[Transform] = None,
        target_transform: Optional[Transform] = None,
        collate_fn: Callable = collate_fn_default,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform or Identity()
        self.target_transform = target_transform or Identity()
        self.collate_fn = collate_fn

        self.train_dataset = None  # Dataset: train dataset
        self.val_dataset = None  # Dataset: validation dataset

        self.data_path = None  # str: Path to downloaded dataset in root
        self.train_path = None  # str: Path to train dataset
        self.val_path = None  # str: Path to validation dataset

    def setup(self, stage: Optional[str] = None):
        """Set up datasets, self.train and self.val

        Args:
            stage (str): PyTorch Lightning trainer stage - fit, test, predict.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("setup not implemented")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class TorchsigNarrowbandRFCOCODataModule(RFCOCODataModule):

    def prepare_data(self):
        # download, split, etc...
        print("NarrowbandRFCOCODataModule prepare_data")

    def setup(self, stage=None):

        self.train_dataset = RFCOCODataset(
            root=self.root / "narrowband_clean",
            split="train",
            transform=self.transform,
            target_transform=self.target_transform,
        )

        self.val_dataset = RFCOCODataset(
            root=self.root / "narrowband_clean",
            split="val",
            transform=self.transform,
            target_transform=self.target_transform,
        )


class TorchsigWidebandRFCOCODataModule(RFCOCODataModule):

    def prepare_data(self):
        # download, split, etc...
        print("WidebandRFCOCODataModule prepare_data")

    def setup(self, stage=None):

        self.train_dataset = RFCOCODataset(
            root=self.root / "wideband_clean",
            split="train",
            transform=self.transform,
            target_transform=self.target_transform,
        )

        self.val_dataset = RFCOCODataset(
            root=self.root / "wideband_clean",
            split="val",
            transform=self.transform,
            target_transform=self.target_transform,
        )
