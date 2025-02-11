import json
import os
from typing import Callable, Optional
from pathlib import Path
import lightning.pytorch as pl
from minio import Minio
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_name = "narrowband_impaired"
        self.bucket = "iqdm-ai"
        self.download = True

        self.minio = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            cert_check=os.getenv("MINIO_CERT_CHECK", "true").lower() == "true",
        )

    def prepare_data(self):
        """Download RF COCO dataset from Minio"""
        print("Downloading dataset from Minio")

        if self.download:
            dataset_path = self.root / self.dataset_name
            print(f"Downloading dataset to {dataset_path}")
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Download annotation files
            for split in ["train", "val"]:
                annot_file = f"instances_{split}.json"
                annot_file_local_path = dataset_path / "annotations" / annot_file
                annot_file_local_path.parent.mkdir(parents=True, exist_ok=True)

                self.minio.fget_object(
                    self.bucket,
                    f"{dataset_path}/annotations/{annot_file}",
                    str(annot_file_local_path)
                )

                # Load annotations to get IQ frame filenames
                annotations = json.loads(annot_file_local_path.read_text())

                # Download IQ data files
                for frame in annotations["iq_frames"]:
                    iq_local_path = dataset_path / \
                        split / str(frame["file_name"])
                    iq_local_path.parent.mkdir(parents=True, exist_ok=True)

                    self.minio.fget_object(
                        self.bucket,
                        f"{dataset_path}/{split}/{frame['file_name']}",
                        str(iq_local_path)
                    )
        else:
            raise NotImplementedError(
                "prepare_data not implemented for ahoc dataset generation")

    def setup(self, stage=None):

        if stage == "fit":

            self.train_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )

            self.val_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            raise NotImplementedError(
                "setup not implemented for stage: {stage}")


class TorchsigWidebandRFCOCODataModule(RFCOCODataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_name = "wideband_impaired"
        self.bucket = "iqdm-ai"
        self.download = True

        self.minio = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            cert_check=os.getenv("MINIO_CERT_CHECK", "true").lower() == "true",
        )

    def prepare_data(self):
        """Download RF COCO dataset from Minio"""

        if self.download:
            dataset_path = self.root / self.dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Download annotation files
            for split in ["train", "val"]:
                annot_file = f"instances_{split}.json"
                annot_file_local_path = dataset_path / "annotations" / annot_file
                annot_file_local_path.parent.mkdir(parents=True, exist_ok=True)

                self.minio.fget_object(
                    self.bucket,
                    f"{self.dataset_name}/annotations/{annot_file}",
                    str(annot_file_local_path)
                )

                # Load annotations to get IQ frame filenames
                annotations = json.loads(annot_file_local_path.read_text())

                # Download IQ data files
                for frame in annotations["iq_frames"]:
                    iq_local_path = dataset_path / str(frame["file_name"])
                    iq_local_path.parent.mkdir(parents=True, exist_ok=True)

                    self.minio.fget_object(
                        self.bucket,
                        f"{self.dataset_name}/{frame['file_name']}",
                        str(iq_local_path)
                    )
        else:
            raise NotImplementedError(
                "prepare_data not implemented for ahoc dataset generation")

    def setup(self, stage: str = None):

        if stage == "fit":

            self.train_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )

            self.val_dataset = RFCOCODataset(
                root=self.root / self.dataset_name,
                split="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )

        else:
            raise NotImplementedError(
                "setup not implemented for stage: {stage}")
