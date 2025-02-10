import os
from pathlib import Path
from typing import List, Dict
from minio import Minio
from detectron2.data.datasets import register_coco_instances

from to_coco_dataset import WidebandToSpectrogramCOCO


class DatasetManager:
    def __init__(
        self,
        minio_config: Dict[str, str],
        dataset_name: str = "torchsig_wideband_250_impaired",
    ):

        self.client = Minio(
            minio_config['url'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            cert_check=True
        )

        self.dataset_name = dataset_name
        self.cwd_path = Path(__file__).resolve().parent
        self.datasets_path = self.cwd_path / \
            "datasets" / dataset_name
        self.minio_prefix = Path("datasets") / dataset_name
        self.bucket_name = "iqdm-ai"

    def _needs_download(self, files: List[str]) -> bool:
        """Check if any files need to be downloaded."""
        for file in files:
            file_path = self.datasets_path / file
            if not file_path.exists():
                return True
        return False

    def _download_files(self, files: List[str]) -> None:
        """Download dataset files from MinIO."""
        if not self._needs_download(files):
            print("All files already present. Skipping download.")
            return

        for file in files:
            try:
                dest_path = self.datasets_path / file
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                self.client.fget_object(
                    bucket_name=self.bucket_name,
                    object_name=str(self.minio_prefix /
                                    file).replace("\\", "/"),
                    file_path=str(dest_path)
                )
                print(f"Downloaded {dest_path}")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                raise

    def _get_coco_paths(self) -> Dict[str, Path]:
        """Get COCO dataset paths."""
        return {
            'train_data': self.datasets_path / "coco_" / self.dataset_name / "/train",
            'train_json': self.datasets_path / "coco_" / self.dataset_name / "/coco/annotations/instances_train.json",
            'val_data': self.datasets_path / "coco_" / self.dataset_name / "/val",
            'val_json': self.datasets_path / "coco_" / self.dataset_name / "annotations/instances_val.json"
        }

    def _convert_to_coco(self) -> None:
        """Convert downloaded dataset to COCO format."""
        converter = WidebandToSpectrogramCOCO(root_dir=str(self.datasets_path))
        for split in ["train", "val"]:
            converter.convert(split)

    def _register_datasets(self, paths: Dict[str, Path]) -> None:
        """Register datasets with detectron2."""
        register_coco_instances(
            "wideband_train",
            {},
            str(paths['train_json']),
            str(paths['train_data'])
        )
        register_coco_instances(
            "wideband_val",
            {},
            str(paths['val_json']),
            str(paths['val_data'])
        )

    def setup(self) -> None:
        """Main setup function."""
        files = [
            "wideband_impaired_train/lock.mdb",
            "wideband_impaired_train/data.mdb",
            "wideband_impaired_val/lock.mdb",
            "wideband_impaired_val/data.mdb"
        ]

        self._download_files(files)
        self._convert_to_coco()
        paths = self._get_coco_paths()
        self._register_datasets(paths)


def setup_datasets():
    """Entry point for dataset setup."""
    minio_config = {
        'url': os.environ['MINIO_URL'],
        'access_key': os.environ['MINIO_ACCESS_KEY'],
        'secret_key': os.environ['MINIO_SECRET_ACCESS_KEY']
    }

    manager = DatasetManager(minio_config)
    manager.setup()
