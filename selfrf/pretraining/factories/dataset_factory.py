from typing import Dict, Type

from selfrf.pretraining.config import BaseConfig
from selfrf.data.data_modules import (
    RFCOCODataModule,
    TorchsigNarrowbandRFCOCODataModule,
    TorchsigWidebandRFCOCODataModule
)
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform


class DatasetFactory:
    _dataset_registry: Dict[DatasetType, Type[RFCOCODataModule]] = {
        DatasetType.TORCHSIG_NARROWBAND: TorchsigNarrowbandRFCOCODataModule,
        DatasetType.TORCHSIG_WIDEBAND: TorchsigWidebandRFCOCODataModule
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> RFCOCODataModule:
        """Create dataset from config"""
        dataset_type = DatasetType(config.dataset)
        dataset_class = cls._dataset_registry[dataset_type]

        return dataset_class(
            root=config.root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transform=build_transform(config),
            target_transform=build_target_transform(config),
            collate_fn=build_collate_fn(config)
        )


def build_dataset(config: BaseConfig) -> RFCOCODataModule:
    return DatasetFactory.create_dataset(config)
