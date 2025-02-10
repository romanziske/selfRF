from dataclasses import dataclass
from typing import Dict, Type, Callable
import torch

from selfrf.pretraining.config import TrainingConfig, BaseConfig
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.models.iq_models import build_resnet1d
from selfrf.models.spectrogram_models import build_resnet2d, build_vit
from selfrf.models.ssl_models import BYOL
from selfrf.pretraining.utils.enums import BackboneType, SSLModelType


@dataclass(frozen=True)  # makes the dataclass immutable
class BackboneConfig:
    backbone_type: BackboneType
    is_spectrogram: bool


class ModelFactory:
    _backbone_registry: Dict[BackboneConfig, Callable] = {
        BackboneConfig(BackboneType.RESNET50, False): lambda **kwargs: build_resnet1d(version="50", input_channels=2, ** kwargs),
        BackboneConfig(BackboneType.RESNET50, True): lambda **kwargs: build_resnet2d(version="50", input_channels=1, ** kwargs),
        BackboneConfig(BackboneType.VIT_B, True): lambda **kwargs: build_vit(version="b", input_channels=1, ** kwargs),
    }

    _ssl_registry: Dict[SSLModelType, Type] = {
        SSLModelType.BYOL: BYOL,
    }

    @classmethod
    def create_backbone(cls, config: BaseConfig) -> torch.nn.Module:
        """Create backbone from config"""
        backbone_config = BackboneConfig(
            backbone_type=BackboneType(config.backbone),
            is_spectrogram=config.spectrogram
        )
        builder = cls._backbone_registry[backbone_config]
        return builder(
            n_features=config.embedding_dim
        )

    @classmethod
    def create_ssl_model(cls, config: TrainingConfig) -> torch.nn.Module:
        """Create SSL model from config"""
        backbone = cls.create_backbone(config)
        ssl_type = SSLModelType(config.ssl_model)
        ssl_model = cls._ssl_registry[ssl_type]
        return ssl_model(
            backbone=backbone,
            num_ftrs=config.embedding_dim,
            batch_size_per_device=config.batch_size,
            use_online_linear_eval=config.online_linear_eval,
            num_classes=len(get_class_list(config))
        )


def build_backbone(config: BaseConfig) -> torch.nn.Module:
    return ModelFactory.create_backbone(config)


def build_ssl_model(config: TrainingConfig) -> torch.nn.Module:
    return ModelFactory.create_ssl_model(config)
