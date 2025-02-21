"""
Configuration for ViTDet Huge model.

This module provides configuration builders for the ViTDet-H model, which is
the largest variant of ViTDet with:
- Increased embedding dimension (1280 vs 1024/768)
- More transformer layers (32 vs 24/12)
- 16 attention heads
- Higher drop path rate (0.5 vs 0.4/0.1)
- Modified window attention pattern
- Custom learning rate decay (0.9 vs 0.8/0.7)
"""

from functools import partial
from selfrf.finetuning.detection.detectron2.config import Detectron2Config

from .mask_rcnn_vitdet_b import (
    build_vitdet_b_lr_multiplier_config,
    build_vitdet_b_model_config,
    build_vitdet_b_training_config,
    build_vitdet_b_optimizer_config,
    get_vit_lr_decay_rate,
)


def build_vitdet_h_model_config():
    """Build ViTDet-H model configuration.

    Modifies the base ViTDet-B configuration with Huge model specific settings:
    - 1280 embedding dimension
    - 32 transformer layers
    - 16 attention heads
    - 0.5 drop path rate
    - Custom window attention pattern with global attention at layers 7,15,23,31
    - Initialized from MAE pretrained weights

    Returns:
        LazyConfig: Model configuration for ViTDet-H
    """
    model = build_vitdet_b_model_config()

    # Architecture settings
    model.backbone.net.embed_dim = 1280
    model.backbone.net.depth = 32
    model.backbone.net.num_heads = 16
    model.backbone.net.drop_path_rate = 0.5

    # Window attention pattern
    model.backbone.net.window_block_indexes = (
        list(range(0, 7)) + list(range(8, 15)) +
        list(range(16, 23)) + list(range(24, 31))
    )

    return model


def build_vitdet_h_optimizer_config():
    """Build optimizer configuration for ViTDet-H.

    Modifies the base optimizer config with:
    - Higher learning rate decay (0.9)
    - Adjusted for 32 layers
    - No weight decay normalization

    Returns:
        LazyConfig: Optimizer configuration
    """
    optimizer = build_vitdet_b_optimizer_config()
    optimizer.params.lr_factor_func = partial(
        get_vit_lr_decay_rate,
        lr_decay_rate=0.9,
        num_layers=32
    )
    optimizer.params.overrides = {}
    optimizer.params.weight_decay_norm = None
    return optimizer


def build_vitdet_h_training_config(config: Detectron2Config):
    """Build training configuration for ViTDet-H.

   Uses the same training configuration as ViTDet-B.
    Args:
        config: Base configuration parameters

    Returns:
        LazyConfig: Training configuration
    """
    return build_vitdet_b_training_config(config)


def build_vitdet_h_lr_multiplier_config(config: Detectron2Config):
    """Build learning rate multiplier configuration for ViTDet-H.

    Uses the same training configuration as ViTDet-B.

    Args:
        config: Base configuration parameters

    Returns:
        LazyConfig: Learning rate multiplier configuration
    """
    return build_vitdet_b_lr_multiplier_config(config)
