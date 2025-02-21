"""
Configuration for ViTDet Large model.

This module provides configuration builders for the ViTDet-L model, which is
a larger variant of ViTDet-B with:
- Increased embedding dimension (1024 vs 768)
- More transformer layers (24 vs 12)
- More attention heads (16 vs 12)
- Higher drop path rate (0.4 vs 0.1)
- Modified window attention pattern
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


def build_vitdet_l_model_config():
    """Build ViTDet-L model configuration.

    Modifies the base ViTDet-B configuration with Large model specific settings:
    - 1024 embedding dimension
    - 24 transformer layers
    - 16 attention heads
    - 0.4 drop path rate
    - Custom window attention pattern with global attention at layers 5,11,17,23

    Returns:
        LazyConfig: Model configuration for ViTDet-L
    """
    model = build_vitdet_b_model_config()
    model.backbone.net.embed_dim = 1024
    model.backbone.net.depth = 24
    model.backbone.net.num_heads = 16
    model.backbone.net.drop_path_rate = 0.4
    # 5, 11, 17, 23 for global attention
    model.backbone.net.window_block_indexes = (
        list(range(0, 5)) + list(range(6, 11)) +
        list(range(12, 17)) + list(range(18, 23))
    )
    return model


def build_vitdet_l_optimizer_config():
    """Build optimizer configuration for ViTDet-L.

    Modifies the base optimizer config with:
    - Higher learning rate decay (0.8 vs 0.7)
    - Adjusted for 24 layers

    Returns:
        LazyConfig: Optimizer configuration
    """
    optimizer = build_vitdet_b_optimizer_config()
    optimizer.params.lr_factor_func = partial(
        get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
    return optimizer


def build_vitdet_l_training_config(config: Detectron2Config):
    """Build training configuration for ViTDet-L.

    Uses the same training configuration as ViTDet-B.

    Args:
        config: Base configuration parameters

    Returns:
        LazyConfig: Training configuration
    """
    return build_vitdet_b_training_config(config)


def build_vitdet_l_lr_multiplier_config(config: Detectron2Config):
    """Build learning rate multiplier configuration for ViTDet-L.

    Uses the same learning rate schedule as ViTDet-B.

    Args:
        config: Base configuration parameters

    Returns:
        LazyConfig: Learning rate multiplier configuration
    """
    return build_vitdet_b_lr_multiplier_config(config)
