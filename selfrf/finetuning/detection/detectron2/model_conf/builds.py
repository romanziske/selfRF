"""Module for mapping model types to their build functions."""

from typing import Tuple, Callable

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, ModelType

# Import all build functions
from .mask_rcnn_vitdet_b import (
    build_vitdet_b_model_config,
    build_vitdet_b_training_config,
    build_vitdet_b_lr_multiplier_config,
    build_vitdet_b_optimizer_config,
)
from .maks_rcnn_vitdet_l import (
    build_vitdet_l_model_config,
    build_vitdet_l_optimizer_config,
    build_vitdet_l_lr_multiplier_config,
    build_vitdet_l_training_config,
)
from .maks_rcnn_vitdet_h import (
    build_vitdet_h_model_config,
    build_vitdet_h_optimizer_config,
    build_vitdet_h_lr_multiplier_config,
    build_vitdet_h_training_config,
)


def get_build_functions(config: Detectron2Config) -> Tuple[Callable, Callable, Callable, Callable]:
    """Get build functions corresponding to the model type.

    Args:
        config: Configuration containing model type

    Returns:
        Tuple of (model, optimizer, lr_multiplier, training) build functions

    Raises:
        ValueError: If model type is not supported
    """
    build_functions = {
        ModelType.VITDET_VIT_B: (
            build_vitdet_b_model_config,
            build_vitdet_b_training_config,
            build_vitdet_b_lr_multiplier_config,
            build_vitdet_b_optimizer_config,
        ),
        ModelType.VITDET_VIT_L: (
            build_vitdet_l_model_config,
            build_vitdet_l_optimizer_config,
            build_vitdet_l_lr_multiplier_config,
            build_vitdet_l_training_config,
        ),
        ModelType.VITDET_VIT_H: (
            build_vitdet_h_model_config,
            build_vitdet_h_optimizer_config,
            build_vitdet_h_lr_multiplier_config,
            build_vitdet_h_training_config,
        ),
    }

    if config.model_type not in build_functions:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    return build_functions[config.model_type]
