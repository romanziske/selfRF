from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
import torch

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from selfrf.finetuning.detection.detectron2.config import Detectron2Config


def build_vitdet_b_model_config():
    """Build model configuration."""
    model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

    model.backbone.net.img_size = 512
    model.backbone.square_pad = 512
    model.backbone.net.in_chans = 1  # Single channel input

    # Disable mask prediction in ROI heads
    model.roi_heads.mask_in_features = None
    model.roi_heads.mask_pooler = None
    model.roi_heads.mask_head = None

    model.pixel_mean = [0.0]  # Single channel mean
    model.pixel_std = [1.0]   # Single channel std
    model.input_format = "L"  # Grayscale format

    return model


def build_vitdet_b_training_config(config: Detectron2Config):
    """Build training configuration."""
    train = model_zoo.get_config("common/train.py").train
    train.amp.enabled = torch.cuda.is_available()
    train.ddp.fp16_compression = True
    train.max_iter = config.max_iter
    return train


def build_vitdet_b_lr_multiplier_config(config: Detectron2Config):
    """Build learning rate multiplier configuration.

    Custom learning rate schedule for RF COCO dataset:
    - Warmup for first 1000 iterations
    - Drop LR at 80% and 90% of training
    """
    # Calculate milestones based on percentages
    milestones = [
        int(0.8 * config.max_iter),  # Drop LR at 80% of training
        int(0.9 * config.max_iter),  # Drop LR at 90% of training
    ]

    return L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=milestones,
            num_updates=config.max_iter,
        ),
        warmup_length=1000 / config.max_iter,  # Fixed 1000 iteration warmup
        warmup_factor=0.001,
    )


def build_vitdet_b_optimizer_config():
    """Build optimizer configuration."""
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    optimizer.params.lr_factor_func = partial(
        get_vit_lr_decay_rate,
        num_layers=12,
        lr_decay_rate=0.7
    )
    optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
    return optimizer
