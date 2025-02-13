from dataclasses import dataclass
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode


@dataclass
class Detectron2Config:
    """
    Configuration for Detectron2 model training.
    """
    # Path to model weights
    weights_path: str = ""
    # Number of classes (61 signal classes)
    num_classes: int = 61
    # Number of training iterations
    max_iter: int = 10  # _000
    # Number of warmup iterations
    warmup_iters: int = 4000
    # Base learning rate
    base_lr: float = 0.0001
    # Number of images per batch
    ims_per_batch: int = 2
    # Checkpoint period
    checkpoint_period: int = 1000
    # Gradient clipping value
    clip_value: float = 1.0
    # Gradient clipping type
    clip_type: str = "value"


def build_detectron2_config(config: Detectron2Config = Detectron2Config()) -> CfgNode:
    """Get Detectron2 config with custom parameters.

    Args:
        config: Custom configuration parameters

    Returns:
        CfgNode: Detectron2 configuration
    """
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("rfcoco_train",)
    cfg.DATASETS.TEST = ("rfcoco_val",)
    cfg.MIN_SIZE_TRAIN = (512,)  # Keep fixed size

    # Model parameters
    cfg.MODEL.WEIGHTS = config.weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    cfg.MODEL.DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [0.0]  # Already normalized
    cfg.MODEL.PIXEL_STD = [1.0]   # No scaling needed

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.WARMUP_ITERS = config.warmup_iters
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.CHECKPOINT_PERIOD = config.checkpoint_period
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = config.clip_value
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = config.clip_type

    return cfg
