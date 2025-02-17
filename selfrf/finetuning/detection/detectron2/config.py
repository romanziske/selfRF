from dataclasses import fields, dataclass
import argparse
import torch

from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo

# Default values as constants
DEFAULT_DOWNLOAD = False
DEFAULT_WEIGHTS_PATH = ""
DEFAULT_NUM_CLASSES = 61
DEFAULT_MAX_ITER = 90_000
DEFAULT_WARMUP_ITERS = 4000
DEFAULT_BASE_LR = 0.0001
DEFAULT_IMS_PER_BATCH = 8
DEFAULT_CHECKPOINT_PERIOD = 1000
DEFAULT_CLIP_VALUE = 1.0
DEFAULT_CLIP_TYPE = "norm"


@dataclass
class Detectron2Config:
    """Configuration for Detectron2 model training."""
    download: bool = DEFAULT_DOWNLOAD
    weights_path: str = DEFAULT_WEIGHTS_PATH
    num_classes: int = DEFAULT_NUM_CLASSES
    max_iter: int = DEFAULT_MAX_ITER
    warmup_iters: int = DEFAULT_WARMUP_ITERS
    base_lr: float = DEFAULT_BASE_LR
    ims_per_batch: int = DEFAULT_IMS_PER_BATCH
    checkpoint_period: int = DEFAULT_CHECKPOINT_PERIOD
    clip_value: float = DEFAULT_CLIP_VALUE
    clip_type: str = DEFAULT_CLIP_TYPE


def add_detectron2_config_args(parser: argparse.ArgumentParser) -> None:
    """Add Detectron2 specific arguments to parser."""
    parser.add_argument(
        '--download',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_DOWNLOAD,
        help='Download dataset model weights'
    )
    parser.add_argument(
        '--weights-path',
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help='Path to pretrained model weights'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help='Number of classes to detect'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=DEFAULT_MAX_ITER,
        help='Maximum number of training iterations'
    )
    parser.add_argument(
        '--warmup-iters',
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=DEFAULT_BASE_LR,
        help='Base learning rate'
    )
    parser.add_argument(
        '--ims-per-batch',
        type=int,
        default=DEFAULT_IMS_PER_BATCH,
        help='Images per batch'
    )
    parser.add_argument(
        '--checkpoint-period',
        type=int,
        default=DEFAULT_CHECKPOINT_PERIOD,
        help='Checkpoint save frequency'
    )
    parser.add_argument(
        '--clip-value',
        type=float,
        default=DEFAULT_CLIP_VALUE,
        help='Gradient clipping value'
    )
    parser.add_argument(
        '--clip-type',
        type=str,
        choices=['value', 'norm'],
        default=DEFAULT_CLIP_TYPE,
        help='Gradient clipping type'
    )


def print_config(config: Detectron2Config) -> None:
    """Print config in a structured format"""
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'

    print(f"\n{BLUE}Detectron2 Configuration:{ENDC}")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {CYAN}{field.name}:{ENDC} {GREEN}{value}{ENDC}")


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
    # Disable all augmentations
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)  # Only one size, no range
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.INPUT.RANDOM_FLIP = "none"

    # Model parameters
    cfg.MODEL.WEIGHTS = config.weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
