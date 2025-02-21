
from enum import Enum, unique
from typing import NamedTuple, Optional
from dataclasses import fields, dataclass
import argparse
import torch

from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg, CfgNode


from typing import Optional, NamedTuple
from enum import Enum


class ModelConfig(NamedTuple):
    """Model configuration with an optional config path and unique identifier.

    Attributes:
        path: Path to config file. Only required for non-lazy configs.
        is_lazy: Indicates if model uses LazyConfig system.
        identifier: Unique identifier to differentiate configs.
    """
    path: Optional[str] = None  # Optional for lazy configs
    is_lazy: bool = False
    identifier: str = ""


class ModelType(Enum):
    """Supported model architectures with their config paths."""
    FASTER_RCNN_R50_FPN = ModelConfig(
        path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        is_lazy=False,
        identifier="faster_rcnn_r50_fpn"
    )
    VITDET_VIT_B = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_b"
    )
    VITDET_VIT_L = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_l"
    )
    VITDET_VIT_H = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_h"
    )

    @property
    def config_path(self) -> str:
        """Get config path if available.

        Returns:
            str: Path to config file

        Raises:
            ValueError: If trying to access path for lazy config
        """
        if self.is_lazy_config:
            raise ValueError(
                f"Config path not available for lazy config: {self.name}"
            )
        return self.value.path

    @property
    def is_lazy_config(self) -> bool:
        return self.value.is_lazy

    @classmethod
    def from_string(cls, name: str) -> 'ModelType':
        """Get ModelType enum from string.

        Args:
            name: String representation of model type (e.g., 'vitdet-vit-l')

        Returns:
            ModelType: Corresponding enum value

        Raises:
            ValueError: If model type string is not recognized
        """
        try:
            # Convert name to uppercase and replace hyphens with underscores
            enum_name = name.upper().replace('-', '_')
            model_type = cls[enum_name]
            return model_type
        except KeyError:
            # Show available model types in error message
            valid_types = [str(t) for t in cls]
            raise ValueError(
                f"Unknown model type: '{name}'. "
                f"Valid types are: {', '.join(valid_types)}"
            )


# Default values as constants
DEFAULT_MODEL_TYPE = ModelType.FASTER_RCNN_R50_FPN
DEFAULT_DOWNLOAD = False
DEFAULT_DATASET = "wideband_impaired"
DEFAULT_WEIGHTS_PATH = ""
DEFAULT_NUM_CLASSES = 61
DEFAULT_MAX_ITER = 90_000
DEFAULT_BASE_LR = 0.0001
DEFAULT_IMS_PER_BATCH = 8
DEFAULT_CHECKPOINT_PERIOD = 1000


@dataclass
class Detectron2Config:
    """Configuration for Detectron2 model training."""
    root: str = ""
    model_type: ModelType = DEFAULT_MODEL_TYPE
    dataset_name: str = DEFAULT_DATASET
    download: bool = DEFAULT_DOWNLOAD
    weights_path: str = DEFAULT_WEIGHTS_PATH
    num_classes: int = DEFAULT_NUM_CLASSES
    max_iter: int = DEFAULT_MAX_ITER
    base_lr: float = DEFAULT_BASE_LR
    ims_per_batch: int = DEFAULT_IMS_PER_BATCH
    checkpoint_period: int = DEFAULT_CHECKPOINT_PERIOD


def add_detectron2_config_args(parser: argparse.ArgumentParser) -> None:
    """Add Detectron2 specific arguments to parser."""
    parser.add_argument(
        '--root',
        type=str,
        default='',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--model-type',
        type=ModelType.from_string,
        choices=list(ModelType),
        default=DEFAULT_MODEL_TYPE,
        help='Model architecture type (e.g., vitdet-vit-l, vitdet-vit-b)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=DEFAULT_DATASET,
        help='Name of dataset'
    )
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
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = config.checkpoint_period

    return cfg
