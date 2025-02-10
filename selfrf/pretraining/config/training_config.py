import argparse
from dataclasses import dataclass

from selfrf.pretraining.utils.enums import SSLModelType
from .base_config import BaseConfig, add_base_config_args

DEFUALT_ONLINE_LINEAR_EVAL = False
DEFAULT_SSL_MODEL = SSLModelType.BYOL
DEFAULT_TRAINING_PATH = './train'
DEFAULT_NUM_EPOCHS = 100


@dataclass
class TrainingConfig(BaseConfig):
    online_linear_eval: bool = DEFUALT_ONLINE_LINEAR_EVAL
    ssl_model: SSLModelType = DEFAULT_SSL_MODEL
    training_path: str = DEFAULT_TRAINING_PATH
    num_epochs: int = DEFAULT_NUM_EPOCHS


def add_training_config_args(parser: argparse.ArgumentParser) -> None:
    add_base_config_args(parser)
    parser.add_argument(
        '--online-linear-eval',
        type=lambda x: x.lower() == 'true',
        default=DEFUALT_ONLINE_LINEAR_EVAL
    )
    parser.add_argument(
        '--ssl-model',
        type=lambda x: SSLModelType(x),
        choices=list(SSLModelType),
        default=DEFAULT_SSL_MODEL
    )
    parser.add_argument(
        '--training-path',
        type=str,
        default=DEFAULT_TRAINING_PATH
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=DEFAULT_NUM_EPOCHS
    )


def parse_training_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Training Config")
    add_training_config_args(parser)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))
