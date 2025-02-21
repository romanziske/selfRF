import torch
import argparse
import os

from detectron2.utils.logger import setup_logger

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, print_config
from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.lazy_trainer import do_train_lazy
from selfrf.finetuning.detection.detectron2.trainer import do_train

setup_logger()


def train(config: Detectron2Config):
    """Register datasets with detectron2."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config.root):
        config.root = os.path.abspath(config.root)

    register_rfcoco_dataset(config.root, config.dataset_name, download=True)

    if config.model_type.is_lazy_config:
        do_train_lazy(config)
    else:
        do_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train(config)
