import torch
import argparse
import os
from pathlib import Path

from detectron2.data import build_detection_train_loader

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, build_detectron2_config, print_config
from selfrf.finetuning.detection.detectron2.debug import HighLossDetector
from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.trainer import SpectrogramTrainer

dataset_root = Path("./datasets/wideband")


def train(config: Detectron2Config):
    """Register datasets with detectron2."""

    torch.autograd.set_detect_anomaly(True)

    register_rfcoco_dataset(dataset_root, "wideband_impaired", download=True)

    cfg = build_detectron2_config(config)

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = SpectrogramTrainer(cfg)
    trainer.register_hooks([HighLossDetector(
        trainer.model, build_detection_train_loader(cfg), "debug_images")])
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(
        weights_path=args.weights_path,
        num_classes=args.num_classes,
        max_iter=args.max_iter,
        warmup_iters=args.warmup_iters,
        base_lr=args.base_lr,
        ims_per_batch=args.ims_per_batch,
        checkpoint_period=args.checkpoint_period,
        clip_value=args.clip_value,
        clip_type=args.clip_type
    )

    print_config(config)
    train(config)
