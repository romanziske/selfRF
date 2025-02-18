import torch
import argparse
import os

from detectron2.data import build_detection_train_loader
from detectron2.utils.events import get_event_storage

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, build_detectron2_config, print_config
from selfrf.finetuning.detection.detectron2.debug import HighLossDetector
from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.trainer import SpectrogramTrainer, rfcoco_mapper


def train(config: Detectron2Config):
    """Register datasets with detectron2."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config.root):
        config.root = os.path.abspath(config.root)

    torch.autograd.set_detect_anomaly(True)

    register_rfcoco_dataset(config.root, config.dataset_name, download=True)

    cfg = build_detectron2_config(config)

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create debug directory
    debug_dir = os.path.join(cfg.OUTPUT_DIR, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)

    trainer = SpectrogramTrainer(cfg)

    # Create hook with lower threshold to catch issues earlier
    debug_hook = HighLossDetector(
        model=trainer.model,
        dataloader=build_detection_train_loader(
            cfg,
            mapper=rfcoco_mapper
        ),
        output_dir=".",
        loss_threshold=10,
    )

    # Register as first hook to ensure it runs
    trainer._hooks = [debug_hook] + trainer._hooks
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except Exception as e:
        print(f"⚠️ Training crashed with error: {str(e)}")
        # Force hook to save current state
        if hasattr(debug_hook, 'after_step'):
            try:
                storage = get_event_storage()
                if storage.latest():
                    debug_hook.after_step()
            except Exception as hook_error:
                print(f"Failed to save debug state: {hook_error}")
        raise  # Re-raise the original exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train(config)
