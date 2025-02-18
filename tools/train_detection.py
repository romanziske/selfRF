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

    # Store last processed batch for debugging
    last_batch = None

    def store_batch(x, **kwargs):
        nonlocal last_batch
        last_batch = x
        return x

    train_loader = build_detection_train_loader(
        cfg,
        mapper=rfcoco_mapper,
        aspect_ratio_grouping=False,  # Disable for easier debugging
    )
    train_loader._map_worker = lambda x: store_batch(rfcoco_mapper(x))

    # Create debug hook
    debug_hook = HighLossDetector(
        model=trainer.model,
        dataloader=train_loader,
        output_dir=debug_dir,
        loss_threshold=10,
    )

    trainer._hooks = [debug_hook] + trainer._hooks
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        print(f"⚠️ Training crashed with error: {str(e)}")

        if last_batch is not None:
            print("\nDebug information for last processed batch:")
            for item in last_batch:
                if "file_name" in item:
                    print(f"Image path: {item['file_name']}")
                if "image" in item:
                    print(f"Image shape: {item['image'].shape}")
                if "instances" in item:
                    print(f"Number of instances: {len(item['instances'])}")
                print("---")
        else:
            print("No batch information available")

        raise  # Re-raise the exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train(config)
