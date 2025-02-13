import os
from pathlib import Path


from selfrf.finetuning.detection.detectron2.config import build_detectron2_config
from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.trainer import SpectrogramTrainer

dataset_root = Path("./datasets/wideband")


def train():
    """Register datasets with detectron2."""
    register_rfcoco_dataset(dataset_root)

    cfg = build_detectron2_config()

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = SpectrogramTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    train()
