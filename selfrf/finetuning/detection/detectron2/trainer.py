import os
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config
from selfrf.finetuning.detection.detectron2.mapper import rf_coco_evaluation_mapper, rfcoco_mapper


class SpectrogramTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=rfcoco_mapper
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """ Build evaluator for validation dataset """
        return COCOEvaluator(
            dataset_name,
            output_dir=cfg.OUTPUT_DIR / Path("eval"),
            tasks=("bbox",),
            distributed=False
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """ Build test loader using the same mapper as training """
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=rf_coco_evaluation_mapper
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """ Run evaluation during training """
        return super().test(cfg, model, evaluators)


def do_train(config: Detectron2Config):

    cfg = build_detectron2_config(config)

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = SpectrogramTrainer(cfg)

    trainer.resume_or_load(resume=False)

    trainer.train()
