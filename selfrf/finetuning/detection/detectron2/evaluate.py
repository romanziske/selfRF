from pathlib import Path
from typing import List
import torch
from tqdm import tqdm

from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config
from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.trainer import rfcoco_mapper
from selfrf.finetuning.detection.detectron2.visualizer import visualize_sample


def evaluate_model(root: Path,
                   weights_path: Path,
                   visualize: bool = False):
    # Register dataset
    register_rfcoco_dataset(root=root)
    # Setup config
    cfg = build_detectron2_config(Detectron2Config(weights_path=weights_path))

    # Setup evaluator
    output_dir = Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator(
        "rfcoco_val",
        tasks=("bbox",),
        distributed=False,
        output_dir=str(output_dir)
    )

    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        metadata = MetadataCatalog.get("rfcoco_val")

    # Run evaluation
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    evaluator.reset()

    val_loader = DatasetCatalog.get("rfcoco_val")
    for d in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
        # Load image from file path
        # Use mapper to process data
        processed_dict = rfcoco_mapper(d)
        img: torch.Tensor = processed_dict["image"]

        # run prediction
        with torch.no_grad():
            # Apply pre-processing to image.
            height, width = img.shape[:2]
            img.to(cfg.MODEL.DEVICE)

            inputs = {"image": img, "height": height, "width": width}

            output = model([inputs])[0]

            if visualize:
                visualize_sample(img,
                                 metadata,
                                 output["instances"],
                                 vis_dir / f"{d['image_id']}.png")

        evaluator.process(
            inputs=[{
                "file_name": d["file_name"],
                "image_id": d["image_id"],
                "height": d["height"],
                "width": d["width"]
            }],
            outputs=[{"instances": output["instances"]}]
        )

    # Compute metrics
    results = evaluator.evaluate()
    print(results)
    return results
