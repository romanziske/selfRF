from pathlib import Path
import cv2
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import detection_utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer


def setup_cfg(weights_path: Path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 61
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [128.0]  # Center data
    cfg.MODEL.PIXEL_STD = [128.0]   # Scale to ~[-1,1]

    return cfg


def evaluate_model(weights_path: Path):
    # Register dataset
    register_coco_instances(
        "wideband_val",
        {},
        "datasets/wideband/coco/annotations/instances_val.json",
        "datasets/wideband/coco/val"
    )

    # Setup config
    cfg = setup_cfg(weights_path)

    # Setup evaluator
    output_dir = Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator(
        "wideband_val",
        tasks=("bbox",),
        distributed=False,
        output_dir=str(output_dir)
    )

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    metadata = MetadataCatalog.get("wideband_val")

    # Run evaluation
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    evaluator.reset()

    val_loader = DatasetCatalog.get("wideband_val")
    for d in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
        # Load image from file path
        img_orig = detection_utils.read_image(d["file_name"], format="L")
        if img_orig is None:
            print(f"Could not load image: {d['file_name']}")
            continue

        # run prediction
        with torch.no_grad():
            # Apply pre-processing to image.
            height, width = img_orig.shape[:2]
            img = torch.as_tensor(img_orig.transpose(2, 0, 1))
            img.to(cfg.MODEL.DEVICE)

            inputs = {"image": img, "height": height, "width": width}

            output = model([inputs])[0]

            if False:
                visualizer = Visualizer(
                    img_orig, metadata=metadata, scale=1.0)  # Original HWC image
                vis = visualizer.draw_instance_predictions(output["instances"])

                # Show image
                cv2.imshow("COCO Visualization", vis.get_image()[:, :, ::-1])
                cv2.waitKey(0)

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

    return results


if __name__ == "__main__":
    weights_path = "output/model_final.pth"
    evaluate_model(weights_path)
