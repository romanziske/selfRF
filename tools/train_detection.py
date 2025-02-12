import copy
import os
import random
from typing import Dict
import cv2
from matplotlib import pyplot as plt
import torch
from pathlib import Path
import numpy as np

from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import detection_utils, build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode

from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.transforms.target_transforms import DescToBBoxCOCO
from torchsig.utils.types import Signal, create_signal_data

from selfrf.data.data_modules import TorchsigWidebandRFCOCODataModule
from selfrf.pretraining.config.base_config import BaseConfig
from selfrf.pretraining.factories.transform_factory import TransformFactory


class_list = torchsig_signals.class_list
dataset_root = Path("./datasets/wideband")

spectrogram_transform = TransformFactory.create_spectrogram_transform(
    BaseConfig(nfft=512))

target_transform = DescToBBoxCOCO(class_list)


def mapper(dataset_dict: Dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    annotations = dataset_dict["annotations"]

    # Load NumPy array instead of image
    iq_data = np.load(dataset_dict["file_name"])

    signal = Signal(
        data=create_signal_data(samples=iq_data),
        metadata=annotations,
    )

    # apply data transform
    transformed_signal = spectrogram_transform(signal)
    # (1, H, W)
    spectrogram_tensor: np.ndarray = transformed_signal["data"]["samples"]
    height, width = spectrogram_tensor.shape[1:]

    dataset_dict["image"] = spectrogram_tensor
    dataset_dict["height"] = height
    dataset_dict["width"] = width

    # apply target transform
    transformed_targets = target_transform(transformed_signal["metadata"])

    labels, boxes = transformed_targets["labels"], transformed_targets["boxes"]
    detectron_annotations = []
    for index, annotation in enumerate(annotations):
        # Get relative coordinates
        x_rel, y_rel, w_rel, h_rel = boxes[index]

        # Convert to absolute pixel coordinates
        x_pix = int(x_rel * width)  # width
        y_pix = int(y_rel * height)  # height
        w_pix = int(w_rel * width)
        h_pix = int(h_rel * height)

        detectron_annotations.append({
            "id": annotation["id"],
            "image_id": annotation["iq_frame_id"],
            "category_id": labels[index],
            "bbox": [x_pix, y_pix, w_pix, h_pix],
            "bbox_mode": BoxMode.XYWH_ABS,  # XYWH format
            "area": float(w_pix * h_pix),
            "iscrowd": 0,
        })

    # overwrite annotations
    dataset_dict["annotations"] = detectron_annotations

    # create instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        detectron_annotations,
        image_size=(height, width)
    )

    return dataset_dict


class Trainer(DefaultTrainer):
    @ classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=mapper
        )


def register_rfcoco_dataset():
    """Register RF COCO format dataset with detectron2"""

    datamodule = TorchsigWidebandRFCOCODataModule(
        root=dataset_root,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train = datamodule.train_dataset
    val = datamodule.val_dataset

    DatasetCatalog.register(
        "rfcoco_train",
        lambda: train.get_image_detection_dicts()
    )

    DatasetCatalog.register(
        "rfcoco_val",
        lambda: val.get_image_detection_dicts()
    )

    MetadataCatalog.get("rfcoco_train").thing_classes = class_list
    MetadataCatalog.get("rfcoco_val").thing_classes = class_list


def visualize_dataset():
    metadata = MetadataCatalog.get("rfcoco_train")
    dataset_dicts = DatasetCatalog.get("rfcoco_train")
    output_dir = Path("datasets/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = random.sample(dataset_dicts, k=10)

    for i, d in enumerate(samples):
        processed_dict = mapper(d)
        img: torch.Tensor = processed_dict["image"]
        # Convert (C,H,W) to (H,W,C)
        img = img.permute(1, 2, 0)

        # Convert [0,1] float to [0,255] uint8 for visualization
        img_uint8 = img * 255

        visualizer = Visualizer(img_uint8,
                                metadata=metadata,
                                scale=1.0,
                                instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_dataset_dict(processed_dict)

        # Save visualization
        vis_img = vis.get_image()
        plt.imsave(output_dir / f"sample_{i}.png", vis_img)
        print(f"Saved sample_{i}.png")

        # Optional: still display
        plt.figure(figsize=(10, 10))
        plt.axis('off')


def train_detector():
    """Register datasets with detectron2."""
    register_rfcoco_dataset()

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("rfcoco_train",)
    cfg.DATASETS.TEST = ("rfcoco_val",)
    cfg.MIN_SIZE_TRAIN = (512,)  # Keep fixed size

    # Model parameters
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 61  # Number of signal classes
    cfg.MODEL.DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [0.0]  # Already normalized
    cfg.MODEL.PIXEL_STD = [1.0]   # No scaling needed

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_ITERS = 4000

    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    train_detector()
