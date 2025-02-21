

from pathlib import Path
import random
from typing import Dict

import matplotlib.pyplot as plt
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from selfrf.finetuning.detection.detectron2.register import register_rfcoco_dataset
from selfrf.finetuning.detection.detectron2.mapper import rfcoco_mapper


def visualize_dataset(n_sampels=100):
    register_rfcoco_dataset(Path(
        "/Users/roman/Repositories/selfRF/datasets/wideband"), "wideband_impaired", download=True)
    metadata = MetadataCatalog.get("rfcoco_train")
    dataset_dicts = DatasetCatalog.get("rfcoco_train")
    output_dir = Path("datasets/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = random.sample(dataset_dicts, k=n_sampels)

    for i, d in enumerate(samples):
        processed_dict = rfcoco_mapper(d)
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


def visualize_sample(img, metadata, predictions: Dict, path: Path):
    visualizer = Visualizer(img,
                            metadata=metadata,
                            scale=1.0,
                            instance_mode=ColorMode.IMAGE_BW)
    vis = visualizer.draw_instance_predictions(predictions)
    vis_img = vis.get_image()
    plt.imshow(vis_img)
    plt.axis('off')
    plt.savefig(path)


if __name__ == "__main__":
    visualize_dataset()
    plt.show()
