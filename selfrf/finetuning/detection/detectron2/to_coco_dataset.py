import os
import json
from typing import Literal, Tuple
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.transforms import Spectrogram, Normalize
from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.transforms.target_transforms import DescToBBoxCOCO
from torchsig.transforms.transforms import Compose, SpectrogramImage


def collate_fn(batch):
    data, labels = zip(*batch)
    data_tensor = np.stack(data)
    label_list = np.stack(labels)
    return data_tensor, label_list


class WidebandToSpectrogramCOCO:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.coco_dir = self.root_dir / "coco"
        self.coco_dir.mkdir(parents=True, exist_ok=True)

        self.class_list = torchsig_signals.class_list
        self.family_dict = torchsig_signals.family_dict

        transform = Compose(
            [
                Spectrogram(
                    nperseg=512,
                    noverlap=0,
                    nfft=512,
                    mode='psd',
                ),
                Normalize(norm=np.inf, flatten=True),
                SpectrogramImage(),
            ]
        )

        target_transform = DescToBBoxCOCO(self.class_list)

        self.datamodule = WidebandDataModule(
            root=self.root_dir,
            qa=False,
            impaired=True,
            transform=transform,
            target_transform=target_transform,
            batch_size=8,
            num_workers=4,
            num_classes=len(self.class_list),
            collate_fn=collate_fn
        )
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

    def convert(self, split: Literal["train", "val"]) -> Path:

        print(f"Converting {split} dataset to COCO format")
        # Setup paths
        images_dir, coco_json_path = self._setup_coco_dir(split)

        # Initialize COCO format
        coco_dict = self._create_coco_dict()

        # Get dataloader
        dataloader = (self.datamodule.train_dataloader()
                      if split == "train"
                      else self.datamodule.val_dataloader())

        ann_id = 0
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
            for img_idx, (img, target) in enumerate(zip(data, targets)):
                img_id = batch_idx * dataloader.batch_size + img_idx

                # Save image
                file_name = f"{split}_{img_id:06d}.png"
                img_path = os.path.join(images_dir, file_name)
                cv2.imwrite(img_path, img)

                # Add image info
                coco_dict["images"].append({
                    "id": img_id,
                    "file_name": file_name,
                    "height": img.shape[-2],
                    "width": img.shape[-1],
                })

                # [N, 4] tensor in format [x, y, width, height]
                boxes = target["boxes"]
                for box_idx, box in enumerate(boxes):
                    x, y, w, h = box.tolist()

                    # convert relative to pixel values
                    x_pix = int(x * img.shape[-1])
                    y_pix = int(y * img.shape[-2])
                    w_pix = int(w * img.shape[-1])
                    h_pix = int(h * img.shape[-2])

                    coco_dict["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": target["labels"][box_idx].item() + 1,
                        "bbox": [x_pix, y_pix, w_pix, h_pix],
                        "area": float(w_pix * h_pix),
                        "iscrowd": 0,
                    })
                    ann_id += 1

        # Save COCO json
        with open(coco_json_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)

        return coco_json_path

    def _setup_coco_dir(self, split: Literal["train", "val"]) -> Tuple[Path, Path]:
        # createa image directory
        images_dir = self.coco_dir / split
        images_dir.mkdir(parents=True, exist_ok=True)

        # create annotation directory
        annotation_dir = self.coco_dir / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)

        # create path tho coco json file
        coco_json_path = annotation_dir / f"instances_{split}.json"

        return images_dir, coco_json_path

    def _create_coco_dict(self) -> dict:
        return {
            "categories": self._create_categories(),
            "images": [],
            "annotations": [],
        }

    def _create_categories(self):
        """Create COCO categories with supercategories"""
        return [
            {
                "id": idx + 1,
                "name": class_name,
                "supercategory": self.family_dict[class_name]
            }
            for idx, class_name in enumerate(self.class_list)
        ]


def convert_dataset():
    converter = WidebandToSpectrogramCOCO("datasets/wideband")

    converter.convert("train")
    converter.convert("val")


if __name__ == "__main__":
    convert_dataset()
