import copy
import os
import torch

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils

from dataset_manager import setup_datasets


def mapper(dataset_dict):
    # it will be modified by code below
    dataset_dict = copy.deepcopy(dataset_dict)

    # read image and store as torch tensor
    image = detection_utils.read_image(dataset_dict["file_name"], format="L")
    # convert to writable array
    image_shape = image.shape[:2]  # (h, w, c) -> (h, w)

    # transform the image to tensor (c, h, w)
    image_tensor = torch.tensor(image)
    dataset_dict["image"] = torch.as_tensor(image_tensor.permute(2, 0, 1))

    # annotations to detectron2 instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        dataset_dict["annotations"], image_size=image_shape)

    return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=mapper
        )


def train_detector():

    setup_datasets()

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("wideband_train",)
    cfg.DATASETS.TEST = ("wideband_val",)
    cfg.MIN_SIZE_TRAIN = (512,)  # Keep fixed size

    # Model parameters
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 61  # Number of signal classes
    cfg.MODEL.DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [128.0]  # Center data
    cfg.MODEL.PIXEL_STD = [128.0]   # Scale to ~[-1,1]

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_ITERS = 4000

    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = ()
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000
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
