
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog

from selfrf.data.data_modules import TorchsigWidebandRFCOCODataModule


def register_rfcoco_dataset(root: Path):
    """Register RF COCO format dataset with detectron2"""

    datamodule = TorchsigWidebandRFCOCODataModule(
        root=root,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    train = datamodule.train_dataset
    val = datamodule.val_dataset

    DatasetCatalog.register(
        "rfcoco_train",
        train.get_image_detection_dicts
    )

    DatasetCatalog.register(
        "rfcoco_val",
        val.get_evaluation_dicts
    )

    MetadataCatalog.get("rfcoco_train").thing_classes = train.class_list
    MetadataCatalog.get("rfcoco_val").thing_classes = val.class_list
