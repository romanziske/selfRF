import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils


def visualize_coco_dataset():
    # Register dataset
    register_coco_instances(
        "wideband_train",
        {},
        "datasets/torchsig_wideband_250_impaired/coco_clean/annotations/instances_train.json",
        "datasets/torchsig_wideband_250_impaired/coco_clean/train"
    )

    # Get metadata
    metadata = MetadataCatalog.get("wideband_train")
    dataset_dicts = DatasetCatalog.get("wideband_train")

    # Visualize 3 random samples
    for d in dataset_dicts:
        img = detection_utils.read_image(d["file_name"], format="L")

        visualizer = Visualizer(img, metadata=metadata,
                                scale=1.0, instance_mode=2)
        vis = visualizer.draw_dataset_dict(d)

        # Show image
        cv2.imshow("COCO Visualization", vis.get_image())
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_coco_dataset()
