import io
import os
import cv2
from dotenv import load_dotenv
from minio import Minio
import numpy as np
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog

from selfrf.finetuning.detection.detectron2.trainer import rfcoco_mapper


class HighLossDetector(HookBase):
    def __init__(self, model, dataloader, output_dir, loss_threshold=10.0):
        self.model = model
        self.dataloader = iter(dataloader)  # Iterator over the dataset
        self.output_dir = output_dir
        self.loss_threshold = loss_threshold  # Adjust this threshold if needed
        os.makedirs(self.output_dir, exist_ok=True)

        load_dotenv()
        self.minio = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
        )
        self.bucket = "iqdm-ai"

    def after_step(self):
        storage = get_event_storage()
        # Get the actual loss value from the tuple
        loss = storage.latest()["total_loss"][0] if isinstance(
            storage.latest()["total_loss"], tuple
        ) else storage.latest()["total_loss"]

        if loss > self.loss_threshold:  # If loss is too high
            batch = next(self.dataloader)  # Get the next batch
            self.save_debug_image(batch, loss)

    def save_debug_image(self, batch, loss):
        dataset_dicts = batch  # Detectron2 uses dataset dicts
        for data in dataset_dicts:
            data = rfcoco_mapper(data)  # Map data to model input

            image = data["image"].permute(
                1, 2, 0).cpu().numpy()  # Convert tensor to NumPy
            image = (image * 255).astype(np.uint8)  # Convert to 0-255 range

            # Get bounding boxes and class labels
            metadata = MetadataCatalog.get(
                "rfcoco_train")  # Replace with your dataset
            class_names = metadata.thing_classes if hasattr(
                metadata, "thing_classes") else []
            bboxes = data.get("instances", None)

            if bboxes:
                for i, box in enumerate(bboxes.gt_boxes.tensor.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    label = class_names[bboxes.gt_classes[i]
                                        ] if class_names else f"Class {bboxes.gt_classes[i]}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image_name = f"high_loss_{loss:.2f}.jpg"
            # Convert image to bytes for MinIO upload
            is_success, buffer = cv2.imencode(".jpg", image)
            if is_success:
                file_bytes = io.BytesIO(buffer)

                # Upload to MinIO
                try:
                    self.minio.put_object(
                        self.bucket,
                        f"debug-images/{image_name}",
                        file_bytes,
                        file_bytes.getbuffer().nbytes,
                        content_type="image/jpeg"
                    )
                    print(
                        f"⚠️ High-loss image saved: minio=s3://{self.bucket}/iqdm-ai/debug-images/{image_name}")
                except Exception as e:
                    print(f"Failed to upload to MinIO: {e}")
            else:
                print("Failed to encode image")
