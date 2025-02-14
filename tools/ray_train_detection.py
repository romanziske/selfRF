import os
import pyarrow
import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from train_detection import train


def train_on_ray():

    ray.init()

    
    fs = pyarrow.fs.S3FileSystem(
        endpoint_override=os.environ['MINIO_ENDPOINT'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train,
        run_config=RunConfig(
            name="detectron2_training",
            storage_filesystem=fs,
            storage_path="iqdm-ai/training",
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    train_on_ray()
