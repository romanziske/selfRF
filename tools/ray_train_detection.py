import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from train_detection import train_detector


def train_on_ray():

    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_detector,
        run_config=RunConfig(
            name="detectron2_training",
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=False
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    train_on_ray()
