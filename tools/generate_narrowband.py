from dotenv import load_dotenv
from minio import Minio
from typing import List
import click
import os

from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from torchsig.utils.dataset import collate_fn
from torchsig.datasets.signal_classes import torchsig_signals

from selfrf.data.data_generators import DatasetLoader, DatasetCreator, RFCOCODatasetWriter
from selfrf.data.storage import MinioBackend, FilesystemBackend
load_dotenv()

modulation_list = torchsig_signals.class_list


def get_backend(to_bucket: bool = False):
    if not to_bucket:
        return FilesystemBackend()

    return MinioBackend(
        client=Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            cert_check=os.getenv("MINIO_SECURE", "true").lower() == "true",
        ),
        bucket=os.getenv("MINIO_BUCKET")
    )


def generate(root: str, configs: List[conf.NarrowbandConfig], num_workers: int, num_samples_override: int, num_iq_samples_override: int = -1, batch_size: int = 32, to_bucket: bool = False):
    for config in configs:
        num_samples = config.num_samples if num_samples_override <= 0 else num_samples_override
        num_iq_samples = config.num_iq_samples if num_iq_samples_override <= 0 else num_iq_samples_override

        print(
            f'batch_size -> {batch_size} num_samples -> {num_samples}, config -> {config}')

        split = "train" if "train" in config.name else "val"
        dataset_name = config.name.removesuffix(f"_{split}")

        if split == "val" and num_samples_override >= 0:
            # adjust for validation set
            num_samples = int(num_samples_override * 0.1)

        if num_samples < len(modulation_list):
            raise ValueError(
                f"Number of samples {num_samples} must be greater than number of modulation classes {len(modulation_list)}")

        ds = ModulationsDataset(
            level=config.level,
            num_samples=num_samples,
            num_iq_samples=num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )

        dataset_loader = DatasetLoader(
            ds,
            seed=12345678,
            collate_fn=collate_fn,
            num_workers=num_workers,
            batch_size=batch_size
        )

        creator = DatasetCreator(
            path=os.path.join(
                root, config.name
            ),
            loader=dataset_loader,
            writer=RFCOCODatasetWriter(
                path=os.path.join(root, dataset_name),
                storage=get_backend(to_bucket=to_bucket),
                split=split,
            )
        )

        creator.create()


@click.command()
@click.option("--root", default="narrowband", help="Path to generate narrowband datasets")
@click.option("--all", is_flag=True, default=False, help="Generate all versions of narrowband dataset.")
@click.option("--qa", is_flag=True, default=False, help="Generate only QA versions of TorchSigNarrowband dataset.")
@click.option("--num-iq-samples", "num_iq_samples", default=-1, help="Override number of iq samples in wideband dataset.")
@click.option("--batch-size", "batch_size", default=32, help="Override batch size.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
@click.option("--impaired", is_flag=True, default=False, help="Generate impaired dataset. Ignored if --all=True (default)")
@ click.option("--to-bucket", is_flag=True, default=False, help="Upload to Minio bucket.")
def main(root: str, all: bool, qa: bool, impaired: bool, num_workers: int, num_samples: int, num_iq_samples: int, batch_size: int, to_bucket: bool):
    os.makedirs(root, exist_ok=True)

    configs = [
        conf.NarrowbandCleanTrainConfig,
        conf.NarrowbandCleanValConfig,
        conf.NarrowbandImpairedTrainConfig,
        conf.NarrowbandImpairedValConfig,
        conf.NarrowbandCleanTrainQAConfig,
        conf.NarrowbandCleanValQAConfig,
        conf.NarrowbandImpairedTrainQAConfig,
        conf.NarrowbandImpairedValQAConfig,
    ]

    impaired_configs = []
    impaired_configs.extend(configs[2:])
    impaired_configs.extend(configs[-2:])
    if all:
        generate(root, configs, num_workers,
                 num_samples, num_iq_samples, batch_size, to_bucket)

    elif qa:
        generate(root, configs[-4:], num_workers,
                 num_samples, num_iq_samples, batch_size, to_bucket)

    elif impaired:
        generate(root, impaired_configs, num_workers,
                 num_samples, num_iq_samples, batch_size, to_bucket)

    else:
        generate(root, configs[:2], num_workers,
                 num_samples, num_iq_samples, batch_size, to_bucket)


if __name__ == "__main__":
    main()
