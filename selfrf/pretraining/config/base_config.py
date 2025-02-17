from dataclasses import dataclass, fields
import argparse
from typing import Optional

import torch

from selfrf.pretraining.utils.enums import BackboneType, DatasetType

# Default values as constants
DEFAULT_DATASET = DatasetType.TORCHSIG_NARROWBAND
DEFAULT_ROOT = './datasets'
DEFAULT_DOWNLOAD = False
DEFAULT_FAMILY = False
DEFAULT_SPECTROGRAM = False
DEFAULT_NFFT = 512
DEFAULT_NOVERLAP = 0
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_BACKBONE = BackboneType.RESNET50
DEFAULT_EMBEDDING_DIM = 2048
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TO_FLOAT_32 = False


@dataclass
class BaseConfig:
    dataset: DatasetType = DEFAULT_DATASET
    root: str = DEFAULT_ROOT
    download: bool = False
    family: bool = DEFAULT_FAMILY

    _num_iq_samples: Optional[int] = None  # Private field for storage

    @property
    def num_iq_samples(self) -> int:
        """Get number of IQ samples based on dataset type."""
        if self._num_iq_samples is not None:
            return self._num_iq_samples
        # Default values optimized for each dataset type
        if self.dataset == DatasetType.TORCHSIG_WIDEBAND:
            return 262144  # samples for wideband
        elif self.dataset == DatasetType.TORCHSIG_NARROWBAND:
            return 4096  # samples for narrowband
        else:
            return 4096  # Default fallback

    @num_iq_samples.setter
    def num_iq_samples(self, value: int):
        """Set custom number of IQ samples."""
        self._num_iq_samples = value

    spectrogram: bool = DEFAULT_SPECTROGRAM
    nfft: int = DEFAULT_NFFT
    noverlap: int = DEFAULT_NOVERLAP

    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS

    backbone: BackboneType = DEFAULT_BACKBONE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM

    device: torch.device = DEFAULT_DEVICE
    to_float_32: bool = DEFAULT_TO_FLOAT_32


def add_base_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--dataset',
        type=lambda x: DatasetType(x),
        choices=list(DatasetType),
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        '--root',
        type=str,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        '--download',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_DOWNLOAD,
    )
    parser.add_argument(
        '--family',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_FAMILY,
    )
    parser.add_argument(
        '--num-iq-samples',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--spectrogram',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_SPECTROGRAM,
    )
    parser.add_argument(
        '--nfft',
        type=int,
        default=DEFAULT_NFFT,
    )
    parser.add_argument(
        '--noverlap',
        type=int,
        default=DEFAULT_NOVERLAP,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
    )
    parser.add_argument(
        '--backbone',
        type=lambda x: BackboneType(x),
        choices=list(BackboneType),
        default=DEFAULT_BACKBONE,
    )
    parser.add_argument(
        '--device',
        type=lambda x: torch.device(x),
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        '--to-float-32',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_TO_FLOAT_32,
    )


def print_config(config: BaseConfig) -> None:
    """Print config in a structured format"""
    # ANSI color codes
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'

    print(f"\n{BLUE}Configuration:{ENDC}")

    # Get fields directly from config instance
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {CYAN}{field.name}:{ENDC} {GREEN}{value}{ENDC}")
