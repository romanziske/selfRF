import argparse
from dataclasses import dataclass

from .base_config import BaseConfig, add_base_config_args

DEFAULT_TSNE = True
DEFAULT_KNN = True
DEFAULT_N_NEIGHBORS = 10
DEFAULT_EVALUATION_PATH = './evaluation'


@dataclass
class EvaluationConfig(BaseConfig):
    model_path: str = '.'
    tsne: bool = DEFAULT_TSNE
    knn: bool = DEFAULT_KNN
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    evaluation_path: str = DEFAULT_EVALUATION_PATH


def add_evaluation_config_args(parser: argparse.ArgumentParser) -> None:
    add_base_config_args(parser)
    parser.add_argument(
        '--model-path',
        type=str,
        default='.',
    )
    parser.add_argument(
        '--tsne',
        type=lambda x: x.lower() == 'true',
        default=True,
    )
    parser.add_argument(
        '--knn',
        type=lambda x: x.lower() == 'true',
        default=True,
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--export-path',
        type=str,
        default='./evaluation',
    )


def parse_evaluation_config() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluation Config")
    add_evaluation_config_args(parser)
    args = parser.parse_args()
    return EvaluationConfig(**vars(args))
