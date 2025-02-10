from typing import Any, Union

import torch

from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.config.training_config import TrainingConfig


def collate_fn(batch: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Custom collate for 2 views"""
    print(batch)

    views, targets = zip(*batch)

    view1s, view2s = zip(*views)

    return (
        torch.stack(view1s),
        torch.stack(view2s)
    ), torch.tensor(targets)


def collate_fn_evaluation(batch):
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return tensors, targets


def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig]):

    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation
    else:
        return collate_fn
