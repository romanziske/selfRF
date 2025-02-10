

from typing import Any, List, Sequence
from copy import deepcopy
import torch
from torchsig.utils.types import SignalMetadata
from torchsig.transforms import Transform, Compose, SignalTransform
from torchsig.transforms.functional import NumericParameter, to_distribution


class ConstantTargetTransform(Transform):

    def __init__(self, constant: Any) -> None:
        super().__init__()
        self.constant = constant

    def __call__(self, metadata: List[SignalMetadata]) -> Any:
        return self.constant
