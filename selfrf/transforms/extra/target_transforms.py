from typing import Any, List
from torchsig.utils.types import SignalMetadata
from torchsig.transforms import Transform


class ConstantTargetTransform(Transform):

    def __init__(self, constant: Any) -> None:
        super().__init__()
        self.constant = constant

    def __call__(self, metadata: List[SignalMetadata]) -> Any:
        return self.constant
