from torch import nn
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

__all__ = ["build_resnet2d"]


class SelectStage(nn.Module):
    """Selects features from a specific ResNet stage."""

    def __init__(self, stage: str = "res5"):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]


def build_detectron2_resnet(
    input_channels: int,
    n_features: int,
    version: str = "50",
) -> nn.Module:
    """Build ResNet backbone using Detectron2."""

    # Initialize config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        f"COCO-Detection/faster_rcnn_R_{version}_FPN_3x.yaml"
    ))

    # Configure model
    cfg.MODEL.WEIGHTS = ""  # No pretrained weights
    cfg.MODEL.PIXEL_MEAN = [128.0] * input_channels  # Zero mean per channel
    cfg.MODEL.PIXEL_STD = [128.0] * input_channels   # Unit std per channel
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.DEVICE = "cpu"

    # Build model
    det_model = build_model(cfg)

    # Create backbone with pooling just like in timm
    return nn.Sequential(
        det_model.backbone.bottom_up,
        SelectStage("res5"),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2048, n_features)
    )


def build_resnet2d(
    input_channels: int,
    n_features: int,
    version: str = "50",
):
    """Constructs and returns a version of the ResNet model.
    Args:

        input_channels (int):
            Number of input channels; should be 2 for complex spectrograms

        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        version (str):
            Specifies the version of resnet to use, e.g., '18', '34' or '50'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """

    return build_detectron2_resnet(
        input_channels=input_channels,
        n_features=n_features,
        version=version
    )
