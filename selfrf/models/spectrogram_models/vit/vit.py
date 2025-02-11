
from typing import Dict, Union
import torch
import torch.nn as nn
from detectron2.modeling.backbone.vit import ViT

__all__ = ["build_vit"]


class ViTWrapper(nn.Module):
    """
    A wrapper class for Vision Transformer (ViT) models that adds a classification head.

    This class wraps a Vision Transformer model and adds a classification head consisting of
    Layer Normalization followed by a Linear layer. It's designed to output feature embeddings
    of a specified dimension.

        vit (ViT): A Vision Transformer model instance.
        n_features (int): Number of output features (dimension of the output embedding).
        embed_dim (int): Dimension of the input embedding from the ViT model.

    Attributes:
        vit (ViT): The wrapped Vision Transformer model.
        head (nn.Sequential): Classification head consisting of LayerNorm and Linear layer.

    """

    def __init__(self, vit: ViT, n_features: int, embed_dim: int):
        super().__init__()
        self.vit: ViT = vit
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, n_features).
        """
        # Forward through the backbone.
        features: Union[dict, torch.Tensor] = self.vit(x)
        # If the backbone returns a dictionary, extract the "cls" token.
        if isinstance(features, dict):
            cls_token = features["cls"]
        else:
            cls_token = features  # Assume it's the cls token already.
        return self.head(cls_token)


def build_vit(
    input_channels: int,
    n_features: int,
    version: str = "b",
    img_size: int = 512,
    patch_size: int = 16,
    drop_path_rate: float = 0.1,
    window_size: int = 14,
    mlp_ratio: int = 4,
    window_block_indexes: list = [0, 1, 3, 4, 6, 7, 9, 10],
    use_rel_pos: bool = True,
) -> nn.Module:
    """
    Builds a ViT-based classifier that uses the "cls" token for classification.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB).
        n_features (int): Number of output classes.
        version (str): Model version key ("b", "l", or "h").
        img_size (int): Input image size (default: 512).
        patch_size (int): Patch size (default: 16).
        drop_path_rate (float): Drop path rate (default: 0.1).
        window_size (int): Window size (default: 14).
        mlp_ratio (int): MLP ratio (default: 4).
        window_block_indexes (list): Indexes for window blocks.
        use_rel_pos (bool): Whether to use relative positional embeddings.

    Returns:
        nn.Module: An instance of ViTClassifier.
    """
    model_configs: Dict[str, Dict[str, int]] = {
        "b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "h": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
    }

    if version not in model_configs:
        raise ValueError(
            f"Invalid version '{version}'. Expected one of: {list(model_configs.keys())}")

    config = model_configs[version]

    backbone = ViT(
        in_chans=input_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        drop_path_rate=drop_path_rate,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        window_block_indexes=window_block_indexes,
        use_rel_pos=use_rel_pos,
        out_feature="cls",
    )

    return ViTWrapper(backbone, n_features, config["embed_dim"])
