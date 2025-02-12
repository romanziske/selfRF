from typing import Literal
import numpy as np
import cv2


def amplitude_scale(tensor: np.ndarray, scale: float) -> np.ndarray:
    """Applies an amplitude scaling to the input tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        scale: (:obj:`float`):
            Scaling factor

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone an amplitude scaling

    """
    return tensor * scale


def spectrogram_image(
    tensor: np.ndarray,
    black_hot: bool = True,
    normalize_max: Literal[1, 255] = 255,
) -> np.ndarray:
    """Convert spectrogram tensor to image.

    Args:
        tensor: Input spectrogram tensor
        black_hot: Use black-hot colormap
        normalize_255: If True normalize to [0,255], else [0,1]

    Returns:
        np.ndarray: Normalized spectrogram image
    """
    spec = 10 * np.log10(tensor + np.finfo(np.float32).tiny)

    # Normalize to target range
    img = cv2.normalize(spec, None, 0, normalize_max, cv2.NORM_MINMAX)

    # invert
    if black_hot:
        if normalize_max == 255:
            # For uint8 images (0-255):
            img = img.astype(np.uint8)  # Convert to uint8 first
            img = cv2.bitwise_not(img)  # Inverts bits: 0->255, 255->0
        else:
            # For float images (0-1):
            img = normalize_max - img  # Simple subtraction: 0->1, 1->0

    return img
