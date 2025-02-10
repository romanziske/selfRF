import numpy as np

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