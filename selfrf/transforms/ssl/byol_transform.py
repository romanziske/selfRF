import numpy as np
from torch import Tensor
from typing import Optional

from torchsig import transforms as T


from ..extra import AmplitudeScale, MultiViewTransform


class BYOLView1Transform(T.Transform):
    def __init__(self,
                 max_time_shift: int = 500,
                 max_freq_shift: float = 0.15,
                 tr_prob: float = 0.5,
                 si_prob: float = 0.5,
                 min_snr_db: float = -80,
                 max_snr_db: float = -20,
                 min_amplitude_scale: float = -6,
                 max_amplitude_scale: float = 6,
                 max_phase_shift_rad: float = np.pi/4,
                 tensor_transform: T.SignalTransform = T.ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T.RandomTimeShift((-max_time_shift, max_time_shift)),
            T.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            T.RandomApply(T.TimeReversal(), tr_prob),
            T.RandomApply(T.SpectralInversion(), si_prob),
            T.AddNoise((min_snr_db, max_snr_db)),
            AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            T.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = T.Compose(transforms=transforms)

    def __call__(self, data: np.ndarray[np.complex64]) -> Tensor:
        return self.transform(data)


class BYOLView2Transform(T.Transform):
    def __init__(self,
                 max_time_shift: int = 100_000,
                 max_freq_shift: float = 0.25,
                 tr_prob: float = 0.3,
                 si_prob: float = 0.3,
                 min_snr_db: float = -80,
                 max_snr_db: float = -20,
                 min_amplitude_scale: float = 9999,
                 max_amplitude_scale: float = 10000,
                 max_phase_shift_rad: float = np.pi/8,
                 tensor_transform: T.SignalTransform = T.ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T.RandomTimeShift((-max_time_shift, max_time_shift)),
            T.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            T.RandomApply(T.TimeReversal(), tr_prob),
            T.RandomApply(T.SpectralInversion(), si_prob),
            T.AddNoise((min_snr_db, max_snr_db)),
            AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            T.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = T.Compose(transforms=transforms)

    def __call__(self, data: np.ndarray[np.complex64]) -> Tensor:
        return self.transform(data)


class BYOLTransform(MultiViewTransform):
    def __init__(
        self,
        view_1_transform: Optional[BYOLView1Transform] = None,
        view_2_transform: Optional[BYOLView2Transform] = None,
        tensor_transform: T.SignalTransform = T.ComplexTo2D(),
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform(
            tensor_transform=tensor_transform)
        view_2_transform = view_2_transform or BYOLView2Transform(
            tensor_transform=tensor_transform)
        super().__init__(transforms=[view_1_transform, view_2_transform])
