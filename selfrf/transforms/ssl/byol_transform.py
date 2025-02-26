from typing import Optional
import numpy as np
from torch import Tensor

from torchsig.utils.types import Signal

from ..extra import torchsig_legacy_transforms as T_LEGACY
from ..extra import MultiViewTransform


class BYOLView1Transform(T_LEGACY.Transform):
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
                 tensor_transform: T_LEGACY.SignalTransform = T_LEGACY.ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            T_LEGACY.RandomApply(T_LEGACY.TimeReversal(), tr_prob),
            T_LEGACY.RandomApply(T_LEGACY.SpectralInversion(), si_prob),
            T_LEGACY.AddNoise((min_snr_db, max_snr_db)),
            # AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = T_LEGACY.Compose(transforms=transforms)

    def __call__(self, signal: Signal) -> Tensor:

        return self.transform(signal)


class BYOLView2Transform(T_LEGACY.Transform):
    def __init__(self,
                 max_time_shift: int = 1000,
                 max_freq_shift: float = 0.15,
                 tr_prob: float = 0.3,
                 si_prob: float = 0.3,
                 min_snr_db: float = -80,
                 max_snr_db: float = -20,
                 min_amplitude_scale: float = -10,
                 max_amplitude_scale: float = 10,
                 max_phase_shift_rad: float = np.pi/8,
                 tensor_transform: T_LEGACY.SignalTransform = T_LEGACY.ComplexTo2D(),
                 ) -> None:
        super().__init__()

        transforms = [
            T_LEGACY.RandomTimeShift((-max_time_shift, max_time_shift)),
            T_LEGACY.RandomFrequencyShift((-max_freq_shift, max_freq_shift)),
            T_LEGACY.RandomApply(T_LEGACY.TimeReversal(), tr_prob),
            T_LEGACY.RandomApply(T_LEGACY.SpectralInversion(), si_prob),
            T_LEGACY.AddNoise((min_snr_db, max_snr_db)),
            # AmplitudeScale((min_amplitude_scale, max_amplitude_scale)),
            T_LEGACY.RandomPhaseShift((0, max_phase_shift_rad)),
            tensor_transform,
        ]

        self.transform = T_LEGACY.Compose(transforms=transforms)

    def __call__(self, signal: Signal) -> Tensor:

        return self.transform(signal)


class BYOLTransform(MultiViewTransform):
    def __init__(
        self,
        view_1_transform: Optional[BYOLView1Transform] = None,
        view_2_transform: Optional[BYOLView2Transform] = None,
        tensor_transform: T_LEGACY.SignalTransform = T_LEGACY.ComplexTo2D(),
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform(
            tensor_transform=tensor_transform,
        )

        view_2_transform = view_2_transform or BYOLView2Transform(
            tensor_transform=tensor_transform,
        )

        super().__init__(transforms=[view_1_transform, view_2_transform])
