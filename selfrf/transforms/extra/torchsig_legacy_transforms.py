"""TorchSig legacy Transforms from v0.6.0
"""

from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import torch
from scipy import signal as sp
from copy import deepcopy
import numpy as np

from torchsig.utils.types import Signal
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import *
from torchsig.utils.dsp import MAX_SIGNAL_UPPER_EDGE_FREQ, MAX_SIGNAL_LOWER_EDGE_FREQ
from torchsig.transforms import Transform
from torchsig.transforms import functional as F

import selfrf.transforms.extra.torchsig_legacy_functional as F_LEGACY
from .torchsig_legacy_functional import (
    FloatParameter,
    IntParameter,
    NumericParameter,
    to_distribution,
    uniform_continuous_distribution,
)

__all__ = [
    "Transform",
    "Compose",
    "MultiViewTransform",
    "Identity",
    "Lambda",
    "FixedRandom",
    "RandomApply",
    "SignalTransform",
    "Concatenate",
    "TargetConcatenate",
    "RandAugment",
    "RandChoice",
    "Normalize",
    "RandomResample",
    "TargetSNR",
    "AddNoise",
    "ImpulseInterferer",
    "RandomPhaseShift",
    "InterleaveComplex",
    "ToTensor",
    "Real",
    "Imag",
    "ComplexMagnitude",
    "WrappedPhase",
    "DiscreteFourierTransform",
    "ChannelConcatIQDFT",
    "ContinuousWavelet",
    "ReshapeTransform",
    "RandomTimeShift",
    "TimeCrop",
    "TimeReversal",
    "AmplitudeReversal",
    "AmplitudeScale",
    "RandomFrequencyShift",
    "RandomDelayedFrequencyShift",
    "LocalOscillatorDrift",
    "GainDrift",
    "AutomaticGainControl",
    "IQImbalance",
    "RollOff",
    "SpectralInversion",
    "Clip",
    "RandomConvolve",
    "DatasetBasebandMixUp",
    "DatasetBasebandCutMix",
    "DatasetWidebandCutMix",
    "DatasetWidebandMixUp",
    "SpectrogramRandomResizeCrop",
    "SpectrogramPatchShuffle",
    "SpectrogramTranslation",
    "SpectrogramMosaicDownsample",
    "SpectrogramImage",
]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class Identity(Transform):
    """Just passes the data -- surprisingly useful in pipelines

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Identity()

    """

    def __init__(self, **kwargs) -> None:
        super(Identity, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        return data


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Lambda(lambda x: x**2)  # A transform that squares all inputs.

    """

    def __init__(self, func: Callable, **kwargs) -> None:
        super(Lambda, self).__init__(**kwargs)
        self.func = func

    def __call__(self, data: Any) -> Any:
        return self.func(data)


class FixedRandom(Transform):
    """Restricts a randomized transform to apply only a fixed set of seeds.
    For example, this could be used to add noise randomly from among 1000
    possible sets of noise or add fading from 1000 possible channels.

    Args:
        transform (:obj:`Callable`):
            transform to be called

        num_seeds (:obj:`int`):
            number of possible random seeds to use

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.FixedRandom(ST.AddNoise(), num_seeds=10)

    """

    def __init__(self, transform: Transform, num_seeds: int, **kwargs) -> None:
        super(FixedRandom, self).__init__(**kwargs)
        self.transform = transform
        self.num_seeds = num_seeds
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transform={}, ".format(str(transform))
            + "num_seeds={}".format(num_seeds)
            + ")"
        )

    def __call__(self, data: Any) -> Any:
        seed = self.random_generator.choice(self.num_seeds)
        orig_state = (
            np.random.get_state()
            # we do not want to somehow fix other random number generation processes.
        )
        np.random.seed(seed)
        data = self.transform(data)
        # return numpy back to its previous state
        np.random.set_state(orig_state)
        return data


class RandomApply(Transform):
    """Randomly applies a set of transforms with probability p

    Args:
        transform (``Transform`` objects):
            transform to randomly apply

        probability (:obj:`float`):
            In [0, 1.0], the probability with which to apply a transform

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.RandomApply(ST.AddNoise(noise_power_db=10), probability=.5)  # Add 10dB noise with probability .5

    """

    def __init__(
        self,
        transform: Callable,
        probability: float,
        **kwargs,
    ) -> None:
        super(RandomApply, self).__init__(**kwargs)
        self.transform = transform
        self.probability = probability
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transform={}, ".format(str(transform))
            + "probability={}".format(probability)
            + ")"
        )

    def __call__(self, data: Any) -> Any:
        return self.transform(data) if self.random_generator.random() < self.probability else data


class TargetConcatenate(Transform):
    """Concatenates Target Transforms into a Tuple

    Args:
        transforms (list of ``Transform`` objects):
            List of transforms to concatenate

    """

    def __init__(self, transforms: List[Transform], **kwargs) -> None:
        super(TargetConcatenate, self).__init__(**kwargs)
        self.transforms = transforms
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + ")"
        )

    def __call__(self, target: Any) -> Any:
        return tuple([transform(target) for transform in self.transforms])


class SignalTransform(Transform):
    """An abstract base class which explicitly only operates on Signal data"""

    def __init__(self, **kwargs,) -> None:
        super(SignalTransform, self).__init__(**kwargs)

    def __call__(self, signal: Signal) -> Signal:
        parameters = self.parameters()

        signal = self.convert_to_signal(signal)
        signal = self.transform_data(signal, parameters)
        return self.transform_meta(signal, parameters)

    def convert_to_signal(self, signal: Any) -> Signal:
        if is_signal(signal):
            return signal
        metadata_list = [create_modulated_rf_metadata(
            **vals) for vals in signal["metadata"]]
        return create_signal(
            data=create_signal_data(samples=signal["data"]["samples"]),
            metadata=metadata_list
        )

    def parameters(self) -> tuple:
        return tuple()

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        return signal


class Concatenate(SignalTransform):
    """Inputs a list of SignalTransforms and applies each to the input data
    independently then concatenates the outputs along the specified dimension.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to apply and concatenate.

        concat_dim (:obj:`int`):
            Dimension along which to concatenate the outputs from each
            transform

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = Concatenate([ST.AddNoise(10), ST.DiscreteFourierTransform()], concat_dim=0)

    """

    def __init__(
        self,
        transforms: List[Transform],
        concat_dim: int = 0,
        **kwargs,
    ) -> None:
        super(Concatenate, self).__init__(**kwargs)
        self.transforms = transforms
        self.concat_dim = concat_dim
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "concat_dim={}".format(concat_dim)
            + ")"
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = np.concatenate(
            [
                transform(deepcopy(signal["data"]["samples"]))
                for transform in self.transforms
            ],
            axis=self.concat_dim,
        )
        return signal


class RandAugment(SignalTransform):
    """RandAugment transform loosely based on:
    `"RandAugment: Practical automated data augmentation with a reduced search space" <https://arxiv.org/pdf/1909.13719.pdf>`_.

    Args:
        transforms (list of `Transform` objects):
            List of transforms to choose from

        num_transforms (:obj: `int`):
            Number of transforms to randomly select

        allow_multiple_same (:obj: `bool`):
            Boolean specifying if multiple of the same transforms can be
            selected from the input list. Implemented as the `replace`
            parameter in numpy's random choice method.

    """

    def __init__(
        self,
        transforms: List[Callable],
        num_transforms: int = 2,
        allow_multiple_same: bool = False,
        **kwargs,
    ) -> None:
        super(RandAugment, self).__init__(**kwargs)
        self.transforms = transforms
        self.num_transforms = num_transforms
        self.allow_multiple_same = allow_multiple_same
        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "num_transforms={}, ".format(num_transforms)
            + "allow_multiple_same={}".format(allow_multiple_same)
            + ")"
        )

    def parameters(self) -> tuple:
        return tuple(
            self.random_generator.choice(
                len(self.transforms),
                size=self.num_transforms,
                replace=self.allow_multiple_same,
            )
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        for t in [self.transforms[idx] for idx in params]:
            signal = t(signal)
        return signal


class RandChoice(SignalTransform):
    """RandChoice inputs a list of transforms and their associated
    probabilities. When called, a single transform will be sampled from the
    list using the probabilities provided, and then the selected transform
    will operate on the input data.

    Args:
        transforms (List[SignalTransform]):
            List of transforms to sample from and then apply.

        probabilities (Optional[np.ndarray], optional):
            Probabilities used when sampling the above list of transforms. Defaults to None.

    """

    def __init__(
        self,
        transforms: List[SignalTransform],
        probabilities: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super(RandChoice, self).__init__(**kwargs)
        self.transforms = transforms
        probabilities = probabilities if isinstance(
            probabilities, np.ndarray) else np.array(probabilities)
        self.probabilities = np.ones(len(
            self.transforms)) / len(self.transforms) if probabilities is None else probabilities

        if np.sum(self.probabilities) != 1.0:
            self.probabilities /= np.sum(self.probabilities)

        transform_strings: str = ",".join([str(t) for t in transforms])
        self.string: str = (
            self.__class__.__name__
            + "("
            + "transforms=[{}], ".format(transform_strings)
            + "probabilities=[{}]".format(self.probabilities)
            + ")"
        )

    def parameters(self) -> tuple:
        return tuple(
            (
                self.random_generator.choice(
                    len(self.transforms),
                    p=self.probabilities,
                ),
            )
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return self.transforms[params[0]](signal)


class Normalize(SignalTransform):
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize

        flatten :obj:`flatten`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(
        self,
        norm: Optional[Union[int, float, Literal["fro", "nuc"]]] = 2,
        flatten: bool = False,
    ) -> None:
        super(Normalize, self).__init__()
        self.norm = norm
        self.flatten = flatten
        self.string: str = (
            self.__class__.__name__
            + "("
            + "norm={}, ".format(norm)
            + "flatten={}".format(flatten)
            + ")"
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F.normalize(
            signal["data"]["samples"], self.norm, self.flatten
        )
        return signal


class RandomResample(SignalTransform):
    """Resample using poly-phase rational resampling technique.

    Args:
        rate_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            new_rate = rate_ratio*old_rate

            * If Callable, resamples to new_rate by calling rate_ratio()
            * If int or float, rate_ratio is fixed by value provided
            * If list, rate_ratio is any element in the list
            * If tuple, rate_ratio is in range of (tuple[0], tuple[1])

        num_iq_samples (:obj:`int`):
            Since resampling changes the number of points in a tensor, it is necessary to designate how
            many samples should be returned. In the case more samples are produced, the last num_iq_samples of
            the resampled tensor are returned.  In the case fewer samples are produced, the returned tensor is zero-padded
            to have num_iq_samples.

        keep_samples (:obj:`bool`):
            Despite returning a different number of samples being an issue, return however many samples
            are returned from resampler

    Note:
        When rate_ratio is > 1.0, the resampling algorithm produces more samples than the original tensor.
        When rate_ratio < 1.0, the resampling algorithm produces less samples than the original tensor. Hence,
        it is necessary to specify a number of samples to return from the newly resampled tensor so that there are
        always enough samples to return

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly resample to a new_rate that is between 0.75 and 1.5 times the original rate
        >>> transform = ST.RandomResample((0.75, 1.5), num_iq_samples=4096)
        >>> # Randomly resample to a new_rate that is either 1.5 or 3.0
        >>> transform = ST.RandomResample([1.5, 3.0], num_iq_samples=4096)
        >>> # Resample to a new_rate that is always 1.5
        >>> transform = ST.RandomResample(1.5, num_iq_samples=4096)

    """

    def __init__(
        self,
        rate_ratio: NumericParameter = (1.5, 3.0),
        num_iq_samples: int = 4096,
        keep_samples: bool = False,
        min_time: float = .05,
    ) -> None:
        super(RandomResample, self).__init__()
        self.rate_ratio: Callable = to_distribution(
            rate_ratio, self.random_generator)
        self.num_iq_samples = num_iq_samples
        self.keep_samples = keep_samples
        self.min_time = min_time
        self.string: str = (
            self.__class__.__name__
            + "("
            + "rate_ratio={}, ".format(rate_ratio)
            + "num_iq_samples={}, ".format(num_iq_samples)
            + "keep_samples={}".format(keep_samples)
            + ")"
        )

    def __call__(self, signal: Signal) -> Signal:

        parameters = None

        # check in bounds within 10 tries
        for i in range(10):
            parameters = self.parameters()
            new_rate = self.check_time_freq_bounds(signal, parameters[0])

        # could not fit in 10 tries, don't resample
        new_rate = [1.] if new_rate == -1 else [new_rate]

        signal = self.convert_to_signal(signal)
        signal = self.transform_data(signal, new_rate)

        return self.transform_meta(signal, new_rate)

    def parameters(self) -> tuple:
        return (self.rate_ratio(),)

    def check_bounds(self, meta: dict, try_new_rate: float):
        """Checks single metadata entry is in bounds, returns rate that keeps in bounds.

        Args:
            meta (dict): SignalMetadata entry.
            try_new_rate (float): New resampling rate to check.

        Returns:
            new_rate: Resampling rate that keeps signal in bounds.
        """
        # we assume original signal is within bounds
        assert meta["lower_freq"] is not None
        assert meta["upper_freq"] is not None
        assert meta["bandwidth"] < 1.0

        new_rate = try_new_rate

        test_lf = meta["lower_freq"] / try_new_rate
        test_hf = meta["upper_freq"] / try_new_rate

        if test_lf < MAX_SIGNAL_LOWER_EDGE_FREQ or test_hf > MAX_SIGNAL_UPPER_EDGE_FREQ:  # out of bounds
            new_rate *= 2

        return new_rate

    def check_time_freq_bounds(self, signal: Signal, new_rate: float) -> float:
        """Method checks frequency mins and maxes and adjust the new_rate to ensure
            frequency bounds stay within the (approximately) +-.5 boundary.

        Args:
            signal (Signal): Signal to transform.
            new_rate (float): Possible new_rate to resample Signal.

        Returns:
            float: New rate to resample Signal, within bounds.
        """
        ret_list = []

        for meta in signal["metadata"]:
            new_rate = self.check_bounds(meta, new_rate)
            ret_list.append(new_rate)

        try:
            new_rate = np.max(ret_list)
        except Exception as error:
            for meta in signal['metadata']:
                print(
                    f"{meta['lower_freq']} {meta['upper_freq']} {meta['class_name']}")
            print(error)
            raise ValueError("Unable to run: new_rate = np.max(ret_list)")

        for meta in signal["metadata"]:
            start = meta["start"] * new_rate
            if start > 1 - self.min_time:
                return -1

        return new_rate

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal
        # Apply transform to data
        signal["data"]["samples"] = F_LEGACY.resample(
            signal["data"]["samples"], params[0], self.num_iq_samples, self.keep_samples)

        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        # new_rate = self.check_time_freq_bounds(signal, params[0])
        anti_alias_lpf: bool = False
        for meta in signal["metadata"]:
            meta["num_samples"] *= params[0]
            meta["num_samples"] = int(meta["num_samples"])
            meta["sample_rate"] *= params[0]
            meta["start"] *= params[0]
            meta["stop"] *= params[0]
            meta["stop"] = 1. if meta["stop"] > 1. else meta["stop"]
            meta["duration"] = meta["stop"] - meta["start"]
            # Update frequency descriptions
            # Check freq bounds for cases of partial signals
            meta["lower_freq"] /= params[0]
            meta["upper_freq"] /= params[0]
            meta["center_freq"] /= params[0]
            meta["bandwidth"] /= params[0]
            if is_rf_modulated_metadata(meta):
                meta["samples_per_symbol"] *= params[0]

        # #TODO : re-integrate anti-alias filter.
        # if anti_alias_lpf:
        #     taps = low_pass(cutoff=new_rate * 0.98 / 2, transition_bandwidth=(0.5 - (new_rate * 0.98) / 2) / 4)
        #     convolve_out = sp.convolve(signal["data"]["samples"], taps, mode="full")
        #     lidx = (len(convolve_out) - len(signal["data"]["samples"])) // 2
        #     ridx = lidx + len(signal["data"]["samples"])
        #     signal["data"]["samples"] = convolve_out[lidx:ridx]

        return signal


class TargetSNR(SignalTransform):
    """Adds zero-mean complex additive white Gaussian noise to a provided
    tensor to achieve a target SNR. The provided signal is assumed to be
    entirely the signal of interest. Note that this transform relies on
    information contained within the SignalData object's SignalMetadata. The
    transform also assumes that only one signal is present in the IQ data. If
    multiple signals' SignalMetadatas are detected, the transform will raise a
    warning.

    Args:
        target_snr_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB

            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

        eb_no (:obj:`bool`):
            Defines SNR as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2))*samples_per_symbol/bits_per_symbol.
            Defining SNR this way effectively normalized the noise level with respect to spectral efficiency and
            bandwidth. Normalizing this way is common in comparing systems in terms of power efficiency.
            If True, bits_per_symbol in the the SignalData will be used in the calculation of SNR. To achieve SNR in
            terms of E_b/N_0, samples_per_symbol must also be provided. Defaults to False.

        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB. Defaults to False.

    """

    def __init__(
        self,
        target_snr_db: NumericParameter = (-10, 10),
        eb_no: bool = False,
        **kwargs,
    ) -> None:
        super(TargetSNR, self).__init__(**kwargs)
        self.target_snr = to_distribution(target_snr_db, self.random_generator)
        self.eb_no = eb_no
        self.string = (
            self.__class__.__name__
            + "("
            + "target_snr_db={}, ".format(target_snr_db)
            + "eb_no={}, ".format(eb_no)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.target_snr(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        target_snr_db: float = params[0]
        signal_power_db = 10 * \
            np.log10(np.mean(np.abs(signal["data"]["samples"]) ** 2, axis=0))
        noise_power_db = signal_power_db - target_snr_db

        if not has_modulated_rf_metadata(signal["metadata"]):
            signal["data"]["samples"] = F_LEGACY.awgn(
                signal["data"]["samples"], noise_power_db)
            return signal

        if "ofdm" not in signal["metadata"][0]["class_name"]:
            # EbNo not available for OFDM
            noise_power_db -= (
                10 * np.log10(signal["metadata"][0]["bits_per_symbol"])
                if self.eb_no
                else 0
            )

        if signal["metadata"][0]["samples_per_symbol"] > 0:
            noise_power_db += 10 * \
                np.log10(signal["metadata"][0]["samples_per_symbol"])

        signal["data"]["samples"] = F_LEGACY.awgn(
            signal["data"]["samples"], noise_power_db)

        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        target_snr_db: float = params[0]
        if not has_modulated_rf_metadata(signal["metadata"]):
            return signal

        signal["metadata"][0]["snr"] = target_snr_db
        return signal


class ImpulseInterferer(SignalTransform):
    """Applies an impulse interferer

    Args:
        amp (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling amp()
            * If int or float, amp is fixed at the value provided
            * If list, amp is any element in the list
            * If tuple, amp is in range of (tuple[0], tuple[1])

        pulse_offset (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling phase_offset()
            * If int or float, pulse_offset is fixed at the value provided
            * If list, phase_offset is any element in the list
            * If tuple, phase_offset is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        amp: FloatParameter = (0.1, 100.0),
        pulse_offset: FloatParameter = (0.0, 1),
        **kwargs,
    ) -> None:
        super(ImpulseInterferer, self).__init__(**kwargs)
        self.amp = to_distribution(amp, self.random_generator)
        self.pulse_offset = to_distribution(
            pulse_offset, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "amp={}, ".format(amp)
            + "pulse_offset={}".format(pulse_offset)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.amp(), self.pulse_offset())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        amp, pulse_offset = params
        pulse_offset = np.clip(pulse_offset, 0, 1.0)
        signal["data"]["samples"] = F_LEGACY.impulsive_interference(
            signal["data"]["samples"], amp, pulse_offset
        )
        return signal


class RandomPhaseShift(SignalTransform):
    """Applies a random phase offset to tensor

    Args:
        phase_offset (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling phase_offset()
            * If int or float, phase_offset is fixed at the value provided
            * If list, phase_offset is any element in the list
            * If tuple, phase_offset is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift(uniform_continuous_distribution(-1, 1))
        >>> # Phase Offset from [-pi/2, 0, and pi/2]
        >>> transform = ST.RandomPhaseShift(uniform_discrete_distribution([-.5, 0, .5]))
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift((-1, 1))
        >>> # Phase Offset either -pi/4 or pi/4
        >>> transform = ST.RandomPhaseShift([-.25, .25])
        >>> # Phase Offset is fixed at -pi/2
        >>> transform = ST.RandomPhaseShift(-.5)
    """

    def __init__(
        self,
        phase_offset: FloatParameter = (-1, 1),
        **kwargs,
    ) -> None:
        super(RandomPhaseShift, self).__init__(**kwargs)
        self.phase_offset = to_distribution(
            phase_offset, self.random_generator)
        self.string = self.__class__.__name__ + \
            "(" + "phase_offset={}".format(phase_offset) + ")"

    def parameters(self) -> tuple:
        return (self.phase_offset(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        phase_offset = params[0]
        signal["data"]["samples"] = F.phase_offset(
            signal["data"]["samples"], phase_offset * np.pi
        )

        return signal


class InterleaveComplex(SignalTransform):
    """Converts complex IQ samples to interleaved real and imaginary floating
    point values.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.InterleaveComplex()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.interleave_complex(
            signal["data"]["samples"])
        return signal


class ToTensor(SignalTransform):
    """Converts a numpy array to a PyTorch tensor.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ToTensor()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        # Convert to float32 for MPS compatibility
        signal["data"]["samples"] = torch.from_numpy(
            signal["data"]["samples"].astype(np.float32)
        )
        return signal


class Real(SignalTransform):
    """Takes a vector of complex IQ samples and returns Real portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Real()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.real(signal["data"]["samples"])
        return signal


class Imag(SignalTransform):
    """Takes a vector of complex IQ samples and returns Imaginary portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Imag()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.imag(signal["data"]["samples"])
        return signal


class ComplexMagnitude(SignalTransform):
    """Takes a vector of complex IQ samples and returns the complex magnitude

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ComplexMagnitude()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.complex_magnitude(
            signal["data"]["samples"])
        return signal


class WrappedPhase(SignalTransform):
    """Takes a vector of complex IQ samples and returns wrapped phase (-pi, pi)

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.WrappedPhase()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.wrapped_phase(
            signal["data"]["samples"])
        return signal


class DiscreteFourierTransform(SignalTransform):
    """Calculates DFT using FFT

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.DiscreteFourierTransform()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.discrete_fourier_transform(
            signal["data"]["samples"]
        )
        return signal


class ChannelConcatIQDFT(SignalTransform):
    """Converts the input IQ into 2D tensor of the real & imaginary components
    concatenated in the channel dimension. Next, calculate the DFT using the
    FFT, convert the complex DFT into a 2D tensor of real & imaginary frequency
    components. Finally, stack the 2D IQ and the 2D DFT components in the
    channel dimension.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ChannelConcatIQDFT()

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        dft_data = F_LEGACY.discrete_fourier_transform(
            signal["data"]["samples"])
        iq_data = F.complex_to_2d(signal["data"]["samples"])
        dft_data = F.complex_to_2d(dft_data)
        signal["data"]["samples"] = np.concatenate([iq_data, dft_data], axis=0)
        return signal


class ContinuousWavelet(SignalTransform):
    """Computes the continuous wavelet transform resulting in a Scalogram of
    the complex IQ vector

    Args:
        wavelet (:obj:`str`):
            Name of the mother wavelet.
            If None, wavename = 'mexh'.

        nscales (:obj:`int`):
            Number of scales to use in the Scalogram.
            If None, nscales = 33.

        sample_rate (:obj:`float`):
            Sample rate of the signal.
            If None, fs = 1.0.

    Example:
        >>> import torchsig.transforms as ST
        >>> # ContinuousWavelet SignalTransform using the 'mexh' mother wavelet with 33 scales
        >>> transform = ST.ContinuousWavelet()

    """

    def __init__(
        self, wavelet: str = "mexh", nscales: int = 33, sample_rate: float = 1.0
    ) -> None:
        super(ContinuousWavelet, self).__init__()
        self.wavelet = wavelet
        self.nscales = nscales
        self.sample_rate = sample_rate
        self.string = (
            self.__class__.__name__
            + "("
            + "wavelet={}, ".format(wavelet)
            + "nscales={}, ".format(nscales)
            + "sample_rate={}".format(sample_rate)
            + ")"
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.continuous_wavelet_transform(
            signal["data"]["samples"],
            self.wavelet,
            self.nscales,
            self.sample_rate,
        )
        return signal


class ReshapeTransform(SignalTransform):
    """Reshapes the input data to the specified shape

    Args:
        new_shape (obj:`tuple`):
            The new shape for the input data

    """

    def __init__(self, new_shape: Tuple, **kwargs) -> None:
        super(ReshapeTransform, self).__init__(**kwargs)
        self.new_shape = new_shape
        self.string = (
            self.__class__.__name__ +
            "(" + "new_shape={}".format(new_shape) + ")"
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = signal["data"]["samples"].reshape(
            *self.new_shape)
        return signal


class RandomTimeShift(SignalTransform):
    """Shifts tensor in the time dimension by shift samples. Zero-padding is applied to maintain input size.

    Args:
        shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling shift()
            * If int or float, shift is fixed at the value provided
            * If list, shift is any element in the list
            * If tuple, shift is in range of (tuple[0], tuple[1])

        interp_rate (:obj:`int`):
            Interpolation rate used by internal interpolation filter

        taps_per_arm (:obj:`int`):
            Number of taps per arm used in filter. More is slower, but more accurate.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Shift inputs by range of (-10, 20) samples with uniform distribution
        >>> transform = ST.RandomTimeShift(lambda size: np.random.uniform(-10, 20, size))
        >>> # Shift inputs by normally distributed time shifts
        >>> transform = ST.RandomTimeShift(lambda size: np.random.normal(0, 10, size))
        >>> # Shift by discrete set of values
        >>> transform = ST.RandomTimeShift(lambda size: np.random.choice([-10, 5, 10], size))
        >>> # Shift by 5 or 10
        >>> transform = ST.RandomTimeShift([5, 10])
        >>> # Shift by random amount between 5 and 10 with uniform probability
        >>> transform = ST.RandomTimeShift((5, 10))
        >>> # Shift fixed at 5 samples
        >>> transform = ST.RandomTimeShift(5)

    """

    def __init__(
        self,
        shift: NumericParameter = (-10, 10),
        interp_rate: int = 100,
        taps_per_arm: int = 24,
        min_time: float = .05,
        **kwargs,
    ) -> None:
        super(RandomTimeShift, self).__init__(**kwargs)
        self.shift = to_distribution(shift, self.random_generator)
        self.interp_rate = interp_rate
        self.min_time = min_time
        num_taps = int(taps_per_arm * interp_rate)

        self.taps = (
            sp.firwin(num_taps, 1.0 / interp_rate, width=1.0 /
                      (interp_rate / 4.0), scale=True)
            * interp_rate
        )
        self.string = (
            self.__class__.__name__
            + "("
            + "shift={}, ".format(shift)
            + "interp_rate={}, ".format(interp_rate)
            + "taps_per_arm={}".format(taps_per_arm)
            + ")"
        )

    def parameters(self) -> tuple:
        return (float(self.shift()),)

    def check_time_bounds(self, signal: Signal, params: tuple) -> float:
        """
            Method checks new start and stop times to ensure signal is not cropped out
            of view
        """
        stop_shift_list = []
        start_shift_list = []
        shift = params[0]
        for meta in signal["metadata"]:
            temp_start = meta["start"] + shift / data_shape(signal["data"])[0]
            temp_stop = meta["stop"] + shift / data_shape(signal["data"])[0]
            if temp_start > 1. or temp_stop < 0.:
                if temp_start > 1.:
                    new_shift = (1 - self.min_time -
                                 meta["start"]) * data_shape(signal["data"])[0]
                    start_shift_list.append(new_shift)
                else:
                    new_shift = (1 - self.min_time -
                                 meta["stop"]) * data_shape(signal["data"])[0]
                    stop_shift_list.append(new_shift)
            else:
                start_shift_list.append(shift)
                stop_shift_list.append(shift)

        if len(start_shift_list) == 0:
            start_shift_list = stop_shift_list

        if len(stop_shift_list) == 0:
            stop_shift_list = start_shift_list
        try:
            min_shift = np.max(
                (np.min(start_shift_list), np.min(stop_shift_list)))
            max_shift = np.min(
                (np.max(start_shift_list), np.max(stop_shift_list)))
        except:
            breakpoint()

        return (np.random.uniform(min_shift, max_shift, 1)[0],)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        params_new = self.check_time_bounds(signal, params)
        integer_part, decimal_part = divmod(params_new[0], 1)
        integer_time_shift: int = int(integer_part) if integer_part else 0
        """     float_decimal_part: float = float(
            decimal_part) if decimal_part else 0.0
        # Apply data transformation
        if float_decimal_part != 0:
            signal["data"]["samples"] = F_LEGACY.fractional_shift(
                signal["data"]["samples"],
                self.taps,
                self.interp_rate,
                # this needed to be negated to be consistent with the previous implementation
                -float_decimal_part,
            ) """
        signal["data"]["samples"] = F_LEGACY.time_shift(
            signal["data"]["samples"], integer_time_shift)

        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        params_new = self.check_time_bounds(signal, params)
        shift = params_new[0]
        for i, meta in enumerate(signal["metadata"]):
            meta["start"] += shift / data_shape(signal["data"])[0]
            meta["stop"] += shift / data_shape(signal["data"])[0]
            meta["start"] = np.clip(meta["start"], a_min=0.0, a_max=1.0)
            meta["stop"] = np.clip(meta["stop"], a_min=0.0, a_max=1.0)
            meta["duration"] = meta["stop"] - meta["start"]
            meta["num_samples"] = int(
                meta["duration"] * len(signal["data"]["samples"]))

        signal["metadata"] = [
            value for value in signal["metadata"] if value["duration"] > 0.]
        if len(signal["metadata"]) == 0:
            print("empty metadata!!!")

        return signal


class TimeCrop(SignalTransform):
    """Crops a tensor in the time dimension to the specified length. Optional
    crop techniques include: start, center, end, & random

    Args:
        crop_type (:obj:`str`):
            Type of cropping to perform. Options are: `start`, `center`, `end`,
            and `random`. `start` crops the input tensor such that the first
            `length` samples are returned. `center` crops the input tensor such
            that the center `length` samples are returned. `end` crops the
            input tensor such that the last `length` samples are returned.
            `random` crops randomly in the range `[0,length-1]`.

        length (:obj:`int`):
            Number of samples to include.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Crop inputs to first 256 samples
        >>> transform = ST.TimeCrop(crop_type='start', length=256)
        >>> # Crop inputs to center 512 samples
        >>> transform = ST.TimeCrop(crop_type='center', length=512)
        >>> # Crop inputs to last 1024 samples
        >>> transform = ST.TimeCrop(crop_type='end', length=1024)
        >>> # Randomly crop any 2048 samples from input
        >>> transform = ST.TimeCrop(crop_type='random', length=2048)

    """

    def __init__(
        self,
        crop_type: str = "random",
        crop_length: int = 256,
        signal_length: int = 1024,
    ) -> None:
        super(TimeCrop, self).__init__()
        self.crop_type = crop_type
        self.crop_length = crop_length
        self.signal_length = signal_length
        if self.crop_type not in ("start", "center", "end", "random"):
            raise ValueError(
                "Crop type must be: `start`, `center`, `end`, or `random`")

        self.string = (
            self.__class__.__name__
            + "("
            + "crop_type={}, ".format(crop_type)
            + "length={}".format(crop_length)
            + ")"
        )

    def parameters(self) -> tuple:
        if self.crop_type == "start":
            start = 0
        elif self.crop_type == "end":
            start = self.signal_length - self.crop_length
        elif self.crop_type == "center":
            start = (self.signal_length - self.crop_length) // 2
        elif self.crop_type == "random":
            start = np.random.randint(0, self.signal_length - self.crop_length)

        return start, self.crop_length

    def check_time_bounds(self, signal: Signal, params: tuple) -> float:
        """
            Method checks new start and stop times to ensure signal is not cropped out
            of view
        """
        start_list = []
        start, crop_length = params

        for meta in signal["metadata"]:
            original_start_sample = meta["start"] * \
                data_shape(signal["data"])[0]
            original_stop_sample = meta["stop"] * data_shape(signal["data"])[0]
            # new_start_sample = original_start_sample - start
            new_start_sample = start
            new_stop_sample = original_stop_sample - start
            start_clip = np.clip(
                float(new_start_sample / crop_length), a_min=0.0, a_max=1.0)
            stop_clip = np.clip(
                float(new_stop_sample / crop_length), a_min=0.0, a_max=1.0)
            duration = stop_clip - start_clip
            if duration < .001:
                start_list.append(
                    int(meta["start"] * data_shape(signal["data"])[0]))
            else:
                start_list.append(int(new_start_sample))

        return np.min(start_list), crop_length

    def transform_data(self, signal: Signal, params: tuple) -> Signal:

        if len(signal["metadata"]) == 0:
            return signal

        if len(signal["data"]["samples"]) == self.crop_length:
            return signal

        params = self.check_time_bounds(signal, params)
        # if signal["metadata"][0]["num_samples"] < self.crop_length:
        if data_shape(signal["data"])[0] < self.crop_length:
            raise ValueError(
                "Input data length {} is less than requested length {}".format(
                    data_shape(signal["data"])[0], self.crop_length
                )
            )

        signal["data"]["samples"] = F_LEGACY.time_crop(
            signal["data"]["samples"], params[0], self.crop_length)
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        params = self.check_time_bounds(signal, params)
        start, crop_length = params
        for meta in signal["metadata"]:
            original_start_sample = meta["start"] * \
                data_shape(signal["data"])[0]
            original_stop_sample = meta["stop"] * data_shape(signal["data"])[0]
            new_start_sample = original_start_sample - start
            new_stop_sample = original_stop_sample - start
            meta["start"] = np.clip(
                float(new_start_sample / crop_length), a_min=0.0, a_max=1.0)
            meta["stop"] = np.clip(
                float(new_stop_sample / crop_length), a_min=0.0, a_max=1.0)
            meta["duration"] = meta["stop"] - meta["start"]
            meta["num_samples"] = crop_length

        return signal


class AmplitudeReversal(SignalTransform):
    """Applies an amplitude reversal to the input tensor by applying a value of
    -1 to each sample. Effectively the same as a static phase shift of pi

    """

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.amplitude_reversal(
            signal["data"]["samples"])
        return signal


class AmplitudeScale(SignalTransform):
    """Scales the amplitude of the input tensor

    Args:
        scale (:py:class:`~Callable`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The scaling factor to apply.
            * If Callable, produces a sample by calling scale()
            * If float, scale is fixed at the value provided  
            * If list, scale is any element in the list
            * If tuple, scale is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Fixed scale of 2.0
        >>> transform = ST.AmplitudeScale(2.0)
        >>> # Random scale between 0.5 and 2.0
        >>> transform = ST.AmplitudeScale((0.5, 2.0))
    """

    def __init__(
        self,
        scale: NumericParameter = (0.5, 2.0),
        **kwargs
    ) -> None:
        super(AmplitudeScale, self).__init__(**kwargs)
        self.scale = to_distribution(scale, self.random_generator)
        self.string = f"{self.__class__.__name__}(scale={scale})"

    def parameters(self) -> tuple:
        return (float(self.scale()),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        scale_value = params[0]
        signal["data"]["samples"] = F_LEGACY.amplitude_scale(
            signal["data"]["samples"], scale_value)
        return signal


class RandomFrequencyShift(SignalTransform):
    """Shifts each tensor in freq by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling freq_shift()
            * If int or float, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = ST.RandomFrequencyShift(freq_shift=(-0.25, 0.25))
        >>> # Frequency shift inputs always fs/10
        >>> transform = ST.RandomFrequencyShift(freq_shift=0.1)
        >>> # Frequency shift inputs with normal distribution with stdev .1
        >>> transform = ST.RandomFrequencyShift(freq_shift=lambda size: np.random.normal(0, .1, size))
        >>> # Frequency shift inputs with either -fs/4 or fs/4 (discrete)
        >>> transform = ST.RandomFrequencyShift(freq_shift=[-.25, .25])

    """

    def __init__(self, freq_shift: NumericParameter = (-0.5, 0.5), **kwargs) -> None:
        super(RandomFrequencyShift, self).__init__(**kwargs)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.string = (
            self.__class__.__name__ +
            "(" + "freq_shift={}".format(freq_shift) + ")"
        )

    def parameters(self) -> tuple:
        return (self.freq_shift(),)

    def check_freq_bounds(self, signal: Signal, freq_shift: float) -> float:
        """
            Method checks frequency mins and maxes and adjust the new_rate to ensure
            frequency bounds stay within the +-.5 boundary.
        """
        ret_list = []
        for meta in signal["metadata"]:
            test_lf = meta["lower_freq"] + freq_shift
            test_hf = meta["upper_freq"] + freq_shift
            if test_lf < -.5 or test_hf > .5:
                if test_lf < -.5:
                    new_shift = -.5 - meta['lower_freq']
                else:
                    new_shift = .5 - meta['upper_freq']
                ret_list.append(new_shift)
            else:
                ret_list.append(freq_shift)
        return find_nearest(ret_list, 0.)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal
        freq_shift = self.check_freq_bounds(signal, params[0])
        signal["data"]["samples"] = F_LEGACY.freq_shift(
            signal["data"]["samples"], freq_shift)

        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        freq_shift = self.check_freq_bounds(signal, params[0])
        for meta in signal["metadata"]:
            # Check bounds for partial signals
            meta["lower_freq"] += freq_shift
            meta["upper_freq"] += freq_shift
            meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
            meta["center_freq"] = meta["lower_freq"] + meta["bandwidth"] * 0.5

        return signal


class RandomDelayedFrequencyShift(SignalTransform):
    """Apply a delayed frequency shift to the input data

        Args:
             start_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
                start_shift sets the start time of the delayed shift
                * If Callable, produces a sample by calling start_shift()
                * If int, start_shift is fixed at the value provided
                * If list, start_shift is any element in the list
                * If tuple, start_shift is in range of (tuple[0], tuple[1])

            freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
                freq_shift sets the translation along the freq-axis
                * If Callable, produces a sample by calling freq_shift()
                * If int, freq_shift is fixed at the value provided
                * If list, freq_shift is any element in the list
                * If tuple, freq_shif@pytest.mark.parametrize(
    t is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        start_shift: FloatParameter = (0.1, 0.9),
        freq_shift: FloatParameter = (-0.2, 0.2),
    ) -> None:
        super(RandomDelayedFrequencyShift, self).__init__()
        self.start_shift = to_distribution(start_shift, self.random_generator)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "start_shift={}, ".format(start_shift)
            + "freq_shift={}".format(freq_shift)
            + ")"
        )

    def parameters(self) -> tuple:
        freq_shift = 0
        while freq_shift < 0.05 and freq_shift > -0.05:
            freq_shift = self.freq_shift()
        return (self.start_shift(), freq_shift)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        start_shift, freq_shift = params
        if not has_rf_metadata(signal["metadata"]):
            num_iq_samples = data_shape(signal["data"])[0]
            signal["data"]["samples"][
                int(start_shift * num_iq_samples):
            ] = F_LEGACY.freq_shift(
                signal["data"]["samples"][int(start_shift * num_iq_samples):],
                freq_shift,
            )
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        start_shift, freq_shift = params
        avoid_aliasing = False
        new_meta = []
        for meta in signal["metadata"]:
            # If the signal is outside of the shift, we ignore it
            if meta["stop"] < start_shift or meta["start"] > start_shift:
                continue

            # If it starts before, then the first half is unchanged
            # and the second half gets shifted
            if meta["start"] < start_shift:
                meta_first = deepcopy(meta)
                meta_first["stop"] = np.clip(
                    meta_first["stop"], a_min=0.0, a_max=start_shift)
                meta_first["duration"] = meta_first["stop"] - \
                    meta_first["start"]
                new_meta.append(meta_first)

                meta_second = deepcopy(meta)
                meta_second["start"] = start_shift

                meta_second = self.shift_frequency(meta_second, freq_shift)
                avoid_aliasing = self.will_alias(meta_second)
                meta_second = self.clip_frequency(meta_second)
                new_meta.append(meta_second)
                continue

            # signal starts after start_shift
            meta_first = deepcopy(meta)
            meta_first["stop"] = np.clip(
                meta_first["stop"], a_min=0.0, a_max=start_shift)
            meta_first["duration"] = meta_first["stop"] - meta_first["start"]

            # Update freqs for next segment
            meta_first = self.shift_frequency(meta_first, freq_shift)
            avoid_aliasing = self.will_alias(meta_first)
            meta_first = self.clip_frequency(meta_first)
            new_meta.append(meta_first)

            meta_second = deepcopy(meta)
            meta_second["start"] = start_shift
            meta_second["duration"] = meta_second["stop"] - \
                meta_second["start"]
            new_meta.append(meta_second)

        # Perform augmentation
        if avoid_aliasing:
            # If any potential aliasing detected, perform shifting at higher sample rate
            signal["data"]["samples"][
                int(start_shift * data_shape(signal["data"])[0]):
            ] = F_LEGACY.freq_shift_avoid_aliasing(
                signal["data"]["samples"][
                    int(start_shift * data_shape(signal["data"])[0]):
                ],
                freq_shift,
            )
            return signal

        # Otherwise, use faster freq shifter
        signal["data"]["samples"][
            int(start_shift * data_shape(signal["data"])[0]):
        ] = F_LEGACY.freq_shift(
            signal["data"]["samples"][
                int(start_shift * data_shape(signal["data"])[0]):
            ],
            freq_shift,
        )

        return signal

    def shift_frequency(self, meta: SignalMetadata, shift: float):
        meta["lower_freq"] += float(shift)
        meta["upper_freq"] += float(shift)
        meta["center_freq"] += float(shift)
        return meta

    def clip_frequency(self, meta: SignalMetadata):
        meta["lower_freq"] = np.clip(meta["lower_freq"], a_min=-0.5, a_max=0.5)
        meta["upper_freq"] = np.clip(meta["upper_freq"], a_min=-0.5, a_max=0.5)
        meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
        meta["center_freq"] = meta["lower_freq"] + meta["bandwidth"] * 0.5
        return meta

    def will_alias(self, meta: SignalMetadata):
        if (
            meta["lower_freq"] >= 0.5
            or meta["upper_freq"] <= -0.5
            or meta["lower_freq"] < -0.45
            or meta["upper_freq"] > 0.45
        ):
            return True
        return False


class LocalOscillatorDrift(SignalTransform):
    """LocalOscillatorDrift is a transform modelling a local oscillator's drift in frequency by
    a random walk in frequency.

    Args:
        max_drift (FloatParameter, optional):
            [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
        max_drift_rate (FloatParameter, optional):
            [description]. Defaults to uniform_continuous_distribution(0.001,0.01).

    """

    def __init__(
        self,
        max_drift: FloatParameter = (0.005, 0.015),
        max_drift_rate: FloatParameter = (0.001, 0.01),
        **kwargs,
    ) -> None:
        super(LocalOscillatorDrift, self).__init__(**kwargs)
        self.max_drift = to_distribution(max_drift, self.random_generator)
        self.max_drift_rate = to_distribution(
            max_drift_rate, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "max_drift={}, ".format(max_drift)
            + "max_drift_rate={}".format(max_drift_rate)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.max_drift(), self.max_drift_rate())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        max_drift, max_drift_rate = params

        # Apply drift as a random walk.
        random_walk = self.random_generator.choice(
            [-1, 1], size=data_shape(signal["data"])[0]
        )

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        frequency = (
            np.cumsum(random_walk)
            * max_drift_rate
            / np.sqrt(data_shape(signal["data"])[0])
        )

        # Every time frequency hits max_drift, reset to zero.
        while np.argmax(np.abs(frequency) > max_drift):
            idx = np.argmax(np.abs(frequency) > max_drift)
            offset = max_drift if frequency[idx] < 0 else -max_drift
            frequency[idx:] += offset

        complex_phase = np.exp(2j * np.pi * np.cumsum(frequency))
        signal["data"]["samples"] = signal["data"]["samples"] * complex_phase
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        max_drift, max_drift_size = params
        for meta in signal["metadata"]:
            meta["lower_freq"] -= float(max_drift)
            meta["upper_freq"] += float(max_drift)
            meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]

        return signal


class GainDrift(SignalTransform):
    """GainDrift is a transform modelling a front end gain controller's drift in gain by
    a random walk in gain values.

    Args:
        max_drift (FloatParameter, optional):
            [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
        min_drift (FloatParameter, optional):
            [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
        drift_rate (FloatParameter, optional):
            [description]. Defaults to uniform_continuous_distribution(0.001,0.01).

    """

    def __init__(
        self,
        max_drift: FloatParameter = uniform_continuous_distribution(
            0.005, 0.015),
        min_drift: FloatParameter = uniform_continuous_distribution(
            0.005, 0.015),
        drift_rate: FloatParameter = uniform_continuous_distribution(
            0.001, 0.01),
        **kwargs,
    ) -> None:
        super(GainDrift, self).__init__(**kwargs)
        self.max_drift = to_distribution(max_drift, self.random_generator)
        self.min_drift = to_distribution(min_drift, self.random_generator)
        self.drift_rate = to_distribution(drift_rate, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "max_drift={}, ".format(max_drift)
            + "min_drift={}, ".format(min_drift)
            + "drift_rate={}".format(drift_rate)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.max_drift(), self.min_drift(), self.drift_rate())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        max_drift, min_drift, drift_rate = params

        # Apply drift as a random walk.
        random_walk = self.random_generator.choice(
            (-1, 1), size=data_shape(signal["data"])[0]
        )

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        gain = (
            np.cumsum(random_walk) * drift_rate /
            np.sqrt(data_shape(signal["data"])[0])
        )

        # Every time gain hits max_drift, reset to zero
        while np.argmax(gain > max_drift):
            idx = np.argmax(gain > max_drift)
            offset = gain[idx] - max_drift
            gain[idx:] -= offset

        # Every time gain hits min_drift, reset to zero
        while np.argmax(gain < min_drift):
            idx = np.argmax(gain < min_drift)
            offset = min_drift - gain[idx]
            gain[idx:] += offset

        signal["data"]["samples"] = signal["data"]["samples"] * (1 + gain)

        return signal


class IQImbalance(SignalTransform):
    """Applies various types of IQ imbalance to a tensor

    Args:
        iq_amplitude_imbalance_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_amplitude_imbalance()
            * If int or float, iq_amplitude_imbalance is fixed at the value provided
            * If list, iq_amplitude_imbalance is any element in the list
            * If tuple, iq_amplitude_imbalance is in range of (tuple[0], tuple[1])

        iq_phase_imbalance (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_phase_imbalance()
            * If int or float, iq_phase_imbalance is fixed at the value provided
            * If list, iq_phase_imbalance is any element in the list
            * If tuple, iq_phase_imbalance is in range of (tuple[0], tuple[1])

        iq_dc_offset_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_dc_offset()
            * If int or float, iq_dc_offset_db is fixed at the value provided
            * If list, iq_dc_offset is any element in the list
            * If tuple, iq_dc_offset is in range of (tuple[0], tuple[1])

    Note:
        For more information about IQ imbalance in RF systems, check out
        https://www.mathworks.com/help/comm/ref/iqimbalance.html

    Example:
        >>> import torchsig.transforms as ST
        >>> # IQ imbalance with default params
        >>> transform = ST.IQImbalance()

    """

    def __init__(
        self,
        iq_amplitude_imbalance_db: NumericParameter = (0, 3),
        iq_phase_imbalance: NumericParameter = (
            -np.pi * 1.0 / 180.0,
            np.pi * 1.0 / 180.0,
        ),
        iq_dc_offset_db: NumericParameter = (-0.1, 0.1),
    ) -> None:
        super(IQImbalance, self).__init__()
        self.amp_imbalance = to_distribution(
            iq_amplitude_imbalance_db, self.random_generator
        )
        self.phase_imbalance = to_distribution(
            iq_phase_imbalance, self.random_generator
        )
        self.dc_offset = to_distribution(
            iq_dc_offset_db, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "amp_imbalance={}, ".format(iq_amplitude_imbalance_db)
            + "phase_imbalance={}, ".format(iq_phase_imbalance)
            + "dc_offset={}".format(iq_dc_offset_db)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.amp_imbalance(), self.phase_imbalance(), self.dc_offset())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        amp_imb, phase_imb, dc_offset = params
        signal["data"]["samples"] = F.iq_imbalance(
            signal["data"]["samples"], amp_imb, phase_imb, dc_offset
        )
        return signal


class RollOff(SignalTransform):
    """Applies a band-edge RF roll-off effect simulating front end filtering

    Args:
        low_freq (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling low_freq()
            * If int or float, low_freq is fixed at the value provided
            * If list, low_freq is any element in the list
            * If tuple, low_freq is in range of (tuple[0], tuple[1])

        upper_freq (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling upper_freq()
            * If int or float, upper_freq is fixed at the value provided
            * If list, upper_freq is any element in the list
            * If tuple, upper_freq is in range of (tuple[0], tuple[1])

        low_cut_apply (:obj:`float`):
            Probability that the low frequency provided above is applied

        upper_cut_apply (:obj:`float`):
            Probability that the upper frequency provided above is applied

        order (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling order()
            * If int or float, order is fixed at the value provided
            * If list, order is any element in the list
            * If tuple, order is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        low_freq: NumericParameter = (0.00, 0.05),
        upper_freq: NumericParameter = (0.95, 1.00),
        low_cut_apply: float = 0.5,
        upper_cut_apply: float = 0.5,
        order: NumericParameter = (6, 20),
    ) -> None:
        super(RollOff, self).__init__()
        self.low_freq = to_distribution(low_freq, self.random_generator)
        self.upper_freq = to_distribution(upper_freq, self.random_generator)
        self.low_cut_apply = low_cut_apply
        self.upper_cut_apply = upper_cut_apply
        self.order = to_distribution(order, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "low_freq={}, ".format(low_freq)
            + "upper_freq={}, ".format(upper_freq)
            + "upper_cut_apply={}, ".format(upper_cut_apply)
            + "order={}".format(order)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.low_freq(), self.upper_freq(), self.order())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        low_freq, upper_freq, order = params
        low_freq = low_freq if np.random.rand() < self.low_cut_apply else 0.0
        upper_freq = upper_freq if np.random.rand() < self.upper_cut_apply else 1.0
        signal["data"]["samples"] = F_LEGACY.roll_off(
            signal["data"]["samples"], low_freq, upper_freq, int(order)
        )
        return signal


class SpectralInversion(SignalTransform):
    """Applies a spectral inversion"""

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F.spectral_inversion(
            signal["data"]["samples"])
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        for meta in signal["metadata"]:
            # Invert frequency labels
            original_lower = meta["lower_freq"]
            original_upper = meta["upper_freq"]
            meta["lower_freq"] = original_upper * -1
            meta["upper_freq"] = original_lower * -1
            meta["center_freq"] *= -1

        return signal


class Clip(SignalTransform):
    """Clips the input values to a percentage of the max/min values

    Args:
        clip_percentage (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Specifies the percentage of the max/min values to clip
            * If Callable, produces a sample by calling clip_percentage()
            * If int or float, clip_percentage is fixed at the value provided
            * If list, clip_percentage is any element in the list
            * If tuple, clip_percentage is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        clip_percentage: NumericParameter = (0.75, 0.95),
        **kwargs,
    ) -> None:
        super(Clip, self).__init__(**kwargs)
        self.clip_percentage = to_distribution(clip_percentage)
        self.string = (
            self.__class__.__name__
            + "("
            + "clip_percentage={}".format(clip_percentage)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.clip_percentage(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        clip_percentage = params[0]
        signal["data"]["samples"] = F_LEGACY.clip(
            signal["data"]["samples"], clip_percentage)
        return signal


class RandomConvolve(SignalTransform):
    """Convolve a random complex filter with the input data

    Args:
        num_taps (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Number of taps for the random filter
            * If Callable, produces a sample by calling num_taps()
            * If int or float, num_taps is fixed at the value provided
            * If list, num_taps is any element in the list
            * If tuple, num_taps is in range of (tuple[0], tuple[1])

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The effect of the filtered data is dampened using an alpha factor
            that determines the weightings for the summing of the filtered data
            and the original data. `alpha` should be in range `[0,1]` where a
            value of 0 applies all of the weight to the original data, and a
            value of 1 applies all of the weight to the filtered data
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        num_taps: IntParameter = [2, 3, 4, 5],
        alpha: FloatParameter = (0.1, 0.5),
        **kwargs,
    ) -> None:
        super(RandomConvolve, self).__init__(**kwargs)
        self.num_taps = to_distribution(num_taps, self.random_generator)
        self.alpha = to_distribution(alpha, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "num_taps={}, ".format(num_taps)
            + "alpha={}".format(alpha)
            + ")"
        )

    def parameters(self) -> tuple:
        return (int(self.num_taps()), self.alpha())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        num_taps, alpha = params
        signal["data"]["samples"] = F_LEGACY.random_convolve(
            signal["data"]["samples"], num_taps, alpha)
        return signal


class DatasetBasebandMixUp(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the TargetSNR transform and the
    additional `alpha` input to set the difference in SNRs between the two
    examples with the following relationship:

       mixup_sample_snr = main_sample_snr + alpha

    Note that `alpha` is used as an additive value because the SNR values are
    expressed in log scale. Typical usage will be with with alpha values less
    than zero.

    This transform is loosely based on
    `"mixup: Beyond Emperical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_.


    Args:
        dataset :obj:`SignalDataset`:
            A SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalMetadataPassThroughTransform()
        >>> dataset = ModulationsDataset(
                use_class_idx=True,
                level=0,
                num_iq_samples=4096,
                num_samples=5300,
                target_transform=target_transform,
            )
        >>> transform = ST.DatasetBasebandMixUp(dataset=dataset,alpha=(-5,-3))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: NumericParameter = (-5, -3),
    ) -> None:
        super(DatasetBasebandMixUp, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.alpha(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        alpha = params[0]
        if not has_modulated_rf_metadata(signal["metadata"]):
            return signal

        # Calculate target SNR of signal to be inserted
        target_snr_db = signal["metadata"][0]["snr"] + alpha

        # Randomly sample from provided dataset
        idx = np.random.randint(self.dataset_num_samples)
        insert_data, insert_metadata = self.dataset[idx]

        if insert_data.shape[0] != data_shape(signal["data"])[0]:
            raise ValueError(
                "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                Found {}, but expected {} samples".format(
                    insert_data.shape[0], data_shape(signal["data"])[0]
                )
            )

        # Set insert data's SNR
        target_snr_transform = TargetSNR(target_snr_db)
        insert_data = target_snr_transform(insert_data)

        signal["data"]["samples"] = signal["data"]["samples"] + insert_data
        signal["metadata"].extend(insert_metadata)
        return signal


class DatasetBasebandCutMix(SignalTransform):
    """Signal Transform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the TargetSNR transform to match
    the main dataset's examples' SNR and an additional `alpha` input to set the
    relative quantity in time to occupy, where

       cutmix_num_iq_samples = total_num_iq_samples * alpha

    With this transform, the inserted signal replaces the IQ samples of the
    original signal rather than adding to them as the `DatasetBasebandMixUp`
    transform does above.

    This transform is loosely based on
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" <https://arxiv.org/pdf/1905.04899.pdf>`_.

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets import ModulationsDataset
        >>> # Add signals from the `ModulationsDataset`
        >>> target_transform = SignalMetadataPassThroughTransform()
        >>> dataset = ModulationsDataset(
                use_class_idx=True,
                level=0,
                num_iq_samples=4096,
                num_samples=5300,
                target_transform=target_transform,
            )
        >>> transform = ST.DatasetBasebandCutMix(dataset=dataset,alpha=(0.2,0.5))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: NumericParameter = (0.2, 0.5),
    ) -> None:
        super(DatasetBasebandCutMix, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.alpha(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        alpha = params[0]
        if not has_modulated_rf_metadata(signal["metadata"]):
            return signal

        if len(signal["metadata"]) > 1:
            raise ValueError(
                "Expected single `SignalMetadata` but {} were detected.".format(
                    len(signal["metadata"])
                )
            )

        # Randomly sample from provided dataset
        idx = np.random.randint(self.dataset_num_samples)
        insert_data, insert_metadata = self.dataset[idx]

        num_iq_samples = data_shape(signal["data"])[0]

        # Set insert data's SNR
        target_snr_transform = TargetSNR(signal["metadata"][0]["snr"])
        insert_signal = target_snr_transform(
            create_signal(data=insert_data, metadata=insert_metadata)
        )
        insert_data = insert_signal["data"]["samples"]

        # Mask both data examples based on alpha and a random start value
        insert_num_iq_samples = int(alpha * num_iq_samples)
        insert_start = np.random.randint(
            num_iq_samples - insert_num_iq_samples)
        insert_stop = insert_start + insert_num_iq_samples

        # Combine two pieces of data
        signal["data"]["samples"][insert_start:insert_stop] = 0
        insert_data[:insert_start] = 0
        insert_data[insert_stop:] = 0
        signal["data"]["samples"] = signal["data"]["samples"] + insert_data

        # Update SignalMetadata
        if insert_start == 0:
            signal["metadata"][0]["start"] = insert_stop / num_iq_samples
            signal["metadata"][0]["stop"] = 1.0

        if insert_stop == num_iq_samples:
            signal["metadata"][0]["start"] = 0.0
            signal["metadata"][0]["stop"] = insert_start / num_iq_samples

        if insert_start != 0 and insert_stop != num_iq_samples:
            # MetaData becomes two
            new_meta = deepcopy(signal["metadata"][0])
            new_meta["start"] = 0.0
            new_meta["stop"] = insert_start / num_iq_samples
            signal["metadata"][0] = new_meta

            new_meta["start"] = insert_stop / num_iq_samples
            new_meta["stop"] = 1.0
            signal["metadata"].append(new_meta)

        for meta in signal["metadata"]:
            meta["duration"] = meta["stop"] - meta["start"]

        return signal


class DatasetWidebandCutMix(SignalTransform):
    """SignalTransform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using an additional `alpha` input to set
    the relative quantity in time to occupy, where

       cutmix_num_iq_samples = total_num_iq_samples * alpha

    This transform is loosely based on [CutMix: Regularization Strategy to
    Train Strong Classifiers with Localizable Features]
    (https://arxiv.org/pdf/1710.09412.pdf).

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in durations between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets.torchsig_wideband import TorchSigWideband
        >>> # Add signals from the `ModulationsDataset`
        >>> dataset = TorchSigWideband('.')
        >>> transform = ST.DatasetWidebandCutMix(dataset=dataset,alpha=(0.2,0.7))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: NumericParameter = (0.2, 0.7),
    ) -> None:
        super(DatasetWidebandCutMix, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.alpha(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        alpha = params[0]
        idx = np.random.randint(self.dataset_num_samples)
        insert_data, insert_meta = self.dataset[idx]

        num_iq_samples = data_shape(signal["data"])[0]
        if data_shape(signal["data"])[0] != num_iq_samples:
            raise ValueError(
                "Input dataset's `num_iq_samples` does not match main dataset.\n\t\
                Found {}, but expected {} samples".format(
                    insert_data.shape[0], data_shape(signal["data"])[0]
                )
            )

        # Mask both data examples based on alpha and a random start value
        insert_num_iq_samples = int(alpha * num_iq_samples)
        insert_start: int = np.random.randint(
            num_iq_samples - insert_num_iq_samples)
        insert_stop = insert_start + insert_num_iq_samples
        signal["data"]["samples"][insert_start:insert_stop] = 0
        insert_data[:insert_start] = 0.0
        insert_data[insert_stop:] = 0.0
        insert_start //= num_iq_samples
        insert_dur = insert_num_iq_samples / num_iq_samples

        # Create new SignalData object for transformed data
        signal["data"]["samples"] += insert_data

        # Update SignalMetadata
        new_meta = []
        for meta in signal["metadata"]:
            # Update labels
            if (
                meta["start"] > insert_start
                and meta["start"] < insert_start + insert_dur
            ):
                # Label starts within cut region
                if (
                    meta["stop"] > insert_start
                    and meta["stop"] < insert_start + insert_dur
                ):
                    # Label also stops within cut region --> Remove label
                    continue
            #     else:
            #         # Push label stabout_start
            #     and meta["stop"] < insert_start + insert_dur
            # ):
            # Label stops within cut region but does not start in region --> Push stop to begining of cut region
            # meta["stop"] = insert_start
            elif (
                meta["start"] < insert_start
                and meta["stop"] > insert_start + insert_dur
            ):
                # Label traverse cut region --> Split into two labels
                meta_split = deepcopy(meta)
                # Update first label region's stop
                meta["stop"] = insert_start
                # Update second label region's start & append to description collection
                meta_split["start"] = insert_start + insert_dur
                new_meta.append(meta_split)

            # Append SignalMetadata to list
            new_meta.append(meta)

        # Repeat for inserted example's SignalMetadata(s)
        insert_meta = []
        for meta in insert_meta:
            # Update labels
            if meta["stop"] < insert_start or meta["stop"] > insert_start + insert_dur:
                # Label is outside inserted region --> Remove label
                continue
            elif (
                meta["stop"] < insert_start and meta["stop"] < insert_start + insert_dur
            ):
                # Label starts before and ends within region, push start to region start
                meta["stop"] = insert_start
            elif (
                meta["stop"] >= insert_start
                and meta["stop"] > insert_start + insert_dur
            ):
                # Label starts within region and stops after, push stop to region stop
                meta["stop"] = insert_start + insert_dur
            elif (
                meta["stop"] < insert_start and meta["stop"] > insert_start + insert_dur
            ):
                # Label starts before and stops after, push both start & stop to region boundaries
                meta["stop"] = insert_start
                meta["stop"] = insert_start + insert_dur

            # Append SignalMetadata to list
            insert_meta.append(meta)

        # Set output data's SignalMetadata to above list
        signal["metadata"] = new_meta

        return signal


class DatasetWidebandMixUp(SignalTransform):
    """SignalTransform that inputs a dataset to randomly sample from and insert
    into the main dataset's examples, using the `alpha` input to set the
    difference in magnitudes between the two examples with the following
    relationship:

       output_sample = main_sample * (1 - alpha) + mixup_sample * alpha

    This transform is loosely based on [mixup: Beyond Emperical Risk
    Minimization](https://arxiv.org/pdf/1710.09412.pdf).

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the synthetic insertion/mixup

        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            alpha sets the difference in power level between the main dataset
            example and the inserted example
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> from torchsig.datasets.torchsig_wideband import TorchSigWideband
        >>> # Add signals from the `TorchSigWideband` Dataset
        >>> dataset = TorchSigWideband('.')
        >>> transform = ST.DatasetWidebandMixUp(dataset=dataset,alpha=(0.4,0.6))

    """

    def __init__(
        self,
        dataset: SignalDataset,
        alpha: NumericParameter = (0.4, 0.6),
    ) -> None:
        super(DatasetWidebandMixUp, self).__init__()
        self.alpha = to_distribution(alpha, self.random_generator)
        self.dataset = dataset
        self.dataset_num_samples = len(dataset)
        self.string = (
            self.__class__.__name__
            + "("
            + "dataset={}, ".format(dataset)
            + "alpha={}".format(alpha)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.alpha(),)

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        alpha = params[0]
        # Randomly sample from provided dataset
        idx = np.random.randint(self.dataset_num_samples)
        insert_data, insert_meta = self.dataset[idx]

        # Create new SignalData object for transformed data
        signal["data"]["samples"] = (
            signal["data"]["samples"] * (1 - alpha) + insert_data * alpha
        )

        # Update SignalMetadata
        signal["metadata"].extend(insert_meta)
        return signal


class SpectrogramRandomResizeCrop(SignalTransform):
    """The SpectrogramRandomResizeCrop transforms the input IQ data into a
    spectrogram with a randomized FFT size and overlap. This randomization in
    the spectrogram computation results in spectrograms of various sizes. The
    width and height arguments specify the target output size of the transform.
    To get to the desired size, the randomly generated spectrogram may be
    randomly cropped or padded in either the time or frequency dimensions. This
    transform is meant to emulate the Random Resize Crop transform often used
    in computer vision tasks.

    Args:
        nfft (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The number of FFT bins for the random spectrogram.
            * If Callable, nfft is set by calling nfft()
            * If int, nfft is fixed by value provided
            * If list, nfft is any element in the list
            * If tuple, nfft is in range of (tuple[0], tuple[1])
            Defaults to (256, 1024).
        overlap_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`list`, :obj:`tuple`):
            The ratio of the (nfft-1) value to use as the overlap parameter for
            the spectrogram operation. Setting as ratio ensures the overlap is
            a lower value than the bin size.
            * If Callable, nfft is set by calling overlap_ratio()
            * If float, overlap_ratio is fixed by value provided
            * If list, overlap_ratio is any element in the list
            * If tuple, overlap_ratio is in range of (tuple[0], tuple[1])
            Defaults to (0.0, 0.2).
        detrend (Optional[str], optional): 
            _description_. Defaults to "constant".
        scaling (Optional[str], optional): 
            _description_. Defaults to "density".
        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.
        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'complex'.
        width (:obj:`int`):
            Target output width (time) of the spectrogram. Defaults to 512.
        height (:obj:`int`):
            Target output height (frequency) of the spectrogram. Defaults to 512.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Randomly sample NFFT size in range [128,1024] and randomly crop/pad output spectrogram to (512,512)
        >>> transform = ST.SpectrogramRandomResizeCrop(nfft=(128,1024), overlap_ratio=(0.0,0.2), width=512, height=512)

    """

    def __init__(
        self,
        nfft: IntParameter = (256, 1024),
        overlap_ratio: FloatParameter = (0.0, 0.2),
        detrend: Optional[str] = "constant",
        scaling: Optional[str] = "density",
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = "complex",
        width: int = 512,
        height: int = 512,
    ) -> None:
        super(SpectrogramRandomResizeCrop, self).__init__()
        self.nfft = to_distribution(nfft, self.random_generator)
        self.overlap_ratio = to_distribution(
            overlap_ratio, self.random_generator)
        self.detrend: Optional[str] = None if detrend is None else detrend
        self.scaling: Optional[str] = None if scaling is None else scaling
        self.window_fcn = window_fcn
        self.mode = mode
        self.width = width
        self.height = height
        self.string = (
            self.__class__.__name__
            + "("
            + "nfft={}, ".format(nfft)
            + "overlap_ratio={}, ".format(overlap_ratio)
            + "detrend={}".format(self.detrend)
            + "scaling={}".format(self.scaling)
            + "window_fcn={}, ".format(window_fcn)
            + "mode={}, ".format(mode)
            + "width={}, ".format(width)
            + "height={}".format(height)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.nfft(), self.overlap_ratio())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        nfft, overlap_ratio = params
        nfft = int(nfft)
        nperseg = nfft
        noverlap = int(overlap_ratio * (nfft - 1))

        # First, perform the random spectrogram operation
        spec_data = F.spectrogram(
            signal["data"]["samples"],
            nperseg,
            noverlap,
            nfft,
            self.detrend,
            self.scaling,
            self.window_fcn,
            self.mode,
        )
        if self.mode == "complex":
            spec_data = self.spec_to_complex(spec_data)

        # Next, perform the random cropping/padding
        channels, curr_height, curr_width = spec_data.shape
        pad_height, crop_height = False, False
        pad_width, crop_width = False, False
        pad_height_samps, pad_width_samps = 0, 0
        if curr_height < self.height:
            pad_height = True
            pad_height_samps = self.height - curr_height
        elif curr_height > self.height:
            crop_height = True
        if curr_width < self.width:
            pad_width = True
            pad_width_samps = self.width - curr_width
        elif curr_width > self.width:
            crop_width = True

        if pad_height or pad_width:
            pad_height_start = np.random.randint(0, pad_height_samps // 2 + 1)
            pad_height_end = pad_height_samps - pad_height_start + 1
            pad_width_start = np.random.randint(0, pad_width_samps // 2 + 1)
            pad_width_end = pad_width_samps - pad_width_start + 1

            if self.mode == "complex":
                spec_data = self.pad_spec_complex(
                    spec_data,
                    self.pad_func,
                    pad_height_start,
                    pad_height_end,
                    pad_width_start,
                    pad_width_end,
                )
            else:
                spec_data = self.pad_spec(
                    self.pad_func,
                    pad_height_start,
                    pad_height_end,
                    pad_width_start,
                    pad_width_end,
                )

        crop_width_start = np.random.randint(
            0, max(1, curr_width - self.width))
        crop_height_start = np.random.randint(
            0, max(1, curr_height - self.height))
        spec_data = spec_data[
            :,
            crop_height_start: crop_height_start + self.height,
            crop_width_start: crop_width_start + self.width,
        ]
        signal["data"]["samples"] = spec_data

        if not has_rf_metadata(signal["metadata"]):
            return signal

        # Update SignalMetadata
        new_meta = []
        for meta in signal["metadata"]:
            meta = meta_bound_frequency(meta)

            # Update labels based on padding/cropping
            if pad_height:
                meta = meta_pad_height(
                    meta, curr_height, self.height, pad_height_start)

            if crop_height:
                if (
                    meta["lower_freq"] + 0.5
                ) * curr_height >= crop_height_start + self.height or (
                    meta["upper_freq"] + 0.5
                ) * curr_height <= crop_height_start:
                    continue
                meta = self.meta_crop_height(
                    curr_height, crop_height_start, meta)

            if pad_width:
                meta = self.meta_pad_width(curr_width, pad_width_start, meta)

            if crop_width:
                if (
                    meta["start"] * curr_width >= crop_width_start + self.width
                    or meta["stop"] * curr_width <= crop_width_start
                ):
                    continue
                self.meta_crop_width(curr_width, crop_width_start, meta)

            # Append SignalMetadata to list
            new_meta.append(meta)

        signal["metadata"] = new_meta
        return signal

    def pad_func(self, vector, pad_width, iaxis, kwargs):
        vector[: pad_width[0]] = (
            np.random.rand(len(vector[: pad_width[0]])) * kwargs["pad_value"]
        )
        vector[-pad_width[1]:] = (
            np.random.rand(len(vector[-pad_width[1]:])) * kwargs["pad_value"]
        )

    def meta_crop_width(self, curr_width, crop_width_start, meta):
        if meta["start"] * curr_width <= crop_width_start:
            meta["start"] = 0.0
        else:
            meta["start"] = (meta["start"] * curr_width -
                             crop_width_start) / self.width

        if meta["stop"] * curr_width >= crop_width_start + self.width:
            meta["stop"] = 1.0
        else:
            meta["stop"] = (meta["stop"] * curr_width -
                            crop_width_start) / self.width
        meta["duration"] = meta["stop"] - meta["start"]

    def meta_crop_height(self, curr_height, crop_height_start, meta):
        if (meta["lower_freq"] + 0.5) * curr_height <= crop_height_start:
            meta["lower_freq"] = -0.5
        else:
            meta["lower_freq"] = (
                (meta["lower_freq"] + 0.5) * curr_height - crop_height_start
            ) / self.height - 0.5
        if (meta["upper_freq"] + 0.5) * curr_height >= crop_height_start + self.height:
            meta["upper_freq"] = crop_height_start + self.height
        else:
            meta["upper_freq"] = (
                (meta["upper_freq"] + 0.5) * curr_height - crop_height_start
            ) / self.height - 0.5
        meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
        meta["center_freq"] = meta["lower_freq"] + meta["bandwidth"] / 2
        return meta

    def meta_pad_width(
        self, curr_width, pad_width_start, meta: SignalMetadata
    ) -> SignalMetadata:
        meta["start"] = (meta["start"] * curr_width +
                         pad_width_start) / self.width
        meta["stop"] = (meta["stop"] * curr_width +
                        pad_width_start) / self.width
        meta["duration"] = meta["stop"] - meta["start"]
        return meta

    def pad_spec(
        self, pad_func, pad_height_start, pad_height_end, pad_width_start, pad_width_end
    ):
        spec_data = np.pad(
            spec_data,
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            min_value=np.percentile(np.abs(spec_data[0]), 50),
        )

        return spec_data

    def pad_spec_complex(
        self,
        spec_data,
        pad_func,
        pad_height_start,
        pad_height_end,
        pad_width_start,
        pad_width_end,
    ):
        new_data_real = np.pad(
            spec_data[0],
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            pad_value=np.percentile(np.abs(spec_data[0]), 50),
        )
        new_data_imag = np.pad(
            spec_data[1],
            (
                (pad_height_start, pad_height_end),
                (pad_width_start, pad_width_end),
            ),
            pad_func,
            pad_value=np.percentile(np.abs(spec_data[1]), 50),
        )
        spec_data = np.concatenate(
            [
                np.expand_dims(new_data_real, axis=0),
                np.expand_dims(new_data_imag, axis=0),
            ],
            axis=0,
        )
        return spec_data

    def spec_to_complex(self, spec_data):
        new_tensor = np.zeros(
            (2, spec_data.shape[0], spec_data.shape[1]), dtype=np.float32
        )
        new_tensor[0, :, :] = np.real(spec_data).astype(np.float32)
        new_tensor[1, :, :] = np.imag(spec_data).astype(np.float32)
        spec_data = new_tensor
        return spec_data


class SpectrogramPatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `PatchShuffle Regularization <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            patch_size sets the size of each patch to shuffle
            * If Callable, produces a sample by calling patch_size()
            * If int or float, patch_size is fixed at the value provided
            * If list, patch_size is any element in the list
            * If tuple, patch_size is in range of (tuple[0], tuple[1])

        shuffle_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If Callable, produces a sample by calling shuffle_ratio()
            * If int or float, shuffle_ratio is fixed at the value provided
            * If list, shuffle_ratio is any element in the list
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        patch_size: NumericParameter = (2, 16),
        shuffle_ratio: FloatParameter = (0.01, 0.10),
    ) -> None:
        super(SpectrogramPatchShuffle, self).__init__()
        self.patch_size = to_distribution(patch_size, self.random_generator)
        self.shuffle_ratio = to_distribution(
            shuffle_ratio, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "patch_size={}, ".format(patch_size)
            + "shuffle_ratio={}".format(shuffle_ratio)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.patch_size(), self.shuffle_ratio())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        patch_size, shuffle_ratio = params
        signal["data"]["samples"] = F_LEGACY.spec_patch_shuffle(
            signal["data"]["samples"], patch_size, shuffle_ratio
        )
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        return signal


class SpectrogramTranslation(SignalTransform):
    """Transform that inputs a spectrogram and applies a random time/freq
    translation

    Args:
         time_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            time_shift sets the translation along the time-axis
            * If Callable, produces a sample by calling time_shift()
            * If int, time_shift is fixed at the value provided
            * If list, time_shift is any element in the list
            * If tuple, time_shift is in range of (tuple[0], tuple[1])

        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            freq_shift sets the translation along the freq-axis
            * If Callable, produces a sample by calling freq_shift()
            * If int, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        time_shift: IntParameter = (-128, 128),
        freq_shift: IntParameter = (-128, 128),
    ) -> None:
        super(SpectrogramTranslation, self).__init__()
        self.time_shift = to_distribution(time_shift, self.random_generator)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "time_shift={}, ".format(time_shift)
            + "freq_shift={}".format(freq_shift)
            + ")"
        )

    def parameters(self) -> tuple:
        return (self.time_shift(), self.freq_shift())

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        time_shift, freq_shift = params
        signal["data"]["samples"] = F_LEGACY.spec_translate(
            signal["data"]["samples"], time_shift, freq_shift
        )
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        time_shift, freq_shift = params
        new_meta = []
        for meta in signal["metadata"]:
            # Update time fields
            meta["start"] = (
                meta["start"] + time_shift / signal["data"]["samples"].shape[1]
            )
            meta["stop"] = (
                meta["stop"] + time_shift / signal["data"]["samples"].shape[1]
            )
            if meta["start"] >= 1.0 or meta["stop"] <= 0.0:
                continue
            meta["start"] = 0.0 if meta["start"] < 0.0 else meta["start"]
            meta["stop"] = 1.0 if meta["stop"] > 1.0 else meta["stop"]
            meta["duration"] = meta["stop"] - meta["start"]

            # Trim any out-of-capture freq values
            meta = meta_bound_frequency(meta)

            # Update freq fields
            meta["lower_freq"] = (
                meta["lower_freq"] + freq_shift /
                signal["data"]["samples"].shape[2]
            )
            meta["upper_freq"] = (
                meta["upper_freq"] + freq_shift /
                signal["data"]["samples"].shape[2]
            )
            if meta["lower_freq"] >= 0.5 or meta["upper_freq"] <= -0.5:
                continue

            meta = meta_bound_frequency(meta)

            # Append SignalMetadata to list
            new_meta.append(meta)

        # Set output data's SignalMetadata to above list
        signal["metadata"] = new_meta
        return signal


class SpectrogramMosaicDownsample(SignalTransform):
    """The SpectrogramMosaicDownsample transform takes the original input
    tensor and inserts it randomly into one cell of a 2x2 grid of 2x the size
    of the orginal spectrogram input. The `dataset` argument is then read 3x to
    retrieve spectrograms to fill the remaining cells of the 2x2 grid. Finally,
    the 2x oversized stitched spectrograms are downsampled by 2 to become the
    desired, original shape

    Args:
        dataset :obj:`SignalDataset`:
            An SignalDataset of complex-valued examples to be used as a source for
            the mosaic operation

    """

    def __init__(self, dataset: SignalDataset) -> None:
        super(SpectrogramMosaicDownsample, self).__init__()
        self.dataset = dataset
        self.string = self.__class__.__name__ + \
            "(" + "dataset={}".format(dataset) + ")"

    def parameters(self) -> tuple:
        return tuple()

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        channels, height, width = signal["data"]["samples"].shape

        # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
        cell_idx = np.random.randint(0, 4)
        x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
        y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
        full_mosaic = np.empty(
            (channels, height * 2, width * 2),
            dtype=signal["data"]["samples"].dtype,
        )
        full_mosaic[
            :,
            y_idx * height: (y_idx + 1) * height,
            x_idx * width: (x_idx + 1) * width,
        ] = signal["data"]["samples"]

        # Next, fill in the remaining cells with data randomly sampled from the input dataset
        for cell_i in range(4):
            if cell_i == cell_idx:
                # Skip if the original data's cell
                continue
            x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
            y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
            dataset_idx = np.random.randint(len(self.dataset))
            curr_data, curr_signal_desc = self.dataset[dataset_idx]
            full_mosaic[
                :,
                y_idx * height: (y_idx + 1) * height,
                x_idx * width: (x_idx + 1) * width,
            ] = curr_data

        # After the data has been stitched into the large 2x2 gride, downsample by 2
        signal["data"]["samples"] = full_mosaic[:, ::2, ::2]
        return signal

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        if not has_rf_metadata(signal["metadata"]):
            return signal

        # Read shapes
        channels, height, width = signal["data"]["samples"].shape

        # First, create a 2x2 grid of (512+512,512+512) and randomly put the initial data into a grid cell
        cell_idx = np.random.randint(0, 4)
        x_idx = 0 if cell_idx == 0 or cell_idx == 2 else 1
        y_idx = 0 if cell_idx == 0 or cell_idx == 1 else 1
        full_mosaic = np.empty(
            (channels, height * 2, width * 2),
            dtype=signal["data"]["samples"].dtype,
        )
        full_mosaic[
            :,
            y_idx * height: (y_idx + 1) * height,
            x_idx * width: (x_idx + 1) * width,
        ] = signal["data"]["samples"]

        # Update original data's SignalMetadata objects given the cell index
        for meta in signal["metadata"]:
            # Update time fields
            if x_idx == 0:
                meta["start"] /= 2
                meta["stop"] /= 2
                meta["duration"] = meta["stop"] - meta["start"]

            else:
                meta["start"] = meta["start"] / 2 + 0.5
                meta["stop"] = meta["stop"] / 2 + 0.5
                meta["duration"] = meta["stop"] - meta["start"]

            # Update frequency fields
            meta["lower_freq"] = (
                -0.5 if meta["lower_freq"] < -0.5 else meta["lower_freq"]
            )
            meta["upper_freq"] = 0.5 if meta["upper_freq"] > 0.5 else meta["upper_freq"]
            if y_idx == 0:
                meta["lower_freq"] = (meta["lower_freq"] + 0.5) / 2 - 0.5
                meta["upper_freq"] = (meta["upper_freq"] + 0.5) / 2 - 0.5
                meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
                meta["center_freq"] = meta["lower_freq"] + \
                    meta["bandwidth"] * 0.5

            else:
                meta["lower_freq"] = (meta["lower_freq"] + 0.5) / 2
                meta["upper_freq"] = (meta["upper_freq"] + 0.5) / 2
                meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
                meta["center_freq"] = meta["lower_freq"] + \
                    meta["bandwidth"] * 0.5

        # Next, fill in the remaining cells with data randomly sampled from the input dataset
        for cell_i in range(4):
            if cell_i == cell_idx:
                # Skip if the original data's cell
                continue
            x_idx = 0 if cell_i == 0 or cell_i == 2 else 1
            y_idx = 0 if cell_i == 0 or cell_i == 1 else 1
            dataset_idx = np.random.randint(len(self.dataset))
            curr_data, curr_signal_desc = self.dataset[dataset_idx]
            full_mosaic[
                :,
                y_idx * height: (y_idx + 1) * height,
                x_idx * width: (x_idx + 1) * width,
            ] = curr_data

            # Update inserted data's SignalMetadata objects given the cell index
            for meta in signal["metadata"]:
                # Update time fields
                if x_idx == 0:
                    meta["start"] /= 2
                    meta["stop"] /= 2
                    meta["duration"] = meta["stop"] - meta["start"]

                else:
                    meta["start"] = meta["start"] / 2 + 0.5
                    meta["stop"] = meta["stop"] / 2 + 0.5
                    meta["duration"] = meta["stop"] - meta["start"]

                # Update frequency fields
                meta["lower_freq"] = (
                    -0.5 if meta["lower_freq"] < -0.5 else meta["lower_freq"]
                )
                meta["upper_freq"] = (
                    0.5 if meta["upper_freq"] > 0.5 else meta["upper_freq"]
                )
                if y_idx == 0:
                    meta["lower_freq"] = (meta["lower_freq"] + 0.5) / 2 - 0.5
                    meta["upper_freq"] = (meta["upper_freq"] + 0.5) / 2 - 0.5
                    meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
                    meta["center_freq"] = meta["lower_freq"] + \
                        meta["bandwidth"] * 0.5

                else:
                    meta["lower_freq"] = (meta["lower_freq"] + 0.5) / 2
                    meta["upper_freq"] = (meta["upper_freq"] + 0.5) / 2
                    meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
                    meta["center_freq"] = meta["lower_freq"] + \
                        meta["bandwidth"] * 0.5

        # After the data has been stitched into the large 2x2 gride, downsample by 2
        signal["data"]["samples"] = full_mosaic[:, ::2, ::2]
        return signal


class SpectrogramImage(SignalTransform):
    """Transforms SignalData to spectrogram image

    Args:
        None


    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.SpectrogramImage() 

    """

    def __init__(
        self,
    ) -> None:
        super(SpectrogramImage, self).__init__()
        self.string: str = (
            self.__class__.__name__
        )

    def transform_data(self, signal: Signal, params: tuple) -> Signal:
        signal["data"]["samples"] = F_LEGACY.spectrogram_image(
            signal["data"]["samples"],
        )
        return signal
