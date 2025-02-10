from enum import Enum


class BackboneType(Enum):
    RESNET50 = "resnet50"
    VIT_B = "vitb"


class SSLModelType(Enum):
    BYOL = "byol"


class DatasetType(Enum):
    TORCHSIG_NARROWBAND = "narrowband"
    TORCHSIG_WIDEBAND = "wideband"


class TransformType(Enum):
    SPECTROGRAM = "spectrogram"
    IQ = "iq"
