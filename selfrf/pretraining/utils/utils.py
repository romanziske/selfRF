from typing import Any

from torchsig.datasets.signal_classes import torchsig_signals


def get_class_list(config: Any) -> list:
    if config.family:
        # Get unique sorted family names from class-family dict
        return sorted(list(set(torchsig_signals.family_dict.values())))
    return torchsig_signals.class_list
