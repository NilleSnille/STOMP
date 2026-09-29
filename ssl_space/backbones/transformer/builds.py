from typing import Dict, Any

from .masked_vits import make_dropped_divided_space_time_vit
from .decoders import make_decoder_masked_divided_space_time_vit
from .load_weights import load_weights_dst


def dropped_dst_vit(method, arch_kwargs: Dict[str, Any], pretrained_kwargs: Dict[str, Any]):
    network = make_dropped_divided_space_time_vit(**arch_kwargs)
    if pretrained_kwargs:
        network = load_weights_dst(network, **pretrained_kwargs)
    return network


def decoder_masked_dst_vit(method, arch_kwargs: Dict[str, Any], pretrained_kwargs: Dict[str, Any]):
    network = make_decoder_masked_divided_space_time_vit(**arch_kwargs)
    if pretrained_kwargs:
        raise NotImplementedError("The decoder is trained from scratch.")
    return network
