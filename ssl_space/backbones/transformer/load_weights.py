from typing import Dict, Any
import re
import torch
import torch.nn as nn


def load_weights_dst(network: nn.Module, **kwargs):

    pretrain_method: str = kwargs.get('pretrain_method', None)
    path: str = kwargs.get('path', None)
    
    if pretrain_method is None:
        raise ValueError("Provide pretraining method, not None")
    
    if path is None:
        raise ValueError("Provide the path for pretraining weights")
    
    if pretrain_method.lower() == "generic":
        return _load_generic_dst(network, pretrain_method, path)

    if pretrain_method.lower() == "kinetics400":
        return _load_kinetics400_dst(network, pretrain_method, path)

    raise ValueError(f"{pretrain_method} not a known pretraining method to load from.")


def _load_generic_dst(network: nn.Module, pretrain_method: str, path: str):
    ckpt: Dict[str, Any] = torch.load(path, map_location='cpu')['model_state_dict']
    ckpt_backbone, ckpt_momentum = {}, {}
    for k, v in ckpt.items():
        if k.endswith(".masked_embed") or k == "masked_embed":
            print("Not loading masked embeds into DST - if you need them, load a masked DST.")
            continue
        
        if k.startswith('momentum_backbone.'): ckpt_momentum[k[len('momentum_backbone.'):]] = v
        elif k.startswith('backbone.'): ckpt_backbone[k[len('backbone.'):]] = v
        else: print(f'unknown parameter group {k}')
    
    if ckpt_momentum:
        print("loading from momentum backbone...")
        msg = network.load_state_dict(ckpt_momentum, strict=True)
    elif ckpt_backbone:
        print("loading from backbone...")
        msg = network.load_state_dict(ckpt_backbone, strict=True)
    else:
        raise ValueError(f"No compatible weights found in checkpoint: {path}")
    print(f"Weights loaded from ({pretrain_method}|{path}) with msg: {msg}")
    return network


def _load_kinetics400_dst(network: nn.Module, pretrain_method: str, path: str):
    ckpt: Dict[str, Any] = torch.load(path, map_location="cpu")
    mapped_ckpt = {}

    for k, v in ckpt.items():
        if k.startswith('backbone.'): k = k[len('backbone.'):]
        if k.startswith('head.'): continue
        # top-level renaming
        if k == "cls_token": k = "cls_tokens"
        elif k == "pos_embed": k = "pos_embeds"
        elif k == "time_embed": k = "time_embeds"
        # rename space attention from attn -> space attn
        k = re.sub(r'^blocks\.(\d+)\.attn\.', r'blocks.\1.space_attn.', k)
        mapped_ckpt[k] = v

    msg = network.load_state_dict(mapped_ckpt, strict=True)
    print(f"Weights loaded from ({pretrain_method}|{path}) with msg: {msg}")
    return network
