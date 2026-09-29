from typing import Tuple
from functools import partial
import torch
import torch.nn as nn
import einops


from .vits import (
    DividedSpaceTimeViT,
)


class Decoder_MaskedDividedSpaceTimeViT(DividedSpaceTimeViT):
    def __init__(
        self,
        img_size:  Tuple = (224, 224),
        patch_size: Tuple = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12, 
        num_heads: int = 12,
        num_cls_tokens: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False, 
        qk_scale: float|None = None,
        feat_drop_p: float = 0.0,
        attn_drop_p: float = 0.0,
        path_drop_p: float = 0.0,
        dropout_p: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        num_frames: int = 8,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_cls_tokens=num_cls_tokens,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            feat_drop_p=feat_drop_p,
            attn_drop_p=attn_drop_p,
            path_drop_p=path_drop_p,
            dropout_p=dropout_p,
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_frames=num_frames,
        )
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.masked_embed, std=0.02)
        self.patch_embed = nn.Identity()

    def patchify(self, x):
        raise NotImplementedError("Decoder expects tokens, not pixels.")
    
    def interp_pos_embeddings(self, x_patches_h: int, x_patches_w: int) -> torch.Tensor:
        cls_pos_embeds = self.pos_embeds[0, :self.num_cls_tokens, :].unsqueeze(0)
        patch_pos_embeds = self.pos_embeds[0, self.num_cls_tokens:, :].unsqueeze(0)
        
        # Reshape into the EXPECTED shape (1, embed_dim, H_num_patches, W_num_patches).
        patch_pos_embeds = patch_pos_embeds.transpose(1, 2).reshape(1, self.embed_dim, self.H_num_patches, self.W_num_patches)
        # Interpolate to ACTUAL shape of x, i.e. (1, embed_dim, x_patches_h, x_patches_w).
        interp_patch_pos_embeds: torch.Tensor = nn.functional.interpolate(
            patch_pos_embeds, 
            size=(x_patches_h, x_patches_w), 
            mode='bilinear', 
            align_corners=False,
        )
        # Reshape
        interp_patch_pos_embeds = interp_patch_pos_embeds.flatten(2).transpose(1, 2)
        # Prepend CLS positions back
        interp_pos_embeds = torch.cat((cls_pos_embeds, interp_patch_pos_embeds), dim=1)
        return interp_pos_embeds

    def interp_time_embeddings(self, T: int) -> torch.Tensor:
        time_embeds = self.time_embeds.transpose(1, 2)
        interp_time_embeds: torch.Tensor = nn.functional.interpolate(
            time_embeds, 
            size=(T), 
            mode='linear', 
            align_corners=False,
        )
        interp_time_embeds = interp_time_embeds.transpose(1, 2)
        return interp_time_embeds

    def embed(
        self, x: torch.Tensor, 
        mask: torch.Tensor,
        x_patches_h: int,
        x_patches_w: int,
        B: int, 
        T: int,
    )  -> torch.Tensor:
        assert x.dim() == 3 and x.shape[0] == B*T and x.shape[1] == x_patches_h*x_patches_w and x.shape[2] == self.embed_dim
        if mask is not None:
            x_hw = einops.rearrange(x, 'bt (hp wp) c -> bt hp wp c', hp=x_patches_h, wp=x_patches_w).contiguous()
            x_hw[mask] = self.masked_embed.to(x_hw.dtype)  # masked_embed can be [1, C] or [1,1,C]
            x = einops.rearrange(x_hw, 'bt hp wp c -> bt (hp wp) c')

        if (x_patches_h != self.H_num_patches) or (x_patches_w != self.W_num_patches):
            pos_full = self.interp_pos_embeddings(x_patches_h=x_patches_h, x_patches_w=x_patches_w)
        else:
            pos_full = self.pos_embeds
        
        pos_cls = pos_full[:, :self.num_cls_tokens, :]                             # [1, num_cls, C]
        pos_patch = pos_full[:, self.num_cls_tokens:, :]                           # [1, N, C]

        x = self.pos_drop(x + pos_patch) 
        x = einops.rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if T != self.time_embeds.size(1):
            time_embeds = self.interp_time_embeddings(T=T)
        else:
            time_embeds = self.time_embeds
        x = self.time_drop(x + time_embeds)
        x = einops.rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
        cls_tokens = self.pos_drop(self.cls_tokens.expand(B, -1, -1) + pos_cls)  # pos_cls broadcasts   
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward_features(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor, 
            blocks: nn.ModuleList,
            x_patches_h: int,
            x_patches_w: int,
            B: int,
            T: int,
    ):
        x = self.embed(
            x, 
            mask, 
            x_patches_h=x_patches_h, 
            x_patches_w=x_patches_w, 
            B=B, 
            T=T
        )

        x, attn = self.push(
            blocks=blocks,
            x=x,
            B=B,
            T=T,
            x_patches_h=x_patches_h,
            x_patches_w=x_patches_w,
            get_attn=False,
        )
        # remove cls, we do not use it downstream ever
        x = x[:, self.num_cls_tokens:]
        return x   

    def forward(
        self, 
        x, 
        mask: torch.Tensor, 
        x_patches_h: int,
        x_patches_w: int,
        B: int, 
        T: int, 
    ):
        return self.forward_features(
            x, 
            mask, 
            self.blocks,
            x_patches_h=x_patches_h, 
            x_patches_w=x_patches_w,
            B=B,
            T=T, 
        )

### Factory ####


def make_decoder_masked_divided_space_time_vit(
    img_size:  Tuple = (224, 224),
    patch_size: Tuple = (16, 16),
    in_chans: int = 3,
    embed_dim: int = 768,
    depth: int = 12, 
    num_heads: int = 12,
    num_cls_tokens: int = 1,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False, 
    qk_scale: float|None = None,
    feat_drop_p: float = 0.0,
    attn_drop_p: float = 0.0,
    path_drop_p: float = 0.0,
    dropout_p: float = 0.0,
    act_layer: str = 'gelu',
    norm_layer: str = 'layernorm',
    num_frames: int = 8
) -> Decoder_MaskedDividedSpaceTimeViT:

    if act_layer == 'gelu':
        act_layer = nn.GELU
    else:
        raise ValueError(f"expected a string in (gelu), got {act_layer}")
    
    if norm_layer == 'layernorm':
        norm_layer = partial(nn.LayerNorm, eps=1.0e-6)
    else:
        raise ValueError(f"expected a string in (layernorm), got {norm_layer}")
    
    network = Decoder_MaskedDividedSpaceTimeViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth, 
        num_heads=num_heads,
        num_cls_tokens=num_cls_tokens,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        feat_drop_p=feat_drop_p,
        attn_drop_p=attn_drop_p,
        path_drop_p=path_drop_p,
        dropout_p=dropout_p,
        act_layer=act_layer,
        norm_layer=norm_layer,
        num_frames=num_frames
    )

    return network
