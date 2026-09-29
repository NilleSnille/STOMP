from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import einops

from .vits import (
    BaseTimeViT,
)

from .vit_blocks import (
    DroppedDividedSpaceTimeBlock
)


class DroppedDividedSpaceTimeViT(BaseTimeViT):
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
        interpolation_mode_space: str = "nearest",
        interpolation_mode_time: str = "nearest",
    ):
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size,
            in_chans=in_chans, 
            embed_dim=embed_dim,
            num_cls_tokens=num_cls_tokens, 
            feat_drop_p=feat_drop_p, 
            dropout_p=dropout_p, 
            norm_layer=norm_layer,
            num_frames=num_frames
        )

        assert interpolation_mode_space in {"nearest", "bilinear"}
        assert interpolation_mode_time in {"nearest", "linear"}
        self.interpolation_mode_space = interpolation_mode_space
        self.interpolation_mode_time = interpolation_mode_time

        dpr = [x.item() for x in torch.linspace(0, path_drop_p, depth)]
        self.blocks = nn.ModuleList(
            [
                DroppedDividedSpaceTimeBlock(
                    dim=embed_dim, 
                    num_heads=num_heads, 
                    num_cls_tokens=num_cls_tokens, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    attn_drop_p=attn_drop_p, 
                    feat_drop_p=feat_drop_p, 
                    path_drop_p=dpr[i],
                    act_layer=act_layer, 
                    norm_layer=norm_layer,                        
                ) for i in range(depth)
            ]
        )

        self.apply(self._init_weights)

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1
    
    def interp_pos_embeddings(self, x_patches_h: int, x_patches_w: int) -> torch.Tensor:
        cls_pos_embeds = self.pos_embeds[0, :self.num_cls_tokens, :].unsqueeze(0)
        patch_pos_embeds = self.pos_embeds[0, self.num_cls_tokens:, :].unsqueeze(0)
        
        # Reshape into the EXPECTED shape (1, embed_dim, H_num_patches, W_num_patches).
        patch_pos_embeds = patch_pos_embeds.transpose(1, 2).reshape(1, self.embed_dim, self.H_num_patches, self.W_num_patches)
        # Interpolate to ACTUAL shape of x, i.e. (1, embed_dim, x_patches_h, x_patches_w).
        if self.interpolation_mode_space == "nearest":
            interp_patch_pos_embeds: torch.Tensor = nn.functional.interpolate(
                patch_pos_embeds, 
                size=(x_patches_h, x_patches_w), 
                mode='nearest', 
            )
        elif self.interpolation_mode_space == "bilinear":
            interp_patch_pos_embeds: torch.Tensor = nn.functional.interpolate(
                patch_pos_embeds, 
                size=(x_patches_h, x_patches_w), 
                mode='bilinear', 
                align_corners=False,
            )
        else: raise ValueError(f"unexpected interpolaiton mode for space: {self.interpolation_mode_space}")
        # Reshape
        interp_patch_pos_embeds = interp_patch_pos_embeds.flatten(2).transpose(1, 2)
        # Prepend CLS positions back
        interp_pos_embeds = torch.cat((cls_pos_embeds, interp_patch_pos_embeds), dim=1)
        return interp_pos_embeds

    def interp_time_embeddings(self, T: int) -> torch.Tensor:
        time_embeds = self.time_embeds.transpose(1, 2)
        if self.interpolation_mode_time == "nearest":
            interp_time_embeds: torch.Tensor = nn.functional.interpolate(
                time_embeds, 
                size=(T), 
                mode='nearest', 
            )
        elif self.interpolation_mode_time == "linear":
            interp_time_embeds: torch.Tensor = nn.functional.interpolate(
                time_embeds, 
                size=(T), 
                mode='linear', 
                align_corners=False,
            )
        else: raise ValueError(f"unexpected interpolation mode for time: {self.interpolation_mode_time}")

        interp_time_embeds = interp_time_embeds.transpose(1, 2)
        return interp_time_embeds

    def embed(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        """
        Args:
            x: [B, C, T, H, W]
            mask: [B*T, Hp, Wp], True=masked, False=visible. We drop masked patches (tube-consistent expected).

        Returns:
            tokens: [B, num_cls + (N_used*T), C]  where N_used=N (unmasked) or N_vis (masked)
            Hp, Wp
            vis_idx [B, N_vis]
        """
        assert mask is not None, "DroppedDividedSpaceTimeViT.embed expects a mask; use fast-path super().embed for unmasked."
        B, C_in, T, H, W = x.shape

        # ---- Patchify ----
        # patchify returns per-frame tokens: [B*T, N, C], N=Hp*Wp
        x_tok, Hp, Wp = self.patchify(x)
        C = x_tok.size(-1)

        # reshape to [B, T, N, C] for time embedding logic and gather
        x_btnc = einops.rearrange(x_tok, '(b t) n c -> b t n c', b=B, t=T)

        # ---- Compute visible indices (tube-consistent) ----
        mask_btn = einops.rearrange(mask, '(b t) h w -> b t (h w)', b=B, t=T)  # [B, T, N]
        mask_bn = mask_btn[:, 0]                                               # [B, N]
        keep_bn = ~mask_bn                                                     # [B, N] True=visible
        if not (mask_btn == mask_bn[:, None, :]).all():
            raise ValueError("Mask must be tube-consistent across time.")

        # number visible per sample (should be equal across batch)
        n_vis_per = keep_bn.sum(dim=1)
        N_vis = int(n_vis_per.min().item())
        # Optional: enforce consistency
        if not torch.all(n_vis_per == N_vis):
            raise ValueError("Masking must yield the same number of visible patches per video in the batch.")

        vis_idx = keep_bn.float().argsort(dim=1, descending=True)[:, :N_vis]   # [B, N_vis]

        # gather visible tokens along spatial dim
        vis_idx_btn = einops.repeat(vis_idx, 'b nvis -> b t nvis', t=T)        # [B, T, N_vis]
        x_btnvisc = x_btnc.gather(
            dim=2,
            index=vis_idx_btn[..., None].expand(B, T, N_vis, C)
        )  # [B, T, N_vis, C]

        # ---- Add spatial positional embeddings (for original grid, gathered to visible indices) ----
        if (Hp != self.H_num_patches) or (Wp != self.W_num_patches):
            pos_full = self.interp_pos_embeddings(x_patches_h=Hp, x_patches_w=Wp)  # [1, num_cls + N, C]
        else:
            pos_full = self.pos_embeds                                             # [1, num_cls + N, C]

        pos_cls = pos_full[:, :self.num_cls_tokens, :]                             # [1, num_cls, C]
        pos_patch = pos_full[:, self.num_cls_tokens:, :]                           # [1, N, C]

        # gather spatial pos for visible indices: [B, N_vis, C]
        pos_patch_b = pos_patch.expand(B, -1, -1)                                  # [B, N, C]
        pos_patch_vis = pos_patch_b.gather(
            dim=1,
            index=vis_idx[..., None].expand(B, N_vis, C)
        )  # [B, N_vis, C]

        # broadcast spatial pos across time and add
        pos_patch_btnvisc = einops.repeat(pos_patch_vis, 'b nvis c -> b t nvis c', t=T)  # [B, T, N_vis, C]
        x_btnvisc = self.pos_drop(x_btnvisc + pos_patch_btnvisc)

        # ---- Add temporal embeddings (unchanged, broadcast across N_vis) ----
        if T != self.time_embeds.size(1):
            time_embeds = self.interp_time_embeddings(T=T)                          # [1, T, C]
        else:
            time_embeds = self.time_embeds                                          # [1, T, C]

        x_btnvisc = self.time_drop(x_btnvisc + time_embeds[:, :, None, :])          # [B, T, N_vis, C]

        # ---- Flatten patches to (N_vis*T) and prepend one CLS per video ----
        x_patches = einops.rearrange(x_btnvisc, 'b t nvis c -> b (nvis t) c')        # [B, N_vis*T, C]

        cls_tokens = self.cls_tokens.expand(B, -1, -1)                               # [B, num_cls, C]
        cls_tokens = self.pos_drop(cls_tokens + pos_cls.expand(B, -1, -1))           # add CLS pos emb (same grid)
        x_out = torch.cat((cls_tokens, x_patches), dim=1)                            # [B, num_cls + N_vis*T, C]
        return x_out, Hp, Wp, vis_idx
        
    def forward_features(self, x: torch.Tensor, mask: torch.Tensor|None, blocks: nn.ModuleList, get_attn: bool=False, get_all: bool=False, get_info: bool=False):
        B, C, T, H, W = x.shape

        if mask is None:
            # Fast path: no mask -> use the parent implementation (early exit)
            x, x_patches_h, x_patches_w = super().embed(x)
        else:
            x, x_patches_h, x_patches_w, vis_idx = self.embed(x, mask)
        
        x, attn = self.push(
            blocks=blocks,
            x=x,
            B=B,
            T=T,
            x_patches_h=x_patches_h,
            x_patches_w=x_patches_w,
            get_attn=get_attn
        )

        if get_all:
            out_feats = x
        else:
            out_feats = x[:, :self.num_cls_tokens]
            if self.num_cls_tokens == 1:
                out_feats = out_feats.squeeze(1)

        if get_info is True:
            info = {
                "batch_size": B,
                "num_frames": T,
                "num_patches_h": x_patches_h,
                "num_patches_w": x_patches_w,
                "attention": attn,
            }
            if mask is not None:
                info["vis_idx"] = vis_idx
            return out_feats, info
        else:
            return out_feats

    def forward(self, x, mask: torch.Tensor|None=None, get_attn = False, get_all = False, get_info = False):
        return self.forward_features(x, mask, self.blocks, get_attn=get_attn, get_all=get_all, get_info=get_info)


### Factory ####


def make_dropped_divided_space_time_vit(
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
    num_frames: int = 8,
    interpolation_mode_space: str = "nearest",
    interpolation_mode_time: str = "nearest",
) -> DroppedDividedSpaceTimeViT:

    if act_layer == 'gelu':
        act_layer = nn.GELU
    else:
        raise ValueError(f"expected a string in (gelu), got {act_layer}")
    
    if norm_layer == 'layernorm':
        norm_layer = partial(nn.LayerNorm, eps=1.0e-6)
    else:
        raise ValueError(f"expected a string in (layernorm), got {norm_layer}")
    
    network = DroppedDividedSpaceTimeViT(
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
        interpolation_mode_space=interpolation_mode_space,
        interpolation_mode_time=interpolation_mode_time,
    )

    return network
