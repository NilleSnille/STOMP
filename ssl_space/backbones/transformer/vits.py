from typing import Tuple


import torch
import torch.nn as nn
import einops

from .vit_utils import (
    PatchEmbed,
    trunc_normal_,
)
from .vit_blocks import (
    DividedSpaceTimeBlock,
)

class BaseViT(nn.Module):
    """
    Base class for vision transformer.
    """
    def __init__(
        self,
        img_size:  Tuple = (224, 224),
        patch_size: Tuple = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        num_cls_tokens: int = 1,
        feat_drop_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.num_features = self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.H_num_patches = img_size[0] // patch_size[0]
        self.W_num_patches = img_size[1] // patch_size[1]
        self.num_patches = self.H_num_patches * self.W_num_patches

        # CLS tokens.
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, embed_dim))
        trunc_normal_(self.cls_tokens, std=.02)

        # Positional Embeddings
        self.pos_embeds = nn.Parameter(torch.zeros(1, num_cls_tokens + self.num_patches, embed_dim))
        trunc_normal_(self.pos_embeds, std=.02)
        self.pos_drop = nn.Dropout(feat_drop_p)

        # Normalisation Layer
        self.norm = norm_layer(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embeds', 'cls_tokens'}
    
    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor,int, int]: 
        return self.patch_embed(x)

    def interp_pos_embeddings(self, x_patches_h: int, x_patches_w: int) -> torch.Tensor:
        cls_pos_embeds = self.pos_embeds[0, :self.num_cls_tokens, :].unsqueeze(0)
        patch_pos_embeds = self.pos_embeds[0, self.num_cls_tokens:, :].unsqueeze(0)
        
        # Reshape into the EXPECTED shape (1, embed_dim, H_num_patches, W_num_patches).
        patch_pos_embeds = patch_pos_embeds.transpose(1, 2).reshape(1, self.embed_dim, self.H_num_patches, self.W_num_patches)
        # Interpolate to ACTUAL shape of x, i.e. (1, embed_dim, x_patches_h, x_patches_w).
        interp_patch_pos_embeds: torch.Tensor = nn.functional.interpolate(patch_pos_embeds, size=(x_patches_h, x_patches_w), mode='nearest')
        # Reshape
        interp_patch_pos_embeds = interp_patch_pos_embeds.flatten(2).transpose(1, 2)
        # Prepend CLS positions back
        #interp_pos_embeds = torch.cat((cls_pos_embeds, interp_patch_pos_embeds))
        interp_pos_embeds = torch.cat((cls_pos_embeds, interp_patch_pos_embeds), dim=1)
        return interp_pos_embeds

    def embed(self, x: torch.Tensor) -> Tuple[torch.Tensor,int,int]:
        x, x_patches_h, x_patches_w = self.patchify(x)
        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if (x_patches_h != self.H_num_patches) or (x_patches_w != self.W_num_patches):
            pos_embeds = self.interp_pos_embeddings(x_patches_h=x_patches_h, x_patches_w=x_patches_w)
        else:
            pos_embeds = self.pos_embeds

        x = self.pos_drop(x + pos_embeds)

        return x, x_patches_h, x_patches_w
    
    def push(
        self, 
        blocks: nn.ModuleList, 
        x: torch.Tensor, 
        B: int, 
        T: int, 
        x_patches_h: int, 
        x_patches_w: int, 
        get_attn: bool
    ) -> Tuple[torch.Tensor,torch.Tensor|None]:
        
        attn_last = None
        
        if get_attn is False:
        
            for blk in blocks:
                x = blk(x, B, T, x_patches_h, x_patches_w, return_attn=False)
        
        else:
            '''
            This is inefficent and potentially missaligned - here the attention is computed by forwarding over x (partly) again. Uneccsary and not 
            consistent if stchasticity is used (should not be used, but often is in implementations).
            for i, blk in enumerate(blocks):
                if i < len(blocks) - 1:
                    x = blk(x, B, T, x_patches_h, x_patches_w, return_attn=False)
                    t = x
                else:
                    # return attention of the last block
                    x = blk(x, B, T, x_patches_h, x_patches_w, return_attn=False)
                    attn_last = blk(t, B, T, x_patches_h, x_patches_w, return_attn=True)
            '''
            for i, blk in enumerate(blocks):
                if i < len(blocks) - 1:
                    x = blk(x, B, T, x_patches_h, x_patches_w, return_attn=False)
                else:
                    x, attn_last = blk(x, B, T, x_patches_h, x_patches_w, return_attn=True)

        x = self.norm(x)
        return x, attn_last
        
    def forward_features(
            self, 
            x: torch.Tensor, 
            blocks: nn.ModuleList, 
            get_attn: bool=False, 
            get_all: bool=False, 
            get_info: bool=False
        ) -> Tuple[torch.Tensor,torch.Tensor|None]:
        B, C, T, H, W = x.shape
        x, x_patches_h, x_patches_w = self.embed(x)

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
                "attention": attn
            }
            return out_feats, info
        else:
            return out_feats
    
    def forward(self, x: torch.Tensor, get_attn: bool=False, get_all: bool=False, get_info: bool=False) -> Tuple[torch.Tensor,torch.Tensor|None]:
        raise NotImplementedError("Implement in leaf classes")

class BaseTimeViT(BaseViT):
    def __init__(
        self,
        img_size:  Tuple = (224, 224),
        patch_size: Tuple = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        num_cls_tokens: int = 1,
        feat_drop_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        num_frames: int = 8,
    ):
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            num_cls_tokens=num_cls_tokens, 
            feat_drop_p=feat_drop_p, 
            dropout_p=dropout_p, 
            norm_layer=norm_layer
        )

        self.time_embeds = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(feat_drop_p)


    @torch.jit.ignore
    def no_weight_decay(self):
        s = super().no_weight_decay()
        s.add('time_embeds')
        return s

    def interp_time_embeddings(self, T: int) -> torch.Tensor:
        time_embeds = self.time_embeds.transpose(1, 2)
        interp_time_embeds: torch.Tensor = nn.functional.interpolate(time_embeds, size=(T), mode='nearest')
        interp_time_embeds = interp_time_embeds.transpose(1, 2)
        return interp_time_embeds

    def embed(self, x: torch.Tensor) -> Tuple[torch.Tensor,int,int]:
        B, C, T, H, W = x.shape
        x, x_patches_h, x_patches_w = super().embed(x)

        cls_tokens = x[:B, :self.num_cls_tokens, :]
        x = x[:, self.num_cls_tokens:]
        x = einops.rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if T != self.time_embeds.size(1):
            time_embeds = self.interp_time_embeddings(T=T)
        else:
            time_embeds = self.time_embeds
        x = self.time_drop(x + time_embeds)
        x = einops.rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, x_patches_h, x_patches_w
    
    def forward_features(self, x: torch.Tensor, blocks: nn.ModuleList, get_attn: bool=False, get_all: bool=False, get_info: bool=False):
        B, C, T, H, W = x.shape
        x, x_patches_h, x_patches_w = self.embed(x)

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
                
            return out_feats, info
        else:
            return out_feats


class DividedSpaceTimeViT(BaseTimeViT):
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
            num_cls_tokens=num_cls_tokens, 
            feat_drop_p=feat_drop_p, 
            dropout_p=dropout_p, 
            norm_layer=norm_layer,
            num_frames=num_frames
        )

        dpr = [x.item() for x in torch.linspace(0, path_drop_p, depth)]
        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(
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


    def forward(self, x, get_attn = False, get_all = False, get_info = False):
        return self.forward_features(x, self.blocks, get_attn=get_attn, get_all=get_all, get_info=get_info)

### Factory ####
