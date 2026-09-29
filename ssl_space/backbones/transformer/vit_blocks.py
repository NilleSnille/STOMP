# Portions adapted and modified for STOMP from DINO/timm (Apache-2.0).
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2019, 2020 Ross Wightman.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

from typing import Tuple

import torch
import torch.nn as nn
import einops

from .vit_utils import DropPath

class MLP(nn.Module):
    def __init__(
        self, 
        in_features:        int, 
        hidden_features:    int|None = None, 
        out_features:       int|None = None,
        act_layer:          nn.Module = nn.GELU, 
        drop_p:             float = 0.
    ):
        '''
        A simple feed-forward network (“MLP”) with one hidden layer.

        Args:
            hidden_features:    Defaults to in_features.
            out_features:       Defaults to in_features.
            act_layer:          The activation function to use. Defaults to GELU.
            drop_rate:          Dropout probability applied after activation and after output. Defaults to 0.
            
        '''
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    '''
    Multi-head self-attention layer.

    Applies query-key-value projection, scaled dot-product attention, then output projection.

    TO-DO: use torch.nn.functional.scaled_dot_product_attention for faster computation. 
    '''

    def __init__(
        self, 
        dim:            int, 
        num_heads:      int = 8, 
        qkv_bias:       bool = False, 
        qk_scale:       float|None = None, 
        attn_drop_p:    float = 0., 
        feat_drop_p:    float = 0.,
    ):
        '''
        Args:
            dim:            Total dimension of input and output features.
            num_heads:      Number of attention heads.
            qkv_bias:       If True, add bias to QKV projections.
            qk_scale:       Manual scale factor for q⋅k; defaults to 1/√(head_dim).
            attn_drop_p:    Dropout probability on attention weights.
            feat_drop_p:    Dropout probability on activations (the features in the network).
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.feat_drop = nn.Dropout(feat_drop_p)
        self.attn_drop = nn.Dropout(attn_drop_p)

    def forward(self, x: torch.Tensor, return_attn: bool=False) -> torch.Tensor|Tuple[torch.Tensor,torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.feat_drop(x)

        if return_attn:
            return x, attn
        else: 
            return x

class BaseBlock(nn.Module):
    '''
    Base class for blocks, defining common modules.
    '''
    def __init__(
        self, 
        dim:            int, 
        num_heads:      int, 
        num_cls_tokens: int = 1,
        mlp_ratio:      float = 4., 
        qkv_bias:       bool = False, 
        qk_scale:       float|None = None,  
        attn_drop_p:    float = 0.,
        feat_drop_p:    float = 0.,
        path_drop_p:    float = 0., 
        act_layer:      nn.Module = nn.GELU, 
        norm_layer:     nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_cls_tokens = num_cls_tokens
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_prob=path_drop_p) if path_drop_p > 0. else nn.Identity()

        self.space_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_p=attn_drop_p,
            feat_drop_p=feat_drop_p,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            out_features=dim,
            act_layer=act_layer,
            drop_p=feat_drop_p,
        )
    
    def forward(self, x: torch.Tensor, B: int, T: int, x_patches_h: int, x_patches_w: int, return_attn: bool=False):
        '''
        Forward method is overridden by child classes. 
        x has shape (B*T, num_cls_tokens + x_patches_h*x_patches_w, embed_dim).    
        '''
        raise NotImplementedError



class DividedSpaceTimeBlock(BaseBlock):
    def __init__(
            self, 
            dim, 
            num_heads,
            num_cls_tokens=1,
            mlp_ratio=4, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop_p=0, 
            feat_drop_p=0, 
            path_drop_p=0, 
            act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm,
        ):
        super().__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop_p=attn_drop_p, 
            feat_drop_p=feat_drop_p, 
            path_drop_p=path_drop_p, 
            act_layer=act_layer, 
            norm_layer=norm_layer,
            num_cls_tokens=num_cls_tokens,
        )

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_p=attn_drop_p,
            feat_drop_p=feat_drop_p,
        )
        self.temporal_fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, B: int, T: int, x_patches_h: int, x_patches_w: int, return_attn: bool=False) -> torch.Tensor:
        assert x.shape == (B, self.num_cls_tokens + T*(x_patches_h*x_patches_w), self.dim)

        # Temporal
        xt = x[:, self.num_cls_tokens:, :] # drop cls tokens. xt.shape = (B, T*(x_patches_h*x_patches_w), embed_dim=dim)
        xt = einops.rearrange(xt, 'b (h w t) d -> (b h w) t d', b=B, h=x_patches_h, w=x_patches_w, t=T)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = einops.rearrange(res_temporal, '(b h w) t d -> b (h w t) d', b=B, h=x_patches_h, w=x_patches_w, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:, self.num_cls_tokens:, :] + res_temporal

        # Spatial
        init_cls_tokens = x[:, :self.num_cls_tokens, :] # (B, num_cls_tokens, embed_dim) 
        cls_tokens = init_cls_tokens.repeat(1, T, 1) # (B, num_cls_tokens * T, embed_dim)
        cls_tokens = einops.rearrange(cls_tokens, 'b (t k) d -> (b t) k d', b=B, t=T, k=self.num_cls_tokens) # (B*T, num_cls_tokens, embed_dim)

        xs = xt
        xs = einops.rearrange(xs, 'b (h w t) d -> (b t) (h w) d', b=B, h=x_patches_h, w=x_patches_w, t=T)
        # prepend CLS tokens along dim=1
        xs = torch.cat((cls_tokens, xs), dim=1)

        if return_attn:
            res_spatial, attn = self.space_attn(self.norm1(xs), return_attn=True)
            res_spatial = self.drop_path(res_spatial)
        else:
            res_spatial = self.drop_path(self.space_attn(self.norm1(xs), return_attn=False))
        
        cls_tokens = res_spatial[:, :self.num_cls_tokens, :] # (B*T, num_cls_tokens, embed_dim)
        cls_tokens = einops.rearrange(cls_tokens, '(b t) k d -> b t k d', b=B, t=T) # (B,T,num_cls_tokens, embed_dim)
        # average over frames
        cls_tokens = torch.mean(cls_tokens, 1, keepdim=True).squeeze(1) # (B, num_cls_tokens, embed_dim)
        
        res_spatial = res_spatial[:, self.num_cls_tokens:, :]
        res_spatial = einops.rearrange(res_spatial, '(b t) (h w) d -> b (h w t) d', b=B, h=x_patches_h, w=x_patches_w, t=T) # (B, T*(x_patches_h*x_patches_w), embed_dim=dim)
        res = res_spatial
        x = xt

        # MLP
        x = torch.cat((init_cls_tokens, x), 1) + torch.cat((cls_tokens, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attn:
            return x, attn.detach()
        else:
            return x
        
class DroppedDividedSpaceTimeBlock(BaseBlock):
    def __init__(
            self, 
            dim, 
            num_heads,
            num_cls_tokens=1,
            mlp_ratio=4, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop_p=0, 
            feat_drop_p=0, 
            path_drop_p=0, 
            act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm,
        ):
        super().__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop_p=attn_drop_p, 
            feat_drop_p=feat_drop_p, 
            path_drop_p=path_drop_p, 
            act_layer=act_layer, 
            norm_layer=norm_layer,
            num_cls_tokens=num_cls_tokens,
        )

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_p=attn_drop_p,
            feat_drop_p=feat_drop_p,
        )
        self.temporal_fc = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        B: int,
        T: int,
        x_patches_h: int,
        x_patches_w: int,
        return_attn: bool = False
    ):
        # x: [B, num_cls + (N_used*T), dim]
        assert x.shape[0] == B and x.shape[2] == self.dim, f"Unexpected shape: {x.shape}"
        assert x.shape[1] >= self.num_cls_tokens, f"Unexpected shape: {x.shape}"

        # Infer number of spatial tokens actually present
        L_patch = x.shape[1] - self.num_cls_tokens
        assert L_patch % T == 0, f"Patch length {L_patch} not divisible by T={T}"
        N_used = L_patch // T  # = Hp*Wp when unmasked, = N_vis when dropped-masked

        # ----- Temporal -----
        # xt: [B, N_used*T, D] -> [(B*N_used), T, D]
        xt = x[:, self.num_cls_tokens:, :]
        xt_bt = einops.rearrange(xt, 'b (n t) d -> (b n) t d', b=B, n=N_used, t=T)

        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt_bt)))
        # back: [(B*N_used), T, D] -> [B, (N_used*T), D]
        res_temporal = einops.rearrange(res_temporal, '(b n) t d -> b (n t) d', b=B, n=N_used, t=T)
        res_temporal = self.temporal_fc(res_temporal)

        xt = x[:, self.num_cls_tokens:, :] + res_temporal  # [B, N_used*T, D]

        # ----- Spatial -----
        init_cls_tokens = x[:, :self.num_cls_tokens, :]  # [B, K, D]

        # replicate CLS per frame for spatial attention: [B, K] -> [B*T, K]
        cls_tokens = init_cls_tokens.repeat(1, T, 1)  # [B, (T*K), D]
        cls_tokens = einops.rearrange(cls_tokens, 'b (t k) d -> (b t) k d', b=B, t=T, k=self.num_cls_tokens)  # [B*T, K, D]

        # patches per frame: [B, N_used*T, D] -> [B*T, N_used, D]
        xs = einops.rearrange(xt, 'b (n t) d -> (b t) n d', b=B, n=N_used, t=T)

        # prepend CLS, attend spatially within each frame
        xs = torch.cat((cls_tokens, xs), dim=1)  # [B*T, K+N_used, D]

        if return_attn:
            res_spatial, attn = self.space_attn(self.norm1(xs), return_attn=True)
            res_spatial = self.drop_path(res_spatial)
        else:
            res_spatial = self.drop_path(self.space_attn(self.norm1(xs), return_attn=False))

        # Extract CLS tokens from each frame and average over frames -> [B, K, D]
        cls_tokens = res_spatial[:, :self.num_cls_tokens, :]  # [B*T, K, D]
        cls_tokens = einops.rearrange(cls_tokens, '(b t) k d -> b t k d', b=B, t=T)  # [B, T, K, D]
        cls_tokens = cls_tokens.mean(dim=1)  # [B, K, D]

        # Extract patch tokens and restore to [B, N_used*T, D]
        res_spatial = res_spatial[:, self.num_cls_tokens:, :]  # [B*T, N_used, D]
        res_spatial = einops.rearrange(res_spatial, '(b t) n d -> b (n t) d', b=B, n=N_used, t=T)  # [B, N_used*T, D]

        # ----- Residual + MLP -----
        x = torch.cat((init_cls_tokens, xt), dim=1) + torch.cat((cls_tokens, res_spatial), dim=1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attn:
            return x, attn.detach()
        else:
            return x