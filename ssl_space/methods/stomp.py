# Portions adapted and modified for STOMP from DINO (Apache-2.0).
# Copyright (c) Facebook, Inc. and its affiliates.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

from typing import Any, Dict, List, Tuple
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import einops

from ssl_space.methods.base import BaseMethodMomentum
from ssl_space.utils.momentum import initialize_momentum_params
from ssl_space.utils.misc import trunc_normal_
from ssl_space.utils.distributed import get_world_size
from ssl_space.backbones.transformer import decoder_masked_dst_vit

class STOMP(BaseMethodMomentum):
    def __init__(self, cfg, ipe):
        super().__init__(cfg, ipe)

        proj_num_layers: int = self.cfg.method.proj_num_layers
        proj_use_bn: bool = self.cfg.method.proj_use_bn
        proj_norm_last_layer = self.cfg.method.proj_norm_last_layer

        proj_in_dim: int = self.features_dim
        proj_hidden_dim: int = self.cfg.method.proj_hidden_dim
        proj_bottleneck_dim: int = self.cfg.method.proj_bottleneck_dim
        proj_output_dim: int = self.cfg.method.proj_output_dim

        inv_warmup_teacher_temp = self.cfg.method.inv_warmup_teacher_temp # invloss
        inv_teacher_temp = self.cfg.method.inv_teacher_temp
        inv_warmup_teacher_temp_epochs = self.cfg.method.inv_warmup_teacher_temp_epochs
        inv_student_temp = self.cfg.method.inv_student_temp
        inv_center_momentum = self.cfg.method.inv_center_momentum 

        self.freeze_last_layer_epochs: int = self.cfg.method.freeze_last_layer_epochs

        # Create projection head(s)
        self.projector = DINOHead(
            in_dim=proj_in_dim,
            out_dim=proj_output_dim, 
            use_bn=proj_use_bn, 
            norm_last_layer=proj_norm_last_layer,
            nlayers=proj_num_layers,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim
        )

        self.momentum_projector = DINOHead(
            in_dim=proj_in_dim,
            out_dim=proj_output_dim, 
            use_bn=proj_use_bn, 
            norm_last_layer=True,
            nlayers=proj_num_layers,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim
        )
        
        initialize_momentum_params(self.projector, self.momentum_projector)

        self.invloss = InvLoss(
            out_dim=proj_output_dim,
            ncrops=self.num_crops,
            warmup_teacher_temp=inv_warmup_teacher_temp,
            warmup_teacher_temp_epochs=inv_warmup_teacher_temp_epochs,
            teacher_temp = inv_teacher_temp,
            nepochs=self.cfg.optimizer.max_epochs,
            student_temp=inv_student_temp,
            center_momentum=inv_center_momentum,
            global_crops=self.num_large_crops,
            num_tokens=1
        )

        # Global mask generator (generates teacher attention guided masks)
        self.global_mask_generator = partial(
            STOMP.attn_tube_mask,
            candidate_ratio=self.cfg.method.global_mask_candidate_ratio,
            masking_ratio=self.cfg.method.global_mask_ratio,
        )


        ## DECODER
        use_temporal_decoder = self.cfg.method.use_temporal_decoder
        decoder_embed_dim = 384

        if use_temporal_decoder:
            decoder_arch_kwargs = {
                "img_size": [224, 224],
                "patch_size": [16, 16],
                "in_chans": 3,
                "embed_dim": decoder_embed_dim,
                "depth": 4,
                "num_heads": 6,
                "num_cls_tokens": 1,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "qk_scale": None,
                "feat_drop_p": 0.0,
                "attn_drop_p": 0.0,
                "path_drop_p": 0.0,
                "dropout_p": 0.0,
                "act_layer": "gelu",
                "norm_layer": "layernorm",
                "num_frames": 8,
            }
            self.decoder = decoder_masked_dst_vit(
                "",
                decoder_arch_kwargs,
                None,
            )
        else:
            raise ValueError("STOMP uses the divided space-time decoder; set use_temporal_decoder=true.")

        self.decoder_head = nn.Linear(self.decoder.embed_dim, self.momentum_backbone.embed_dim)
        self.encoder_to_decoder = nn.Linear(self.backbone.embed_dim, self.decoder.embed_dim, bias=False)

        num_proto: int = self.cfg.method.num_prototypes
        proto_dim: int = self.momentum_backbone.embed_dim
        sinkhorn_eps: float = self.cfg.method.sinkhorn_eps
        sinkhorn_iters: int = self.cfg.method.sinkhorn_iters
        sinkhorn_logits_temp: float = self.cfg.method.sinkhorn_logits_temp
        sinkhorn_normalise_target: bool = self.cfg.method.sinkhorn_normalise_target
        sinkhorn_soft_targets: bool = self.cfg.method.sinkhorn_soft_targets
        
        # LocLoss module
        self.locloss = LocLoss(
            num_proto=num_proto, 
            proto_dim=proto_dim,
            sk_eps=sinkhorn_eps,
            num_iters=sinkhorn_iters,
            init_distribution='uniform',
            broadcast=False,
            loss_name='xent',
            logits_temp=sinkhorn_logits_temp,
            normalise_target=sinkhorn_normalise_target, 
            world_size=get_world_size(),
            soft_targets=sinkhorn_soft_targets,
        )

        self.loc_strength: float = self.cfg.method.recon_strength
        self.invariance_strength: float = self.cfg.method.invariance_strength

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "decoder", "params": self.decoder.parameters()},
            {"name": "decoder_head", "params": self.decoder_head.parameters()}, 
            {"name": "encoder_to_decoder", "params": self.encoder_to_decoder.parameters()},
            {"name": "prototypes_no_decay", "params": self.locloss.parameters(), "weight_decay": 0.0}, # no weight decay on prototypes
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs
    
    def after_backward_pass(self, batch_idx, epoch_idx):
        super().after_backward_pass(batch_idx, epoch_idx)
        if epoch_idx >= self.freeze_last_layer_epochs:
            return
        
        for n,p in self.projector.named_parameters():
            if "last_layer" in n:
                p.grad = None

    @torch.no_grad()
    def _normalize_prototypes(self):
        w = self.locloss.prototypes.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.locloss.prototypes.data.copy_(w)

    def after_optimizer_step(self, batch_idx, epoch_idx):
        super().after_optimizer_step(batch_idx, epoch_idx)
        self._normalize_prototypes()

    @staticmethod
    @torch.no_grad()
    def attn_tube_mask(
        attention: torch.Tensor, # [B,T, num_patches (spatial)] 
        candidate_ratio: float,
        masking_ratio: float, 
        x_patches_h: int,
        x_patches_w: int,
    ) -> torch.Tensor:

        bs, T, num_patches = attention.size()
        #assert num_patches == x_patches_h * x_patches_w, f"check attention shape"

        attention = attention.mean(1) # shape [B, num_patches]
        N_vis = math.ceil(num_patches * (1.0 - masking_ratio)) # total number of visible patches
        num_top_candidates = int(num_patches * candidate_ratio) # number of patches to chose from.
        #assert N_vis <= num_top_candidates, "candidate_ratio must be >= (1 - masking_ratio)"

        _, top_candidate_idxs = torch.topk(attention, k=num_top_candidates, dim=1)
        chosen_top_idxs = torch.randperm(num_top_candidates)[:N_vis]
        result_random_top = top_candidate_idxs[:, chosen_top_idxs]

        masks_attn = torch.ones((bs, num_patches), dtype=torch.bool, device=attention.device)
        masks_attn.scatter_(dim=-1, index=result_random_top.long(), value=False)
        masks_attn = masks_attn.unsqueeze(1).expand(-1, T, -1).reshape(bs * T, -1)
        mask = masks_attn.reshape(-1, x_patches_h, x_patches_w)
        return mask

    @torch.no_grad()
    def _forward_momentum(self, X: List[torch.Tensor]) -> Dict[str,Any]:
        feats_cls_list, feats_patch_list, info_list = [], [], []
        for x in X:
            feats_i, info_i = self.momentum_backbone(
                    x=x, 
                    mask=None, 
                    get_attn=True, 
                    get_all=True, 
                    get_info=True
            )
            feats_cls_list.append(feats_i[:, 0])
            feats_patch_list.append(feats_i[:, 1:])
            info_list.append(info_i)

        assert len(info_list) == len(X) and len(feats_cls_list) == len(X)
        return {
            "feats_cls_list": feats_cls_list, 
            "feats_patch_list": feats_patch_list,
            "info_list": info_list,
        }
    
    def _forward_train_masked(self, X: List[torch.Tensor], masks: List[torch.Tensor]):
        feats_cls_list = []
        recon_patch_list = []

        for x, mask in zip(X, masks):
            x_orig = x  # [B, C, T, H, W]

            x_enc, info = self.backbone(
                x=x_orig,
                mask=mask,
                get_attn=False,
                get_all=True,
                get_info=True,
            )
            # x_enc: [B, 1 + (N_vis*T), C]   (because dropped tokens)
            # mask:  [B*T, Hp, Wp]            True=masked
            # vis_idx: [B, N_vis]             spatial indices into N=Hp*Wp

            vis_idx: torch.Tensor = info["vis_idx"]  # [B, N_vis]
            B = info["batch_size"]
            T = info["num_frames"]
            Hp = info["num_patches_h"]
            Wp = info["num_patches_w"]
            C = self.features_dim
            N = Hp * Wp

            # ---- CLS features ----
            feats_cls_list.append(x_enc[:, 0])  # [B, C]

            # ----- Decoder ------
            x_vis: torch.Tensor = self.encoder_to_decoder(x_enc[:, self.backbone.num_cls_tokens:]) # [B, N_vis*T, C_d]

            # reshape to [B, T, N_vis, C_d]
            x_vis_btnvisc = einops.rearrange(x_vis, 'b (nvis t) c -> b t nvis c', t=T)
            # allocate dense [B, T, N, C_d]
            N = Hp * Wp
            x_dense_btnc = x_vis_btnvisc.new_zeros((B, T, N, x_vis.size(-1)))
            # scatter using vis_idx along N
            vis_idx_btnvis = einops.repeat(vis_idx, 'b nvis -> b t nvis', t=T)  # [B, T, N_vis]
            x_dense_btnc.scatter_(
                dim=2,
                index=vis_idx_btnvis[..., None].expand(B, T, vis_idx.size(1), x_vis.size(-1)),
                src=x_vis_btnvisc
            )            
            # flatten to [B*T, N, C_d]
            x_dense_btnc = einops.rearrange(x_dense_btnc, 'b t n c -> (b t) n c')

            x_rec = self.decoder(
                x=x_dense_btnc,
                mask=mask,
                x_patches_h=Hp,
                x_patches_w=Wp,
                B=B,
                T=T,
            ) # [B, N*T, C_d]

            recon_patch_list.append(x_rec)
        
        return {"feats_cls_list": feats_cls_list, "recon_patch_list": recon_patch_list}

    def _forward_train(self, X: List[torch.Tensor]):
        # we group the inputs based on shapes to make fewer kernel calls
        feats_cls_list = []
        start_idx = 0
        for i in range(len(X)):
            last = (i == len(X) - 1)
            if last or X[i].shape[1:] != X[i+1].shape[1:]:
                feats_cls_list.append(
                    self.backbone(torch.cat(X[start_idx:i+1]))
                )
                start_idx = i + 1
        feats_cls = torch.cat(feats_cls_list)
        return {"feats_cls": feats_cls}

    def forward(
        self, 
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor], 
        batch_idx: int, 
        epoch_idx: int
    ) -> Dict[str,Any]:
        
        # online branch - get the cls outputs for the invariance loss.
        _, X, _ = batch

        # forward large crops through momentum branch. Also collect the attention maps, and build global masks.     
        with torch.no_grad():
            glo_masks = []

            momentum_outs = self._forward_momentum(X[:self.num_large_crops])
            for il in momentum_outs["info_list"]:
                attn: torch.Tensor  = il["attention"]
                batch_size: int     = il["batch_size"]
                num_frames: int     = il["num_frames"]
                x_patches_h: int    = il["num_patches_h"]
                x_patches_w: int    = il["num_patches_w"]
                
                attn_cls = attn[:, :, 0, 1:].mean(1).reshape(batch_size, num_frames, x_patches_h*x_patches_w).detach()
                masks_attn = self.global_mask_generator(
                    attn_cls, x_patches_h=x_patches_h, x_patches_w=x_patches_w
                )
                glo_masks.append(masks_attn)

        online_glo_outs = self._forward_train_masked(X[:self.num_large_crops], masks=glo_masks)
        online_loc_outs = self._forward_train(X[self.num_large_crops:])

        momentum_outs = {f"momentum_{k}": v for k, v in momentum_outs.items()}
        online_glo_outs = {f"online_glo_{k}": v for k, v in online_glo_outs.items()}
        online_loc_outs = {f"online_loc_{k}": v for k, v in online_loc_outs.items()}

        outs = {}
        outs.update(momentum_outs)
        outs.update(online_glo_outs)
        outs.update(online_loc_outs)
        outs["glo_masks"] = glo_masks

        return outs

    def learning_algorithm(self, outs, batch_idx: int, epoch_idx: int):
        online_z = self.projector(
            torch.cat(
                [
                    torch.cat(outs["online_glo_feats_cls_list"], dim=0), outs["online_loc_feats_cls"]
                ],  dim=0
            )
        )
        momentum_z = self.momentum_projector(
            torch.cat(
                outs["momentum_feats_cls_list"], dim=0
            )
        )
        invloss: torch.Tensor = self.invloss(online_z, momentum_z, epoch_idx)

        momentum_patch_list:List[torch.Tensor] = outs["momentum_feats_patch_list"] # len = num_large_crops
        online_glo_patch_list:List[torch.Tensor] = outs["online_glo_recon_patch_list"] # len = num_large_crops
        glo_masks:List[torch.Tensor] = outs["glo_masks"] # len = num_large_crops
        info_list: List[Dict[str,Any]] = outs["momentum_info_list"] # len = num_large_crops

        proto_loss: torch.Tensor = self.locloss(
            momentum_patch_list,
            online_glo_patch_list,
            glo_masks,
            info_list,
            self.decoder_head,
        )

        # reconstruct and compute loc loss
        loss = self.invariance_strength * invloss + self.loc_strength * proto_loss

        loss_log = {
            "inv_loss": invloss.detach(),
            "loc_loss": proto_loss.detach() 
        }

        return loss, loss_log

class LocLoss(nn.Module):
    def __init__(
        self,
        num_proto: int,
        proto_dim: int,
        sk_eps: float,
        num_iters: int,
        init_distribution: str, # normal or uniform
        broadcast: bool, # if we want all processes to share prototypes
        world_size: int,
        loss_name: str,
        logits_temp: float,
        normalise_target: bool, 
        soft_targets: bool = True
    ):
        super().__init__()
        self.num_proto = num_proto
        self.proto_dim = proto_dim
        self.sk_eps = sk_eps
        self.num_iters = num_iters
        self.world_size = world_size
        self.logits_temp = logits_temp
        self.normalise_target = normalise_target
        self.soft_targets = soft_targets
        
        # create prototypes.
        if init_distribution == "normal":
            _protos = torch.randn(num_proto, proto_dim)
        elif init_distribution == "uniform":
            _protos = torch.empty(num_proto, proto_dim)
            _sqrt_k = (1./proto_dim)**0.5
            torch.nn.init.uniform_(_protos, -_sqrt_k, _sqrt_k)
        else:
            raise ValueError()
        
        assert _protos is not None
        self.prototypes = torch.nn.Parameter(F.normalize(_protos, dim=1, p=2))

        if broadcast:
            with torch.no_grad():
                self.prototypes.copy_(_protos)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(self.prototypes.data, src=0)

        if loss_name == 'xent' and not self.soft_targets:
            self.loss_func = nn.CrossEntropyLoss()
        elif self.soft_targets:
            self.loss_func = LocLoss.soft_ce
        else:
            raise ValueError(f"Currently expects xent, got {loss_name}")

    @staticmethod    
    def soft_ce(student_logits: torch.Tensor, target_probs: torch.Tensor, temp: float) -> torch.Tensor:
        # student_logits: [N, K]
        # target_probs:   [N, K]  (must sum to 1 over K)
        log_p = F.log_softmax(student_logits / temp, dim=-1)
        return -(target_probs * log_p).sum(dim=-1).mean()

    @staticmethod
    @torch.no_grad()
    def _sinkhorn_knopp(Q: torch.Tensor, nmb_iters: int, world_size) -> torch.Tensor:
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / (B * world_size)

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()    
    def _find_optimal_assignment(self, scores, epsilon, sinkhorn_iterations, world_size) -> torch.Tensor:
        """
        Computes the Sinkhorn matrix Q.
        :param scores: similarity matrix
        :return: Sinkhorn matrix Q
        """
        q = torch.exp(scores / epsilon).t()
        q = self._sinkhorn_knopp(q, sinkhorn_iterations, world_size=world_size)
        return q

    def _compute_score(self, patch_features: torch.Tensor, detach: bool) -> torch.Tensor:
        prototypes = self.prototypes.detach() if detach else self.prototypes
        normalised_patch_features = F.normalize(patch_features, dim=-1, p=2)
        batch_scores = normalised_patch_features @ prototypes.t() # similarity matrix between patches and prototypes: [num_patches, num_proto]
        return batch_scores

    def forward(
        self, 
        momentum_patch_list: List[torch.Tensor], 
        online_patch_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        info_list: List[Dict],
        decoder_head: nn.Linear
    ):
        proto_loss = 0.
        num_views = 0.

        for mp_v, op_v, mask, info in zip(
            momentum_patch_list,
            online_patch_list,
            mask_list,
            info_list,
        ):
            with torch.no_grad():
                B = info["batch_size"]
                T = info["num_frames"]
                H = info["num_patches_h"] 
                W = info["num_patches_w"]

                if self.normalise_target:
                    mean_mp_v = mp_v.mean(dim=-2, keepdim=True)
                    var_mp_v = mp_v.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1.0e-6
                    mp_v = (mp_v - mean_mp_v) / var_mp_v

                mask_bhwt = einops.rearrange(mask, '(b t) h w -> (b h w t)', b=B, t=T, h=H, w=W)
                mv_bhwt_c = einops.rearrange(mp_v, 'b (h w t) c -> (b h w t) c', h=H, w=W, t=T)
                mv_masked = mv_bhwt_c[mask_bhwt] # (num_masked c)

            ov_bhwt_c = einops.rearrange(op_v, 'b (h w t) c -> (b h w t) c', h=H, w=W, t=T) # specifically local_bs * 14 * 14 * 8 or 16 x 768 = 25,088 or 50,176    
            ov_masked = decoder_head(ov_bhwt_c[mask_bhwt]) # (num_masked c)

            scores_o = self._compute_score(patch_features=ov_masked, detach=False)
            scores_m = self._compute_score(patch_features=mv_masked, detach=False)

            q_m = self._find_optimal_assignment(
                scores=scores_m,
                epsilon=self.sk_eps,
                sinkhorn_iterations=self.num_iters,
                world_size=self.world_size,
            ) # soft assignment matrix: [num_patches, num_proto]
            
            q_o = self._find_optimal_assignment(
                scores=scores_o,
                epsilon=self.sk_eps,
                sinkhorn_iterations=self.num_iters,
                world_size=self.world_size,
            ) # soft assignment matrix: [num_patches, num_proto]

            if not self.soft_targets:
                q_m = q_m.argmax(dim=-1) # make hard assignment
                q_o = q_o.argmax(dim=-1)
                proto_loss += self.loss_func(scores_o/self.logits_temp, q_m) + self.loss_func(scores_m/self.logits_temp, q_o) 
                num_views += 1.
            else:
                # Soft-target distillation
                loss_o = self.loss_func(scores_o, q_m, self.logits_temp)

                # Optional: keep the momentum-side term to mirror hard-target objective
                # Note q_m is computed from scores_m, but q_m is detached (no_grad), so this is stable.
                loss_m = self.loss_func(scores_m, q_o, self.logits_temp)

                proto_loss += (loss_o + loss_m)
                num_views += 1.

        proto_loss /= num_views
        return proto_loss

class DINOHead(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        use_bn, 
        norm_last_layer, 
        nlayers, 
        hidden_dim, 
        bottleneck_dim
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class InvLoss(nn.Module):
    # See DINO - invariance loss through centered cross entropy https://arxiv.org/pdf/2104.14294
    def __init__(
        self, 
        out_dim, 
        ncrops, 
        warmup_teacher_temp, 
        teacher_temp,
        warmup_teacher_temp_epochs, 
        nepochs, 
        student_temp,
        center_momentum, 
        global_crops,
        num_tokens,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.num_tokens = num_tokens
        if self.num_tokens == 1:
            self.register_buffer("center", torch.zeros(1, out_dim))
        else:
            raise ValueError("expects num_tokens = 1.")
        
        # warm up for the teacher temperature because
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
            )
        )

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        world_size = get_world_size()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * world_size)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_output: torch.Tensor, teacher_output:torch.Tensor, epoch:int) -> torch.Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0.
        n_loss_terms = 0
        if self.num_tokens == 1:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.n_crops)
            
            teacher_temp = self.teacher_temp_schedule[epoch]
            # center, sharpen, softmax outs
            teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(self.global_crops)

            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
        
        else:
            raise ValueError("expects num_tokens = 1.")
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss