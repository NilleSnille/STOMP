# Portions adapted and modified for STOMP from solo-learn (MIT).
# Copyright 2021, 2023 solo-learn development team.
# See THIRD_PARTY_NOTICES.md and LICENSES/solo-learn-MIT.txt.

import inspect
from typing import Any, Callable, Dict, List, Sequence, Tuple
from omegaconf import OmegaConf, DictConfig
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from ssl_space.backbones import dropped_dst_vit

from ssl_space.utils.knn import WeightedKNNClassifier
from ssl_space.utils.lars import LARS
from ssl_space.utils.misc import (
    remove_bias_and_norm_from_weight_decay, 
    clip_grads,
)
from ssl_space.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from ssl_space.utils.wd_scheduler import LinearWarmupCosineWD, ConstantWD
from ssl_space.utils.metrics import accuracy_at_k, weighted_mean
from ssl_space.utils.momentum import MomentumUpdater, initialize_momentum_params
from ssl_space.utils.distributed import get_world_size

def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

class BaseMethod(nn.Module):

    _BACKBONES = {"dropped_dst_vit": dropped_dst_vit}
    _OPTIMIZERS = {
        "sgd":      torch.optim.SGD,
        "adam":     torch.optim.Adam,
        "adamw":    torch.optim.AdamW,
        "lars":     LARS,
    }
    _LR_SCHEDULERS = [
        "reduce",
        "cosine",
        "step",
        "exponential",
    ]

    _WD_SCHEDULERS = [
        "cosine",
        "constant",
    ]

    def __init__(self, cfg: DictConfig, ipe: int):
        # in the base method we initialise the (main) dataset, optimisers, backbone, 
        # validation methods, and general training logic
        super().__init__()
        self.cfg = cfg
        self.ipe = ipe
        self.max_epochs = self.cfg.optimizer.max_epochs
        self.max_steps = self.max_epochs * self.ipe
        self.clip_grad: float = self.cfg.optimizer.clip_grad # if > 0 clips grads at this value. See "after_backward_pass"
        self.clip_grad_local: bool = self.cfg.optimizer.clip_grad_local # local or global clipping

        # asserts
        assert self.cfg.backbone.name in self._BACKBONES
        assert self.cfg.optimizer.name in self._OPTIMIZERS
        assert self.cfg.lr_scheduler.name in self._LR_SCHEDULERS
        assert self.cfg.lr_scheduler.interval in ["step", "epoch"]
        assert self.cfg.wd_scheduler.name in self._WD_SCHEDULERS
        assert self.cfg.wd_scheduler.interval in ["step", "epoch"]

        ### BACKBONE ###
        
        self.base_model: Callable = self._BACKBONES[self.cfg.backbone.name]
        self.backbone: nn.Module = self.base_model(
            self.cfg.method.name, 
            self.cfg.backbone.arch_kwargs,
            self.cfg.backbone.pretrained_kwargs,
        )

        self.features_dim: int = self.backbone.num_features

        ### DATA-PIPELINE RELATED ###
        num_small_crops, num_large_crops = 0, 0
        for view_cfg in self.cfg.augmentation_train.views.values():
            if view_cfg.is_small is True:
                num_small_crops += view_cfg.num_crops
            else:
                num_large_crops += view_cfg.num_crops
        self.num_small_crops: int = num_small_crops
        self.num_large_crops: int = num_large_crops
        self.num_crops: int = num_small_crops + num_large_crops

        ### Online linear and knn evaluation ###
        self.linear_eval: bool = self.cfg.online_eval.linear_eval
        if self.linear_eval is True:
            self.online_classifier: nn.Module = nn.Linear(self.features_dim, self.cfg.dataset_train.num_classes)
        
        self.knn_eval: bool = self.cfg.online_eval.knn_eval
        if self.knn_eval is True:
            self.knn = WeightedKNNClassifier(
                k=self.cfg.online_eval.knn_k, T=self.cfg.online_eval.knn_temperature, distance_fx=self.cfg.online_eval.knn_distance_func
            )
        
        ### LOGGING ###
        self.validation_step_outputs = []

        ### PERFORMANCE ###
        # changes mem layout of tensors to be more optimised for convolutions
        self.disable_channel_last:bool = self.cfg.performance.disable_channel_last

        ### Optionally scale Learning rates ###
        if self.cfg.optimizer.scale_lr_linear:
            lr_scale_coef: float = (self.cfg.optimizer.batch_size * get_world_size()) / 256. # scales by the global batch size. 
        else:
            lr_scale_coef: float = 1.0
        
        self.lr = self.cfg.optimizer.lr * lr_scale_coef
        if self.linear_eval is True:
            self.linear_eval_lr: float = self.cfg.online_eval.linear_eval_lr * lr_scale_coef

        self.warmup_start_lr = self.cfg.lr_scheduler.warmup_start_lr
        self.end_lr = self.cfg.lr_scheduler.end_lr 

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """
        Defines learnable parameters for the base class. Update with projector parameters if needed, from the 
        child class.
        Returns: List[Dict[str, Any]]: list of dicts containing learnable parameters and possible settings.
        """
        backbone_learnable_parameters = [
            {
                "name": "backbone",
                "params": self.backbone.parameters()
            }
        ]
        
        online_classifier_learnable_parameters = []
        if self.linear_eval is True:
            online_classifier_learnable_parameters.append(
                {
                    "name": "online_classifier_no_decay", # the group is excluded from weight decay
                    "params": self.online_classifier.parameters(),
                    "lr": self.linear_eval_lr,
                    "weight_decay": 0,
                }
            )
        return backbone_learnable_parameters + online_classifier_learnable_parameters

    def configure_optimizers(self) -> Tuple[List, List, List]:
        """
        Collects learnable parameters and configure the optimizer and lr scheduler. Allows to specify parameters that should not be affected by the 
        lr_scheduler (set them as "static_lr"). Se "idxs_no_scheduler" below. Weight decay scheduling currently decays any parameters 
        not discarded by "remove_bias_and_norm_from_weight_decay".
        Returns: Tuple[List, List]: two lists containing the optimizer and scheduler.
        """

        learnable_params = self.learnable_params
        if self.cfg.optimizer.exclude_bias_n_norm_wd is True:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        idxs_no_scheduler = [i for i,m in enumerate(learnable_params) if m.pop("static_lr", False)]

        # create the optimizer
        opt_cfg = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
        optimizer = (opt_cls := self._OPTIMIZERS[self.cfg.optimizer.name])(
            learnable_params,
            lr=self.lr,
            **{k: opt_cfg[k] for k in inspect.signature(opt_cls).parameters if k in opt_cfg and k != "lr"}
        )
        
        if self.cfg.lr_scheduler.name == "cosine":
            max_warmup_steps = (
                self.cfg.lr_scheduler.warmup_epochs * (self.ipe)
                if self.cfg.lr_scheduler.interval == "step"
                else self.cfg.lr_scheduler.warmup_epochs
            )
            max_scheduler_steps = (
                self.ipe * self.cfg.optimizer.max_epochs
                if self.cfg.lr_scheduler.interval == "step"
                else self.cfg.optimizer.max_epochs
            )

            lr_scheduler = {
                "lr_scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.cfg.lr_scheduler.warmup_epochs > 0 else self.lr,
                    eta_min=self.end_lr
                ),
                "interval": self.cfg.lr_scheduler.interval,
                "frequency": 1,
            }
        elif self.cfg.lr_scheduler.name == "step":
            lr_scheduler = MultiStepLR(optimizer, self.cfg.lr_scheduler.lr_decay_steps)
        else:
            raise ValueError(f"LR Scheduler {self.cfg.lr_scheduler.name} not supported.")
        
        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr, 
                get_lr = lr_scheduler['lr_scheduler'].get_lr if isinstance(lr_scheduler, dict) else lr_scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(lr_scheduler, dict):
                lr_scheduler["lr_scheduler"].get_lr = partial_fn
            else:
                lr_scheduler.get_lr = partial_fn

        if self.cfg.wd_scheduler.name == "cosine":
            max_warmup_steps_wd = (
                self.cfg.wd_scheduler.warmup_epochs * (self.ipe)
                if self.cfg.wd_scheduler.interval == "step"
                else self.cfg.wd_scheduler.warmup_epochs
            )
            max_scheduler_steps_wd = (
                self.ipe * self.cfg.optimizer.max_epochs
                if self.cfg.wd_scheduler.interval == "step"
                else self.cfg.optimizer.max_epochs
            )

            wd_scheduler = {
                "wd_scheduler": LinearWarmupCosineWD(
                    optimizer,
                    warmup_start_wd=self.cfg.wd_scheduler.warmup_start_wd if max_warmup_steps_wd > 0 else None,
                    wd_end=self.cfg.wd_scheduler.end_wd,
                    warmup_steps=max_warmup_steps_wd,
                    max_steps=max_scheduler_steps_wd
                ),
                "interval": self.cfg.wd_scheduler.interval,
                "frequency": 1,
            }
        
        if self.cfg.wd_scheduler.name == "constant":
            wd_scheduler = {
                "wd_scheduler": ConstantWD(
                    optimizer
                ),
                "interval": self.cfg.wd_scheduler.interval,
                "frequency": 1,    
            }
            
        return [optimizer], [lr_scheduler], [wd_scheduler]
    
    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer):
        try: 
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()

    def group_by_resolution(self, X: List[torch.Tensor]) -> List[torch.Tensor]:
        X_grouped, start = [], 0
        for i in range(len(X)):
            if i == len(X)-1 or X[i].shape[1:] != X[i+1].shape[1:]:
                X_grouped.append(torch.cat(X[start:i+1], dim=0))
                start = i + 1
        return X_grouped

    def move_batch_to_device(self, batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor]) -> Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor]:
        """
        Utility function to move the batch to the same device (where the model is defined).
        """
        if len(batch) == 3:
            indexes, crops, targets = batch
            targets = targets.to(self.device, non_blocking=True) 
            if self.disable_channel_last:
                crops = [crop.to(self.device, non_blocking=True) for crop in crops]
            else:
                crops = [crop.to(self.device, non_blocking=True, memory_format=torch.channels_last) for crop in crops]
            return (indexes, crops, targets)
        else:
            raise ValueError("Batch should consist of 3 items")

    def _forward_train(self, X: List[torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        Child classes **should** override this method to implement method specific logic.
        For example, add the projector outputs to the outs dict. 
        Args:
            X (List[torch.Tensor]): batch of images in list format.
        Returns:
            Dict[str,torch.Tensor]: dict with features and possibly projections etc.
        """
        feats = torch.cat([self.backbone(x) for x in X])
        return {"feats": feats}
    
    @torch.no_grad()
    def _forward_val(self, X: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward method for validation over a single tensor. 
        Child classes **can** override this method to implement method specific logic. 
        Always decorate any overide with no_grad()
        Args:
            x (torch.Tensor): batch of images in tensor format.
        Returns:
            Dict[str,torch.Tensor]: dict with features, possibly projections etc.
        """
        feats = torch.cat([self.backbone(x) for x in X])
        return {"feats": feats}
    
    def _classifier_compute(self, classifier: nn.Module, feats: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor,float,float]:
        """
        Computes the classifier loss and accuracy stats. 
        **Detaches features before passing to classifier** 
        Args:
            feats (torch.Tensor): tensor with features.
        Returns:
            Tuple[torch.Tensor,float,float] 
        """
        with torch.no_grad():
            mask = targets != -1          # keep only valid samples
            feats_valid   = feats[mask]
            targets_valid = targets[mask]
    
        logits: torch.Tensor = classifier(feats_valid.detach()) # IMPORTANT - DETACH FEATURES BEFORE PASSING
        cls_loss = F.cross_entropy(logits, targets_valid)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets_valid, top_k=(1, top_k_max))
        return cls_loss, acc1, acc5

    def forward(
        self, 
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor], 
        batch_idx: int, 
        epoch_idx: int,
    ) -> Dict[str,Any]:
        
        """
        Main training step, forwarding one batch. 
        Should not be overridden - instead the child classes override the substeps in this function.
        The child class should override:
            **_forward_train**
        Args:
            batch (Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor]): a batch of data in the format of [indexes, [x], Y], where
                [x] is a list of size self.num_crops containing batches of images. Assumes batch has been
                moved to the correct device already
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the features, optionally classification loss, acc@1 and acc@5.
        """

        _, X, targets = batch
        outs = self._forward_train(X)

        if self.linear_eval is True:
            # only on the large crops
            cls_loss, acc1, acc5 = self._classifier_compute(
                classifier=self.online_classifier,
                feats=outs["feats"][: targets.size(0)*self.num_large_crops], 
                targets=targets.repeat(self.num_large_crops)
            )
            outs.update({"cls_loss": cls_loss, "acc1": acc1, "acc5": acc5})

        if self.knn_eval is True: 
            # update the memory bank in the knn classifier with the new features
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn.update(
                train_features=outs["feats"][: targets.size(0)*self.num_large_crops][mask].detach(),
                train_targets=targets[mask],
            )

        return outs

    def learning_algorithm(
        self, 
        outs: Dict[str, Any], 
        epoch_index: int,
        global_step: int,
    ) -> Tuple[torch.Tensor,Dict[str,Any]]:
        """
        Return the loss computed according to some method - implemented by child class. Also return a dict with logging - detach vals before returning.
        """
        raise NotImplementedError("Must be overridden by child class implementation")

    @torch.no_grad()
    def validation_step(
        self,
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor],
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        
        """
        Validation step. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.
        This will be skipped and return {metrics: batch_size} if neither knn_eval or an online classifier is used.

        Args:
            batch (Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor]): a batch of data in the format of [indexes, [x], Y], where
                [x] is a list of size self.num_crops containing batches of images. Assumes batch has been
                moved to the correct device already
            batch_idx (int): index of the batch.
            update_validation_step_outputs (bool): whether or not to append the metrics to validation_step_outputs

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), and optionally the classification loss and accuracies
            if knn/online classifier is used.
        """
        _, X, targets = batch
        assert len(X) == 1, f"len X ({len(X)} should be 1 during validation"
        metrics = {"batch_size": targets.size(0)}        
        if self.linear_eval is False and self.knn_eval is False:
            return metrics
        
        out = self._forward_val(X)
        
        if self.linear_eval is True:
            val_cls_loss, val_acc1, val_acc5 = self._classifier_compute(
                classifier=self.online_classifier,
                feats=out["feats"], 
                targets=targets
            )
            metrics.update({"val_loss": val_cls_loss, "val_acc1": val_acc1, "val_acc5": val_acc5})
            if update_validation_step_outputs:
                self.validation_step_outputs.append(metrics)
        
        if self.knn_eval is True:
            self.knn.update(test_features=out["feats"].detach(), test_targets=targets.detach())

        return metrics

    def on_validation_epoch_end(self, clear_validation_step_outputs=True):
        """
        Averages the losses and accuracies of all the validation batches.
        Computes (and empties) the the knn classifier.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        This function clear all the knn states, do not call if multiple validations
        needs to be run using the same train features!
        """
        
        log = {}
        if self.linear_eval is True:
            val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
            val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
            val_acc5 = weighted_mean(self.validation_step_outputs, "val_acc5", "batch_size")
            log.update({"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5})

        if self.knn_eval is True:
            # this will clear all bank states, be careful if running validation on multiple datasets.
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        if clear_validation_step_outputs:
            self.validation_step_outputs.clear()

        return log
    
    def before_backward_pass(self, outs: Dict[str, Any], batch_idx: int, epoch_idx: int) -> torch.Tensor:
        """
        Preprocess losses. Adds the cls loss to the ssl loss for bp.
        **Return the loss to backpropagate over**
        """
        bp_loss = outs["ssl_loss"]
        if self.linear_eval is True:         
            bp_loss += outs["cls_loss"]
        return bp_loss

    def after_backward_pass(self, batch_idx: int, epoch_idx: int) -> None:
        clip_grads(self.learnable_params, self.clip_grad, self.clip_grad_local)
        
    def after_optimizer_step(self, batch_idx: int, epoch_idx: int):
        """
        For example, update EMA.
        """
        pass
    
class BaseMethodMomentum(BaseMethod):
    def __init__(self, cfg, ipe):
        """
        Adds shared momentum arguments, basic training steps
        for the momentum backbone. Implements momentum update using exponential moving average (EMA) and 
        cosine annealing of the weighting decrease coefficient.
        """
        super().__init__(cfg, ipe)
        

        ### MOMENTUM BACKBONE ###

        self.momentum_backbone: nn.Module = self.base_model(
            self.cfg.method.name, 
            self.cfg.momentum_backbone.arch_kwargs,
            {},
        )

        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # initilise the momentum updater
        self.momentum_updater = MomentumUpdater(
            base_tau=self.cfg.momentum.base_tau,
            final_tau=self.cfg.momentum.final_tau,
            use_eman=self.cfg.momentum.use_eman,
        )

        ### Momentum linear evaluation ###
        self.linear_eval_momentum: bool = self.cfg.online_eval.linear_eval_momentum
        if self.linear_eval_momentum is True:
            self.momentum_classifier: nn.Module = nn.Linear(self.features_dim, self.cfg.dataset_train.num_classes)
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """
        Defines learnable parameters for the base class. Update with projector parameters if needed, in the 
        child class.
        Returns: List[Dict[str, Any]]: list of dicts containing learnable parameters and possible settings.
        """
        momentum_classifier_learnable_params = []
        if self.linear_eval_momentum is True:
            momentum_classifier_learnable_params.append(
                {
                    "name": "momentum_classifier_no_decay",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.linear_eval_lr,
                    "weight_decay": 0,   
                }
            )
        return super().learnable_params + momentum_classifier_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[nn.Module,nn.Module]]:
        """
        Defines base momentum pairs that will be updated using EMA. If for example projectors are used, 
        override this method in the child class and make sure the online and momentum projectors are 
        treated as an additional pair. 
        **Check the order is correct** - expect online first in the order of each tuple. See MomentumUpdater class

        Returns:
            List[Tuple[nn.Module,nn.Module]]: list of momentum pairs (two element tuples).
        """
        return [(self.backbone, self.momentum_backbone)]

    def train(self, mode: bool=True):
        """
        Sets the non-momentum modules in train mode.
        Ensures momentum modules are always in eval mode.
        """
        super().train(mode)
        for _, m in self.momentum_pairs:
            m.eval()
        return self

    @torch.no_grad()
    def _forward_momentum(self, X: List[torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        Momentum backbone forward method. Assumes a list of tensors or a tensor with only one tensor.
        **Only pass global views**.
        Child classes **should** override this method to implement method specific logic.
        For example, add the projector outputs to the outs dict. 
        Args:
            X (List[torch.Tensor]): list of global views.
        Returns:
            Dict: dict with features, projectors etc.
        """
        feats = torch.cat([self.momentum_backbone(x) for x in X])
        return {"feats": feats}
    
    @torch.no_grad()
    def _forward_momentum_val(self, x: torch.Tensor) -> Dict[str,torch.Tensor]:
        return {"feats": self.momentum_backbone(x)}
    
    def forward(
        self, 
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor], 
        batch_idx: int, 
        epoch_idx: int
    ) -> Dict[str,Any]:
        """
        Main training step for momentum methods, forwarding one batch. 
        Should not be overridden - instead the child classes override the substeps in this function.
        The child class should override:
            **_forward_train**
            **_forward_momentum**
            **_forward_val**
        Args:
            batch (Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor]): a batch of data in the format of [indexes, [x], Y], where
                [x] is a list of size self.num_crops containing batches of images. Assumes batch has been
                moved to the correct device already
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the features, optionally classification loss, acc@1 and acc@5.
        """
        # online branch
        outs = super().forward(batch, batch_idx, epoch_idx)
        _, X, targets = batch
        # forward large crops through momentum branch
        
        with torch.no_grad():
            momentum_outs = self._forward_momentum(X[:self.num_large_crops])

        if self.linear_eval_momentum is True:
            cls_loss, acc1, acc5 = self._classifier_compute(
                classifier=self.momentum_classifier,
                feats=momentum_outs["feats"][: targets.size(0)*self.num_large_crops],
                targets=targets.repeat(self.num_large_crops)
            )
            momentum_outs.update({"cls_loss": cls_loss, "acc1": acc1, "acc5": acc5})

        momentum_outs = {f"momentum_{k}": v for k, v in momentum_outs.items()}
        outs.update(momentum_outs)
        return outs
        
    @torch.no_grad()
    def validation_step(
        self,
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor],
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        
        metrics = super().validation_step(batch, update_validation_step_outputs=False)
        if self.linear_eval_momentum is True:
            _, X, targets = batch
            out = self._forward_momentum_val(X[0])
            val_cls_loss, val_acc1, val_acc5 = self._classifier_compute(
                classifier=self.momentum_classifier,
                feats=out["feats"],
                targets=targets
            )
            metrics.update(
                {"momentum_val_loss": val_cls_loss, "momentum_val_acc1": val_acc1, "momentum_val_acc5": val_acc5}
            )
        
            if update_validation_step_outputs:
                self.validation_step_outputs.append(metrics)
        
        return metrics

    def on_validation_epoch_end(self, clear_validation_step_outputs=True):
        log = super().on_validation_epoch_end(clear_validation_step_outputs=False)
        if self.linear_eval_momentum:
            val_loss = weighted_mean(self.validation_step_outputs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(self.validation_step_outputs, "momentum_val_acc1", "batch_size")
            val_acc5 = weighted_mean(self.validation_step_outputs, "momentum_val_acc5", "batch_size")
            log.update({"momentum_val_loss": val_loss, "momentum_val_acc1": val_acc1, "momentum_val_acc5": val_acc5})
        
        if clear_validation_step_outputs:
            self.validation_step_outputs.clear()
        
        return log
    
    @torch.no_grad()
    def update_momentum_backbone(self, global_step: int):
        """
        Performs the momentum update using EMA. 
        To be called **after** optimizer. Works on single-GPU and DDP wrapped online models.
        """
        momentum_pairs = self.momentum_pairs
        for mp in momentum_pairs:
            self.momentum_updater.update(*mp)
        
        self.momentum_updater.update_tau(
            cur_step=global_step,
            max_steps=self.max_steps
        )

    def before_backward_pass(self, outs: Dict[str, Any], batch_idx: int, epoch_idx: int) -> torch.Tensor:
        bp_loss = super().before_backward_pass(outs, batch_idx, epoch_idx)
        if self.linear_eval_momentum is True:
            bp_loss += outs["momentum_cls_loss"]
        return bp_loss
        
    def after_optimizer_step(self, batch_idx, epoch_idx):
        """
        To be called **after** optimizer step. Works with single-GPU and DDP wrapped online models.
        """
        global_step = epoch_idx*self.ipe + batch_idx
        self.update_momentum_backbone(global_step=global_step)