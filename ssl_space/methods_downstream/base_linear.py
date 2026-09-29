import inspect
from typing import Any, Callable, Dict, List, Tuple
from omegaconf import OmegaConf, DictConfig

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import CalibrationError

from ssl_space.backbones import dropped_dst_vit
from ssl_space.utils.lars import LARS
from ssl_space.utils.lr_scheduler import LinearWarmupCosineAnnealingLR, ConstantLR
from ssl_space.utils.wd_scheduler import LinearWarmupCosineWD, ConstantWD
from ssl_space.utils.metrics import accuracy_at_k
from ssl_space.utils.misc import remove_bias_and_norm_from_weight_decay
from ssl_space.utils import distributed

class LinearClassifier(nn.Module):
    """Linear layer to train."""
    def __init__(self, dim, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class BaseLinear(nn.Module):

    _BACKBONES = {"dropped_dst_vit": dropped_dst_vit}

    _OPTIMIZERS = {
        "sgd":      torch.optim.SGD,
        "adam":     torch.optim.Adam,
        "adamw":    torch.optim.AdamW,
        "lars":     LARS,
    }
    _LR_SCHEDULERS = [
        "cosine",
        "constant", "no_schedule",
    ]

    _WD_SCHEDULERS = [
        "cosine",
        "constant", "no_schedule",
    ]

    def __init__(
            self, 
            cfg: DictConfig,
            ipe: int,
        ):
        """
        Implements linear evaluation. 
        FIX: IMPLEMENT FINE TUNING AS WELL
        """
        super().__init__()
        self.cfg = cfg
        self.ipe = ipe
        ### BACKBONE ###
        ### BACKBONE ###

        ### METHOD ###
        self.finetune: bool = self.cfg.method_eval.finetune
        self.normalize: bool = self.cfg.method_eval.normalize
        #self.pick_ckpt_by: str = self.cfg.method_eval.pick_ckpt_by # acc1, f1_macro, f1_micro, f1_binary, recall, precision, null. If None, we always take the last ckpt.
        #self.save_dict = {}

        self.base_model: Callable = self._BACKBONES[self.cfg.backbone.name]
        self.backbone: nn.Module = self.base_model(
            self.cfg.method_eval.name, 
            self.cfg.backbone.arch_kwargs,
            self.cfg.backbone.pretrained_kwargs,
        )
        
        self.features_dim: int = self.backbone.num_features

        #### CLASSIFIER ###
        self.num_classes: int = self.cfg.dataset_train.num_classes
        self.classifier = LinearClassifier(self.features_dim, self.num_classes)

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False


        ### PERFORMANCE ###
        self.disable_channel_last: bool = self.cfg.performance.disable_channel_last

        ### LOGGING ###
        self.validation_step_indices = []
        self.validation_step_logits = []
        self.validation_step_targets = []

        ### OPTIMIZER ###
        self.max_epochs = cfg.optimizer.max_epochs
        if self.cfg.optimizer.scale_lr_linear:
            lr_scale_coef: float = (self.cfg.optimizer.batch_size * distributed.get_world_size()) / 256. # scales by the global batch size. 
        else:
            lr_scale_coef: float = 1.0
        
        self.lr = self.cfg.optimizer.lr * lr_scale_coef
        self.backbone_lr_scale = self.cfg.optimizer.backbone_lr_scale
        self.backbone_wd_scale = self.cfg.optimizer.backbone_wd_scale

        self.use_flooding = False
        self.flood_level = 0.05 if self.use_flooding else 0.0
        print(f"flooding: {self.use_flooding} | level: {self.flood_level}")


    @property
    def learnable_params(self) -> List[Dict[str,Any]]:
        '''
        Defines the learnable parameters. If finetuning, the backbone and classifier are learnable. 
        Otherwise, only the classifier are learnable.
        '''
        learnable_params_list = []

        learnable_params_list.append(
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.lr,
                "weight_decay": self.cfg.optimizer.weight_decay
            }
        )

        if self.finetune is True:
            learnable_params_list.append(
                {
                    "name": "backbone",
                    "params": self.backbone.parameters(),
                    "lr": self.lr * self.backbone_lr_scale,
                    "weight_decay": self.cfg.optimizer.weight_decay * self.backbone_wd_scale
                }
            )

        return learnable_params_list

    def configure_optimizers(self) -> Tuple[List, List, List]:
        """
        Collects learnable parameters and configure the optimizer, lr scheduler, wd scheduler. Weight decay scheduling currently decays any parameters 
        not discarded by "remove_bias_and_norm_from_weight_decay".
        Returns: Tuple[List, List, List]: three lists containing the optimizer, lr_scheduler and wd_sceduler.
        """

        learnable_params = self.learnable_params
        if self.cfg.optimizer.exclude_bias_n_norm_wd is True:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

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
                    warmup_start_lr=self.cfg.lr_scheduler.warmup_start_lr if self.cfg.lr_scheduler.warmup_epochs > 0 else None,
                    eta_min=self.cfg.lr_scheduler.end_lr
                ),
                "interval": self.cfg.lr_scheduler.interval,
                "frequency": 1,
            }
        elif self.cfg.lr_scheduler.name in {"constant", "no_schedule"}:
            lr_scheduler = {
                "lr_scheduler": ConstantLR(
                    optimizer
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f"LR Scheduler {self.cfg.lr_scheduler.name} not supported.")

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
        elif self.cfg.wd_scheduler.name in {"constant", "no_schedule"}:
            wd_scheduler = {
                "wd_scheduler": ConstantWD(
                    optimizer
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f"WD Scheduler {self.cfg.wd_scheduler.name} not supported.")
            
        return [optimizer], [lr_scheduler], [wd_scheduler]
    
    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer):
        try: 
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()


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
        '''
        Forward the samples through the backbone and classifier. If self.cfg.method.finetune is False, 
        we do not compute gradients through the backbone. Gradients always enabled for the classifier.
        Args:
            X (List[torch.Tensor]): batch of images in list format.
        Returns:
            Dict[str,torch.Tensor]: dict with backbone features and logits from classifier.
        '''

        with torch.set_grad_enabled(self.finetune):
            feats = torch.cat([self.backbone(x) for x in X])
            if self.normalize:
                feats = F.normalize(feats, p=2)
            
        logits = self.classifier(feats)

        return {"feats": feats, "logits": logits}
    
    @torch.no_grad()
    def _forward_val(self, X: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = torch.cat([self.backbone(x) for x in X])
        if self.normalize:
            feats = F.normalize(feats, p=2)
        
        logits = self.classifier(feats)

        return {"feats": feats, "logits": logits}
    
    def forward(
        self, 
        batch: Tuple[torch.Tensor,List[torch.Tensor],torch.Tensor], 
        batch_idx: int, 
        epoch_idx: int,
    ) -> Dict[str,Any]:
        """Performs forward pass of the possibly frozen backbone and the linear layer.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        _, X, targets = batch
        outs = self._forward_train(X)
        acc1 = accuracy_at_k(outs["logits"].detach(), targets, top_k=(1,))[0]
        outs.update({"acc1": acc1})
        return outs
    
    def learning_algorithm(
        self, 
        outs: Dict[str,torch.Tensor],
        targets: Any,
        batch_idx: int,
        epoch_idx: int, 
    ) -> torch.Tensor:
        """
        Return the loss. Can be overriden by child class.
        """
        if self.use_flooding:
            loss = torch.abs(F.cross_entropy(outs["logits"], targets, ignore_index=-1) - self.flood_level) + self.flood_level 
        else:
            loss = F.cross_entropy(outs["logits"], targets, ignore_index=-1)
        return loss
    
    @torch.no_grad()
    def validation_step(
            self, 
            batch: torch.Tensor
        ) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        indices, X, targets = batch
        assert len(X) == 1, f"len X ({len(X)}) should be 1 during validation"
        
        outs = self._forward_val(X)
        logits = outs["logits"].detach()
        targets = targets.detach()

        self.validation_step_indices.append(indices) 
        self.validation_step_logits.append(logits)
        self.validation_step_targets.append(targets)
        
        return
    
    def on_validation_end(self, dset: Any = None):
        local_logits = torch.cat(self.validation_step_logits)
        local_targets = torch.cat(self.validation_step_targets)
        local_indices = torch.cat(self.validation_step_indices).to(device=local_logits.device)
        self.validation_step_logits.clear()
        self.validation_step_targets.clear()
        self.validation_step_indices.clear()

        glo_logits = distributed.gather_varlen_tensors(local_logits)
        glo_targets = distributed.gather_varlen_tensors(local_targets)
        glo_indices = distributed.gather_varlen_tensors(local_indices)

        if not distributed.is_main_process():
            assert glo_logits is None and glo_targets is None
            return None
        
        assert glo_logits is not None and glo_targets is not None and glo_indices is not None
        loss = nn.functional.cross_entropy(glo_logits, glo_targets)
        
        # move to cpu
        y_true = glo_targets.detach().cpu().numpy()
        y_score = glo_logits.detach().cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        indices = glo_indices.detach().cpu().numpy()
        num_classes = self.num_classes

        if dset is not None:
            majority_vote: Dict[str,Dict[str,Any]] = dset.majority_vote(indices, y_pred)
            videos, vote_res, target_vote = [], [], []

            print("\n" + "="*40)
            print("VIDEO VOTE RESULTS")
            print("="*40)
            print(f"{'Video':<25} {'Vote':<8} {'Target':<8}")
            print("-"*40)

            for video, vote in majority_vote.items():
                videos.append(video)
                vote_val = vote["vote"].item()
                target_val = vote["target"]

                vote_res.append(vote_val)
                target_vote.append(target_val)

                print(f"{video:<25} {vote_val:<8} {target_val:<8}")

            # Convert to numpy arrays for sklearn metrics
            video_predictions = np.array(vote_res)
            video_targets = np.array(target_vote)
            
            # Calculate video-level metrics
            video_accuracy = float(sum(video_predictions == video_targets)) / float(video_targets.shape[0])
            video_f1_macro = f1_score(video_targets, video_predictions, average='macro')
            video_precision_macro = precision_score(video_targets, video_predictions, average='macro')
            video_recall_macro = recall_score(video_targets, video_predictions, average='macro')
            
            print(f"Video-level Metrics:")
            print(f"  Accuracy: {video_accuracy:.4f}")
            print(f"  F1 Macro: {video_f1_macro:.4f}")
            print(f"  Precision Macro: {video_precision_macro:.4f}")
            print(f"  Recall Macro: {video_recall_macro:.4f}")

        acc1 = float(sum(y_true == y_pred)) / float(y_true.shape[0])
        precision_macro = float(precision_score(y_true=y_true, y_pred=y_pred, average='macro'))
        recall_macro = float(recall_score(y_true=y_true, y_pred=y_pred, average='macro'))
        f1_macro = float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
        if num_classes == 2:
            precision_binary = float(precision_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1))
            recall_binary = float(recall_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=1)) 
            f1_binary = float(f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1))
            # Binary ECE (using probability of the positive class)
            probs = torch.softmax(glo_logits, dim=1)
            ece_bin = CalibrationError(task="binary").to(glo_logits.device)
            ece_bin.update(probs[:, 1], glo_targets)
            ece_value = ece_bin.compute().item()

            try:
                roc_auc = roc_auc_score(y_true, probs[:, 1].detach().cpu().numpy())
            except ValueError:
                roc_auc = float("nan")  # occurs if y_true has a single class

        else:
            precision_binary = float("nan")
            recall_binary = float("nan")
            f1_binary = float("nan")
            roc_auc = float("nan")
            probs = torch.softmax(glo_logits, dim=1)
            ece_mul = CalibrationError(task="multiclass", num_classes=num_classes).to(glo_logits.device)
            ece_mul.update(probs, glo_targets)
            ece_value = ece_mul.compute().item()


        metrics = {
            "loss": loss.detach().item(),
            "acc1": acc1,
            "precision_macro": precision_macro,
            "precision_binary": precision_binary,
            "recall_macro": recall_macro,
            "recall_binary": recall_binary,
            "f1_macro": f1_macro,
            "f1_binary": f1_binary,
            "ece": ece_value,
            "roc_auc": roc_auc,
        }

        return metrics