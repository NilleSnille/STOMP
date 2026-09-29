# Scheduler adapted and modified for STOMP from Lightning Bolts, via solo-learn.
# Copyright 2018-2021 William Falcon. Licensed under Apache-2.0.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

import math
import warnings
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


import math
import warnings
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: Optional[float] = None,  # Change to Optional
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float, optional): Learning rate to start the linear warmup. 
                If None and warmup_epochs>0, uses 0.0. If None and warmup_epochs=0, uses base_lrs.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        
        # Handle warmup_start_lr logic properly
        if warmup_start_lr is None:
            if warmup_epochs > 0:
                self.warmup_start_lr = 0.0
            else:
                # No warmup - start from each group's base_lr
                self.warmup_start_lr = None
        else:
            self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # No warmup case - start directly with cosine annealing from base_lrs
        if self.warmup_epochs == 0:
            progress = min(self.last_epoch / self.max_epochs, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
        
        # With warmup case
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
            
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup for each parameter group
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                (self.last_epoch / self.warmup_epochs)
                for base_lr in self.base_lrs
            ]
            
        # Cosine annealing after warmup
        progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Closed form solution for when epoch is passed as param to step()"""
        if self.warmup_epochs == 0:
            progress = min(self.last_epoch / self.max_epochs, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
        
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                (self.last_epoch / self.warmup_epochs)
                for base_lr in self.base_lrs
            ]
        
        progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

class ConstantLR(_LRScheduler):
    """
    Constant learning rate scheduler.

    Keeps the learning rate fixed throughout training. This is useful when you
    want a scheduler object in your loop (to call `.step()` uniformly) but
    don't want the LR to change.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        fixed_lr (float, optional): If provided, override all parameter groups'
            LRs with this value. If None, keep each group's base LR as set on
            the optimizer at construction time. Default: None.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> scheduler = ConstantLR(optimizer)              # use optimizer's base_lrs
        >>> # or force a specific constant LR:
        >>> scheduler = ConstantLR(optimizer, fixed_lr=1e-3)
        >>> for step in range(num_steps):
        ...     # train(...)
        ...     scheduler.step()
    """

    def __init__(
            self, optimizer: Optimizer, 
            fixed_lr: Optional[float] = None, 
            last_epoch: int = -1
        ) -> None:
        self.fixed_lr = fixed_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.fixed_lr is not None:
            return [self.fixed_lr for _ in self.base_lrs]
        # Use the optimizer's original base LRs (constant)
        return [base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> List[float]:
        # Same as get_lr but without the runtime warning; used when `.step(epoch)` is called.
        if self.fixed_lr is not None:
            return [self.fixed_lr for _ in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]