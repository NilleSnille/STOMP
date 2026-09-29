from typing import List
import numpy as np
import warnings
import torch

class LinearWarmupCosineWD:
    """
    Warmup (linear) from warmup_wd_start -> wd_base, then cosine to end_wd,
    applied only to the selected param-group indices.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_start_wd: float|None,
        wd_end: float,
        max_steps: int,
        warmup_steps: int,
    ):
        self.optimizer = optimizer
        self.wd_base = optimizer.defaults.get("weight_decay", None)
        assert self.wd_base is not None, f"Could not retrive weight decay from optimizer {optimizer}."

        self.warmup_start_wd = warmup_start_wd
        self.wd_end = wd_end
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

        # make the schedule
        warmup_schedule = np.array([])
        if warmup_steps > 0:
            assert warmup_start_wd is not None
            warmup_schedule = np.linspace(warmup_start_wd, self.wd_base, warmup_steps)

        iters = np.arange(max_steps - warmup_steps)
        schedule = self.wd_end + 0.5 * ( self.wd_base - self.wd_end ) * ( 1.0 + np.cos( np.pi * iters / len(iters) ) )
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == max_steps
        self.schedule = schedule

        # We register the group indices that should be regularized with decay

        self.wd_indices: List[int] = []
        for pg_idx, param_group in enumerate(optimizer.param_groups):
            gn: str|None = param_group.get("name", None)
            if gn is not None and gn.endswith('_no_decay'):
                # the group has a name, and it is NOT decayed
                continue 
            
            if gn is None:
                warnings.warn("Group without name found")
                # we add it if it has non-zero weight decay specified. Can go wrong
                # if we have passed weight decay = 0 to the optimizer!!!
                if param_group.get("weight_decay", 0.) > 0.:
                    self.wd_indices.append(pg_idx)
            else:
                self.wd_indices.append(pg_idx)

        for pg_idx in self.wd_indices:
            self.optimizer.param_groups[pg_idx]["weight_decay"] = self.schedule[0]

        self.step_idx = 0

    def get_wd(self) -> float:
        return float(self.schedule[self.step_idx])
    
    def step(self):
        idx = min(self.step_idx, self.max_steps - 1)
        wd = float(self.schedule[idx])
        pgs = self.optimizer.param_groups

        for i in self.wd_indices:
            pgs[i]['weight_decay'] = wd

        if self.step_idx >= self.max_steps:
            warnings.warn("LinearWarmupCosineWD: step index exceeded max_steps; using final value.")

        self.step_idx += 1


class ConstantWD:
    """
    Constant wd, applied only to the selected param-group indices.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer
    ):
        self.optimizer = optimizer
        # We register the group indices that should be regularized with decay

        self.wd_indices: List[int] = []
        self.wd_vals: List[float] = []
        for pg_idx, param_group in enumerate(optimizer.param_groups):
            gn: str|None = param_group.get("name", None)
            if gn is not None and gn.endswith('_no_decay'):
                # the group has a name, and it is NOT decayed
                continue 
            
            if gn is None:
                warnings.warn("Group without name found")
                # we add it if it has non-zero weight decay specified. Can go wrong
                # if we have passed weight decay = 0 to the optimizer!!!
                if param_group.get("weight_decay", 0.) > 0.:
                    self.wd_indices.append(pg_idx)
                    self.wd_vals.append(param_group.get("weight_decay", 0.0))
            else:
                self.wd_indices.append(pg_idx)
                self.wd_vals.append(param_group.get("weight_decay", 0.0))
            

        for pg_idx, pg_wd in zip(self.wd_indices, self.wd_vals):
            self.optimizer.param_groups[pg_idx]["weight_decay"] = pg_wd 

    def get_wd(self) -> List[float]:
        return self.wd_vals.copy()
    
    def step(self):
        pgs = self.optimizer.param_groups

        for pg_idx, pg_wd in zip(self.wd_indices, self.wd_vals):
            pgs[pg_idx]['weight_decay'] = pg_wd
