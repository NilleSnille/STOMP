# Portions adapted and modified for STOMP from solo-learn (MIT).
# Copyright 2021, 2023 solo-learn development team.
# See THIRD_PARTY_NOTICES.md and LICENSES/solo-learn-MIT.txt.

import math

import torch
from torch import nn

@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    momentum_net.load_state_dict(state_dict=online_net.state_dict(), strict=True)
    for p in momentum_net.parameters():
        p.requires_grad = False

class MomentumUpdater:

    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0, use_eman: bool = False):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_tau (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_tau (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau
        self.use_eman = use_eman

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        """Performs the momentum update for each param group.

        Args:
            online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
            momentum_net (nn.Module): momentum network (e.g. momentum backbone,
                momentum projection, etc...).
        """
        # 1) EMA update parameters
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data.mul_(self.cur_tau).add_((1-self.cur_tau)*op.detach().data)
        
        # 2) Optional EMAN: update BN running stats only (Cai et.al 2021, https://arxiv.org/pdf/2101.08482)
        if self.use_eman:

            assert not momentum_net.training, (
                "EMAN enabled but momentum_net is in train() mode. "
                "Put the momentum/teacher network in eval() so BN uses running stats."
            )

            for m_o, m_m in zip(online_net.modules(), momentum_net.modules()):
                if isinstance(m_o, self.BN_TYPES):
                    m_m.running_mean.mul_(self.cur_tau).add_((1 - self.cur_tau) * m_o.running_mean)
                    m_m.running_var.mul_(self.cur_tau).add_((1 - self.cur_tau) * m_o.running_var)
                    # Copy int buffer
                    if hasattr(m_o, "num_batches_tracked") and hasattr(m_m, "num_batches_tracked"):
                        m_m.num_batches_tracked.copy_(m_o.num_batches_tracked)

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )