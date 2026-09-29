"""Small CPU checks of the released STOMP path; no videos or weights required."""
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ssl_space.backbones import dropped_dst_vit
from ssl_space.configs.full_config import load_full_config, load_full_config_eval
from ssl_space.methods.stomp import InvLoss, LocLoss, STOMP


@pytest.fixture(autouse=True)
def repository_root(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def test_release_configs_match_paper():
    cfg = load_full_config("configs/methods/stomp/main.yaml")
    assert cfg.method.global_mask_ratio == 0.8
    assert cfg.method.num_prototypes == 1024
    assert cfg.method.sinkhorn_eps == 0.1
    assert cfg.method.sinkhorn_soft_targets is True
    assert cfg.method.use_temporal_decoder is True
    assert cfg.optimizer.max_epochs == 30
    for name in ("finetune", "lp"):
        downstream = load_full_config_eval(f"configs/methods_eval/polypdiag/stomp/{name}.yaml")
        assert downstream.backbone.name == "dropped_dst_vit"


def test_tube_mask_keeps_high_attention_candidates():
    attention = torch.arange(16, dtype=torch.float32).repeat(2, 3, 1)
    mask = STOMP.attn_tube_mask(attention, 0.5, 0.75, 4, 4).reshape(2, 3, 16)
    assert torch.equal(mask[:, 0], mask[:, 1])
    assert torch.equal(mask[:, 1], mask[:, 2])
    assert ((~mask).sum(-1) == 4).all()
    assert mask[:, :, :8].all()


def test_invariance_cross_view_loss_and_center_without_distributed():
    objective = InvLoss(5, 3, 0.1, 0.1, 0, 2, 0.2, 0.9, 2, 1)
    teacher = torch.randn(4, 5)
    student = torch.randn(6, 5, requires_grad=True)
    teacher_probs = (teacher / 0.1).softmax(-1).chunk(2)
    student_logs = (student / 0.2).log_softmax(-1).chunk(3)
    expected = torch.stack([
        -(q * p).sum(-1).mean()
        for i, q in enumerate(teacher_probs)
        for j, p in enumerate(student_logs)
        if i != j
    ]).mean()
    loss = objective(student, teacher, 0)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(objective.center, teacher.mean(0, keepdim=True) * 0.1)
    loss.backward()
    assert torch.isfinite(student.grad).all()


def test_soft_prototype_loss_has_detached_balanced_targets_and_gradients():
    objective = LocLoss(4, 8, 0.1, 10, "uniform", False, 1, "xent", 0.1, True, True)
    scores = torch.linspace(-0.2, 0.2, 128).reshape(32, 4).requires_grad_()
    assignments = objective._find_optimal_assignment(scores, 0.1, 10, 1)
    assert not assignments.requires_grad
    torch.testing.assert_close(assignments.sum(-1), torch.ones(32))
    torch.testing.assert_close(assignments.mean(0), torch.full((4,), 0.25))

    teacher = torch.randn(2, 8, 8, requires_grad=True)
    student = torch.randn(2, 8, 6, requires_grad=True)
    mask = torch.tensor([[[True, False], [True, True]]] * 4)
    head = torch.nn.Linear(6, 8)
    loss = objective(
        [teacher], [student], [mask],
        [dict(batch_size=2, num_frames=2, num_patches_h=2, num_patches_w=2)], head,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert objective.prototypes.grad is not None and torch.isfinite(objective.prototypes.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_reduced_training_step_and_teacher_checkpoint_roundtrip(tmp_path):
    cfg = load_full_config("configs/methods/stomp/main.yaml")
    OmegaConf.set_readonly(cfg, False)
    for backbone in (cfg.backbone, cfg.momentum_backbone):
        backbone.pretrained_kwargs = {}
        backbone.arch_kwargs.img_size = [32, 32]
        backbone.arch_kwargs.embed_dim = 24
        backbone.arch_kwargs.depth = 1
        backbone.arch_kwargs.num_heads = 3
        backbone.arch_kwargs.num_frames = 2
        backbone.arch_kwargs.path_drop_p = 0.0
    cfg.method.proj_hidden_dim = 32
    cfg.method.proj_bottleneck_dim = 12
    cfg.method.proj_output_dim = 16
    cfg.method.num_prototypes = 4
    cfg.augmentation_train.views[0].num_crops = 2
    cfg.augmentation_train.views[1].num_crops = 0
    cfg.augmentation_train.views[2].num_crops = 1
    for key in list(cfg.augmentation_train.views)[3:]:
        cfg.augmentation_train.views[key].num_crops = 0
    model = STOMP(cfg, ipe=1).train()
    views = [torch.randn(2, 3, 2, 32, 32), torch.randn(2, 3, 3, 32, 32), torch.randn(2, 3, 2, 16, 16)]
    batch = (torch.arange(2), views, torch.zeros(2, dtype=torch.long))
    outs = model(batch, 0, 0)
    loss, logs = model.learning_algorithm(outs, 0, 0)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in logs.values())
    optimizers, _, _ = model.configure_optimizers()
    loss.backward()
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.momentum_backbone.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.after_backward_pass(0, 0)
    optimizers[0].step()
    model.after_optimizer_step(0, 0)
    torch.testing.assert_close(model.locloss.prototypes.norm(dim=1), torch.ones(4))

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    restored = dropped_dst_vit("stomp", cfg.backbone.arch_kwargs, {
        "pretrain_method": "generic", "path": str(checkpoint_path),
    }).eval()
    model.momentum_backbone.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(views[0]), model.momentum_backbone(views[0]))
