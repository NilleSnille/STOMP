"""Small CPU regressions for video data loading and released configuration paths."""

import csv
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import av
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from ssl_space.configs.augmentations import load_aug_config
from ssl_space.datasets import make_dataset, polypdiag
from ssl_space.transforms import make_transform_pipeline
from ssl_space.transforms.video import make_video_transform as video

ROOT = Path(__file__).resolve().parents[1]


def test_random_resized_crop_stops_after_valid_candidate():
    torch.manual_seed(0)
    frames = torch.arange(3 * 32 * 48, dtype=torch.float32).reshape(1, 3, 32, 48)
    with patch.object(torch, "randint", wraps=torch.randint) as randint:
        cropped = video.random_resized_crop(frames, size=16, scale=(0.2, 0.3))
        assert randint.call_count == 2
    assert cropped.shape == (1, 3, 16, 16)
    assert torch.isfinite(cropped).all()


def test_temporal_sampling_repeats_short_clips():
    frames = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    sampled = video.temporal_sampling_random(frames, num_windows=8, T=8)
    assert torch.equal(sampled, frames.expand(8, -1, -1, -1))
    with pytest.raises(ValueError, match="empty clip"):
        video.temporal_sampling_random(frames[:0], num_windows=8, T=8)
    with pytest.raises(ValueError, match="positive"):
        video.temporal_sampling_random(frames, num_windows=0, T=8)


def test_resize_uses_configured_size():
    dataset = OmegaConf.load(ROOT / "configs/datasets/video/endofm/train_innorm.yaml")
    augmentation = load_aug_config({
        "stage_type": "pretrain", "modality": "video",
        "views": {0: {"is_small": False, "num_crops": 1, "resize": {"size": 24}}},
    })
    transformed = make_transform_pipeline(dataset, augmentation)(
        torch.ones(1, 3, 10, 12, dtype=torch.uint8)
    )
    assert transformed[0].shape == (1, 3, 24, 24)


def test_manifest_handles_nested_files_without_overwriting(tmp_path):
    (tmp_path / "dataset" / "nested").mkdir(parents=True)
    (tmp_path / "dataset" / "nested" / "one.mp4").touch()
    (tmp_path / "two.mp4").touch()
    command = [sys.executable, str(ROOT / "dataset-curation/endo-fm-data/gencsv.py"), str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True)
    contents = (tmp_path / "train.csv").read_text()
    rows = list(csv.reader(contents.splitlines()))
    assert len(rows) == 2
    for folder, filename, label in rows:
        assert (tmp_path / folder / filename).exists()
        assert label == "-1"
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (tmp_path / "train.csv").read_text() == contents


def test_real_video_pretraining_and_polypdiag_loaders(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    with av.open(str(videos / "sample.mp4"), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 80, 64, "yuv420p"
        for index in range(24):
            pixels = np.full((64, 80, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    (tmp_path / "train.csv").write_text("videos,sample.mp4,-1\n")
    dataset = OmegaConf.load(ROOT / "configs/datasets/video/endofm/train_innorm.yaml")
    dataset.root = str(tmp_path)
    augmentation = load_aug_config(str(ROOT / "configs/augmentations/ssl/video/endofm.yaml"))
    pretraining = make_dataset(dataset, transform=make_transform_pipeline(dataset, augmentation))
    index, views, target = pretraining[0]
    assert index == 0 and target == -1
    expected = [(3, 8, 224, 224), (3, 16, 224, 224)]
    expected += [(3, frames, 96, 96) for frames in (16, 16, 8, 8, 4, 4, 2, 2)]
    assert [tuple(view.shape) for view in views] == expected
    assert all(view.dtype == torch.float32 and torch.isfinite(view).all() for view in views)

    splits = tmp_path / "splits"
    splits.mkdir()
    for split, aug_name in [("train", "train"), ("test", "val")]:
        (splits / f"{split}.txt").write_text("sample.mp4,1\n")
        cfg = OmegaConf.load(ROOT / f"configs/datasets/video/polypdiag/their-normalisation/{split}.yaml")
        cfg.root = str(tmp_path)
        aug = load_aug_config(str(ROOT / f"configs/augmentations/downstream/video/polypdiag/{aug_name}.yaml"))
        downstream = make_dataset(cfg, transform=make_transform_pipeline(cfg, aug))
        with patch.object(polypdiag, "decode", wraps=polypdiag.decode) as decoder:
            index, views, target = downstream[0]
            assert decoder.call_count == 1
        size = 224 if split == "train" else 256
        assert index == 0 and target == 1 and len(views) == 1
        assert views[0].shape == (3, 8, size, size)
        assert len(downstream) == (1 if split == "train" else 10)


def test_validation_metrics_accept_python_float_results():
    from ssl_space.methods_downstream import BaseLinear

    model = BaseLinear.__new__(BaseLinear)
    torch.nn.Module.__init__(model)
    model.num_classes = 2
    model.validation_step_indices = [torch.arange(4)]
    model.validation_step_targets = [torch.tensor([0, 1, 0, 1])]
    model.validation_step_logits = [torch.tensor([[4.0, -4.0], [-4.0, 4.0], [2.0, -2.0], [-2.0, 2.0]])]
    metrics = model.on_validation_end()
    for name in ("acc1", "precision_macro", "recall_macro", "f1_macro", "precision_binary", "recall_binary", "f1_binary", "roc_auc"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["loss"] >= 0.0 and 0.0 <= metrics["ece"] <= 1.0
    assert model.validation_step_indices == []
    assert model.validation_step_targets == []
    assert model.validation_step_logits == []
