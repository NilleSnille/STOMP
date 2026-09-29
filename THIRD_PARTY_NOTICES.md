# Third-party notices

The MIT license in the repository root covers original STOMP contributions. The
adapted third-party portions identified below retain their upstream licenses.
These portions have been modified for STOMP; the source links identify the
upstream implementations rather than claiming the files are unmodified copies.

## solo-learn — MIT

Copyright 2021 solo-learn development team. The upstream source files also carry
Copyright 2023 solo-learn development team.

Adapted portions are present in `ssl_space/utils/lars.py`,
`ssl_space/utils/knn.py`, `ssl_space/utils/momentum.py`,
`ssl_space/utils/metrics.py`, the weight-decay parameter-group helper in
`ssl_space/utils/misc.py`, and learning-rate helper code in
`ssl_space/methods/base.py`.

Sources: [solo-learn utilities](https://github.com/vturrisi/solo-learn/tree/main/solo/utils)
and [base method](https://github.com/vturrisi/solo-learn/blob/main/solo/methods/base.py).
The full permission and warranty notice is retained in
[LICENSES/solo-learn-MIT.txt](LICENSES/solo-learn-MIT.txt).

## DINO — Apache License 2.0

Copyright (c) Facebook, Inc. and its affiliates.

Adapted portions include the projection head and centered cross-view loss in
`ssl_space/methods/stomp.py`, logging and initialization helpers in
`ssl_space/utils/misc.py`, distributed helpers in
`ssl_space/utils/distributed.py`, and transformer initialization/attention helpers
in `ssl_space/backbones/transformer/`.

Sources: [DINO training and loss](https://github.com/facebookresearch/dino/blob/main/main_dino.py),
[vision transformer](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py),
and [utilities](https://github.com/facebookresearch/dino/blob/main/utils.py).
DINO's vision-transformer source also credits the timm library.

## PyTorch Image Models (timm) — Apache License 2.0

Copyright 2019 Ross Wightman. Relevant utility sources also carry
Copyright 2020 Ross Wightman.

Transformer and initialization helpers in
`ssl_space/backbones/transformer/vit_utils.py`,
`ssl_space/backbones/transformer/vit_blocks.py`, and `ssl_space/utils/misc.py`
include adapted timm/DINO components, including stochastic depth and truncated
normal initialization.

Sources: [timm](https://github.com/huggingface/pytorch-image-models),
[stochastic depth](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py),
and [weight initialization](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py).

## SlowFast and Endo-FM — Apache License 2.0

SlowFast: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
The upstream license also identifies Copyright 2019, Facebook, Inc.

Video augmentation functions in
`ssl_space/transforms/video/make_video_transform.py` include adapted portions
from these implementations, including crop, color, and normalization helpers.
Video decoding and container helpers in `ssl_space/datasets/endofm.py` and
`ssl_space/datasets/polypdiag.py` also include adapted portions.

Sources: [SlowFast transforms](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/transform.py),
[SlowFast decoder](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/decoder.py),
[SlowFast video containers](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/video_container.py),
[SlowFast license](https://github.com/facebookresearch/SlowFast/blob/main/LICENSE),
[Endo-FM transforms](https://github.com/med-air/Endo-FM/blob/main/datasets/transform.py),
and [Endo-FM license](https://github.com/med-air/Endo-FM/blob/main/LICENSE).

## Lightning Bolts — Apache License 2.0

Copyright 2018-2021 William Falcon.

`LinearWarmupCosineAnnealingLR` in `ssl_space/utils/lr_scheduler.py` adapts the
Lightning Bolts scheduler also redistributed by solo-learn, with STOMP-specific
warmup and schedule handling.

Sources: [Lightning Bolts scheduler](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/optimizers/lr_scheduler.py),
[upstream license](https://github.com/Lightning-Universe/lightning-bolts/blob/master/LICENSE),
and [solo-learn scheduler attribution](https://github.com/vturrisi/solo-learn/blob/main/solo/utils/lr_scheduler.py).

The full Apache License 2.0 text applying to the Apache-licensed portions above
is retained in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

External dependencies and datasets retain their own licenses and access terms.
