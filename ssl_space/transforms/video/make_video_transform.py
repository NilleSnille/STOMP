# Portions adapted and modified for STOMP from SlowFast/Endo-FM (Apache-2.0).
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

from typing import Tuple, List
from functools import partial
import random
from omegaconf import DictConfig
import math

import numpy as np
import torch
from torchvision import transforms

def random_short_side_scale_jitter(
    images: torch.Tensor, 
    min_size, 
    max_size, 
    inverse_uniform_sampling=False
):
    """
    Perform a spatial short scale jittering on the given images.
    Args:
        images (tensor): images to perform scale jitter. 
            Dimension is `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale, max_scale].
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    if inverse_uniform_sampling:
        size = int(
            round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
        )
    else:
        size = int(round(np.random.uniform(min_size, max_size)))


    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    
    return torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
    
def random_crop(images: torch.Tensor, size):
    """
    Perform random spatial crop on the given images.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped

def resize(images, size, mode="bilinear"):
    if isinstance(size, int):
        new_height, new_width = size, size
    else:
        new_height, new_width = size
    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode=mode,
        align_corners=False,
    )

def random_resized_crop(images, size, scale, ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
    height, width = images.shape[-2:]
    area = height * width
    non_central = False

    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            non_central = True
            break

    if not non_central:
        # fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

    y_offset, x_offset = i, j
    cropped = images[:, :, y_offset: y_offset + h, x_offset: x_offset + w]
    resized = resize(cropped, size=size, mode=interpolation)
    return resized

def horizontal_flip(images: torch.Tensor):
    """
    Perform horizontal flip on the given images.
    Args:
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
    """
    return images.flip((-1))

def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    return images1 * alpha + images2 * (1 - alpha)

def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order RGB.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = images.clone()
    gray_channel = (
        0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray

def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    return blend(images, img_bright, alpha)

def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    return blend(images, img_gray, alpha)

def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    return blend(images, img_gray, alpha)

def color_normalization(images, mean, stddev):
    """
    Perform color normalization on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1] and len(stddev) == images.shape[1], "channel mean not computed properly"
    
    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]

    return out_images

def to_float01(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        x = x.float()
        x = x / 255.0
    return x

def undo_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean

    if tensor.dtype != torch.uint8:
        tensor = tensor * 255.0
        tensor = tensor.to(torch.uint8)

    return tensor

def temporal_sampling_offset(frames: torch.Tensor, start_offset: int, end_offset: int, T: int):
    max_len = frames.shape[0]
    start_idx = 0 + start_offset
    end_idx = max_len - end_offset

    index = torch.linspace(start_idx, end_idx, T)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames

def temporal_sampling_random(frames: torch.Tensor, num_windows: int, T: int):
    max_len = frames.shape[0]
    if max_len == 0:
        raise ValueError("Cannot sample frames from an empty clip.")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive.")
    window_length = max(1, max_len // num_windows)
    start_idx = random.randint(0, window_length-1)
    end_idx = start_idx + window_length
    
    index = torch.linspace(start_idx, end_idx, T)
    index = torch.clamp(index, 0, max_len - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames

def make_composed_transform(dset_cfg: DictConfig, view_cfg: DictConfig):
    ops = []

    if view_cfg.temporal:
        temp_aug = []
        if view_cfg.temporal.offset is not None:
            temp_aug.append(
                partial(
                    temporal_sampling_offset,
                    start_offset=view_cfg.temporal.offset.start_offset,
                    end_offset=view_cfg.temporal.offset.end_offset,
                    T=view_cfg.temporal.T
                )
            )
        
        if view_cfg.temporal.random is not None:
            temp_aug.append(
                partial(
                    temporal_sampling_random,
                    num_windows=view_cfg.temporal.random.num_windows,
                    T=view_cfg.temporal.T
                )
            )
        
        assert len(temp_aug) <= 1, "Should not use more than one temporal augmentation per view"
        ops.extend(temp_aug)

    ops.append(
        to_float01
    )

    if view_cfg.resize is not None:
        ops.append(
            partial(resize, size=view_cfg.resize.size, mode="bilinear")
        )
    
    if view_cfg.rrc is not None:
        ops.append(
            partial(
                random_resized_crop, 
                size=view_cfg.rrc.size, 
                scale=(view_cfg.rrc.min_scale, view_cfg.rrc.max_scale),
                ratio=(3. / 4., 4. / 3.),
                interpolation="bilinear",
            )
        )

    if view_cfg.random_short_side_scale_jitter is not None:
        ops.append(
            partial(
                random_short_side_scale_jitter,
                min_size=view_cfg.random_short_side_scale_jitter.min_size,
                max_size=view_cfg.random_short_side_scale_jitter.max_size,
                inverse_uniform_sampling=view_cfg.random_short_side_scale_jitter.inverse_uniform_sampling
            )
        )

    if view_cfg.random_crop is not None:
        ops.append(
            partial(
                random_crop,
                size=view_cfg.random_crop.size
            )
        )

    if view_cfg.horizontal_flip is not None:
        ops.append(
            transforms.RandomApply(
                [
                    partial(
                        horizontal_flip,
                    )
                ],
                p=view_cfg.horizontal_flip.prob,
            )
        )
    
    if view_cfg.color_jitter is not None:
        ops.append(
            transforms.RandomApply(
                [
                    partial(
                        color_jitter,
                        img_brightness=view_cfg.color_jitter.brightness,
                        img_contrast=view_cfg.color_jitter.contrast,
                        img_saturation = view_cfg.color_jitter.saturation,
                    )
                ],
                p=view_cfg.color_jitter.prob
            )
        )

    if view_cfg.grayscale is not None:
        ops.append(
            transforms.RandomApply(
                [
                    grayscale
                ],
                p=view_cfg.grayscale.prob
            )
        )

    if view_cfg.gaussian_blur is not None:
        raise NotImplementedError("Gaussian Blur not implemented")
    
    if view_cfg.solarization is not None:
        raise NotImplementedError("Solarization not implemented")
    
    # TODO: this is unreadable
    if view_cfg.color_normalize is not None: 
        if view_cfg.color_normalize.do is True: 
            ops.append(
                partial(
                    color_normalization,
                    mean=dset_cfg.normalization.mean, 
                    stddev=dset_cfg.normalization.std
                )
            )
        else:
            pass
    else:
        ops.append(
            partial(
                color_normalization,
                mean=dset_cfg.normalization.mean, 
                stddev=dset_cfg.normalization.std
            )
        )

    ops = transforms.Compose(ops)
    return ops
