from omegaconf import DictConfig
from .image import make_composed_image_transform
from .video import make_composed_video_transform

from .wrappers import NCropAugmentation, FullTransformPipeline

def make_transform_pipeline(dset_cfg: DictConfig, aug_cfg: DictConfig) -> FullTransformPipeline:
    # Before batching, outputs
    # ([a1_0(x), ..., a1_numcrops1(x), ..., aN_0(x), ..., aN_numcropsN(x)], target)

    # after default batching this becomes 
    # [tensor(all a1_0 imgs), ..., tensor(all a1_numcrops1 imgs), ..., tensor(all aN_0 imgs), tensor(all aN_numcropsN imgs)]
    T = []

    # we have an image and one video mode.
    modality = aug_cfg.modality 
    if  modality == "image":

        # Sort outer keys like 0, 1, ... numerically.
        # ensures the views are outputted as expected
        for view_idx in sorted(aug_cfg.views.keys()):
            view_cfg = aug_cfg.views[view_idx]
            T.append(
                NCropAugmentation(
                    transform=make_composed_image_transform(dset_cfg, view_cfg), 
                    num_crops=view_cfg.num_crops
                )
            )

        return FullTransformPipeline(transforms=T)
    
    elif modality == "video":

        # Sort outer keys like 0, 1, ... numerically.
        # ensures the views are outputted as expected
        for view_idx in sorted(aug_cfg.views.keys()):
            view_cfg = aug_cfg.views[view_idx]
            T.append(
                NCropAugmentation(
                    transform=make_composed_video_transform(dset_cfg, view_cfg), 
                    num_crops=view_cfg.num_crops
                )
            )

        return FullTransformPipeline(transforms=T)
    
    else:
        raise ValueError(f"Modalities 'image' and 'video' supported, got {modality}")