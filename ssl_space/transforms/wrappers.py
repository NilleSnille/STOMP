class NCropAugmentation:
    def __init__(self, transform, num_crops):
        """Creates a pipeline that apply a transformation pipeline multiple times.
        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """
        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x):
        """Applies transforms n times to generate n crops.
        Args:
            x (Image): an image in the PIL.Image format.
        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"
    
class FullTransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        """Applies transforms n times to generate n crops.
        Args:
            x (data): some data.
        Returns:
            List[torch.Tensor]: an image/sequence in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)