# Portions adapted and modified for STOMP from SlowFast/Endo-FM (Apache-2.0).
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

import os
import warnings
import random
import math

import av
import numpy as np
import torch
from einops import rearrange

from .base_dataset import BaseDataset

class EndoFM(BaseDataset):
    TARGET = -1
    def __init__(
        self, 
        root: str, 
        split: str,
        num_clips_from_video: int,
        num_frames_from_clip: int,
        sampling_rate: int,
        target_fps: float,
        multithread_decode: bool,
        decoding_backend: str, 
        mode: str,
        num_retries: int,
        transform: object, 
        target_transform: object,
    ):
        super().__init__(transform, target_transform)
        assert split in {'train'}, f"split {split} not supported"
        self.root = root
        self.split = split

        # the number of clips to decode from the video.
        self.num_clips_from_video = num_clips_from_video
        self.num_frames_from_clip = num_frames_from_clip
        self.sampling_rate = sampling_rate
        self.target_fps = target_fps 
        self.multithread_decode = multithread_decode
        self.decoding_backend = decoding_backend
        self.mode = mode
        self.num_retries = num_retries
        
        # construct the loader
        path_to_videos, labels = EndoFM._construct_loader(root, split)
        self._path_to_videos = path_to_videos
        self._labels = labels

    @staticmethod
    def _construct_loader(root, split):
        '''
        Construct the video loader
        '''
        SEPARATOR = ','
        path_to_file = os.path.join(root, "{}.csv".format(split))
        assert os.path.exists(path_to_file), f"{path_to_file} not found."

        path_to_videos = []
        labels = []

        with open(path_to_file, "r") as f:
            for clip_idx, folder_video_label in enumerate(f.read().splitlines()):
                
                folder_video_label = folder_video_label.split(SEPARATOR) 
                assert len(folder_video_label) == 3, f"Expected folder, video, label, got {folder_video_label}"
                folder, video, label = folder_video_label
                path = os.path.join(root, folder, video)

                # add the video num_clips times 
                path_to_videos.append(path)
                labels.append(int(label))
                   
        assert len(path_to_videos) > 0, f"Failed to load anything."
        print(
            f"Constructed endo-fm dataset (size: {len(path_to_videos)}) from {path_to_file}"
        )
        return path_to_videos, labels

    def __len__(self):
        return len(self._path_to_videos)
    
    def getitem_train(self, index):
        """
        We duplicate some code to make this class more readable. Each split gets its own getitem, which is called from
        __getitem__. This is because the logic is difficult to follow otherwise, with many chains of dependent if statements.
        We also do not need the test/validation modes for the time beeing.
        """

        frames = None
        video_container = None
        for i_try in range(self.num_retries):
            video_container = None
            path_to_video = self._path_to_videos[index]
            try:
                video_container = get_video_container(
                    path_to_vid=path_to_video,
                    multi_thread_decode=self.multithread_decode,
                    backend=self.decoding_backend,
                )
            except Exception as e:
                warnings.warn(f"Failed to load video from {self._path_to_videos[index]} with error {e}")

            # select random new video if loading failed
            if video_container is None:
                warnings.warn(
                    f"Failed to meta load video idx {index} from {self._path_to_videos[index]} | trial {i_try+1}"
                )
            
                if i_try > self.num_retries // 2:
                    # try another index
                    index = random.randint(0, len(self._path_to_videos)-1)
                    continue
                else:
                    # try with same index
                    continue
            
            # decode the video (selective decoding).
            frames, fps, decode_all_video = decode(
                container=video_container,
                sampling_rate=self.sampling_rate,
                num_frames=self.num_frames_from_clip,
                clip_idx=-1,
                num_clips=self.num_clips_from_video,
                target_fps=self.target_fps, 
                backend=self.decoding_backend
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                warnings.warn(
                    f"Failed to decode video idx {index} from {self._path_to_videos[index]} | trial {i_try+1}"
                )
                
                if i_try > self.num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                else:
                    continue
            
            else:
                # we have sucessfully decoded frames. break the loop, we dont need more tries.
                break
            
        # If the frames are still None, we were not succefull
        if frames is None:
            raise RuntimeError(f"Failed to decode frames after {self.num_retries} attempts.")
      
        # make the temporal and spatial augmentation
        frames = rearrange(frames, "t h w c -> t c h w")
        frames, _ = self.do_transform(frames, None) # we get a list of augmented samples. We do the temporal augmentation here as well.
        # reorder output dims again to get channel first.
        frames = [rearrange(view, "t c h w -> c t h w") for view in frames]

        return frames, self.TARGET

    def __getitem__(self, index):
        if self.split == "train":
            return self.getitem_train(index)
        else:
            raise NotImplementedError(f"Get item for split {self.split} not implemented")

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Args:
        video_size (int): number of decoded frames to choose from.
        clip_size (int): desired window length in frames.
        clip_idx (int): if -1 perform random jitter sampling. If > -1, uniformly split
            the video to num_clips. Select the start and end indices of the
            clip_idx-th clip.
        num_clips (int): overall number of clips to uniformly sample from the given
            video for testing. Not used if clip_idx = -1.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        # We pass clip_idx = temporal_sample_index = -1
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def get_video_container(path_to_vid: str, multi_thread_decode: bool, backend: str):
    '''
    Given the path to the video, return a container for that video.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options currently only include `pyav.
    Returns:
        container (container): video container.
    '''
    if backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError(f"Unknown backend {backend}.")

def pyav_decode_stream(
    container, 
    start_pts, 
    end_pts, 
    stream, 
    stream_name, 
    buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts

def pyav_decode(
    container, 
    sampling_rate, 
    num_frames, 
    clip_idx, 
    num_clips, 
    target_fps, 
    start=None, 
    end=None, 
    duration=None
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.

    fps = float(container.streams.video[0].average_rate) # The average frame rate of this video stream. 

    orig_duration = duration # None
    
    # the time-base, for example (1/30). Often the stream has a time-base of 1/fps.
    # Its essentially the time one frame "covers".
    tb = float(container.streams.video[0].time_base) # The unit of time (in fractional seconds) in which timestamps are expressed.

    # The duration of this stream, in time-base units. 
    duration: int = container.streams.video[0].duration

    # the number of frames the stream contains.
    frames_length: int = container.streams.video[0].frames
    
    if duration is None and orig_duration is not None:
        # this is a fallback for when we passed a duration, but we could not 
        # verify using container.streams.video[0].duration. We do not use this, 
        # as we do not know duration a priori (orig_duration=None).
        duration = orig_duration / tb
    
    if duration is None or frames_length <= 0:
        # Fall back to full decoding when the stream has no usable frame count.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # we can perform selective decoding. We compute an intervall in pts units that
        # covers a sufficient window of the stream to sample num_frames 
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            video_size=frames_length,
            clip_size=sampling_rate * num_frames / target_fps * fps, # 64
            clip_idx=clip_idx,
            num_clips=num_clips
        )

        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    if start is not None and end is not None:
        decode_all_video = False # we pass start, end = None

    frames = None
    if container.streams.video:
        if start is None and end is None:
            # we are here. 
            video_frames, max_pts = pyav_decode_stream(
                container,
                video_start_pts, 
                video_end_pts,
                container.streams.video[0],
                {"video": 0},
            )
        else:
            timebase = duration / frames_length # ????
            start_i = start
            end_i = end
            video_frames, max_pts = pyav_decode_stream(
                container,
                start_i,
                end_i,
                container.streams.video[0],
                {"video": 0},
            )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))

    return frames, fps, decode_all_video

def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=1,
    target_fps=30,
    backend="pyav",
    start=None,
    end=None,
    duration=None
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        temporal_aug: bool
        two_token: bool
        rand_fr: bool
    Returns:
        frames (tensor): decoded frames from the video.
    """

    # Currently support two decoders: 1) PyAV, and 2) TorchVision.

    assert clip_idx >= -1, f"not valid clip_idx {clip_idx}"
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container=container,
                sampling_rate=sampling_rate,
                num_frames=num_frames,
                clip_idx=clip_idx,
                num_clips=num_clips, 
                target_fps=target_fps,
                start=start,
                end=end,
                duration=duration,
            )
        else:
            raise NotImplementedError(f"Decoding backend {backend} not implemented.")

    except Exception as e:
        print(f"Failed decode by {backend} with exception: {e}")
        return None
    
    if frames.size(0) == 0:
        return None, None, None
    
    return frames, fps, decode_all_video
    





        

