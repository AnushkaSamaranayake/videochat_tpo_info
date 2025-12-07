# videochat_tpo/video_io.py

from typing import Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from torchvision import transforms


def load_video_tensor(
    path: str,
    num_frames: int = 16,
    size: int = 224,
) -> torch.Tensor:
    """
    Load a video from disk and return tensor of shape [B=1, T, C, H, W].

    - Uniformly samples `num_frames` frames across the whole video.
    - Resizes to `size x size`.
    - Normalizes roughly like CLIP/ImageNet (this matches many ViT backbones).
    """

    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video {path} has no frames.")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3], uint8

    # torchvision transforms expect [H, W, C]
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # [C, H, W], 0-1
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    frame_tensors = [transform(f) for f in frames]  # each [C, H, W]
    video_tensor = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]

    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)  # [1, T, C, H, W]
    return video_tensor
