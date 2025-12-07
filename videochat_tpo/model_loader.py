# videochat_tpo/model_loader.py

from typing import Tuple

import torch
from transformers import AutoModel

from .processor import create_videochat_tpo_processor, VideoChatTPOProcessor

from .task_hooks import SegmentationHook, DetectionHook, TrackingHook

def load_videochat_tpo(
    model_name: str = "OpenGVLab/VideoChat-TPO",
    llm_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, VideoChatTPOProcessor]:
    """
    Loads:
      - VideoChat-TPO model (MultiModalLLM_PT via HF remote code)
      - VideoChatTPOProcessor tokenizer wrapper

    Returns (model, processor).
    """

    processor = create_videochat_tpo_processor(base_model_name=llm_name, device=device)

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        _tokenizer=processor,  # <-- important: passed into MultiModalLLM_PT.__init__
    )

    segmentation_hook = SegmentationHook(device=device)
    detection_hook = DetectionHook(device=device)
    tracking_hook = TrackingHook(device=device)

    model.to(device)
    model.eval()

    return model, processor
