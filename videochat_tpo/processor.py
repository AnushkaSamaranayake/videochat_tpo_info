# videochat_tpo/processor.py

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from .special_tokens import (
    special_tokens,
    IMG_TOKEN,
    VID_TOKEN,
    BOX_START,
    TIME_START,
    TIME_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    TRACK_PLACEHOLDER,
    TRACK_START,
    TRACK_START_BOX,
)


class VideoChatTPOProcessor:
    """
    A thin wrapper around a base text tokenizer (Mistral) that:
      - Adds special vision / task tokens.
      - Implements `build_input_ids` used by MultiModalLLM_PT.generate_answer.
      - Exposes attributes expected by the model, e.g. .box_token, .temp_token, etc.

    It behaves like a tokenizer: __len__, __call__, batch_decode, convert_tokens_to_ids, etc.
    """

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerBase,
        n_query: int = 64,
        v_query: int = 64,
        device: str = "cuda",
    ):
        self.tokenizer = base_tokenizer
        self.device = device
        self.n_query = n_query
        self.v_query = v_query

        # Ensure pad token exists: follow common practice (pad = eos if missing).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add task-related special tokens to vocab
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.tokenizer.add_tokens(self.special_tokens)

        # Store ids for various markers used by the model
        self.box_start_token = self.convert_tokens_to_ids([BOX_START])[0]
        self.time_start_token = self.convert_tokens_to_ids([TIME_START])[0]
        self.temp_token = self.convert_tokens_to_ids([TIME_PLACEHOLDER])[0]
        self.box_token = self.convert_tokens_to_ids([BOXES_PLACEHOLDER])[0]
        self.track_box_token = self.convert_tokens_to_ids([TRACK_START_BOX])[0]
        self.track_token = self.convert_tokens_to_ids([TRACK_PLACEHOLDER])[0]
        self.track_start_token = self.convert_tokens_to_ids([TRACK_START])[0]

        # Convenience alias (used in temporal decoder)
        self.temp_place_ids = self.temp_token

    # ----------- generic forwarding -----------

    def __len__(self) -> int:
        return len(self.tokenizer)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        tokenizer = super().__getattribute__("tokenizer")
        if hasattr(tokenizer, name):
            return getattr(tokenizer, name)

        raise AttributeError(f"'VideoChatTPOProcessor' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    # ----------- image / video helpers -----------

    @staticmethod
    def prepare_image_input(images: torch.Tensor) -> torch.Tensor:
        """
        The model expects raw image/video tensors; we don't do tokenization here.
        """
        return images

    # ----------- text <-> image/video alignment -----------

    def _prepare_text_input(
        self,
        text: List[str],
        max_length: int,
        add_special_tokens: bool,
        truncation: bool,
        padding: str = "longest",
        return_tensors: str = "pt",
        image_placeholder: str = IMG_TOKEN,
        video_placeholder: str = VID_TOKEN,
    ) -> Dict[str, torch.Tensor]:
        """
        Parses a single conversation string with visual placeholders like:

            "... <Video>[<VID_PLH>]</Video> ..."

        and builds:
          - input_ids for all text.
          - attention_mask.
          - a binary index vector marking where visual tokens should be injected.

        This replicates the logic from TPO's tokenizer.py.
        """
        assert len(text) == 1, "Only single-string batches are supported here."
        text = text[0]

        start = 0
        total_len = 0
        input_ids_chunks: List[torch.Tensor] = []
        attention_chunks: List[torch.Tensor] = []
        index_chunks: List[torch.Tensor] = []

        while True:
            index1 = text.find(image_placeholder, start)
            index2 = text.find(video_placeholder, start)

            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)

            if index == -1:
                # No more visual placeholders – tokenize remaining text
                sub = text[start:]
                inputs = self.tokenizer(
                    sub,
                    max_length=max_length - total_len,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    padding=padding,
                    return_tensors=return_tensors,
                )
            else:
                # Tokenize text up to placeholder
                sub = text[start:index]
                inputs = self.tokenizer(
                    sub,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    padding="longest",
                    return_tensors=return_tensors,
                )

            ids = inputs.input_ids[0]
            mask = inputs.attention_mask[0]

            input_ids_chunks.append(ids)
            attention_chunks.append(mask)
            index_chunks.append(torch.zeros_like(ids))  # 0 for text tokens

            total_len += ids.shape[0]

            if index != -1:
                # Insert a block of "vision tokens" (just placeholders here).
                image_block_ids = torch.zeros(self.n_query, dtype=torch.long)
                image_block_mask = torch.ones(self.n_query, dtype=torch.long)
                index_block = torch.ones(self.n_query, dtype=torch.long)  # 1 = visual

                input_ids_chunks.append(image_block_ids)
                attention_chunks.append(image_block_mask)
                index_chunks.append(index_block)

                # Continue scanning after the placeholder token itself
                start = index + len(image_placeholder)
            else:
                # Finished
                break

        input_ids = torch.cat(input_ids_chunks).long()
        attention_mask = torch.cat(attention_chunks).long()
        index = torch.cat(index_chunks).to(torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "index": index,
        }

    def build_input_ids(
        self,
        text: List[str],
        max_length: int,
        add_special_tokens: bool,
        truncation: bool,
        padding: str,
        return_tensors: str,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        require_image: bool = False,
        require_video: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        API expected by MultiModalLLM_PT.generate_answer:

        Returns a dict with:
          - input_ids           (1D tensor)
          - attention_mask      (1D tensor)
          - image_index         (1D bool tensor or None)
          - video_index         (1D bool tensor or None)
          - image               (tensor or None)
          - video               (tensor or None)
        """

        if image is not None:
            image = self.prepare_image_input(image)

        if video is not None:
            video = self.prepare_image_input(video)

        inputs = self._prepare_text_input(
            text=text,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            image_placeholder=IMG_TOKEN,
            video_placeholder=VID_TOKEN,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        index = inputs["index"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_index": index if (image is not None or require_image) else None,
            "video_index": index if (video is not None or require_video) else None,
            "image": image,
            "video": video,
        }


def create_videochat_tpo_processor(
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    device: str = "cuda",
) -> VideoChatTPOProcessor:
    """
    Convenience factory:
      - loads the base text tokenizer
      - wraps it as VideoChatTPOProcessor
    """
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
    )
    processor = VideoChatTPOProcessor(base_tokenizer, device=device)
    return processor
