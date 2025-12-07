# videochat_tpo/special_tokens.py

# Visual placeholder tokens used in conversation text
IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

# Tokens actually used inside the LLM for generic “image/video” embeddings
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

# Task-specific special tokens (boxes, time, tracking)
BOX_START = "<box_begin>"
TIME_START = "<time_begin>"
TIME_PLACEHOLDER = "<temp>"
BOXES_PLACEHOLDER = "<boxes>"
TRACK_START = "<track_begin>"
TRACK_PLACEHOLDER = "<tracking>"
TRACK_START_BOX = "<track_box>"

# This is the set that is added to the base tokenizer vocab
special_tokens = [
    BOX_START,
    TIME_START,
    TIME_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    TRACK_START,
    TRACK_PLACEHOLDER,
    TRACK_START_BOX,
]
