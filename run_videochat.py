# run_videochat.py

import argparse
import torch

from videochat_tpo.model_loader import load_videochat_tpo
from videochat_tpo.video_io import load_video_tensor


def main():
    parser = argparse.ArgumentParser(description="Run VideoChat-TPO inference on a video.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to an input video file (e.g., .mp4).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe what happens in this video in detail.",
        help="User question / prompt about the video.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a helpful assistant that understands videos.",
        help="System / instruction prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    args = parser.parse_args()

    device = args.device

    print("[*] Loading VideoChat-TPO model and tokenizer...")
    model, processor = load_videochat_tpo(device=device)

    print(f"[*] Loading and preprocessing video: {args.video}")
    video_tensor = load_video_tensor(args.video, num_frames=16, size=224)
    video_tensor = video_tensor.to(device)

    # Build a simple “message” to mimic the paper’s usage patterns.
    # `msg` is a context description. `user_prompt` is the actual question.
    msg = "Please watch the video and answer the user's question based on its content."
    user_prompt = args.question
    instruction = args.instruction

    print("[*] Running inference...")
    with torch.no_grad():
        response, _ = model.generate_answer(
            tokenizer=processor,
            instruction=instruction,
            msg=msg,
            user_prompt=user_prompt,
            media_type="video",
            video_tensor=video_tensor,
            image_tensor=None,
            answer_prompt=None,
            chat_history=[],
            return_history=False,
            debug=False,
            generation_config={},
        )

    print("\n=== Model answer ===")
    print(response)


if __name__ == "__main__":
    main()
