from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .model import MoEConfig, SimpleMoELanguageModel


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_dir: Path, device: torch.device, tokenizer_name: str | None = None) -> tuple[SimpleMoELanguageModel, object]:
    with (checkpoint_dir / "config.json").open() as f:
        config = MoEConfig.from_dict(json.load(f))
    # Older interrupted checkpoints may not include tokenizer files because the
    # first crash happened before tokenizer.save_pretrained(). Fall back to GPT-2.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or checkpoint_dir)
    model = SimpleMoELanguageModel(config)
    state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained VerySimpleMoE checkpoint")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/verysimplemoe"))
    p.add_argument("--tokenizer", default=None, help="Tokenizer name/path. Use gpt2 for interrupted checkpoints without tokenizer files.")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, tokenizer = load_model(args.checkpoint, device, args.tokenizer)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
