from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .model import ARCH_PRESETS, MoEConfig, SimpleMoELanguageModel


class FineWebTokenDataset(IterableDataset):
    """Streams FineWeb text and emits fixed-length next-token prediction chunks."""

    def __init__(
        self,
        tokenizer,
        block_size: int,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: str = "sample-10BT",
        split: str = "train",
        shuffle_buffer: int = 10_000,
        seed: int = 1337,
        text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.text_column = text_column

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer > 0:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + worker_id)

        eos = self.tokenizer.eos_token_id
        token_buffer: list[int] = []
        for row in ds:
            text = row.get(self.text_column)
            if not text:
                continue
            # FineWeb rows can exceed GPT-2 tokenizer.model_max_length (1024),
            # but we chunk tokens ourselves below, so suppress that irrelevant
            # tokenizer warning instead of truncating the document.
            token_buffer.extend(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))
            token_buffer.append(eos)

            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[: self.block_size + 1]
                del token_buffer[: self.block_size]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    out_dir: Path,
    model: SimpleMoELanguageModel,
    tokenizer,
    trainer_state: dict,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(trainer_state, out_dir / "trainer_state.pt")
    with (out_dir / "config.json").open("w") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    with (out_dir / "train_args.json").open("w") as f:
        # argparse contains pathlib.Path values such as --out-dir.
        json.dump(vars(args), f, indent=2, default=str)
    tokenizer.save_pretrained(out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VerySimpleMoE on FineWeb")
    p.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="v1", help="Architecture preset. v1 preserves the original model.")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/verysimplemoe"))
    p.add_argument("--resume-from", type=Path, default=None, help="Resume model/optimizer state from a checkpoint directory")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")

    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)

    # Model overrides. Leave unset to use the selected architecture preset.
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--n-experts", type=int, default=None)
    p.add_argument("--active-experts", type=int, default=None, help="Top-k experts selected per token")
    p.add_argument("--expert-hidden-size", type=int, default=None)
    p.add_argument("--aux-loss-coef", type=float, default=None)
    p.add_argument("--router-noise-std", type=float, default=None)
    p.add_argument("--router-z-loss-coef", type=float, default=None)

    # Phased expert training. This restricts the router to a moving subset of
    # experts and only keeps optimizer state for that subset.
    p.add_argument("--train-experts-per-phase", type=int, default=0, help="0/all disables phased expert training")
    p.add_argument("--expert-phase-steps", type=int, default=500)
    p.add_argument("--expert-phase-stride", type=int, default=None, help="Default: half of --train-experts-per-phase")
    return p.parse_args()


def build_config(args: argparse.Namespace, tokenizer) -> MoEConfig:
    if args.resume_from is not None and (args.resume_from / "config.json").exists():
        with (args.resume_from / "config.json").open() as f:
            config = MoEConfig.from_dict(json.load(f))
        print(f"Loaded model config from {args.resume_from / 'config.json'}")
        return config

    values = MoEConfig().to_dict()
    values.update(ARCH_PRESETS[args.arch])
    values["vocab_size"] = len(tokenizer)

    overrides = {
        "block_size": args.block_size,
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "n_experts": args.n_experts,
        "active_experts": args.active_experts,
        "expert_hidden_size": args.expert_hidden_size,
        "aux_loss_coef": args.aux_loss_coef,
        "router_noise_std": args.router_noise_std,
        "router_z_loss_coef": args.router_z_loss_coef,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return MoEConfig(**values)


def get_phase_active_expert_ids(
    step: int,
    n_experts: int,
    train_experts_per_phase: int,
    phase_steps: int,
    phase_stride: int | None,
) -> list[int] | None:
    if train_experts_per_phase <= 0 or train_experts_per_phase >= n_experts:
        return None
    if phase_steps <= 0:
        raise ValueError("expert_phase_steps must be > 0")
    stride = phase_stride if phase_stride is not None else max(1, train_experts_per_phase // 2)
    phase = max(0, step - 1) // phase_steps
    start = (phase * stride) % n_experts
    return [(start + i) % n_experts for i in range(train_experts_per_phase)]


def set_expert_trainability(model: SimpleMoELanguageModel, active_expert_ids: Sequence[int] | None) -> None:
    active = None if active_expert_ids is None else set(active_expert_ids)
    for block in model.blocks:
        for expert_id, expert in enumerate(block.moe.experts):
            requires_grad = active is None or expert_id in active
            for param in expert.parameters():
                param.requires_grad = requires_grad


def expert_parameters(model: SimpleMoELanguageModel, active_expert_ids: Sequence[int] | None = None) -> list[torch.nn.Parameter]:
    active = None if active_expert_ids is None else set(active_expert_ids)
    params: list[torch.nn.Parameter] = []
    for block in model.blocks:
        for expert_id, expert in enumerate(block.moe.experts):
            if active is None or expert_id in active:
                params.extend(p for p in expert.parameters() if p.requires_grad)
    return params


def shared_parameters(model: SimpleMoELanguageModel) -> list[torch.nn.Parameter]:
    expert_param_ids = {id(param) for block in model.blocks for expert in block.moe.experts for param in expert.parameters()}
    return [param for param in model.parameters() if id(param) not in expert_param_ids and param.requires_grad]


def trainable_parameters(model: SimpleMoELanguageModel) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def make_optimizer(params: Sequence[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer | None:
    params = [param for param in params if param.requires_grad]
    if not params:
        return None
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def set_optimizer_lr(optimizer: torch.optim.Optimizer | None, lr: float) -> None:
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


def zero_optimizer(optimizer: torch.optim.Optimizer | None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)


def step_optimizer(optimizer: torch.optim.Optimizer | None) -> None:
    if optimizer is not None:
        optimizer.step()


def summarize_router_stats(router_stats: list[dict[str, torch.Tensor]]) -> dict[str, float] | None:
    if not router_stats:
        return None
    entropies: list[float] = []
    loads: list[torch.Tensor] = []
    for stats in router_stats:
        entropies.append(float(stats["entropy"]))
        eligible = stats["eligible_expert_ids"].long()
        loads.append(stats["load"].index_select(0, eligible))
    load_values = torch.cat(loads)
    return {
        "entropy": sum(entropies) / len(entropies),
        "load_min": float(load_values.min()),
        "load_max": float(load_values.max()),
        "dead": float((load_values == 0).sum()),
        "eligible": float(load_values.numel()),
    }


def make_trainer_state(
    step: int,
    use_phased_experts: bool,
    optimizer: torch.optim.Optimizer | None,
    shared_optimizer: torch.optim.Optimizer | None,
    expert_optimizer: torch.optim.Optimizer | None,
    active_expert_ids: Sequence[int] | None,
) -> dict:
    if use_phased_experts:
        return {
            "step": step,
            "optimizers": {
                "shared": None if shared_optimizer is None else shared_optimizer.state_dict(),
                "experts": None if expert_optimizer is None else expert_optimizer.state_dict(),
            },
            "active_expert_ids": None if active_expert_ids is None else list(active_expert_ids),
        }
    return {"step": step, "optimizer": None if optimizer is None else optimizer.state_dict()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = build_config(args, tokenizer)
    if args.train_experts_per_phase > 0 and args.train_experts_per_phase < config.active_experts:
        raise ValueError("train_experts_per_phase must be >= active_experts")
    if config.active_experts > config.n_experts:
        raise ValueError("active_experts must be <= n_experts")

    raw_model = SimpleMoELanguageModel(config).to(device)

    trainer_state = None
    start_step = 0
    if args.resume_from is not None:
        model_path = args.resume_from / "model.pt"
        trainer_state_path = args.resume_from / "trainer_state.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint model: {model_path}")
        print(f"Resuming from {args.resume_from}")
        raw_model.load_state_dict(torch.load(model_path, map_location=device))
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location=device)
            start_step = int(trainer_state.get("step", 0))
            print(f"Loaded trainer state at step {start_step}")
        else:
            print("No trainer_state.pt found; resuming model weights only")

    expert_counts = raw_model.expert_parameter_counts()
    expert_param_count = expert_counts[0][0] if expert_counts and expert_counts[0] else 0
    print(f"Device: {device}")
    print(f"Architecture: {config.arch}")
    print(f"Total parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    print(f"MoE: {config.n_experts} experts, top-{config.active_experts} active per token")
    print(f"Expert parameters per expert: {expert_param_count:,}")
    print(f"Expert parameters per MoE layer: {expert_counts}")
    if config.d_model == 500 and config.expert_hidden_size == 500:
        assert all(count == 500_000 for layer in expert_counts for count in layer)
    if config.d_model == 500 and config.expert_hidden_size == 1000:
        assert all(count == 1_000_000 for layer in expert_counts for count in layer)

    use_phased_experts = 0 < args.train_experts_per_phase < config.n_experts
    current_active_expert_ids = get_phase_active_expert_ids(
        start_step + 1,
        config.n_experts,
        args.train_experts_per_phase,
        args.expert_phase_steps,
        args.expert_phase_stride,
    )
    set_expert_trainability(raw_model, current_active_expert_ids if use_phased_experts else None)
    if use_phased_experts:
        print(
            f"Phased expert training: {args.train_experts_per_phase}/{config.n_experts} experts per phase, "
            f"phase_steps={args.expert_phase_steps}, active_expert_ids={current_active_expert_ids}"
        )
        if args.compile:
            print("Note: torch.compile may recompile when the expert phase changes.")

    dataset = FineWebTokenDataset(
        tokenizer=tokenizer,
        block_size=config.block_size,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        text_column=args.text_column,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    data_iter = iter(loader)

    optimizer: torch.optim.Optimizer | None = None
    shared_optimizer: torch.optim.Optimizer | None = None
    expert_optimizer: torch.optim.Optimizer | None = None
    if use_phased_experts:
        shared_optimizer = make_optimizer(shared_parameters(raw_model), args)
        expert_optimizer = make_optimizer(expert_parameters(raw_model, current_active_expert_ids), args)
    else:
        optimizer = make_optimizer(trainable_parameters(raw_model), args)

    if trainer_state is not None:
        if use_phased_experts and "optimizers" in trainer_state:
            optimizers_state = trainer_state["optimizers"]
            if shared_optimizer is not None and optimizers_state.get("shared") is not None:
                shared_optimizer.load_state_dict(optimizers_state["shared"])
                print("Loaded shared optimizer state")
            saved_active = trainer_state.get("active_expert_ids")
            if saved_active == current_active_expert_ids and expert_optimizer is not None and optimizers_state.get("experts") is not None:
                expert_optimizer.load_state_dict(optimizers_state["experts"])
                print("Loaded active expert optimizer state")
            elif use_phased_experts:
                print("Rebuilt active expert optimizer for current phase")
        elif not use_phased_experts and optimizer is not None and trainer_state.get("optimizer") is not None:
            optimizer.load_state_dict(trainer_state["optimizer"])
            print("Loaded optimizer state")
        else:
            print("Optimizer state format does not match current training mode; starting optimizers fresh")

    train_model = torch.compile(raw_model) if args.compile else raw_model  # type: ignore[assignment]
    use_amp = args.amp and device.type == "cuda"

    train_model.train()
    zero_optimizer(optimizer)
    zero_optimizer(shared_optimizer)
    zero_optimizer(expert_optimizer)
    start = time.time()
    running_loss = 0.0
    running_lm_loss = 0.0
    running_aux_loss = 0.0
    running_load_balance_loss = 0.0
    running_router_z_loss = 0.0
    running_router_entropy = 0.0
    router_stat_count = 0
    last_router_summary: dict[str, float] | None = None

    progress = tqdm(range(start_step + 1, args.max_steps + 1), desc="training", initial=start_step, total=args.max_steps)
    for step in progress:
        if use_phased_experts:
            next_active_expert_ids = get_phase_active_expert_ids(
                step,
                config.n_experts,
                args.train_experts_per_phase,
                args.expert_phase_steps,
                args.expert_phase_stride,
            )
            if next_active_expert_ids != current_active_expert_ids:
                current_active_expert_ids = next_active_expert_ids
                set_expert_trainability(raw_model, current_active_expert_ids)
                expert_optimizer = make_optimizer(expert_parameters(raw_model, current_active_expert_ids), args)
                zero_optimizer(shared_optimizer)
                zero_optimizer(expert_optimizer)
                progress.write(f"Expert phase changed at step {step}: active_expert_ids={current_active_expert_ids}")

        lr_scale = min(1.0, step / max(1, args.warmup_steps))
        current_lr = args.lr * lr_scale
        set_optimizer_lr(optimizer, current_lr)
        set_optimizer_lr(shared_optimizer, current_lr)
        set_optimizer_lr(expert_optimizer, current_lr)

        step_loss = 0.0
        step_lm_loss = 0.0
        step_aux_loss = 0.0
        step_load_balance_loss = 0.0
        step_router_z_loss = 0.0
        collect_router_stats = step % args.log_every == 0
        for _ in range(args.grad_accum_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = train_model(
                        input_ids,
                        labels=labels,
                        active_expert_ids=current_active_expert_ids if use_phased_experts else None,
                        collect_router_stats=collect_router_stats,
                    )
                    loss = out["loss"] / args.grad_accum_steps
            else:
                out = train_model(
                    input_ids,
                    labels=labels,
                    active_expert_ids=current_active_expert_ids if use_phased_experts else None,
                    collect_router_stats=collect_router_stats,
                )
                loss = out["loss"] / args.grad_accum_steps

            loss.backward()
            step_loss += float(loss.detach().cpu())
            step_lm_loss += float((out["lm_loss"] / args.grad_accum_steps).detach().cpu())
            step_aux_loss += float((out["aux_loss"] / args.grad_accum_steps).detach().cpu())
            step_load_balance_loss += float((out["load_balance_loss"] / args.grad_accum_steps).detach().cpu())
            step_router_z_loss += float((out["router_z_loss"] / args.grad_accum_steps).detach().cpu())
            summary = summarize_router_stats(out.get("router_stats", [])) if collect_router_stats else None
            if summary is not None:
                running_router_entropy += summary["entropy"]
                router_stat_count += 1
                last_router_summary = summary

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(raw_model), args.max_grad_norm)
        step_optimizer(optimizer)
        step_optimizer(shared_optimizer)
        step_optimizer(expert_optimizer)
        zero_optimizer(optimizer)
        zero_optimizer(shared_optimizer)
        zero_optimizer(expert_optimizer)

        running_loss += step_loss
        running_lm_loss += step_lm_loss
        running_aux_loss += step_aux_loss
        running_load_balance_loss += step_load_balance_loss
        running_router_z_loss += step_router_z_loss

        if step % args.log_every == 0:
            denom = args.log_every
            elapsed = max(time.time() - start, 1e-9)
            toks_per_sec = (step - start_step) * args.batch_size * args.grad_accum_steps * config.block_size / elapsed
            avg_loss = running_loss / denom
            avg_lm_loss = running_lm_loss / denom
            avg_aux_loss = running_aux_loss / denom
            avg_load_balance_loss = running_load_balance_loss / denom
            avg_router_z_loss = running_router_z_loss / denom
            postfix = {
                "loss": f"{avg_loss:.3f}",
                "ppl": f"{math.exp(min(avg_lm_loss, 20)):.1f}",
                "lm": f"{avg_lm_loss:.3f}",
                "aux": f"{avg_aux_loss:.3f}",
                "lb": f"{avg_load_balance_loss:.3f}",
                "z": f"{avg_router_z_loss:.4f}",
                "tok_s": f"{toks_per_sec:.0f}",
            }
            if router_stat_count > 0 and last_router_summary is not None:
                postfix.update(
                    {
                        "r_ent": f"{running_router_entropy / router_stat_count:.2f}",
                        "load": f"{last_router_summary['load_min']:.2f}-{last_router_summary['load_max']:.2f}",
                        "dead": f"{int(last_router_summary['dead'])}/{int(last_router_summary['eligible'])}",
                    }
                )
            progress.set_postfix(**postfix)
            running_loss = 0.0
            running_lm_loss = 0.0
            running_aux_loss = 0.0
            running_load_balance_loss = 0.0
            running_router_z_loss = 0.0
            running_router_entropy = 0.0
            router_stat_count = 0
            last_router_summary = None

        if step % args.save_every == 0:
            trainer_state_to_save = make_trainer_state(
                step,
                use_phased_experts,
                optimizer,
                shared_optimizer,
                expert_optimizer,
                current_active_expert_ids if use_phased_experts else None,
            )
            save_checkpoint(args.out_dir, raw_model, tokenizer, trainer_state_to_save, args)

    trainer_state_to_save = make_trainer_state(
        args.max_steps,
        use_phased_experts,
        optimizer,
        shared_optimizer,
        expert_optimizer,
        current_active_expert_ids if use_phased_experts else None,
    )
    save_checkpoint(args.out_dir, raw_model, tokenizer, trainer_state_to_save, args)
    print(f"Saved checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
