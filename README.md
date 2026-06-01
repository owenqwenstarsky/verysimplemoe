# VerySimpleMoE

A working PyTorch implementation of a tiny decoder-only Mixture-of-Experts language model trained from HuggingFace FineWeb.

## Architectures

The original architecture is preserved as the `v1` preset:

- 12 experts total per MoE layer
- 6 active experts per token (`top-k=6` router)
- each expert has exactly 500,000 parameters
  - expert MLP: `Linear(500 -> 500, bias=False)`, GELU, `Linear(500 -> 500, bias=False)`
  - params: `500*500 + 500*500 = 500,000`
- GPT-style causal self-attention before the MoE block
- learned top-k router plus load-balancing auxiliary loss
- GPT-2 tokenizer by default

The new research architecture is available as `v2-32x1m`:

- 32 experts total per MoE layer
- 4 active experts per token by default (`top-k=4` router)
- each expert has exactly 1,000,000 parameters
  - expert MLP: `Linear(500 -> 1000, bias=False)`, GELU, `Linear(1000 -> 500, bias=False)`
  - params: `500*1000 + 1000*500 = 1,000,000`
- router noise during training for exploration
- router z-loss to keep router logits stable
- optional phased expert training so only a subset of experts are trainable/routeable at a time

The default still uses `n_layers=1`, which means the model has one MoE block. If you raise `--n-layers`, each layer gets its own full expert set.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train v1 on FineWeb

This streams FineWeb, so it does not download the full dataset first.

```bash
verysimplemoe-train \
  --arch v1 \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --out-dir checkpoints/verysimplemoe-v1 \
  --max-steps 1000 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --block-size 256
```

## Train v2: 32 experts, 1M params per expert

Recommended laptop-friendly command:

```bash
verysimplemoe-train \
  --arch v2-32x1m \
  --out-dir checkpoints/verysimplemoe-v2-32x1m \
  --train-experts-per-phase 16 \
  --expert-phase-steps 500 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --block-size 256
```

Phased expert training means:

- only 16 of the 32 experts are eligible for routing in a given phase
- only those 16 experts have gradients enabled
- the expert optimizer is rebuilt per phase, so Adam state is kept only for the current expert subset
- the phase window overlaps by default using a half-window stride, e.g. `0-15`, `8-23`, `16-31`, `24-31 + 0-7`

Useful router options:

```bash
--active-experts 4          # top-k experts per token
--router-noise-std 0.1      # train-time router exploration
--router-z-loss-coef 1e-4   # router logit stabilization
--aux-loss-coef 0.01        # load-balancing loss
```

## Tiny CPU smoke test

```bash
verysimplemoe-train --device cpu --max-steps 5 --batch-size 1 --grad-accum-steps 1 --block-size 64
```

For a tiny v2 smoke test:

```bash
verysimplemoe-train \
  --device cpu \
  --arch v2-32x1m \
  --train-experts-per-phase 16 \
  --max-steps 5 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --block-size 64
```

Useful speed flags on NVIDIA GPUs:

```bash
verysimplemoe-train --amp --compile --max-steps 10000 --batch-size 16
```

Note: with phased expert training, `torch.compile` may recompile when the active expert phase changes.

## Resume after interruption

If training stops after a checkpoint save begins, resume from the checkpoint directory:

```bash
verysimplemoe-train \
  --resume-from checkpoints/verysimplemoe-v2-32x1m \
  --out-dir checkpoints/verysimplemoe-v2-32x1m \
  --max-steps 1000 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --block-size 256 \
  --train-experts-per-phase 16
```

`--max-steps` is the final target step count, not additional steps.

## Generate with the final model

```bash
verysimplemoe-generate \
  --checkpoint checkpoints/verysimplemoe-v2-32x1m \
  --prompt "The future of open language models is" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50
```

You can also run modules without installing scripts:

```bash
PYTHONPATH=src python -m verysimplemoe.train --max-steps 100
PYTHONPATH=src python -m verysimplemoe.generate --prompt "Hello"
```

## Files

- `src/verysimplemoe/model.py` - model, MoE router, experts, generation
- `src/verysimplemoe/train.py` - FineWeb streaming trainer, phased expert training, checkpointing
- `src/verysimplemoe/generate.py` - checkpoint loader and text generation CLI
