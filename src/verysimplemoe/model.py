from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """Configuration for a tiny decoder-only MoE language model.

    The default values are the original v1 architecture. Keep these defaults
    stable so old experiments/checkpoints remain reproducible.

    v1 experts have exactly 500,000 trainable parameters:

        Linear(500 -> 500, bias=False) + Linear(500 -> 500, bias=False)
        = 500 * 500 + 500 * 500 = 500,000
    """

    arch: str = "v1"

    vocab_size: int = 50257
    block_size: int = 256
    n_layers: int = 1
    d_model: int = 500
    n_heads: int = 10
    dropout: float = 0.1

    n_experts: int = 12
    active_experts: int = 6
    expert_hidden_size: int = 500
    aux_loss_coef: float = 0.01
    router_noise_std: float = 0.0
    router_z_loss_coef: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MoEConfig":
        return cls(**data)


ARCH_PRESETS: dict[str, dict[str, int | float | str]] = {
    "v1": {
        "arch": "v1",
        "d_model": 500,
        "n_heads": 10,
        "n_experts": 12,
        "active_experts": 6,
        "expert_hidden_size": 500,
        "router_noise_std": 0.0,
        "router_z_loss_coef": 0.0,
    },
    "v2-32x1m": {
        "arch": "v2-32x1m",
        "d_model": 500,
        "n_heads": 10,
        "n_experts": 32,
        "active_experts": 4,
        "expert_hidden_size": 1000,
        "router_noise_std": 0.1,
        "router_z_loss_coef": 1e-4,
    },
}


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float):
        super().__init__()
        # No biases: with d_model=hidden_size=500 this is exactly 500k params.
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        if config.active_experts > config.n_experts:
            raise ValueError("active_experts must be <= n_experts")
        self.n_experts = config.n_experts
        self.active_experts = config.active_experts
        self.aux_loss_coef = config.aux_loss_coef
        self.router_noise_std = config.router_noise_std
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(config.d_model, config.expert_hidden_size, config.dropout)
            for _ in range(config.n_experts)
        )

    def expert_parameter_counts(self) -> list[int]:
        return [sum(p.numel() for p in expert.parameters()) for expert in self.experts]

    def forward(
        self,
        x: torch.Tensor,
        active_expert_ids: Optional[Sequence[int] | torch.Tensor] = None,
        collect_router_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        bsz, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)

        router_logits = self.router(flat_x)  # [tokens, experts]
        if active_expert_ids is None:
            eligible_expert_ids = torch.arange(self.n_experts, device=flat_x.device)
            eligible_logits = router_logits
        else:
            eligible_expert_ids = torch.as_tensor(active_expert_ids, dtype=torch.long, device=flat_x.device)
            if eligible_expert_ids.numel() < self.active_experts:
                raise ValueError("active_expert_ids must contain at least active_experts entries")
            eligible_logits = router_logits.index_select(dim=-1, index=eligible_expert_ids)

        if self.training and self.router_noise_std > 0:
            eligible_logits = eligible_logits + torch.randn_like(eligible_logits) * self.router_noise_std

        eligible_probs = F.softmax(eligible_logits, dim=-1)
        topk_probs, topk_local_idx = torch.topk(eligible_probs, self.active_experts, dim=-1)
        topk_idx = eligible_expert_ids.index_select(0, topk_local_idx.reshape(-1)).reshape_as(topk_local_idx)
        # Normalize selected expert weights so the mixed expert output scale is stable.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_pos, choice_pos = torch.where(topk_idx == expert_id)
            if token_pos.numel() == 0:
                continue
            expert_in = flat_x.index_select(0, token_pos)
            expert_out = expert(expert_in)
            weights = topk_probs[token_pos, choice_pos].unsqueeze(-1).to(expert_out.dtype)
            flat_out.index_add_(0, token_pos, expert_out * weights)

        # Switch Transformer-style load balancing term, computed only over the
        # currently eligible experts. This matters when training a 16-of-32
        # expert phase: the router should balance the phase, not unreachable
        # experts.
        n_eligible = eligible_expert_ids.numel()
        importance_local = eligible_probs.mean(dim=0)
        one_hot_local = F.one_hot(topk_local_idx, num_classes=n_eligible).float().sum(dim=1)
        load_local = one_hot_local.mean(dim=0) / self.active_experts
        load_balance_loss = n_eligible * torch.sum(importance_local * load_local) * self.aux_loss_coef

        if self.router_z_loss_coef > 0:
            router_z_loss = torch.mean(torch.logsumexp(eligible_logits, dim=-1).pow(2)) * self.router_z_loss_coef
        else:
            router_z_loss = x.new_zeros(())

        router_stats = None
        if collect_router_stats:
            importance = torch.zeros(self.n_experts, device=flat_x.device, dtype=importance_local.dtype)
            load = torch.zeros(self.n_experts, device=flat_x.device, dtype=load_local.dtype)
            importance.index_copy_(0, eligible_expert_ids, importance_local.detach())
            load.index_copy_(0, eligible_expert_ids, load_local.detach())
            entropy = -(eligible_probs.detach() * eligible_probs.detach().clamp_min(1e-9).log()).sum(dim=-1).mean()
            router_stats = {
                "eligible_expert_ids": eligible_expert_ids.detach().cpu(),
                "importance": importance.detach().cpu(),
                "load": load.detach().cpu(),
                "entropy": entropy.detach().cpu(),
            }

        return flat_out.reshape(bsz, seq_len, d_model), load_balance_loss, router_z_loss, router_stats


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = TopKMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        active_expert_ids: Optional[Sequence[int] | torch.Tensor] = None,
        collect_router_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        x = x + self.attn(self.ln1(x))
        moe_out, load_balance_loss, router_z_loss, router_stats = self.moe(
            self.ln2(x),
            active_expert_ids=active_expert_ids,
            collect_router_stats=collect_router_stats,
        )
        x = x + moe_out
        return x, load_balance_loss, router_z_loss, router_stats


class SimpleMoELanguageModel(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def expert_parameter_counts(self) -> list[list[int]]:
        return [block.moe.expert_parameter_counts() for block in self.blocks]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_expert_ids: Optional[Sequence[int] | torch.Tensor] = None,
        collect_router_stats: bool = False,
    ) -> dict[str, Any]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)

        load_balance_loss = x.new_zeros(())
        router_z_loss = x.new_zeros(())
        router_stats = []
        for block in self.blocks:
            x, block_load_balance, block_router_z, block_stats = block(
                x,
                active_expert_ids=active_expert_ids,
                collect_router_stats=collect_router_stats,
            )
            load_balance_loss = load_balance_loss + block_load_balance
            router_z_loss = router_z_loss + block_router_z
            if collect_router_stats and block_stats is not None:
                router_stats.append(block_stats)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        lm_loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            loss = lm_loss + load_balance_loss + router_z_loss

        aux_loss = load_balance_loss + router_z_loss
        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "router_stats": router_stats,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.block_size :]
            logits = self(idx_cond)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
