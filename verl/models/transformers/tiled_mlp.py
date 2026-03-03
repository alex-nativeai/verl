# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP2-compatible TiledMLP implementation for memory-efficient MLP computation.

This module provides a tiled MLP implementation that reduces peak memory usage
by processing the MLP forward/backward pass in chunks (tiles). This is particularly
useful for large models with FSDP2 training.
"""

import threading
from typing import Any, Optional

import torch
import torch.nn as nn
`

class GradientAccumulator:
    """Gradient accumulator for TiledMLP (FSDP compatible).

    This class manages gradient accumulation across multiple shards during
    the backward pass of TiledMLP. It ensures correct gradient computation
    when processing input in chunks.
    """

    def __init__(self, params: list[torch.nn.Parameter], total_shards: int, dtype: torch.dtype = None):
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()

        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(self.grad_accumulation_dtype)
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(param, dtype=self.grad_accumulation_dtype)

    def install_hooks(self, is_last_shard: bool):
        """Install gradient hooks for the current shard."""
        self._remove_hooks()

        def create_hook(param):
            def hook(grad):
                with self.lock:
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    self.accumulated_grads[param] += grad_to_accum_dtype

                    if is_last_shard:
                        param.grad = None  # Critical: prevent double accumulation
                        final_grad = self.accumulated_grads[param].to(param.dtype)
                        return final_grad
                    return None

            return hook

        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def cleanup(self):
        """Cleanup hooks and resources."""
        self._remove_hooks()


class TiledMLP(torch.autograd.Function):
    """TiledMLP implementation for memory-efficient MLP computation.

    This autograd function processes MLP forward/backward in tiles (chunks)
    to reduce peak memory usage. Compatible with FSDP2.
    """

    @staticmethod
    def forward(ctx, fn, module, x, shards, compute_params):
        ctx.fn = fn
        ctx.module = module
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        # Split on dim=-2 (seqlen dimension) following Liger Kernel style
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)
        return output_unsharded

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        module = ctx.module
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        # Flatten to [bs*seqlen, hidden_size]
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)

        # Pre-allocate input gradient
        x_grad = torch.zeros_like(x)

        # Split on dim=0
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        grad_accumulator = GradientAccumulator(compute_params, shards, dtype=x.dtype)

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            # narrow(0, ...) creates a contiguous view that can receive gradients
            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step)

            is_last_shard = i + 1 == shards
            grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                output = fn(module, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        grad_accumulator.cleanup()
        del grad_accumulator

        # Restore original shape
        x_grad = x_grad.view(x_shape_orig) if x_requires_grad else None
        return (None, None, x_grad, None, None)


def _mlp_forward_fn(module, x):
    """Forward function for LlamaMLP / Qwen2MLP / Qwen3MLP style."""
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


# ============================================================================
# Monkey Patch Functions
# ============================================================================

# Model type to MLP class mapping
_MODEL_TYPE_TO_MLP_CLASS = {
    "llama": ("transformers.models.llama.modeling_llama", "LlamaMLP"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
    "qwen2_5": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),  # Qwen2.5 uses Qwen2 MLP
    "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
}


def _patch_gpt_oss_experts_class(experts_class: type[nn.Module], num_shards: int):
    """Patch GPT-OSS MoE experts forward with token-chunked computation.

    GPT-OSS uses MoE experts instead of dense SwiGLU MLP modules. This patch reduces
    peak activation memory by chunking per-expert token processing.
    """

    original_forward = experts_class.forward

    def _flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        if x.ndim == 2:
            return x, None
        if x.ndim == 3:
            bsz, seqlen, hidden = x.shape
            return x.reshape(-1, hidden), (bsz, seqlen, hidden)
        raise RuntimeError(f"Unsupported hidden_states rank for GPT-OSS experts: {x.ndim}")

    def _flatten_router(x: torch.Tensor, shape: tuple[int, ...] | None) -> torch.Tensor:
        if shape is None:
            return x
        bsz, seqlen, _hidden = shape
        return x.reshape(bsz * seqlen, x.shape[-1])

    def tiled_forward(
        self: Any,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fallback to upstream behavior for unexpected invocation patterns.
        if router_indices is None or routing_weights is None:
            return original_forward(self, hidden_states, router_indices, routing_weights)

        hidden_flat, restore_shape = _flatten_tokens(hidden_states)
        router_idx_flat = _flatten_router(router_indices, restore_shape)
        routing_w_flat = _flatten_router(routing_weights, restore_shape)

        next_states = torch.zeros_like(hidden_flat, dtype=hidden_flat.dtype, device=hidden_flat.device)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_idx_flat, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t.item())
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            chunk_size = max(1, (token_idx.numel() + num_shards - 1) // num_shards)
            for start in range(0, token_idx.numel(), chunk_size):
                end = min(start + chunk_size, token_idx.numel())
                tok = token_idx[start:end]
                pos = top_k_pos[start:end]

                current_state = hidden_flat[tok]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate_up_proj_bias = getattr(self, "gate_up_proj_bias", None)
                if gate_up_proj_bias is not None:
                    gate_up = gate_up + gate_up_proj_bias[expert_idx]
                # Handle both GPT-OSS variants:
                # - older/newer classes exposing `_apply_gate`
                # - classes without `_apply_gate` (inline gate math)
                if hasattr(self, "_apply_gate"):
                    gated_output = self._apply_gate(gate_up)
                else:
                    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    limit = getattr(self, "limit", 7.0)
                    alpha = getattr(self, "alpha", 1.702)
                    gate = gate.clamp(min=None, max=limit)
                    up = up.clamp(min=-limit, max=limit)
                    glu = gate * torch.sigmoid(gate * alpha)
                    gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx]
                down_proj_bias = getattr(self, "down_proj_bias", None)
                if down_proj_bias is not None:
                    out = out + down_proj_bias[expert_idx]
                # Support both GPT-OSS router score layouts across transformers versions.
                # Prefer expert-axis indexing when ambiguous (e.g., top_k == num_experts).
                if routing_w_flat.shape[1] == self.num_experts:
                    weights = routing_w_flat[tok, expert_idx]
                elif routing_w_flat.shape[1] == router_idx_flat.shape[1]:
                    weights = routing_w_flat[tok, pos]
                else:
                    return original_forward(self, hidden_states, router_indices, routing_weights)

                weighted_output = out * weights[:, None]
                next_states.index_add_(0, tok, weighted_output.to(hidden_flat.dtype))

        if restore_shape is None:
            return next_states

        bsz, seqlen, hidden = restore_shape
        return next_states.reshape(bsz, seqlen, hidden)

    experts_class.forward = tiled_forward


def _patch_qwen3_moe_experts_class(experts_class: type[nn.Module], num_shards: int):
    """Patch Qwen3MoE experts forward with token-chunked computation.

    This keeps the upstream Qwen3MoeExperts.forward signature and return shape:
    forward(hidden_states, top_k_index, top_k_weights) -> hidden_states.
    """

    original_forward = experts_class.forward

    def _flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        if x.ndim == 2:
            return x, None
        if x.ndim == 3:
            bsz, seqlen, hidden = x.shape
            return x.reshape(-1, hidden), (bsz, seqlen, hidden)
        raise RuntimeError(f"Unsupported hidden_states rank for Qwen3Moe experts: {x.ndim}")

    def _flatten_router(x: torch.Tensor, shape: tuple[int, ...] | None) -> torch.Tensor:
        if shape is None:
            return x
        bsz, seqlen, _hidden = shape
        return x.reshape(bsz * seqlen, x.shape[-1])

    def tiled_forward(
        self: Any,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor | None = None,
        top_k_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fallback to upstream behavior for unexpected invocation patterns.
        if top_k_index is None or top_k_weights is None:
            return original_forward(self, hidden_states, top_k_index, top_k_weights)

        hidden_flat, restore_shape = _flatten_tokens(hidden_states)
        top_k_index_flat = _flatten_router(top_k_index, restore_shape)
        top_k_weights_flat = _flatten_router(top_k_weights, restore_shape)

        final_hidden_states = torch.zeros_like(hidden_flat)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index_flat, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t.item())
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            chunk_size = max(1, (token_idx.numel() + num_shards - 1) // num_shards)
            for start in range(0, token_idx.numel(), chunk_size):
                end = min(start + chunk_size, token_idx.numel())
                tok = token_idx[start:end]
                pos = top_k_pos[start:end]

                current_state = hidden_flat[tok]
                gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights_flat[tok, pos, None]
                final_hidden_states.index_add_(0, tok, current_hidden_states.to(final_hidden_states.dtype))

        if restore_shape is None:
            return final_hidden_states

        bsz, seqlen, hidden = restore_shape
        return final_hidden_states.reshape(bsz, seqlen, hidden)

    experts_class.forward = tiled_forward


def apply_tiled_mlp_monkey_patch(
    num_shards: int = 4,
    model_type: Optional[str] = None,
):
    """Apply TiledMLP monkey patch based on model_type.

    This function MUST be called BEFORE model instantiation to take effect.
    It patches the MLP classes in transformers library to use TiledMLP for
    memory-efficient computation during training.

    Args:
        num_shards: Number of shards to split the input into. Higher values
                   reduce peak memory but may slightly impact performance.
        model_type: The model type string (e.g., "llama", "qwen2", "qwen3").
                   If None, patches all supported model types.

    Returns:
        List of patched class names.
    """
    if model_type is None:
        types_to_patch = [*list(_MODEL_TYPE_TO_MLP_CLASS.keys()), "gpt_oss", "qwen3_moe"]
    elif model_type in ("gpt_oss", "qwen3_moe"):
        types_to_patch = [model_type]
    elif model_type in _MODEL_TYPE_TO_MLP_CLASS:
        types_to_patch = [model_type]
    else:
        raise ValueError(
            f"TiledMLP does not support model_type='{model_type}'. "
            f"Supported types: {list(_MODEL_TYPE_TO_MLP_CLASS.keys())}, gpt_oss, qwen3_moe. "
            f"For SwiGLU-style MLPs, you can add support by extending _MODEL_TYPE_TO_MLP_CLASS "
            f"in verl/models/transformers/tiled_mlp.py"
        )

    patched_classes = []

    for mtype in types_to_patch:
        try:
            import importlib

            if mtype == "gpt_oss":
                module = importlib.import_module("transformers.models.gpt_oss.modeling_gpt_oss")
                experts_class = getattr(module, "GptOssExperts")
                _patch_gpt_oss_experts_class(experts_class, num_shards)
                if "GptOssExperts" not in patched_classes:
                    patched_classes.append("GptOssExperts")
                continue

            if mtype == "qwen3_moe":
                module = importlib.import_module("transformers.models.qwen3_moe.modeling_qwen3_moe")
                # Newer transformers: packed expert weights in Qwen3MoeExperts.
                if hasattr(module, "Qwen3MoeExperts"):
                    experts_class = getattr(module, "Qwen3MoeExperts")
                    _patch_qwen3_moe_experts_class(experts_class, num_shards)
                    if "Qwen3MoeExperts" not in patched_classes:
                        patched_classes.append("Qwen3MoeExperts")
                # Older/community variants: experts are Qwen3MoeMLP modules.
                elif hasattr(module, "Qwen3MoeMLP"):
                    mlp_class = getattr(module, "Qwen3MoeMLP")
                    _patch_mlp_class(mlp_class, _mlp_forward_fn, num_shards)
                    if "Qwen3MoeMLP" not in patched_classes:
                        patched_classes.append("Qwen3MoeMLP")
                else:
                    raise AttributeError("Neither Qwen3MoeExperts nor Qwen3MoeMLP found")
                continue

            module_path, class_name = _MODEL_TYPE_TO_MLP_CLASS[mtype]
            module = importlib.import_module(module_path)
            mlp_class = getattr(module, class_name)
            _patch_mlp_class(mlp_class, _mlp_forward_fn, num_shards)
            if class_name not in patched_classes:
                patched_classes.append(class_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not patch {mtype} MLP: {e}")

    if patched_classes:
        print(f"TiledMLP monkey patch applied to: {', '.join(patched_classes)} (shards={num_shards})")

    return patched_classes


def _patch_mlp_class(mlp_class: type[nn.Module], forward_fn, num_shards: int):
    """Patch a single MLP class to use TiledMLP."""

    def tiled_forward(self, x):
        compute_params = [p for p in self.parameters() if p.requires_grad]
        return TiledMLP.apply(forward_fn, self, x, num_shards, compute_params)

    mlp_class.forward = tiled_forward
