# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import importlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch


def _build_dummy_mlp_class(class_name: str) -> type[nn.Module]:
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 8, bias=False)
            self.up_proj = nn.Linear(4, 8, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    DummyMLP.__name__ = class_name
    return DummyMLP


def _build_dummy_moe_experts_class(class_name: str) -> type[nn.Module]:
    class DummyExperts(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, hidden_states, top_k_index=None, top_k_weights=None):
            return hidden_states

    DummyExperts.__name__ = class_name
    return DummyExperts


def _build_numeric_dummy_moe_experts_class(class_name: str) -> type[nn.Module]:
    class DummyExperts(nn.Module):
        def __init__(self, hidden_size: int = 4, intermediate_size: int = 8, num_experts: int = 3):
            super().__init__()
            self.num_experts = num_experts
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, intermediate_size * 2, hidden_size))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size))
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index=None, top_k_weights=None):
            if top_k_index is None or top_k_weights is None:
                raise ValueError("top_k_index/top_k_weights are required")

            restore_shape = hidden_states.shape if hidden_states.ndim == 3 else None
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            top_k_index_flat = top_k_index.reshape(hidden_flat.shape[0], -1)
            top_k_weights_flat = top_k_weights.reshape(hidden_flat.shape[0], -1)

            out = torch.zeros_like(hidden_flat)
            for tok in range(hidden_flat.shape[0]):
                token_state = hidden_flat[tok : tok + 1]
                for pos in range(top_k_index_flat.shape[1]):
                    expert_idx = int(top_k_index_flat[tok, pos].item())
                    gate, up = torch.nn.functional.linear(token_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    hidden = self.act_fn(gate) * up
                    expert_out = torch.nn.functional.linear(hidden, self.down_proj[expert_idx])
                    out[tok] += (expert_out * top_k_weights_flat[tok, pos]).squeeze(0)

            if restore_shape is None:
                return out
            return out.reshape(*restore_shape)

    DummyExperts.__name__ = class_name
    return DummyExperts


def _collect_param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def test_tiled_mlp_qwen3_5_direct_patch(monkeypatch):
    qwen35_module_path = "transformers.models.qwen3_5.modeling_qwen3_5"
    qwen35_class = _build_dummy_mlp_class("Qwen3_5MLP")
    qwen35_module = SimpleNamespace(Qwen3_5MLP=qwen35_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_module_path:
            return qwen35_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    original_forward = qwen35_class.forward
    patched = apply_tiled_mlp_monkey_patch(num_shards=2, model_type="qwen3_5")

    assert "Qwen3_5MLP" in patched
    assert qwen35_class.forward is not original_forward


def test_tiled_mlp_qwen3_5_fallback_to_qwen3(monkeypatch):
    qwen35_module_path = "transformers.models.qwen3_5.modeling_qwen3_5"
    qwen3_module_path = "transformers.models.qwen3.modeling_qwen3"
    qwen3_class = _build_dummy_mlp_class("Qwen3MLP")
    qwen3_module = SimpleNamespace(Qwen3MLP=qwen3_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_module_path:
            raise ImportError(module_path)
        if module_path == qwen3_module_path:
            return qwen3_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    original_forward = qwen3_class.forward
    patched = apply_tiled_mlp_monkey_patch(num_shards=2, model_type="qwen3_5")

    assert "Qwen3MLP" in patched
    assert qwen3_class.forward is not original_forward


def test_tiled_mlp_qwen3_5_moe_direct_patch(monkeypatch):
    qwen35_moe_module_path = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    qwen35_moe_class = _build_dummy_moe_experts_class("Qwen3_5MoeExperts")
    qwen35_moe_module = SimpleNamespace(Qwen3_5MoeExperts=qwen35_moe_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_moe_module_path:
            return qwen35_moe_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    original_forward = qwen35_moe_class.forward
    patched = apply_tiled_mlp_monkey_patch(num_shards=2, model_type="qwen3_5_moe")

    assert "Qwen3_5MoeExperts" in patched
    assert qwen35_moe_class.forward is not original_forward


def test_tiled_mlp_qwen3_5_moe_fallback_to_qwen3_moe(monkeypatch):
    qwen35_moe_module_path = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    qwen3_moe_module_path = "transformers.models.qwen3_moe.modeling_qwen3_moe"
    qwen3_moe_class = _build_dummy_moe_experts_class("Qwen3MoeExperts")
    qwen3_moe_module = SimpleNamespace(Qwen3MoeExperts=qwen3_moe_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_moe_module_path:
            raise ImportError(module_path)
        if module_path == qwen3_moe_module_path:
            return qwen3_moe_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    original_forward = qwen3_moe_class.forward
    patched = apply_tiled_mlp_monkey_patch(num_shards=2, model_type="qwen3_5_moe")

    assert "Qwen3MoeExperts" in patched
    assert qwen3_moe_class.forward is not original_forward


def test_tiled_mlp_qwen3_5_dense_numerical_equivalence(monkeypatch):
    torch.manual_seed(7)
    qwen35_module_path = "transformers.models.qwen3_5.modeling_qwen3_5"
    qwen35_class = _build_dummy_mlp_class("Qwen3_5MLP")
    qwen35_module = SimpleNamespace(Qwen3_5MLP=qwen35_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_module_path:
            return qwen35_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    x_ref = torch.randn(2, 5, 4, dtype=torch.float32, requires_grad=True)
    x_tiled = x_ref.detach().clone().requires_grad_(True)
    ref = qwen35_class()
    y_ref = ref(x_ref)
    loss_ref = y_ref.square().mean()
    loss_ref.backward()
    ref_grads = _collect_param_grads(ref)

    apply_tiled_mlp_monkey_patch(num_shards=3, model_type="qwen3_5")
    tiled = qwen35_class()
    tiled.load_state_dict(ref.state_dict())
    y_tiled = tiled(x_tiled)
    loss_tiled = y_tiled.square().mean()
    loss_tiled.backward()
    tiled_grads = _collect_param_grads(tiled)

    assert torch.allclose(y_ref.detach(), y_tiled.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_ref.grad, x_tiled.grad, atol=1e-6, rtol=1e-6)
    assert ref_grads.keys() == tiled_grads.keys()
    for name in ref_grads:
        assert torch.allclose(ref_grads[name], tiled_grads[name], atol=1e-6, rtol=1e-6)


def test_tiled_mlp_qwen3_5_moe_numerical_equivalence(monkeypatch):
    torch.manual_seed(11)
    qwen35_moe_module_path = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    qwen35_moe_class = _build_numeric_dummy_moe_experts_class("Qwen3_5MoeExperts")
    qwen35_moe_module = SimpleNamespace(Qwen3_5MoeExperts=qwen35_moe_class)

    def fake_import_module(module_path: str):
        if module_path == qwen35_moe_module_path:
            return qwen35_moe_module
        raise ImportError(module_path)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    hidden_ref = torch.randn(2, 4, 4, dtype=torch.float32, requires_grad=True)
    hidden_tiled = hidden_ref.detach().clone().requires_grad_(True)
    top_k_index = torch.randint(0, 3, (2, 4, 2), dtype=torch.long)
    top_k_weights = torch.rand(2, 4, 2, dtype=torch.float32)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    ref = qwen35_moe_class()
    y_ref = ref(hidden_ref, top_k_index, top_k_weights)
    loss_ref = y_ref.square().mean()
    loss_ref.backward()
    ref_grads = _collect_param_grads(ref)

    apply_tiled_mlp_monkey_patch(num_shards=3, model_type="qwen3_5_moe")
    tiled = qwen35_moe_class()
    tiled.load_state_dict(ref.state_dict())
    y_tiled = tiled(hidden_tiled, top_k_index, top_k_weights)
    loss_tiled = y_tiled.square().mean()
    loss_tiled.backward()
    tiled_grads = _collect_param_grads(tiled)

    assert torch.allclose(y_ref.detach(), y_tiled.detach(), atol=2e-5, rtol=2e-5)
    assert torch.allclose(hidden_ref.grad, hidden_tiled.grad, atol=2e-5, rtol=2e-5)
    assert ref_grads.keys() == tiled_grads.keys()
    for name in ref_grads:
        assert torch.allclose(ref_grads[name], tiled_grads[name], atol=2e-5, rtol=2e-5)
