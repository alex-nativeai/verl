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

import pytest

from verl.utils.fsdp_utils import _normalize_fsdp_wrap_layer_names


def test_normalize_fsdp_wrap_layer_names_from_string():
    assert _normalize_fsdp_wrap_layer_names("Qwen3_5DecoderLayer") == {"Qwen3_5DecoderLayer"}


def test_normalize_fsdp_wrap_layer_names_from_set_of_strings():
    names = {"Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"}
    assert _normalize_fsdp_wrap_layer_names(names) == names


def test_normalize_fsdp_wrap_layer_names_from_set_of_classes():
    class DecoderLayer:
        pass

    class VisionBlock:
        pass

    assert _normalize_fsdp_wrap_layer_names({DecoderLayer, VisionBlock}) == {"DecoderLayer", "VisionBlock"}


def test_normalize_fsdp_wrap_layer_names_rejects_invalid_item():
    with pytest.raises(TypeError):
        _normalize_fsdp_wrap_layer_names({"Qwen3_5DecoderLayer", 123})
