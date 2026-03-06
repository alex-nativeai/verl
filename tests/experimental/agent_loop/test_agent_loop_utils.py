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
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text


def test_build_gpt_oss_tool_response_text_optional_generation_prompt():
    messages = [{"role": "tool", "content": "Conversation ended."}]
    tool_call_names = ["terminate"]

    with_generation_prompt = build_gpt_oss_tool_response_text(messages, tool_call_names)
    without_generation_prompt = build_gpt_oss_tool_response_text(
        messages, tool_call_names, include_generation_prompt=False
    )

    assert with_generation_prompt.endswith("<|start|>assistant")
    assert not without_generation_prompt.endswith("<|start|>assistant")
