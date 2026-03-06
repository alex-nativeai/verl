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
import asyncio

from verl.experimental.agent_loop.tool_parser import GptOssToolParser


class _DummyTokenizer:
    pad_token = "<pad>"

    def __init__(self, text: str):
        self._text = text

    def decode(self, _ids, skip_special_tokens=False):
        return self._text


def test_gpt_oss_tool_parser_ignores_calls_inside_analysis_channel():
    text = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "thinking "
        "<|start|>assistant<|channel|>commentary to=functions.explore_sql "
        "<|constrain|>json<|message|>{\"sql\": \"SELECT 1\"}<|call|>"
        "<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.terminate "
        "<|constrain|>json<|message|>{}<|call|>"
    )
    parser = GptOssToolParser(_DummyTokenizer(text))
    content, function_calls, parse_errors = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert len(function_calls) == 1
    assert function_calls[0].name == "terminate"
    assert function_calls[0].arguments == "{}"
    assert "explore_sql" not in content
    assert len(parse_errors) == 0
