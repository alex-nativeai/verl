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

from verl.experimental.agent_loop.tool_parser import HermesToolParser


class _DummyTokenizer:
    def __init__(self, text: str):
        self._text = text

    def decode(self, _ids):
        return self._text


def test_hermes_tool_parser_ignores_tool_call_inside_think():
    text = (
        "<think>\n"
        "planning <tool_call>{\"name\":\"explore_sql\",\"arguments\":{\"sql\":\"SELECT 1\"}}</tool_call>\n"
        "</think>\n"
        "<tool_call>{\"name\":\"terminate\",\"arguments\":{}}</tool_call>"
    )
    parser = HermesToolParser(_DummyTokenizer(text))
    content, function_calls, parse_errors = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert len(function_calls) == 1
    assert function_calls[0].name == "terminate"
    assert len(parse_errors) == 0

    # Inner tool call is preserved because it was inside <think>.
    assert "explore_sql" in content
    # Extracted top-level tool call should be removed from remaining content.
    assert "\"terminate\"" not in content
