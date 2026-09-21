# SPDX-License-Identifier: Apache-2.0
"""Nemotron-Nano-v2 <TOOLCALL> parser for vLLM 0.29+.

The parser shipped with the Hugging Face model still imports
`vllm.entrypoints.openai.tool_parsers`, which 0.29 removed. This file
registers the same `nemotron_json` name against the current ToolParser API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager

logger = init_logger(__name__)


class NemotronJSONToolParser(ToolParser):
    tool_call_start_token = "<TOOLCALL>"
    tool_call_end_token = "</TOOLCALL>"
    tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            raw = self.tool_call_regex.findall(model_output)[0].strip()
            if not raw.startswith("["):
                raw = "[" + raw
            if not raw.endswith("]"):
                raw = raw + "]"
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]

            tool_calls: list[ToolCall] = []
            for tool_call in parsed:
                name = tool_call.get("name")
                if not name:
                    continue
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                )

            content = model_output[: model_output.find(self.tool_call_start_token)]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error extracting Nemotron tool call from: %s", model_output)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ):
        return None


ToolParserManager.register_module(name="nemotron_json", module=NemotronJSONToolParser)
