"""Tool calling functionality for Snowflake chat models - PATCHED VERSION.

PATCHES APPLIED (Format Fixes Only - NO conversation trimming):
1. Fix assistant message format: Include top-level 'content' field per Snowflake REST API spec
2. Fix tool_results: Ensure tool name is correctly extracted from tool_call_id mapping
3. Truncate individual tool outputs (>4000 chars) - NOT conversation history
4. Extract Cortex Inference metadata (request_id, finish_reason, guard_tokens) for OTel observability

WHY 400 BAD REQUEST OCCURS:
- Snowflake REST API expects assistant messages with tool_calls to have BOTH:
  - "content": "text..." (top-level string)
  - "content_list": [{"type": "tool_use", ...}] (tool_use blocks only)
- Original langchain-snowflake puts text INSIDE content_list, not as top-level field
- This format mismatch causes 400 Bad Request on multi-turn conversations

OTEL OBSERVABILITY:
- Extracts request_id from response 'id' field -> maps to gen_ai.response.id (standard semconv)
- Extracts finish_reason from choices[0].finish_reason -> maps to gen_ai.response.finish_reasons
- Extracts guard_tokens from usage.guard_tokens -> maps to snowflake.inference.guard_tokens
"""

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from .._error_handling import SnowflakeErrorHandler

# SnowflakeMetadataFactory import removed - using inline metadata dicts

logger = logging.getLogger(__name__)

# Maximum characters for a SINGLE tool output (prevents malformed payloads, NOT conversation trimming)
MAX_TOOL_OUTPUT_CHARS = 8000


def _truncate_tool_output(content: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate individual tool output if extremely large.

    This is NOT about context limits (200k tokens is huge).
    This prevents malformed JSON payloads when a single tool returns massive output.
    """
    if len(content) <= max_chars:
        return content

    # Keep first and last parts with truncation indicator
    half = max_chars // 2 - 30
    return content[:half] + "\n\n... [tool output truncated for payload size] ...\n\n" + content[-half:]


class SnowflakeTools:
    """Mixin class for Snowflake tool calling functionality."""

    def bind_tools(
        self,
        tools: Sequence[BaseTool | Callable | dict],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model using Snowflake Cortex Complete native tool calling.

        Args:
            tools: A list of tools to bind to the model
            **kwargs: Additional arguments including tool_choice

        Returns:
            A new ChatSnowflake instance with tools bound and REST API enabled
        """
        # Convert LangChain tools to Snowflake tool format
        snowflake_tools = []

        for tool_obj in tools:
            openai_tool = convert_to_openai_tool(tool_obj)
            tool_name = openai_tool["function"]["name"]

            # Convert OpenAI format to Snowflake format exactly as per official documentation
            input_schema = openai_tool["function"]["parameters"].copy()

            # Ensure properties have descriptions (required for tool calling to work properly)
            if "properties" in input_schema:
                for prop_name, prop_def in input_schema["properties"].items():
                    if "description" not in prop_def:
                        # Add default description if missing
                        prop_def["description"] = f"The {prop_name} parameter"

            tool_spec = {
                "type": "generic",
                "name": tool_name,
                "description": openai_tool["function"]["description"],
                "input_schema": input_schema,
            }

            snowflake_tools.append({"tool_spec": tool_spec})

        # Handle tool_choice (processed for LangChain compatibility but not sent to Snowflake)
        tool_choice = kwargs.pop("tool_choice", "auto")
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool" and "name" in tool_choice:
                tool_choice = tool_choice["name"]
            else:
                tool_choice = "auto"
        elif tool_choice not in ["auto", "none", "any", "required"]:
            tool_choice = "auto"

        # Warn users if they're trying to use tool_choice other than "auto"
        if tool_choice != "auto":
            SnowflakeErrorHandler.log_info(
                "tool_choice compatibility",
                f"tool_choice='{tool_choice}' is processed for LangChain compatibility but "
                f"is not supported by Snowflake Cortex REST API. The model will auto-select "
                f"tools regardless of this setting. Use 'auto' to suppress this warning.",
                logger,
            )

        # Create a new instance with bound tools, tool_choice, AND REST API enabled
        return self.model_copy(
            update={
                "_bound_tools": snowflake_tools,
                "_original_tools": list(tools),
                "_tool_choice": tool_choice,
                "_use_rest_api": True,
                "max_tokens": self.max_tokens,
            }
        )

    def _build_enhanced_system_prompt(self, tools: list[dict[str, Any]] | None = None) -> str:
        """Build an enhanced system prompt for better tool calling and reasoning."""
        if not tools or not self._has_tools():
            return """You are a helpful, knowledgeable assistant. Provide accurate, thoughtful responses 
                        based on the information available to you. If you're uncertain about something, 
                        acknowledge that uncertainty rather than guessing."""

        # Extract tool descriptions for the prompt
        tool_descriptions = []
        for tool in tools:
            tool_spec = tool.get("tool_spec", {})
            name = tool_spec.get("name", "unknown")
            description = tool_spec.get("description", "No description available")
            tool_descriptions.append(f"- **{name}**: {description}")

        tool_list = "\n".join(tool_descriptions)

        return f"""You are an intelligent assistant with access to tools. 
                    Your role is to help users by understanding their needs and 
                    using available tools when appropriate.

## Available Tools:
{tool_list}

## Guidelines for Tool Usage:

1. **Analyze First**: Carefully understand what the user is asking for and determine 
if any tools can help provide a better answer.

2. **Choose Wisely**: Select the most appropriate tool(s) based on:
   - The user's specific question or request
   - The capabilities and purpose of each tool
   - The likelihood that the tool will provide relevant, accurate information

3. **Use Tools Effectively**: When calling tools:
   - Provide clear, specific parameters that will yield useful results
   - Call tools in a logical sequence if multiple tools are needed
   - Use tool results to inform your final response

4. **Reasoning Process**: 
   - If multiple tools could be relevant, choose the most specific and appropriate one
   - If a tool doesn't provide the expected information, acknowledge this and either 
   try an alternative approach or explain the limitation
   - Combine tool results with your knowledge to provide comprehensive, accurate answers

5. **Response Quality**:
   - Always provide clear, well-structured responses
   - Cite tool results when they inform your answer
   - If tools are not needed or relevant, provide direct answers based on your knowledge
   - Be transparent about when you're using tools vs. providing direct knowledge

6. **Error Handling**: If a tool fails or returns unexpected results, explain this to the user 
and provide alternative information when possible.

Remember: Use tools to enhance your responses, not replace thoughtful analysis. 
The goal is to provide the most helpful, accurate, and relevant information to the user."""

    def _group_consecutive_tool_messages(self, messages: list[BaseMessage]) -> list[BaseMessage | list[ToolMessage]]:
        """Group consecutive ToolMessage objects together."""
        if not getattr(self, "group_tool_messages", True):
            return messages

        result = []
        current_tool_group = []

        for message in messages:
            if isinstance(message, ToolMessage):
                current_tool_group.append(message)
            else:
                if current_tool_group:
                    result.append(current_tool_group)
                    current_tool_group = []
                result.append(message)

        if current_tool_group:
            result.append(current_tool_group)

        return result

    def _get_tool_name_from_id(self, tool_call_id: str, messages: list[BaseMessage]) -> str:
        """PATCH: Extract tool name from previous AIMessage's tool_calls by matching tool_call_id."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id == tool_call_id:
                        return tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
        return "unknown"

    def _process_tool_message_group(self, tool_messages: list[ToolMessage], all_messages: list[BaseMessage] = None) -> dict[str, Any]:
        """Process a group of ToolMessage objects into a single user message.

        PATCHED:
        - Use actual tool name from tool_call_id mapping (fixes "unknown" name issue)
        - Truncate extremely large individual tool outputs
        """
        tool_results = []

        for tool_msg in tool_messages:
            # PATCH: Get actual tool name from AIMessage tool_calls if not set
            tool_name = getattr(tool_msg, "name", None)
            if not tool_name or tool_name == "unknown":
                tool_name = self._get_tool_name_from_id(tool_msg.tool_call_id, all_messages or [])

            # Truncate extremely large tool outputs (prevents malformed JSON, not context limits)
            content = _truncate_tool_output(str(tool_msg.content))

            tool_results.append(
                {
                    "type": "tool_results",
                    "tool_results": {
                        "tool_use_id": tool_msg.tool_call_id,
                        "name": tool_name,
                        "content": [{"type": "text", "text": content}],
                    },
                }
            )

        return {"role": "user", "content_list": tool_results}

    def _process_single_message(self, message: BaseMessage) -> dict[str, Any]:
        """Process a single message (non-ToolMessage) into API format.

        CRITICAL PATCH: Fix assistant message format per Snowflake REST API spec.

        Snowflake expects (from official docs):
        {
            "role": "assistant",
            "content": "I'll help you check the weather.",  // TOP-LEVEL text
            "content_list": [
                {
                    "type": "tool_use",
                    "tool_use": { "tool_use_id": "...", "name": "...", "input": {...} }
                }
            ]
        }

        Original langchain-snowflake sends:
        {
            "role": "assistant",
            "content_list": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
        }

        This format mismatch causes 400 Bad Request on multi-turn conversations!
        """
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}

        elif isinstance(message, AIMessage):
            if hasattr(message, "tool_calls") and message.tool_calls:
                # PATCH: Build message per Snowflake REST API spec exactly
                result: dict[str, Any] = {"role": "assistant"}

                # PATCH: Add text content as TOP-LEVEL 'content' field (per Snowflake API spec)
                if message.content:
                    result["content"] = str(message.content)

                # PATCH: Build content_list with ONLY tool_use blocks (not text!)
                content_list: list[dict[str, Any]] = []

                for tool_call in message.tool_calls:
                    # Handle both dict and ToolCall object formats
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id", "") or ""
                        tool_call_name = tool_call["name"]
                        tool_call_args = tool_call["args"]
                    else:
                        tool_call_id = getattr(tool_call, "id", None) or ""
                        tool_call_name = getattr(tool_call, "name", None)
                        tool_call_args = getattr(tool_call, "args", {})

                    # Ensure we have a valid ID
                    if not tool_call_id:
                        tool_call_id = f"toolu_{id(tool_call)}"

                    # Process input
                    if isinstance(tool_call_args, dict) or isinstance(tool_call_args, (str, list)):
                        tool_input = tool_call_args
                    else:
                        tool_input = tool_call_args if isinstance(tool_call_args, (dict, list)) else str(tool_call_args)

                    content_list.append(
                        {
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": str(tool_call_id),
                                "name": tool_call_name,
                                "input": tool_input,
                            },
                        }
                    )

                result["content_list"] = content_list
                return result
            else:
                return {"role": "assistant", "content": message.content or ""}

        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}

        else:
            content = str(message.content) if hasattr(message, "content") else str(message)
            return {"role": "user", "content": content}

    def _build_rest_api_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Build REST API payload.

        PATCHED: Pass all_messages to tool message processor for proper name lookup.
        NO conversation trimming - Claude has 200k token context window!
        """
        api_messages: list[dict[str, Any]] = []

        # Get bound tools for enhanced system prompt
        bound_tools = None
        if hasattr(self, "_bound_tools") and self._bound_tools:
            bound_tools = [{"tool_spec": tool} for tool in self._bound_tools]

        # Add enhanced system message when tools are available
        if self._has_tools():
            enhanced_prompt = self._build_enhanced_system_prompt(bound_tools)
            api_messages.append({"role": "system", "content": enhanced_prompt})

        # Group messages properly
        grouped_messages = self._group_consecutive_tool_messages(messages)

        # Process each group or individual message
        for item in grouped_messages:
            if isinstance(item, list):  # Tool message group
                # PATCH: Pass all messages for tool name lookup
                user_message = self._process_tool_message_group(item, messages)
                api_messages.append(user_message)
            else:  # Single message
                api_message = self._process_single_message(item)
                api_messages.append(api_message)

        # Build payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "disable_parallel_tool_use": self.disable_parallel_tool_use,
        }

        # Add tools if bound
        if self._has_tools() and hasattr(self, "_bound_tools"):
            payload["tools"] = self._bound_tools

        return payload

    def _parse_rest_api_response(self, response_or_data, original_messages: list[BaseMessage]) -> ChatResult:
        """Parse REST API response and handle tool calls."""

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage_data: dict[str, Any] = {}
        tool_input_buffers: dict[str, str] = {}

        # Handle different input types
        if isinstance(response_or_data, dict):
            response_data = response_or_data
        else:
            try:
                response_data = response_or_data.json()
            except Exception:
                response_data = {
                    "choices": [{"message": {"content": str(response_or_data)}}],
                    "usage": {},
                }

        # PATCH: Check if this is a non-streaming response (has "message" not "delta")
        # Non-streaming responses use a different format that _parse_direct_response handles
        choices = response_data.get("choices", [])
        if choices and "message" in choices[0] and "delta" not in choices[0]:
            return self._parse_direct_response(response_data)

        # Process the response data (streaming format with delta)
        for choice in choices:
            delta = choice.get("delta", {})
            message = choice.get("message", {})

            # Get content type from delta
            delta_type = None
            if "content" in delta:
                delta_type = "text"
            elif "content_list" in delta:
                content_list = delta.get("content_list", [])
                for item in content_list:
                    if "tool_use_id" in item or item.get("type") == "tool_use" or "input" in item:
                        delta_type = "tool_use"
                        break

            # Process based on delta_type
            if delta_type == "text":
                if "content" in delta:
                    content_parts.append(delta["content"])

            elif delta_type == "tool_use":
                if "tool_use_id" in delta and "name" in delta:
                    tool_call_obj = ToolCall(
                        name=delta["name"],
                        args={},
                        id=delta["tool_use_id"],
                    )
                    tool_calls.append(tool_call_obj)
                    tool_input_buffers[delta["tool_use_id"]] = ""

                if "input" in delta and tool_calls:
                    recent_tool_id = tool_calls[-1]["id"]
                    input_chunk = delta["input"]
                    if recent_tool_id in tool_input_buffers:
                        tool_input_buffers[recent_tool_id] += input_chunk
                        try:
                            accumulated = tool_input_buffers[recent_tool_id]
                            parsed_args = json.loads(accumulated)
                            tool_calls[-1]["args"] = parsed_args
                        except json.JSONDecodeError:
                            pass

            # Handle content_list
            content_list = delta.get("content_list", [])
            for item in content_list:
                if item.get("type") == "text" and "text" in item:
                    content_parts.append(item["text"])
                elif item.get("type") == "tool_use" and "tool_use" in item:
                    tool_use = item["tool_use"]
                    tool_use_id = tool_use.get("tool_use_id")
                    tool_name = tool_use.get("name")

                    if tool_use_id and tool_name:
                        existing_call = None
                        for tc in tool_calls:
                            if tc["id"] == tool_use_id:
                                existing_call = tc
                                break

                        if not existing_call:
                            tool_call_obj = ToolCall(
                                name=tool_name,
                                args=tool_use.get("input") or {},
                                id=tool_use_id,
                            )
                            tool_calls.append(tool_call_obj)

                elif "tool_use_id" in item and "name" in item:
                    existing_call = None
                    for tc in tool_calls:
                        if tc["id"] == item["tool_use_id"]:
                            existing_call = tc
                            break

                    if not existing_call:
                        from langchain_core.messages import tool_call

                        tool_call_obj = tool_call(name=item["name"], args={}, id=item["tool_use_id"])
                        tool_calls.append(tool_call_obj)
                        tool_input_buffers[item["tool_use_id"]] = ""

        if "usage" in response_data:
            usage_data.update(response_data["usage"])

        full_content = "".join(content_parts)

        # PATCH: Extract finish_reason from streaming response
        finish_reason = None
        for choice in choices:
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]
                break

        # PATCH: Build response_metadata with Cortex Inference observability fields
        response_metadata = {
            "model_name": response_data.get("model", self.model),
        }
        
        # PATCH: Add request_id for OTel observability (maps to gen_ai.response.id)
        if "id" in response_data:
            response_metadata["snowflake_request_id"] = response_data["id"]
        
        # PATCH: Add finish_reason for OTel observability (maps to gen_ai.response.finish_reasons)
        if finish_reason:
            response_metadata["finish_reason"] = finish_reason
        elif tool_calls:
            response_metadata["finish_reason"] = "tool_calls"
        else:
            response_metadata["finish_reason"] = "stop"
        
        # PATCH: Add guard_tokens for OTel observability (Snowflake-specific)
        if usage_data.get("guard_tokens") is not None:
            response_metadata["snowflake_guard_tokens"] = usage_data["guard_tokens"]

        ai_message = AIMessage(
            content=full_content,
            tool_calls=tool_calls if tool_calls else [],
            additional_kwargs={"usage": usage_data},
            response_metadata=response_metadata,
        )

        # Create metadata dict inline (avoiding dependency on create_metadata)
        metadata = {
            "model": response_data.get("model", self.model),
            "usage": usage_data,
        }

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output=metadata,
        )

    async def _parse_rest_api_response_async(self, response_or_data, original_messages: list[BaseMessage]) -> ChatResult:
        """Async wrapper for parsing REST API response.

        The actual parsing is CPU-bound and doesn't require async I/O,
        so this simply delegates to the sync version.
        """
        return self._parse_rest_api_response(response_or_data, original_messages)

    def _parse_direct_response(self, data: dict[str, Any]) -> ChatResult:
        """Parse a direct JSON response (non-streaming format) with content_list support.
        
        PATCHED: Extracts Cortex Inference metadata for OTel observability:
        - request_id from response 'id' field
        - finish_reason from choices[0].finish_reason
        - guard_tokens from usage.guard_tokens
        """
        full_content = ""
        tool_calls = []
        usage_data = {}
        finish_reason = None

        if "usage" in data:
            usage_data = data["usage"]

        choices = data.get("choices", [])
        for choice in choices:
            # PATCH: Extract finish_reason for OTel observability
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]
            
            message = choice.get("message", choice.get("delta", {}))
            if message:
                if message.get("role") == "assistant" or "content" in message or "content_list" in message:
                    if "content_list" in message:
                        for item in message["content_list"]:
                            if item.get("type") == "text" and "text" in item:
                                full_content += item["text"]
                            elif item.get("type") == "tool_use" and "tool_use" in item:
                                tool_use = item["tool_use"]
                                tool_calls.append(
                                    {
                                        "id": tool_use.get("tool_use_id", f"call_{len(tool_calls)}"),
                                        "name": tool_use.get("name"),
                                        "args": tool_use.get("input") or {},
                                    }
                                )
                    else:
                        full_content += message.get("content", "")

                    if "tool_calls" in message:
                        tool_calls.extend(message["tool_calls"])

        # PATCH: Build response_metadata with Cortex Inference observability fields
        # These fields are extracted by OTel instrumentation in callback_handler.py
        response_metadata = {
            "model_name": data.get("model", getattr(self, "model", "unknown")),
        }
        
        # PATCH: Add request_id for OTel observability (maps to gen_ai.response.id)
        if "id" in data:
            response_metadata["snowflake_request_id"] = data["id"]
        
        # PATCH: Add finish_reason for OTel observability (maps to gen_ai.response.finish_reasons)
        if finish_reason:
            response_metadata["finish_reason"] = finish_reason
        elif tool_calls:
            response_metadata["finish_reason"] = "tool_calls"
        else:
            response_metadata["finish_reason"] = "stop"
        
        # PATCH: Add guard_tokens for OTel observability (Snowflake-specific)
        if usage_data.get("guard_tokens") is not None:
            response_metadata["snowflake_guard_tokens"] = usage_data["guard_tokens"]

        ai_message = AIMessage(
            content=full_content,
            tool_calls=tool_calls,
            additional_kwargs={"usage": usage_data},
            response_metadata=response_metadata,
        )

        # Create metadata dict inline (avoiding dependency on create_metadata)
        metadata = {
            "model": data.get("model", getattr(self, "model", "unknown")),
            "usage": usage_data,
        }

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output=metadata,
        )

    def _should_use_rest_api(self) -> bool:
        """Determine whether to use REST API or SQL function based on tool usage.

        The ChatSnowflake class supports dual API modes:
        - REST API: Required for tool calling, streaming, and advanced features
        - SQL Function: Simpler approach for basic chat completions

        Returns:
            True if REST API should be used, False for SQL function
        """
        # Use REST API if explicitly enabled via _use_rest_api attribute
        if getattr(self, "_use_rest_api", False):
            return True

        # Use REST API if tools are bound (tools require REST API)
        if self._has_tools():
            return True

        # Default to SQL function for basic chat
        return False

    def _has_tools(self) -> bool:
        """Check if the model has bound tools.

        Returns:
            True if tools are bound to the model, False otherwise
        """
        return hasattr(self, "_bound_tools") and bool(self._bound_tools)
