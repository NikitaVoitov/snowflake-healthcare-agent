"""
PATCHED Streaming functionality for Snowflake chat models.

FIXES:
1. ToolCallChunk support - yields proper AIMessageChunk with tool_call_chunks for tool_use deltas
2. Fake streaming fix - always uses REST API for native streaming (SQL function can't stream)
3. Better error handling and logging for streaming

To apply this patch:
    cp patches/langchain_snowflake_streaming_patched.py \
       .venv/lib/python3.13/site-packages/langchain_snowflake/chat_models/streaming.py
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGenerationChunk

from .._connection.rest_client import RestApiClient, RestApiRequestBuilder
from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeStreaming:
    """Mixin class for Snowflake streaming functionality.

    PATCHED: Now properly handles tool_use deltas using ToolCallChunk.
    """

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions from Snowflake Cortex.

        Args:
            messages: List of messages to send to the model
            stop: List of stop sequences (not supported by Cortex)
            run_manager: Callback manager for the run
            **kwargs: Additional keyword arguments

        Yields:
            ChatGenerationChunk: Streaming chunks of the response

        PATCHED: Always uses REST API for native streaming. SQL function path
        is now deprecated for streaming as it cannot truly stream.
        """
        try:
            # Check if streaming is disabled
            if getattr(self, "disable_streaming", False):
                logger.debug("Streaming disabled, falling back to generate")
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                if result.generations:
                    content = result.generations[0].message.content
                    tool_calls = getattr(result.generations[0].message, "tool_calls", [])
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=content,
                            tool_calls=tool_calls,
                            usage_metadata=result.generations[0].message.usage_metadata,
                            response_metadata=result.generations[0].message.response_metadata,
                        )
                    )
                    yield chunk
                return

            # PATCHED: Always prefer REST API for true streaming
            if self._should_use_rest_api():
                # Use native streaming via REST API with ToolCallChunk support
                for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # PATCHED: Log warning and use REST API anyway for native streaming
                logger.warning(
                    "SQL function does not support native streaming. "
                    "Switching to REST API for true token-by-token streaming. "
                    "Set disable_streaming=True if you want batch response."
                )
                # Try REST API first, fall back to simulated if it fails
                try:
                    for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
                        yield chunk
                except Exception as rest_error:
                    logger.warning(f"REST API streaming failed: {rest_error}. Falling back to simulated streaming.")
                    # Fallback: Simulate streaming by chunking the complete response
                    result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

                    if result.generations:
                        content = result.generations[0].message.content

                        # Split content into chunks for streaming effect
                        chunk_size = max(1, len(content) // 20)  # Aim for ~20 chunks

                        for i in range(0, len(content), chunk_size):
                            chunk_content = content[i : i + chunk_size]

                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content=chunk_content,
                                    usage_metadata=(result.generations[0].message.usage_metadata if i == 0 else None),
                                    response_metadata=(result.generations[0].message.response_metadata if i == 0 else {}),
                                )
                            )

                            yield chunk

                            if run_manager:
                                run_manager.on_llm_new_token(chunk_content)

        except Exception as e:
            # Use centralized error handling to create consistent error response
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response by delegating to sync _stream method.

        This eliminates code duplication by using the sync implementation
        with asyncio.to_thread() for non-blocking execution.
        """
        # Determine streaming method based on tool requirements
        try:
            # Check if streaming is disabled
            if getattr(self, "disable_streaming", False):
                logger.debug("Streaming disabled, falling back to async generate")
                result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                if result.generations:
                    content = result.generations[0].message.content
                    tool_calls = getattr(result.generations[0].message, "tool_calls", [])
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=content,
                            tool_calls=tool_calls,
                            usage_metadata=result.generations[0].message.usage_metadata,
                            response_metadata=result.generations[0].message.response_metadata,
                        )
                    )
                    yield chunk
                return

            if self._should_use_rest_api():
                # Use native async REST API streaming with aiohttp and ToolCallChunk
                async for chunk in self._astream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # PATCHED: Try async REST API first
                logger.warning(
                    "SQL function does not support native streaming. "
                    "Switching to REST API for true token-by-token streaming."
                )
                try:
                    async for chunk in self._astream_via_rest_api(messages, run_manager, **kwargs):
                        yield chunk
                except Exception as rest_error:
                    logger.warning(f"Async REST API streaming failed: {rest_error}. Falling back to sync.")
                    # For SQL-based streaming, we currently delegate to sync method since
                    # Snowflake doesn't support native SQL streaming, only batch results
                    def sync_stream():
                        return list(self._stream(messages, stop, run_manager, **kwargs))

                    chunks = await asyncio.to_thread(sync_stream)
                    for chunk in chunks:
                        yield chunk

        except Exception as e:
            # Use centralized error handling for consistent async streaming errors
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="async stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk

    def _stream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions via REST API with native streaming support.

        PATCHED: Now yields AIMessageChunk with ToolCallChunk for tool_use deltas!
        This enables true token-by-token streaming even when tools are bound.
        """
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters directly
            payload.update(
                {
                    "temperature": getattr(self, "temperature", 0.7),
                    "max_tokens": getattr(self, "max_tokens", 4096),
                    "top_p": getattr(self, "top_p", 1.0),
                }
            )

            # Get session for centralized REST API client
            session = self._get_session()

            # Use centralized REST API client for streaming
            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            # Track tool call index for combining chunks
            tool_call_index = 0
            current_tool_id = None

            # Use centralized streaming
            for chunk_json in RestApiClient.make_sync_streaming_request(request_config, "streaming Cortex Complete"):
                if chunk_json:
                    # Parse JSON chunk and extract content
                    try:
                        chunk_data = json.loads(chunk_json)

                        # Handle Cortex Complete streaming format
                        if isinstance(chunk_data, dict):
                            # Check for choices array (standard format)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                for choice in choices:
                                    delta = choice.get("delta", {})
                                    delta_type = delta.get("type")

                                    # ✅ FIX 1: Handle TEXT chunks (works today)
                                    if delta_type == "text" or (not delta_type and "content" in delta):
                                        chunk_content = delta.get("content", "")
                                        if chunk_content:
                                            chunk = ChatGenerationChunk(
                                                message=AIMessageChunk(content=chunk_content),
                                                generation_info={"stream": True, "delta_type": "text"},
                                            )
                                            if run_manager:
                                                run_manager.on_llm_new_token(chunk_content)
                                            yield chunk

                                    # ✅ FIX 2: Handle TOOL_USE chunks with ToolCallChunk!
                                    elif delta_type == "tool_use":
                                        tool_use_id = delta.get("tool_use_id") or delta.get("id")
                                        tool_name = delta.get("name")
                                        tool_input = delta.get("input", "")

                                        # If new tool call starts, increment index
                                        if tool_use_id and tool_use_id != current_tool_id:
                                            if current_tool_id is not None:
                                                tool_call_index += 1
                                            current_tool_id = tool_use_id

                                        # Create ToolCallChunk - LangChain will combine them!
                                        tool_call_chunk = ToolCallChunk(
                                            name=tool_name,  # "get_weather" or None
                                            args=tool_input if isinstance(tool_input, str) else json.dumps(tool_input),  # '{"city":' (partial JSON!)
                                            id=tool_use_id,  # "toolu_abc123" or None
                                            index=tool_call_index,  # For combining
                                        )

                                        chunk = ChatGenerationChunk(
                                            message=AIMessageChunk(
                                                content="",  # Empty content for tool calls
                                                tool_call_chunks=[tool_call_chunk],
                                            ),
                                            generation_info={"stream": True, "delta_type": "tool_use"},
                                        )

                                        # Notify about tool call progress
                                        if run_manager and tool_name:
                                            run_manager.on_llm_new_token(f"[Tool: {tool_name}]")

                                        yield chunk

                                    # Handle content_block_stop (tool call complete)
                                    elif delta_type == "content_block_stop":
                                        # Emit empty chunk to signal tool call complete
                                        yield ChatGenerationChunk(
                                            message=AIMessageChunk(content=""),
                                            generation_info={"stream": True, "delta_type": "content_block_stop"},
                                        )

                            else:
                                # Fallback: direct content field (legacy format)
                                chunk_content = chunk_data.get("content", "")
                                if chunk_content:
                                    chunk = ChatGenerationChunk(
                                        message=AIMessageChunk(content=chunk_content),
                                        generation_info={"stream": True},
                                    )
                                    if run_manager:
                                        run_manager.on_llm_new_token(chunk_content)
                                    yield chunk
                        else:
                            chunk_content = str(chunk_data)
                            if chunk_content:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=chunk_content),
                                    generation_info={"stream": True},
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(chunk_content)
                                yield chunk

                    except (json.JSONDecodeError, TypeError):
                        # Fallback: treat as plain text
                        chunk_content = chunk_json
                        if chunk_content:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=chunk_content),
                                generation_info={"stream": True},
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(chunk_content)
                            yield chunk

        except Exception as e:
            # Use centralized error handling
            error_content = f"Streaming error: {str(e)}"
            logger.error(f"REST API streaming error: {e}", exc_info=True)
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk

    async def _astream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completions via REST API using aiohttp for true async.

        PATCHED: Now yields AIMessageChunk with ToolCallChunk for tool_use deltas!
        """
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters directly
            payload.update(
                {
                    "temperature": getattr(self, "temperature", 0.7),
                    "max_tokens": getattr(self, "max_tokens", 4096),
                    "top_p": getattr(self, "top_p", 1.0),
                }
            )

            # Get session for centralized REST API client
            session = self._get_session()

            # Use centralized REST API client for streaming
            from .._connection.rest_client import RestApiClient, RestApiRequestBuilder

            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            # Track tool call index for combining chunks
            tool_call_index = 0
            current_tool_id = None

            # Use centralized async streaming
            async for chunk_json in RestApiClient.make_async_streaming_request(
                request_config, "async streaming Cortex Complete"
            ):
                if chunk_json:
                    # Parse JSON chunk and extract content
                    try:
                        chunk_data = json.loads(chunk_json)

                        # Handle Cortex Complete streaming format
                        if isinstance(chunk_data, dict):
                            # Check for choices array (standard format)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                for choice in choices:
                                    delta = choice.get("delta", {})
                                    delta_type = delta.get("type")

                                    # ✅ FIX 1: Handle TEXT chunks
                                    if delta_type == "text" or (not delta_type and "content" in delta):
                                        chunk_content = delta.get("content", "")
                                        if chunk_content:
                                            chunk = ChatGenerationChunk(
                                                message=AIMessageChunk(content=chunk_content),
                                                generation_info={"stream": True, "delta_type": "text"},
                                            )
                                            if run_manager:
                                                await run_manager.on_llm_new_token(chunk_content)
                                            yield chunk

                                    # ✅ FIX 2: Handle TOOL_USE chunks with ToolCallChunk!
                                    elif delta_type == "tool_use":
                                        tool_use_id = delta.get("tool_use_id") or delta.get("id")
                                        tool_name = delta.get("name")
                                        tool_input = delta.get("input", "")

                                        # If new tool call starts, increment index
                                        if tool_use_id and tool_use_id != current_tool_id:
                                            if current_tool_id is not None:
                                                tool_call_index += 1
                                            current_tool_id = tool_use_id

                                        # Create ToolCallChunk
                                        tool_call_chunk = ToolCallChunk(
                                            name=tool_name,
                                            args=tool_input if isinstance(tool_input, str) else json.dumps(tool_input),
                                            id=tool_use_id,
                                            index=tool_call_index,
                                        )

                                        chunk = ChatGenerationChunk(
                                            message=AIMessageChunk(
                                                content="",
                                                tool_call_chunks=[tool_call_chunk],
                                            ),
                                            generation_info={"stream": True, "delta_type": "tool_use"},
                                        )

                                        if run_manager and tool_name:
                                            await run_manager.on_llm_new_token(f"[Tool: {tool_name}]")

                                        yield chunk

                                    elif delta_type == "content_block_stop":
                                        yield ChatGenerationChunk(
                                            message=AIMessageChunk(content=""),
                                            generation_info={"stream": True, "delta_type": "content_block_stop"},
                                        )

                            else:
                                chunk_content = chunk_data.get("content", "")
                                if chunk_content:
                                    chunk = ChatGenerationChunk(
                                        message=AIMessageChunk(content=chunk_content),
                                        generation_info={"stream": True},
                                    )
                                    if run_manager:
                                        await run_manager.on_llm_new_token(chunk_content)
                                    yield chunk
                        else:
                            chunk_content = str(chunk_data)
                            if chunk_content:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=chunk_content),
                                    generation_info={"stream": True},
                                )
                                if run_manager:
                                    await run_manager.on_llm_new_token(chunk_content)
                                yield chunk

                    except (json.JSONDecodeError, TypeError):
                        chunk_content = chunk_json
                        if chunk_content:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=chunk_content),
                                generation_info={"stream": True},
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(chunk_content)
                            yield chunk

        except Exception as e:
            # Use centralized error handling
            error_content = f"Async streaming error: {str(e)}"
            logger.error(f"Async REST API streaming error: {e}", exc_info=True)
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk

