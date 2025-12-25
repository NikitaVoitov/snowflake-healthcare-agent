## Summary

The README claims **"Streaming - Real-time response streaming for better UX"**, but token-level streaming **silently fails** when tools are bound. The streaming code only extracts text content and completely ignores tool call deltas.

**This is fixable!** LangChain has built-in `ToolCallChunk` support for streaming partial tool calls, but `langchain-snowflake` doesn't use it.

## Environment

- `langchain-snowflake`: latest from main branch
- `langchain-core`: 0.3.x
- Python: 3.11+
- Snowflake Cortex: claude-3-5-sonnet

---

## Root Cause: `_stream_via_rest_api` extracts text  only

**File**: `streaming.py`, lines 169-170

```python
# The streaming code ONLY looks for "content" field
chunk_content = chunk_data.get("content", "")  # ← Tool deltas IGNORED!
```

When Cortex Complete returns tool calls, the response contains `tool_use` deltas (not `content`), which are **silently dropped**.

### What Cortex Complete Returns

**Text chunks** (extracted correctly):
```json
{"choices": [{"delta": {"type": "text", "content": "Let me check the weather"}}]}
```

**Tool call chunks** (silently ignored):
```json
{"choices": [{"delta": {"type": "tool_use", "tool_use_id": "abc123", "name": "get_weather"}}]}
{"choices": [{"delta": {"type": "tool_use", "input": "{\"city\": \"Paris\"}"}}]}
```

The streaming code does `chunk_data.get("content", "")` which returns empty string for tool deltas → **yields nothing** → LangChain throws "No generations found in stream".

---

## Behavior Matrix

| Scenario | Streaming Works? | Token-by-Token? | Notes |
|----------|------------------|-----------------|-------|
| Text only (REST API) | ✅ Yes | ✅ Yes | Works correctly |
| Text only (SQL function) | ⚠️ Fake | ⚠️ Simulated | Full response chunked into ~20 parts |
| **Tools bound → text response** | ✅ Yes | ✅ Yes | If LLM returns text only |
| **Tools bound → tool call** | ❌ No | ❌ No | Tool deltas silently dropped |

---

## Steps to Reproduce

```python
from langchain_core.tools import tool
from langchain_snowflake import ChatSnowflake

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F"

llm = ChatSnowflake(model="claude-3-5-sonnet", session=session)
llm_with_tools = llm.bind_tools([get_weather])

messages = [("human", "What's the weather in Paris?")]

# This fails when LLM decides to call the tool
chunks = []
for chunk in llm_with_tools.stream(messages):
    chunks.append(chunk)
    print(chunk.content, end="")

# Result: Either empty output or "No generations found in stream" error
```

---

## Code Analysis

### Issue 1: Streaming Only Extracts `content`

**File**: `streaming.py`, lines 162-184

```python
def _stream_via_rest_api(self, messages, ...):
    payload["stream"] = True  # Enable native streaming
    
    for chunk_json in RestApiClient.make_sync_streaming_request(...):
        chunk_data = json.loads(chunk_json)
        
        # ❌ BUG: Only extracts "content", ignores tool_use deltas
        chunk_content = chunk_data.get("content", "")
        
        if chunk_content:  # Empty for tool calls → nothing yielded
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))
```

### Issue 2: `_parse_streaming_chunk` Not Used in Streaming Path

**File**: `tools.py` has `_parse_streaming_chunk()` that DOES handle tool deltas correctly:

```python
def _parse_streaming_chunk(self, data, content_parts, tool_calls, ...):
    delta_type = delta.get("type")
    
    if delta_type == "text":
        content_parts.append(delta["content"])
    elif delta_type == "tool_use":
        # Handles tool calls properly!
        tool_calls.append(ToolCall(...))
```

But this method is only used by `_parse_rest_api_response()` for **non-streaming** response parsing, NOT by `_stream_via_rest_api()`.

### Issue 3: Simulated Streaming is Fake

**File**: `streaming.py`, lines 51-74

```python
# When SQL function is used (no REST API):
result = self._generate(messages, ...)  # Gets FULL response first
content = result.generations[0].message.content
chunk_size = max(1, len(content) // 20)  # Split into ~20 chunks

for i in range(0, len(content), chunk_size):
    yield ChatGenerationChunk(...)  # "Streaming" pre-generated content
```

This is cosmetic chunking, not real streaming.

---

## The Fix: Use `ToolCallChunk` (LangChain Built-in!)

LangChain has `ToolCallChunk` specifically designed for streaming partial tool calls:

```python
from langchain_core.messages import ToolCallChunk

# Partial args can be streamed as strings!
left = [ToolCallChunk(name="foo", args='{"a":', index=0)]
right = [ToolCallChunk(name=None, args='1}', index=0)]

# LangChain auto-combines chunks!
(AIMessageChunk(tool_call_chunks=left) + AIMessageChunk(tool_call_chunks=right))
# → tool_call_chunks=[ToolCallChunk(name="foo", args='{"a":1}', index=0)]
```

---

## Requested Changes

### 1. Fix `_stream_via_rest_api` to Use `ToolCallChunk`

```python
from langchain_core.messages import AIMessageChunk, ToolCallChunk

def _stream_via_rest_api(self, messages, ...):
    tool_call_index = 0
    
    for chunk_json in RestApiClient.make_sync_streaming_request(...):
        chunk_data = json.loads(chunk_json)
        choices = chunk_data.get("choices", [])
        
        for choice in choices:
            delta = choice.get("delta", {})
            delta_type = delta.get("type")
            
            # ✅ Handle TEXT chunks (works today)
            if delta_type == "text":
                content = delta.get("content", "")
                if content:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=content))
            
            # ✅ NEW: Handle TOOL_USE chunks with ToolCallChunk!
            elif delta_type == "tool_use":
                tool_call_chunk = ToolCallChunk(
                    name=delta.get("name"),        # "get_weather" or None
                    args=delta.get("input", ""),   # '{"city":' (partial JSON string!)
                    id=delta.get("tool_use_id"),   # "toolu_abc123" or None
                    index=tool_call_index,
                )
                
                if delta.get("tool_use_id"):
                    tool_call_index += 1
                
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[tool_call_chunk]  # ← THE KEY!
                    )
                )
```

**Why this works:** LangChain's `AIMessageChunk.__add__` automatically combines `tool_call_chunks` by index, and parses complete JSON into `tool_calls` when ready.


**Current** (`streaming.py` lines 51-74): Generates full response, then splits into ~20 chunks (fake streaming)

```python
# ❌ FAKE: Gets full response first, then chunks it
result = self._generate(messages, ...)  # Blocks until complete!
content = result.generations[0].message.content
chunk_size = max(1, len(content) // 20)

for i in range(0, len(content), chunk_size):
    yield ChatGenerationChunk(...)  # Cosmetic chunking
```

**Proposed**: Use REST API for all streaming (SQL function doesn't support native streaming)

```python
def _stream(self, messages, ...):
    # Always use REST API for streaming - SQL function can't stream
    if not self._should_use_rest_api():
        logger.info("Switching to REST API for native streaming support")
    
    # Use REST API streaming path
    for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
        yield chunk
```

Or document clearly that SQL function path doesn't support real streaming.

### 3. Update README.md

**Current** (misleading):
```markdown
- **Streaming** - Real-time response streaming for better UX
```### 2. Fix Fake Streaming for SQL Function Path


**Proposed** (accurate):
```markdown
- **Streaming** - Real-time token streaming via REST API. SQL function uses simulated streaming.
```

### 4. Add `disable_streaming` Parameter

Document and add explicit parameter to `ChatSnowflake`:

```python
class ChatSnowflake:
    disable_streaming: bool = Field(
        default=False,
        description="Disable streaming for tool calling compatibility"
    )
```

### 5. Add Warning When Streaming + Tools (Optional, fix above is better)

```python
def _stream(self, messages, ...):
    if self._has_tools():
        logger.warning(
            "Streaming with tools bound may produce incomplete results. "
            "Consider using invoke() or setting disable_streaming=True."
        )
```

---

## Expected Result After Fix

| Scenario | Current | After Fix |
|----------|---------|-----------|
| Text streaming (REST API) | ✅ Works | ✅ Works |
| Text streaming (SQL function) | ⚠️ Fake (~20 chunks) | ✅ Native via REST API |
| Tool call start | ❌ Dropped | ✅ `ToolCallChunk(name, id)` yielded |
| Tool call args | ❌ Dropped | ✅ `ToolCallChunk(args)` streamed |
| LangChain accumulation | ❌ Broken | ✅ Auto-combines via `__add__` |
| Final `tool_calls` | ❌ Empty | ✅ Populated when JSON complete |

---