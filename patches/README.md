# LangChain-Snowflake Streaming Patches

These patches fix critical streaming issues in `langchain-snowflake` that prevent token-by-token streaming when tools are bound.

## Problem

The official `langchain-snowflake` has these issues:
1. **Tool deltas dropped**: `_stream_via_rest_api` only extracts `content` field, ignoring `tool_use` deltas
2. **Fake streaming**: SQL function path generates full response, then splits into ~20 chunks
3. **No disable option**: Can't explicitly disable streaming for tool compatibility

## Patches

### 1. `langchain_snowflake_streaming_patched.py`
**Fixes:**
- ✅ Uses `ToolCallChunk` to properly yield tool_use deltas  
- ✅ LangChain auto-accumulates partial tool calls via `AIMessageChunk.__add__`
- ✅ Always uses REST API for native streaming (SQL function can't stream)
- ✅ Respects `disable_streaming` parameter

### 2. `langchain_snowflake_base_patched.py`
**Fixes:**
- ✅ Adds `disable_streaming: bool = False` parameter to ChatSnowflake
- ✅ Documents the parameter with examples

## Installation

```bash
# Apply patches
./patches/apply_patches.sh

# To revert
./patches/revert_patches.sh
```

## Expected Behavior After Patches

| Scenario | Before | After |
|----------|--------|-------|
| Text streaming (REST API) | ✅ Works | ✅ Works |
| Text streaming (SQL function) | ⚠️ Fake (~20 chunks) | ✅ Native via REST API |
| Tool call start | ❌ Dropped | ✅ `ToolCallChunk(name, id)` yielded |
| Tool call args | ❌ Dropped | ✅ `ToolCallChunk(args)` streamed |
| LangChain accumulation | ❌ Broken | ✅ Auto-combines via `__add__` |
| Final `tool_calls` | ❌ Empty | ✅ Populated when JSON complete |

## Testing

### Local Testing
```bash
# 1. Apply patches
./patches/apply_patches.sh

# 2. Update llm_service.py to remove disable_streaming=True
# (or set it to False)

# 3. Run LangGraph dev server
uv run langgraph dev

# 4. Test streaming
curl -X POST http://127.0.0.1:2024/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What is the weather?"}, ...}'
```

### SPCS Testing
```bash
# 1. Apply patches (they're already in the Docker image)
# 2. Rebuild and deploy
./scripts/build_and_deploy.sh

# 3. Test via Streamlit app with streaming enabled
```

## Streamlit UI Expectations

With patches applied, you should see:
```
🤔 Agent is thinking...
Let me check the weather for you.     ← TEXT STREAMS TOKEN BY TOKEN!
🔧 Tool call: get_weather             ← TOOL CALL STARTS
📝 Building arguments...              ← PARTIAL ARGS STREAMING
✅ Tool call complete                 ← ARGS COMPLETE
📊 Tool returned: 72°F sunny          ← TOOL RESULT
```

## Files

- `langchain_snowflake_streaming_patched.py` - Patched streaming.py
- `langchain_snowflake_base_patched.py` - Patched base.py
- `langchain_snowflake_tools_patched.py` - Patched tools.py (message format + OTel observability)
- `langchain_snowflake_utils_patched.py` - Patched utils.py (Cortex Inference metadata)
- `langchain_snowflake_retrievers_patched.py` - Patched retrievers.py (request_id propagation)
- `langchain_snowflake_mcp_integration_patched.py` - Patched mcp_integration.py (Tool import fix)
- `apply_patches.sh` - Shell script to apply patches
- `revert_patches.sh` - Shell script to revert patches
- `README.md` - This file

### OTel Observability Patches

The following files are patched to enable OpenTelemetry observability for Snowflake Cortex:

| File | Observability Feature |
|------|----------------------|
| `tools.py` | Extracts `request_id`, `finish_reason`, `guard_tokens` from Cortex Inference API response |
| `utils.py` | `SnowflakeMetadataFactory.create_response_metadata` accepts new observability fields |
| `retrievers.py` | Propagates `_snowflake_request_id` to Document metadata |

These patches enable standard OTel GenAI semconv attributes:
- `gen_ai.response.id` - Request ID from Cortex Inference API
- `gen_ai.response.finish_reasons` - Array of stop reasons (standard semconv)
- `snowflake.inference.guard_tokens` - Snowflake-specific: Cortex Guard token usage

