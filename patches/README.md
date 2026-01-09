# LangChain-Snowflake Patches

These patches enable **LangChain v1 / LangGraph v1** compatibility and fix critical issues in `langchain-snowflake` for production use.

## LangChain v1 Compatibility

**Why patches are needed:**
- PyPI `langchain-snowflake==0.2.2` pins `langgraph>=0.6.4,<0.7.0` which blocks LangGraph v1
- Local editable install removes this constraint
- MCP integration needs updated imports for `langchain>=1.0.0`

**Solution:**
- Use editable install from `original_langchain_snwoflake_repo/`
- Apply patches for LangChain v1 API changes

---

## 🐛 Discovered Bugs & Fixes

### Bug #1: Session Validation Causes "Password is empty" Error

**File:** `langchain_snowflake/_connection/base.py`

**Symptom:**
```
Error in create Snowflake session: 251006: 251006: Password is empty
```

**Root Cause:**
When `SnowflakeConnectionMixin._get_session()` is called, it passes the session to `SnowflakeSessionManager.get_or_create_session()` which runs `SELECT 1` to validate the session. This validation can fail due to:
1. Async context issues (session created in different async context)
2. Warehouse not set on the session
3. Session timeout/expiration edge cases

When validation fails, the session manager tries to create a NEW session using `password=self.password`, which is `None` when using key-pair authentication.

**Fix:** Trust the provided session directly without running validation queries. If a session is explicitly passed, use it as-is.

**Patch File:** `langchain_snowflake_connection_base_patched.py`

---

### Bug #2: JWT Authentication Requires Public Key Fingerprint

**Files:** `langchain_snowflake/retrievers.py`, `langchain_snowflake/tools/analyst.py`

**Symptom:**
```
401 JWT token is invalid
```

**Root Cause:**
When key-pair credentials (`account`, `user`, `private_key_path`) are passed to `RestApiRequestBuilder.cortex_search_request()` or `cortex_analyst_request()`, the library generates a JWT for authentication. However, Snowflake's JWT format requires the **public key fingerprint** in the `iss` claim:

- **Current (broken):** `"iss": f"{account}.{user}".upper()`
- **Required:** `"iss": f"{account}.{user}.SHA256:{PUBLIC_KEY_FP}".upper()`

The `langchain-snowflake` library doesn't extract or compute the public key fingerprint from the private key file.

**Fix:** Don't pass key-pair credentials to Cortex Search/Analyst REST API requests. Instead, rely on **session token authentication** (same as `ChatSnowflake` does for `/cortex/inference:complete`).

**Patch Files:** 
- `langchain_snowflake_retrievers_patched.py` (updated)
- `langchain_snowflake_analyst_patched.py` (new - inline in original)

---

### Bug #3: Streaming Tool Deltas Dropped

**File:** `langchain_snowflake/chat_models/streaming.py`

**Symptom:** Tool calls are silently dropped when streaming is enabled.

**Root Cause:** The `_stream_via_rest_api` method only extracts the `content` field from stream chunks, ignoring `tool_use` deltas entirely.

**Fix:** Use `ToolCallChunk` to properly yield `tool_use` deltas that LangChain can accumulate.

**Patch File:** `langchain_snowflake_streaming_patched.py`

---

### Bug #4: No Way to Disable Streaming

**File:** `langchain_snowflake/chat_models/base.py`

**Symptom:** Can't reliably use tools with streaming because tool deltas are dropped.

**Fix:** Added `disable_streaming: bool = False` parameter to `ChatSnowflake` to force non-streaming behavior.

**Patch File:** `langchain_snowflake_base_patched.py`

---

## Patches

### Session & Authentication Patches

| Patch File | Bug Fixed | Description |
|------------|-----------|-------------|
| `langchain_snowflake_connection_base_patched.py` | Bug #1 | Trust provided session without validation |
| `langchain_snowflake_retrievers_patched.py` | Bug #2 | Use session token auth for Cortex Search |
| (inline in original) | Bug #2 | Use session token auth for Cortex Analyst |

### Streaming & Tool Calling Patches

| Patch File | Bug Fixed | Description |
|------------|-----------|-------------|
| `langchain_snowflake_streaming_patched.py` | Bug #3 | Properly yield tool_use deltas |
| `langchain_snowflake_base_patched.py` | Bug #4 | Add `disable_streaming` parameter |

### OTel Observability Patches

| Patch File | Feature |
|------------|---------|
| `langchain_snowflake_tools_patched.py` | Extract `request_id`, `finish_reason` from Cortex API |
| `langchain_snowflake_utils_patched.py` | Add observability fields to response metadata |
| `langchain_snowflake_retrievers_patched.py` | Propagate `_snowflake_request_id` to Document metadata |

### LangChain v1 Compatibility Patches

| Patch File | Fix |
|------------|-----|
| `langchain_snowflake_mcp_integration_patched.py` | Tool import fix for `langchain>=1.0.0` |

---

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
| Session with key-pair auth | ❌ "Password is empty" | ✅ Uses provided session |
| Cortex Search REST API | ❌ 401 JWT invalid | ✅ Session token auth works |
| Cortex Analyst REST API | ❌ 401 JWT invalid | ✅ Session token auth works |
| Text streaming (REST API) | ✅ Works | ✅ Works |
| Tool call start | ❌ Dropped | ✅ `ToolCallChunk(name, id)` yielded |
| Tool call args | ❌ Dropped | ✅ `ToolCallChunk(args)` streamed |
| Final `tool_calls` | ❌ Empty | ✅ Populated when JSON complete |

---

## Testing

### Local Testing
```bash
# 1. Apply patches
./patches/apply_patches.sh

# 2. Run LangGraph dev server
.venv/bin/langgraph dev

# 3. Test agent
curl -X POST "http://127.0.0.1:2024/threads/{id}/runs/wait" \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "react_healthcare", "input": {...}}'
```

### SPCS Testing
```bash
# 1. Rebuild Docker image (patches are applied in Dockerfile)
./scripts/build_and_deploy.sh

# 2. Test via Streamlit app
```

---

## Files

### Session & Auth
- `langchain_snowflake_connection_base_patched.py` - Patched `_connection/base.py`
- `langchain_snowflake_retrievers_patched.py` - Patched `retrievers.py`

### Streaming & Tools
- `langchain_snowflake_streaming_patched.py` - Patched `chat_models/streaming.py`
- `langchain_snowflake_base_patched.py` - Patched `chat_models/base.py`
- `langchain_snowflake_tools_patched.py` - Patched `chat_models/tools.py`
- `langchain_snowflake_utils_patched.py` - Patched `chat_models/utils.py`

### LangChain v1
- `langchain_snowflake_mcp_integration_patched.py` - Patched `mcp_integration.py`

### Scripts
- `apply_patches.sh` - Shell script to apply patches
- `revert_patches.sh` - Shell script to revert patches
- `README.md` - This file
