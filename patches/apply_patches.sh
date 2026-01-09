#!/bin/bash
# Apply langchain-snowflake patches for streaming, tool calling, and authentication
#
# This script applies patches to fix:
# 1. ToolCallChunk support for streaming tool calls
# 2. Fake streaming fix (always use REST API for native streaming)
# 3. disable_streaming parameter for explicit control
# 4. Message format fix for multi-turn conversations with tools (400 Bad Request fix)
# 5. Non-streaming response parsing (tool_calls not extracted from non-streaming responses)
# 6. Tool import fix for langchain 1.2.0+ compatibility (langchain_core.tools.Tool)
# 7. Retrievers: Add _snowflake_request_id to Document metadata for OTel tracing
# 8. Utils: Add request_id, finish_reason, guard_tokens to response_metadata for OTel observability
# 9. LangChain v1 / LangGraph v1 compatibility
# 10. Session validation bypass (fixes "Password is empty" error with key-pair auth)
# 11. Session token auth for Cortex Search/Analyst (fixes "JWT token is invalid" error)
#
# Usage: ./patches/apply_patches.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check for editable install location FIRST (LangChain v1 uses local source)
EDITABLE_INSTALL_DIR="$PROJECT_DIR/original_langchain_snwoflake_repo/langchain-snowflake/libs/snowflake/langchain_snowflake"
if [ -d "$EDITABLE_INSTALL_DIR/chat_models" ]; then
    echo "📦 Found editable install at: $EDITABLE_INSTALL_DIR"
    LANGCHAIN_SNOWFLAKE_DIR="$EDITABLE_INSTALL_DIR/chat_models"
    LANGCHAIN_SNOWFLAKE_ROOT="$EDITABLE_INSTALL_DIR"
    INSTALL_TYPE="editable"
else
    # Fall back to site-packages for non-editable installs
    VENV_DIR="$PROJECT_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "❌ Virtual environment not found at $VENV_DIR"
        echo "   Please create a virtual environment first: uv venv"
        exit 1
    fi

    # Find site-packages
    SITE_PACKAGES=$(find "$VENV_DIR/lib" -name "site-packages" -type d 2>/dev/null | head -1)
    if [ -z "$SITE_PACKAGES" ]; then
        echo "❌ Could not find site-packages directory"
        exit 1
    fi

    LANGCHAIN_SNOWFLAKE_DIR="$SITE_PACKAGES/langchain_snowflake/chat_models"
    LANGCHAIN_SNOWFLAKE_ROOT="$SITE_PACKAGES/langchain_snowflake"
    if [ ! -d "$LANGCHAIN_SNOWFLAKE_DIR" ]; then
        echo "❌ langchain-snowflake not installed. Install with:"
        echo "   uv pip install langchain-snowflake"
        exit 1
    fi
    INSTALL_TYPE="site-packages"
    echo "📦 Found langchain-snowflake at: $LANGCHAIN_SNOWFLAKE_DIR"
fi

echo "📦 Install type: $INSTALL_TYPE"

# Backup originals (if not already backed up)
if [ ! -f "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py.original" ]; then
    echo "💾 Backing up original streaming.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py" "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py.original"
fi

if [ ! -f "$LANGCHAIN_SNOWFLAKE_DIR/base.py.original" ]; then
    echo "💾 Backing up original base.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/base.py" "$LANGCHAIN_SNOWFLAKE_DIR/base.py.original"
fi

if [ ! -f "$LANGCHAIN_SNOWFLAKE_DIR/tools.py.original" ]; then
    echo "💾 Backing up original tools.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/tools.py" "$LANGCHAIN_SNOWFLAKE_DIR/tools.py.original"
fi

if [ ! -f "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py.original" ]; then
    echo "💾 Backing up original mcp_integration.py..."
    cp "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py" "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py.original"
fi

if [ ! -f "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py.original" ]; then
    echo "💾 Backing up original retrievers.py..."
    cp "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py" "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py.original"
fi

if [ ! -f "$LANGCHAIN_SNOWFLAKE_DIR/utils.py.original" ]; then
    echo "💾 Backing up original utils.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/utils.py" "$LANGCHAIN_SNOWFLAKE_DIR/utils.py.original"
fi

# Connection base.py (for session validation bypass)
CONNECTION_DIR="$LANGCHAIN_SNOWFLAKE_ROOT/_connection"
if [ -d "$CONNECTION_DIR" ]; then
    if [ ! -f "$CONNECTION_DIR/base.py.original" ]; then
        echo "💾 Backing up original _connection/base.py..."
        cp "$CONNECTION_DIR/base.py" "$CONNECTION_DIR/base.py.original"
    fi
fi

# Apply patches
echo "🔧 Applying streaming.py patch (ToolCallChunk + fake streaming fix)..."
cp "$SCRIPT_DIR/langchain_snowflake_streaming_patched.py" "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py"

echo "🔧 Applying base.py patch (disable_streaming parameter)..."
cp "$SCRIPT_DIR/langchain_snowflake_base_patched.py" "$LANGCHAIN_SNOWFLAKE_DIR/base.py"

echo "🔧 Applying tools.py patch (message format fix + non-streaming response parsing)..."
cp "$SCRIPT_DIR/langchain_snowflake_tools_patched.py" "$LANGCHAIN_SNOWFLAKE_DIR/tools.py"

echo "🔧 Applying mcp_integration.py patch (langchain 1.2.0+ Tool import fix)..."
cp "$SCRIPT_DIR/langchain_snowflake_mcp_integration_patched.py" "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py"

echo "🔧 Applying retrievers.py patch (_snowflake_request_id propagation)..."
cp "$SCRIPT_DIR/langchain_snowflake_retrievers_patched.py" "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py"

echo "🔧 Applying utils.py patch (Cortex Inference metadata for OTel observability)..."
cp "$SCRIPT_DIR/langchain_snowflake_utils_patched.py" "$LANGCHAIN_SNOWFLAKE_DIR/utils.py"

# Apply _connection/base.py patch if the directory exists
if [ -d "$CONNECTION_DIR" ]; then
    echo "🔧 Applying _connection/base.py patch (session validation bypass)..."
    cp "$SCRIPT_DIR/langchain_snowflake_connection_base_patched.py" "$CONNECTION_DIR/base.py"
fi

echo ""
echo "✅ Patches applied successfully!"
echo ""
echo "Changes:"
echo "  1. streaming.py: Added ToolCallChunk support for tool call streaming"
echo "  2. streaming.py: Fixed fake streaming (now uses REST API for native SSE)"
echo "  3. base.py: Added disable_streaming parameter"
echo "  4. tools.py: Fixed assistant message format per Snowflake REST API spec"
echo "              (top-level 'content' + 'content_list' for tool_use only)"
echo "              This fixes 400 Bad Request on multi-turn conversations!"
echo "  5. tools.py: Fixed non-streaming response parsing (delegates to _parse_direct_response)"
echo "              This fixes tool_calls not being extracted when stream=False!"
echo "  6. mcp_integration.py: Fixed Tool import for langchain 1.2.0+ compatibility"
echo "              (from langchain_core.tools import Tool)"
echo "  7. retrievers.py: Added _snowflake_request_id to Document metadata"
echo "              Enables OTel tracing of Cortex Search API calls"
echo "  8. utils.py: Added request_id, guard_tokens to SnowflakeMetadataFactory"
echo "  9. tools.py: Extracts Cortex Inference metadata (request_id, finish_reason, guard_tokens)"
echo "              Enables OTel observability for LLM completions via gen_ai.response.id,"
echo "              gen_ai.response.finish_reasons, and snowflake.inference.guard_tokens"
echo " 10. _connection/base.py: Session validation bypass"
echo "              Fixes 'Password is empty' error when using key-pair authentication"
echo " 11. retrievers.py: Use session token auth instead of JWT for Cortex Search"
echo "              Fixes '401 JWT token is invalid' error"
echo ""
echo "To revert patches:"
echo "  ./patches/revert_patches.sh"
echo ""
echo "Now you can test streaming with:"
echo "  uv run langgraph dev  # Local testing"
echo "  # Or rebuild Docker image and deploy to SPCS"

