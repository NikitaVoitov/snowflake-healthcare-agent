#!/bin/bash
# Revert langchain-snowflake patches
#
# This script restores the original langchain-snowflake files.
#
# Usage: ./patches/revert_patches.sh

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
        echo "❌ langchain-snowflake not installed"
        exit 1
    fi
    INSTALL_TYPE="site-packages"
fi

echo "📦 Install type: $INSTALL_TYPE"
echo "📦 Found langchain-snowflake at: $LANGCHAIN_SNOWFLAKE_DIR"

# Restore originals
if [ -f "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py.original" ]; then
    echo "🔄 Restoring original streaming.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py.original" "$LANGCHAIN_SNOWFLAKE_DIR/streaming.py"
else
    echo "⚠️ No backup found for streaming.py"
fi

if [ -f "$LANGCHAIN_SNOWFLAKE_DIR/base.py.original" ]; then
    echo "🔄 Restoring original base.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/base.py.original" "$LANGCHAIN_SNOWFLAKE_DIR/base.py"
else
    echo "⚠️ No backup found for base.py"
fi

if [ -f "$LANGCHAIN_SNOWFLAKE_DIR/tools.py.original" ]; then
    echo "🔄 Restoring original tools.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/tools.py.original" "$LANGCHAIN_SNOWFLAKE_DIR/tools.py"
else
    echo "⚠️ No backup found for tools.py"
fi

if [ -f "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py.original" ]; then
    echo "🔄 Restoring original mcp_integration.py..."
    cp "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py.original" "$LANGCHAIN_SNOWFLAKE_ROOT/mcp_integration.py"
else
    echo "⚠️ No backup found for mcp_integration.py"
fi

if [ -f "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py.original" ]; then
    echo "🔄 Restoring original retrievers.py..."
    cp "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py.original" "$LANGCHAIN_SNOWFLAKE_ROOT/retrievers.py"
else
    echo "⚠️ No backup found for retrievers.py"
fi

if [ -f "$LANGCHAIN_SNOWFLAKE_DIR/utils.py.original" ]; then
    echo "🔄 Restoring original utils.py..."
    cp "$LANGCHAIN_SNOWFLAKE_DIR/utils.py.original" "$LANGCHAIN_SNOWFLAKE_DIR/utils.py"
else
    echo "⚠️ No backup found for utils.py"
fi

# Restore _connection/base.py
CONNECTION_DIR="$LANGCHAIN_SNOWFLAKE_ROOT/_connection"
if [ -d "$CONNECTION_DIR" ] && [ -f "$CONNECTION_DIR/base.py.original" ]; then
    echo "🔄 Restoring original _connection/base.py..."
    cp "$CONNECTION_DIR/base.py.original" "$CONNECTION_DIR/base.py"
else
    echo "⚠️ No backup found for _connection/base.py"
fi

echo ""
echo "✅ Patches reverted successfully!"
echo ""
echo "Remember to also disable streaming in your agent:"
echo "  Set ENABLE_LLM_STREAMING=false (or unset it)"
