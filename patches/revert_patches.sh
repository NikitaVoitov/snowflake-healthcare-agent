#!/bin/bash
# Revert langchain-snowflake streaming patches
#
# This script restores the original langchain-snowflake files.
#
# Usage: ./patches/revert_patches.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find the Python site-packages directory
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
if [ ! -d "$LANGCHAIN_SNOWFLAKE_DIR" ]; then
    echo "❌ langchain-snowflake not installed"
    exit 1
fi

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

echo ""
echo "✅ Patches reverted successfully!"
echo ""
echo "Remember to also disable streaming in your agent:"
echo "  Set ENABLE_LLM_STREAMING=false (or unset it)"

