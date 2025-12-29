#!/bin/bash
# Revert splunk-otel-python-contrib patches
# 
# This script restores the original files that were patched by apply_patches.sh
#
# Usage: ./revert_patches.sh [VENV_PATH]
#   VENV_PATH: Optional path to virtual environment (defaults to .venv)

set -e

VENV_PATH="${1:-.venv}"

# Determine site-packages path
if [ -d "$VENV_PATH" ]; then
    SITE_PACKAGES="$VENV_PATH/lib/python3.*/site-packages"
    SITE_PACKAGES=$(echo $SITE_PACKAGES)  # Expand glob
else
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "   Usage: $0 [VENV_PATH]"
    exit 1
fi

if [ ! -d "$SITE_PACKAGES" ]; then
    echo "❌ site-packages not found at: $SITE_PACKAGES"
    exit 1
fi

echo "📦 Reverting splunk-otel-python-contrib patches..."
echo "   Target: $SITE_PACKAGES"
echo ""

LANGCHAIN_INSTR="$SITE_PACKAGES/opentelemetry/instrumentation/langchain"
UTIL_GENAI="$SITE_PACKAGES/opentelemetry/util/genai"

# Track if any files were reverted
REVERTED=0

# Revert callback_handler.py
if [ -f "$LANGCHAIN_INSTR/callback_handler.py.original" ]; then
    cp "$LANGCHAIN_INSTR/callback_handler.py.original" "$LANGCHAIN_INSTR/callback_handler.py"
    echo "   ✓ Reverted callback_handler.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for callback_handler.py"
fi

# Revert emitters/span.py
if [ -f "$UTIL_GENAI/emitters/span.py.original" ]; then
    cp "$UTIL_GENAI/emitters/span.py.original" "$UTIL_GENAI/emitters/span.py"
    echo "   ✓ Reverted emitters/span.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for emitters/span.py"
fi

# Revert types.py
if [ -f "$UTIL_GENAI/types.py.original" ]; then
    cp "$UTIL_GENAI/types.py.original" "$UTIL_GENAI/types.py"
    echo "   ✓ Reverted types.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for types.py"
fi

# Revert attributes.py
if [ -f "$UTIL_GENAI/attributes.py.original" ]; then
    cp "$UTIL_GENAI/attributes.py.original" "$UTIL_GENAI/attributes.py"
    echo "   ✓ Reverted attributes.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for attributes.py"
fi

# Revert instruments.py
if [ -f "$UTIL_GENAI/instruments.py.original" ]; then
    cp "$UTIL_GENAI/instruments.py.original" "$UTIL_GENAI/instruments.py"
    echo "   ✓ Reverted instruments.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for instruments.py"
fi

# Revert emitters/metrics.py
if [ -f "$UTIL_GENAI/emitters/metrics.py.original" ]; then
    cp "$UTIL_GENAI/emitters/metrics.py.original" "$UTIL_GENAI/emitters/metrics.py"
    echo "   ✓ Reverted emitters/metrics.py"
    REVERTED=1
else
    echo "   ⚠ No backup found for emitters/metrics.py"
fi

# Clear Python cache
echo ""
echo "🧹 Clearing Python cache..."
find "$SITE_PACKAGES/opentelemetry" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ Cache cleared"

echo ""
if [ $REVERTED -eq 1 ]; then
    echo "✅ Patches reverted successfully!"
    echo ""
    echo "To re-apply patches, run: ./apply_patches.sh"
else
    echo "⚠ No patches were applied - nothing to revert"
fi
