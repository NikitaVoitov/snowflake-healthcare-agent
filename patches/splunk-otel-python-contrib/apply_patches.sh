#!/bin/bash
# Apply splunk-otel-python-contrib patches for LangChain instrumentation fixes
# 
# This script applies patches to fix:
# 1. Parent-child span linking
# 2. Model name extraction for Snowflake
# 3. Tool span attributes per GenAI semantic conventions
# 4. Tool call ID extraction
# 5. Provider name inheritance for tools
# 6. Snowflake Cortex Search provider-specific attributes
# 7. Snowflake Cortex Analyst provider-specific attributes
# 8. Search score histogram metric
# 9. Cortex Inference LLM completion attributes (gen_ai.response.*)
# 10. Snowflake Cortex pricing configuration for cost metrics
#
# Usage: ./apply_patches.sh [VENV_PATH]
#   VENV_PATH: Optional path to virtual environment (defaults to .venv)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "📦 Applying splunk-otel-python-contrib patches..."
echo "   Target: $SITE_PACKAGES"
echo ""

# Check if packages are installed
LANGCHAIN_INSTR="$SITE_PACKAGES/opentelemetry/instrumentation/langchain"
UTIL_GENAI="$SITE_PACKAGES/opentelemetry/util/genai"

if [ ! -d "$LANGCHAIN_INSTR" ]; then
    echo "❌ opentelemetry-instrumentation-langchain not installed"
    echo "   Install with: pip install opentelemetry-instrumentation-langchain"
    exit 1
fi

if [ ! -d "$UTIL_GENAI" ]; then
    echo "❌ opentelemetry-util-genai not installed"
    echo "   Install the splunk-otel-python-contrib GenAI utilities first"
    exit 1
fi

# Backup original files if not already backed up
echo "💾 Creating backups..."

if [ ! -f "$LANGCHAIN_INSTR/callback_handler.py.original" ]; then
    cp "$LANGCHAIN_INSTR/callback_handler.py" "$LANGCHAIN_INSTR/callback_handler.py.original"
    echo "   ✓ Backed up callback_handler.py"
fi

if [ ! -f "$UTIL_GENAI/emitters/span.py.original" ]; then
    cp "$UTIL_GENAI/emitters/span.py" "$UTIL_GENAI/emitters/span.py.original"
    echo "   ✓ Backed up emitters/span.py"
fi

if [ ! -f "$UTIL_GENAI/types.py.original" ]; then
    cp "$UTIL_GENAI/types.py" "$UTIL_GENAI/types.py.original"
    echo "   ✓ Backed up types.py"
fi

if [ ! -f "$UTIL_GENAI/attributes.py.original" ]; then
    cp "$UTIL_GENAI/attributes.py" "$UTIL_GENAI/attributes.py.original"
    echo "   ✓ Backed up attributes.py"
fi

if [ ! -f "$UTIL_GENAI/instruments.py.original" ]; then
    cp "$UTIL_GENAI/instruments.py" "$UTIL_GENAI/instruments.py.original"
    echo "   ✓ Backed up instruments.py"
fi

if [ ! -f "$UTIL_GENAI/emitters/metrics.py.original" ]; then
    cp "$UTIL_GENAI/emitters/metrics.py" "$UTIL_GENAI/emitters/metrics.py.original"
    echo "   ✓ Backed up emitters/metrics.py"
fi

if [ ! -f "$UTIL_GENAI/environment_variables.py.original" ]; then
    cp "$UTIL_GENAI/environment_variables.py" "$UTIL_GENAI/environment_variables.py.original"
    echo "   ✓ Backed up environment_variables.py"
fi

if [ ! -f "$UTIL_GENAI/config.py.original" ]; then
    cp "$UTIL_GENAI/config.py" "$UTIL_GENAI/config.py.original"
    echo "   ✓ Backed up config.py"
fi

# Apply patches
echo ""
echo "🔧 Applying patches..."

cp "$SCRIPT_DIR/callback_handler_patched.py" "$LANGCHAIN_INSTR/callback_handler.py"
echo "   ✓ Applied callback_handler.py patch"

cp "$SCRIPT_DIR/span_emitter_patched.py" "$UTIL_GENAI/emitters/span.py"
echo "   ✓ Applied emitters/span.py patch"

cp "$SCRIPT_DIR/types_patched.py" "$UTIL_GENAI/types.py"
echo "   ✓ Applied types.py patch"

cp "$SCRIPT_DIR/attributes_patched.py" "$UTIL_GENAI/attributes.py"
echo "   ✓ Applied attributes.py patch (Snowflake Cortex Search & Analyst attributes)"

cp "$SCRIPT_DIR/instruments_patched.py" "$UTIL_GENAI/instruments.py"
echo "   ✓ Applied instruments.py patch (search score histogram)"

cp "$SCRIPT_DIR/metrics_patched.py" "$UTIL_GENAI/emitters/metrics.py"
echo "   ✓ Applied emitters/metrics.py patch (search score metric recording)"

cp "$SCRIPT_DIR/environment_variables_patched.py" "$UTIL_GENAI/environment_variables.py"
echo "   ✓ Applied environment_variables.py patch (Snowflake pricing env vars)"

cp "$SCRIPT_DIR/config_patched.py" "$UTIL_GENAI/config.py"
echo "   ✓ Applied config.py patch (Settings with pricing configuration)"

# Clear Python cache
echo ""
echo "🧹 Clearing Python cache..."
find "$SITE_PACKAGES/opentelemetry" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ Cache cleared"

echo ""
echo "✅ All patches applied successfully!"
echo ""
echo "Fixes included:"
echo "  1. Parent-child span linking (proper trace hierarchy)"
echo "  2. Model name extraction for ChatSnowflake (gen_ai.request.model)"
echo "  3. Tool span attributes (gen_ai.tool.name, gen_ai.tool.type, etc.)"
echo "  4. Tool call ID extraction (gen_ai.tool.call.id)"
echo "  5. Provider inheritance for tools (gen_ai.provider.name on tool spans)"
echo "     - Works for both multi-agent and single-workflow patterns"
echo "     - Provider propagated to any parent context (Agent/Workflow/Step)"
echo ""
echo "Snowflake Connection Context (shared):"
echo "  6. snowflake.database - Snowflake database name"
echo "  7. snowflake.schema - Snowflake schema name"
echo "  8. snowflake.warehouse - Snowflake warehouse name"
echo ""
echo "Snowflake Cortex Search observability:"
echo "  9. snowflake.cortex_search.request_id - Request ID from Cortex Search API"
echo "  10. snowflake.cortex_search.top_score - Top cosine similarity score"
echo "  11. snowflake.cortex_search.result_count - Number of search results"
echo "  12. snowflake.cortex_search.sources - Search sources (faqs, policies, etc.)"
echo "  13. gen_ai.tool.search.score histogram metric"
echo ""
echo "Snowflake Cortex Analyst observability:"
echo "  14. snowflake.cortex_analyst.request_id - Request ID"
echo "  15. snowflake.cortex_analyst.semantic_model.name - Semantic model path"
echo "  16. snowflake.cortex_analyst.semantic_model.type - FILE_ON_STAGE or SEMANTIC_VIEW"
echo "  17. snowflake.cortex_analyst.sql - Generated SQL (if capture_content)"
echo "  18. snowflake.cortex_analyst.model_names - LLM models used"
echo "  19. snowflake.cortex_analyst.question_category - Query classification"
echo "  20. snowflake.cortex_analyst.verified_query.* - VQR match details"
echo "  21. snowflake.cortex_analyst.warnings_count - Warning count"
echo ""
echo "Snowflake Cortex Inference LLM attributes:"
echo "  22. gen_ai.response.id - Request ID from Cortex Inference API"
echo "  23. gen_ai.response.finish_reasons - Array of stop reasons"
echo "  24. snowflake.inference.guard_tokens - Cortex Guard tokens consumed"
echo ""
echo "Cost Tracking (configurable via env vars):"
echo "  - OTEL_SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE (default: 0.067)"
echo "  - OTEL_SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES (default: 1.7)"
echo "  - OTEL_SNOWFLAKE_CREDIT_PRICE_USD (default: 3.00)"
echo ""
echo "Metrics:"
echo "  - snowflake.cortex_analyst.messages (counter)"
echo "  - snowflake.cortex_search.queries (counter)"
echo "  - snowflake.cortex.credits (counter)"
echo "  - snowflake.cortex.cost (counter, USD)"
echo ""
echo "To revert, run: ./revert_patches.sh"
