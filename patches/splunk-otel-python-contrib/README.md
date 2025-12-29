# Splunk OTel Python Contrib - LangChain Instrumentation Patches

This directory contains patches for `splunk-otel-python-contrib` to fix issues with LangChain/LangGraph instrumentation discovered during healthcare agent development.

## Quick Start

```bash
# From the patches directory
./apply_patches.sh /path/to/.venv

# Or if .venv is in current directory
./apply_patches.sh
```

## Issues Fixed

### 1. Parent-Child Span Linking (PR-ready)

**Problem:** Spans were not linked in a proper parent-child hierarchy, appearing as separate traces instead of a tree structure.

**Root Cause:** The `parent_run_id` UUID was not being resolved to an actual OpenTelemetry `Span` object. The span emitter checked for `parent_span` but it was never set.

**Files Changed:**
- `callback_handler.py`: Added `_resolve_parent_span()` method and set `parent_span` on all invocation types
- `types.py`: Added `parent_span: Optional[Span] = None` to the `GenAI` base class

### 2. Model Name Extraction for Snowflake (PR-ready)

**Problem:** `gen_ai.request.model` was showing as `unknown_model` for ChatSnowflake.

**Root Cause:** The callback handler only checked `invocation_params.model_name`, but ChatSnowflake uses `invocation_params.model`.

**Files Changed:**
- `callback_handler.py`: Extended model name extraction to check both `model_name` and `model` fields

### 3. Tool Span Attributes per GenAI Semantic Conventions (PR-ready)

**Problem:** Tool spans were missing required GenAI semantic convention attributes:
- `gen_ai.tool.name`
- `gen_ai.tool.type`
- `gen_ai.tool.call.id`
- `gen_ai.tool.call.arguments`
- `gen_ai.tool.call.result`

**Root Cause:** The `span.py` emitter wasn't explicitly setting these attributes on `ToolCall` spans.

**Files Changed:**
- `span.py`: Added explicit attribute setting in `_apply_start_attrs()` and created `_finish_tool_call()` method

### 4. Tool Call ID Extraction (PR-ready)

**Problem:** `gen_ai.tool.call.id` was always missing even when the LLM returned tool_call_id.

**Root Cause:** The callback handler looked for `id` in `serialized` or `extra`, but LangGraph ToolNode passes `tool_call_id` differently.

**Files Changed:**
- `callback_handler.py`: Extended `on_tool_start()` to extract `tool_call_id` from multiple locations:
  - `extra.tool_call_id`
  - `metadata.tool_call_id`
  - `inputs["tool_call_id"]`

### 5. Provider Name Inheritance for Tools (PR-ready)

**Problem:** `gen_ai.provider.name` was set on LLM spans but missing on tool spans.

**Root Cause:** Provider was only propagated to `AgentInvocation` context, but:
1. Many LangGraph apps use `Workflow` and `Step` invocations without explicit agent markers
2. LangGraph creates **sibling branches** for model and tools (separate steps), so provider set on model's parent wasn't visible to tool's parent

**Architecture Impact - LangGraph's Sibling Branches:**
```
workflow (root) ← Provider needed HERE for all branches
├── step model_node ← Provider set here by LLM (not visible to siblings!)
│   └── LLM (snowflake)
├── step tools_node ← tool's parent (different branch!)
│   └── tool search_knowledge ← Couldn't find provider before fix
```

**Fix:** Provider is now propagated to ALL ancestors up to the root workflow, so sibling branches can inherit it.

**Files Changed:**
- `callback_handler.py`: 
  - Added `_find_nearest_provider_context()` to search for provider in any parent entity
  - Added `_propagate_provider_to_root()` to propagate provider from LLM to ALL ancestors
  - Updated `on_chat_model_start()` to propagate provider to all ancestors
  - Updated `on_tool_start()` to inherit provider from any parent context

### 6. Snowflake Cortex Search Provider-Specific Attributes (Enhancement)

**Problem:** Snowflake Cortex Search API returns valuable observability metadata that wasn't captured.

**Snowflake API returns:**
- `_snowflake_request_id` - Unique request identifier for API tracing
- `@search_score` - Cosine similarity score for search relevance
- Result count - Number of matching documents
- Sources - Which search services were queried

**Enhancement:**
Added Snowflake-specific span attributes to capture this metadata:
- `snowflake.cortex_search.request_id`
- `snowflake.cortex_search.top_score`
- `snowflake.cortex_search.result_count`
- `snowflake.cortex_search.sources`

**Files Changed:**
- `attributes.py`: Added Snowflake Cortex Search attribute constants
- `span.py`: Added `_extract_snowflake_search_metadata()` and updated `_finish_tool_call()` to apply these attributes

### 7. Search Score Histogram Metric with Pre-Configured Buckets (Enhancement)

**Problem:** No way to monitor search quality trends over time, and default histogram buckets (0-10000) were inappropriate for cosine similarity scores (0.0-1.0).

**Enhancement:**
Added a histogram metric `gen_ai.tool.search.score` with:
- **Pre-configured bucket boundaries** for cosine similarity (0.0, 0.1, 0.2, ... 1.0)
- **Exemplar linking** to correlate metric data points with specific traces
- **Dimensions:**
  - `gen_ai.tool.name` - The search tool (e.g., `search_knowledge`)
  - `gen_ai.provider.name` - The provider (e.g., `snowflake`)
  - `snowflake.cortex_search.sources` - Sources searched

This enables:
- Alerting on low search quality (p50/p95 < threshold)
- Trending search relevance over time
- Correlating search quality with user satisfaction
- **No manual View configuration required** - buckets are pre-configured in the library

**Files Changed:**
- `instruments.py`: Added `tool_search_score_histogram` with `explicit_bucket_boundaries_advisory` for cosine similarity (0.0-1.0)
- `metrics.py`: Added `_extract_search_score_from_tool_response()` and `_record_search_score_metric()` with exemplar context linking

## Patched Files Summary

| File | Target Location | Description |
|------|----------------|-------------|
| `callback_handler_patched.py` | `opentelemetry/instrumentation/langchain/callback_handler.py` | Parent span linking, model extraction, provider propagation |
| `span_emitter_patched.py` | `opentelemetry/util/genai/emitters/span.py` | Tool attributes, Snowflake metadata extraction |
| `types_patched.py` | `opentelemetry/util/genai/types.py` | `parent_span` field for hierarchy |
| `attributes_patched.py` | `opentelemetry/util/genai/attributes.py` | Snowflake Cortex Search attribute constants |
| `instruments_patched.py` | `opentelemetry/util/genai/instruments.py` | Search score histogram with pre-configured buckets |
| `metrics_patched.py` | `opentelemetry/util/genai/emitters/metrics.py` | Search score metric recording with exemplar linking |

## How to Apply Patches

### Option 1: Use the apply script (Recommended)

```bash
# From the patches directory
./apply_patches.sh /path/to/.venv

# Or if .venv is in current directory
./apply_patches.sh
```

### Option 2: Install from local modified source (for development)

```bash
# From the splunk-otel-python-contrib root directory
cd instrumentation-genai/opentelemetry-instrumentation-langchain
pip install -e .

cd ../../util/opentelemetry-util-genai
pip install -e .
```

### Option 3: Copy patched files manually

```bash
# Find your installed package location
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Apply all patches
cp callback_handler_patched.py \
   $SITE_PACKAGES/opentelemetry/instrumentation/langchain/callback_handler.py

cp span_emitter_patched.py \
   $SITE_PACKAGES/opentelemetry/util/genai/emitters/span.py

cp types_patched.py \
   $SITE_PACKAGES/opentelemetry/util/genai/types.py

cp attributes_patched.py \
   $SITE_PACKAGES/opentelemetry/util/genai/attributes.py

cp instruments_patched.py \
   $SITE_PACKAGES/opentelemetry/util/genai/instruments.py

cp metrics_patched.py \
   $SITE_PACKAGES/opentelemetry/util/genai/emitters/metrics.py
```

## How to Revert Patches

```bash
# From the patches directory
./revert_patches.sh /path/to/.venv

# Or if .venv is in current directory
./revert_patches.sh
```

## Testing

After applying patches, verify with a test query that triggers search tools:

### 1. Span Attributes (check in OTel collector logs)

**LLM spans should include:**
- `gen_ai.request.model` - Actual model name (e.g., `claude-3-5-sonnet`)
- `gen_ai.response.model` - Response model name
- `gen_ai.provider.name` - Provider (e.g., `snowflake`)

**Tool spans should include:**
- `gen_ai.tool.name` (e.g., `search_knowledge`)
- `gen_ai.tool.type` (`function`)
- `gen_ai.tool.call.id` (e.g., `tooluse_cJ0D_A_nQp201hv_VbOzcw`)
- `gen_ai.tool.call.arguments` (JSON string of tool input)
- `gen_ai.tool.call.result` (tool output)
- `gen_ai.provider.name` (inherited from LLM, e.g., `snowflake`)

**Snowflake Cortex Search tool spans should also include:**
- `snowflake.cortex_search.request_id`
- `snowflake.cortex_search.top_score`
- `snowflake.cortex_search.result_count`
- `snowflake.cortex_search.sources`

### 2. Histogram Metric (check in OTel collector logs)

```
Name: gen_ai.tool.search.score
Description: Top similarity score from search/retrieval tool operations
Unit: {score}
DataType: Histogram
AggregationTemporality: Cumulative

Data point attributes:
  -> gen_ai.tool.name: Str(search_knowledge)
  -> gen_ai.provider.name: Str(snowflake)
  -> snowflake.cortex_search.sources: Str(transcripts)

ExplicitBounds #0: 0.000000
ExplicitBounds #1: 0.100000
ExplicitBounds #2: 0.200000
...
ExplicitBounds #10: 1.000000

Buckets #5, Count: 2  <- Scores in 0.4-0.5 range
```

### 3. Parent-Child Span Hierarchy

Traces should show proper nesting:
```
workflow react_healthcare
├── step model
│   └── LLM claude-3-5-sonnet
├── step tools
│   └── tool search_knowledge
└── step model
    └── LLM claude-3-5-sonnet
```

## Example OTel Collector Output

```
Span #3
    Trace ID       : 1ee0b6f4f88c813637195748268cfeb3
    Parent ID      : 0446be4472da32a1
    ID             : 506eef71bdbb74a5
    Name           : tool search_knowledge
    Kind           : Client
Attributes:
     -> gen_ai.tool.name: Str(search_knowledge)
     -> gen_ai.tool.call.id: Str(tooluse_ge8XouBYRLGxPKGrVs23Zw)
     -> gen_ai.tool.type: Str(function)
     -> gen_ai.tool.call.arguments: Str({"query": "previous calls for Justin Tran"})
     -> gen_ai.tool.call.result: Str([Knowledge Search] Found 9 results...)
     -> snowflake.cortex_search.request_id: Str(e14a3c0c-c258-4267-a0d5-63075dbe178d)
     -> snowflake.cortex_search.top_score: Double(0.446)
     -> snowflake.cortex_search.result_count: Int(9)
     -> snowflake.cortex_search.sources: Str(transcripts)
     -> gen_ai.provider.name: Str(snowflake)
```

## PR Submission Checklist

- [ ] All patches have corresponding unit tests
- [ ] Changes are backward compatible (additive, not breaking)
- [ ] Documentation updated
- [ ] Follows project coding standards
- [ ] Signed CLA (if required)

## Related Issues

- Parent-child linking: Similar to upstream opentelemetry-python-contrib LangChain instrumentation
- Tool attributes: Per GenAI semantic conventions specification
- Provider inheritance: Enables consistent observability across LLM and tool spans
- Snowflake attributes: Vendor-specific extension following semantic conventions pattern
- Search score histogram: Following pattern from `opentelemetry-instrumentation-openai-v2` for pre-configured bucket boundaries
