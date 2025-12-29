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

| Attribute | Type | Description |
|-----------|------|-------------|
| `snowflake.database` | string | Target database (shared with Cortex Analyst) |
| `snowflake.schema` | string | Target schema (shared with Cortex Analyst) |
| `snowflake.warehouse` | string | Compute warehouse (shared with Cortex Analyst) |
| `snowflake.cortex_search.request_id` | string | Snowflake request ID for correlation |
| `snowflake.cortex_search.top_score` | float | Top cosine similarity score |
| `snowflake.cortex_search.result_count` | int | Number of matching documents |
| `snowflake.cortex_search.sources` | string | Comma-separated sources searched |

**Note:** The `snowflake.database`, `snowflake.schema`, and `snowflake.warehouse` attributes are shared between Cortex Search and Cortex Analyst for consistent observability across both services.

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

### 8. Snowflake Cortex Analyst Provider-Specific Attributes (Enhancement)

**Problem:** Snowflake Cortex Analyst API returns rich metadata for NL→SQL conversion that wasn't captured for observability.

**Cortex Analyst API returns:**
- `request_id` - Unique request identifier for API tracing/correlation
- `metadata.model_names` - LLM models used for SQL generation
- `metadata.question_category` - How the question was classified (CLEAR_SQL, AMBIGUOUS, etc.)
- `confidence.verified_query_used` - Details about matched Verified Query Repository entry
- `warnings` - Any warnings during SQL generation
- `sql` - The generated SQL query

**Enhancement:**
Added 14 Snowflake Cortex Analyst span attributes to capture this metadata:

| Attribute | Type | Description |
|-----------|------|-------------|
| `snowflake.database` | string | Target database |
| `snowflake.schema` | string | Target schema |
| `snowflake.warehouse` | string | Compute warehouse |
| `snowflake.cortex_analyst.request_id` | string | Snowflake request ID for correlation |
| `snowflake.cortex_analyst.semantic_model.name` | string | Semantic model path or view name |
| `snowflake.cortex_analyst.semantic_model.type` | string | `FILE_ON_STAGE` or `SEMANTIC_VIEW` |
| `snowflake.cortex_analyst.sql` | string | Generated SQL (if capture_content enabled) |
| `snowflake.cortex_analyst.model_names` | string | LLM models used (comma-separated) |
| `snowflake.cortex_analyst.question_category` | string | Question classification |
| `snowflake.cortex_analyst.verified_query.name` | string | Matched VQR entry name |
| `snowflake.cortex_analyst.verified_query.question` | string | Original VQR question |
| `snowflake.cortex_analyst.verified_query.sql` | string | VQR SQL template |
| `snowflake.cortex_analyst.verified_query.verified_by` | string | Who verified the query |
| `snowflake.cortex_analyst.warnings_count` | int | Number of warnings |

**Implementation:**
The healthcare app formats tool responses with `@cortex_analyst.*` markers, and the span emitter parses these to set span attributes.

**Files Changed:**
- `attributes.py`: Added 14 Cortex Analyst attribute constants
- `span.py`: Added `_extract_snowflake_analyst_metadata()` function to parse markers from tool responses

### 9. Cortex Inference LLM Completion Attributes (Enhancement)

**Problem:** Snowflake Cortex Inference API (`/api/v2/cortex/inference:complete`) returns metadata that wasn't captured for LLM observability.

**Cortex Inference API returns:**
- `id` - Unique request identifier for API tracing/correlation
- `choices[0].finish_reason` - How the completion finished (`stop`, `length`, `tool_calls`, `tool_use`)
- `usage.guard_tokens` - Tokens consumed by Cortex Guard (content safety) if enabled

**Enhancement:**
Uses standard GenAI semantic convention attributes where available (for portability across providers), plus Snowflake-specific attributes for features unique to Cortex:

| Attribute | Type | Standard? | Description |
|-----------|------|-----------|-------------|
| `gen_ai.response.id` | string | ✅ OTel semconv | Request ID from API (per [gen-ai-spans.md](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)) |
| `gen_ai.response.finish_reasons` | string[] | ✅ OTel semconv | Array of stop reasons (per semconv) |
| `snowflake.inference.guard_tokens` | int | ❌ Snowflake-specific | Guard tokens consumed (Cortex Guard) |

**Implementation:**
1. Modified `langchain-snowflake` to extract `id`, `finish_reason`, and `guard_tokens` from API response and store in `response_metadata`
2. The callback handler extracts these from `response_metadata`:
   - `snowflake_request_id` → `inv.response_id` (maps to `gen_ai.response.id`)
   - `finish_reason` → `inv.attributes["gen_ai.response.finish_reasons"]` (array per semconv)
   - `snowflake_guard_tokens` → `inv.attributes["snowflake.inference.guard_tokens"]`

**Files Changed:**
- `callback_handler.py`: Extract metadata from `response_metadata` in `on_llm_end()`, using standard semconv fields
- `span.py`: Added `gen_ai.response.finish_reasons` and `snowflake.inference.guard_tokens` to `_SPAN_ALLOWED_SUPPLEMENTAL_KEYS`
- `attributes.py`: Added only `SNOWFLAKE_INFERENCE_GUARD_TOKENS` (others are standard semconv)

### 10. Snowflake Cortex Pricing Configuration (Enhancement)

**Problem:** Users need to estimate costs for Cortex API calls but pricing rates can change over time.

**Solution:** Added configurable pricing constants with environment variable overrides.

**Pricing Model (verified from Snowflake Service Consumption Table, Dec 19, 2025):**

| Service | Pricing Model | Default Rate | Environment Variable |
|---------|---------------|--------------|---------------------|
| **Cortex Analyst** | Per Message | **0.067 credits/message** | `OTEL_SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE` |
| **Cortex Search** | Per 1000 Queries | **1.7 credits/1000** | `OTEL_SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES` |
| **Credit Price** | USD | **$3.00/credit** | `OTEL_SNOWFLAKE_CREDIT_PRICE_USD` |

**Important Note:**
> For direct Cortex Analyst calls, **token count does NOT affect cost**. Cost is based purely on message count.
> Token-based pricing only applies when Cortex Analyst is invoked via Cortex Agents.

**Files Changed:**
- `environment_variables.py`: Added pricing env var definitions
- `attributes.py`: Added pricing default constants
- `config.py`: Extended `Settings` dataclass with pricing fields and cost estimation methods
- `instruments.py`: Added cost counters for automatic metrics

**Library-Level Usage (Auto-Instrumentation):**
The pricing configuration is built into `splunk-otel-python-contrib` for universal auto-instrumentation support.

```python
from opentelemetry.util.genai.config import parse_env

settings = parse_env()

# Access pricing rates
print(settings.snowflake_cortex_analyst_credits_per_message)  # 0.067
print(settings.snowflake_cortex_search_credits_per_query)     # 0.0017

# Estimate costs
cost_usd = settings.estimate_cortex_analyst_cost_usd(message_count=100)
# → 100 × 0.067 × $3.00 = $20.10
```

**Cost Metrics (automatically emitted):**
- `snowflake.cortex_analyst.messages` - Message count (counter)
- `snowflake.cortex_search.queries` - Query count (counter)
- `snowflake.cortex.credits` - Credits consumed (counter)
- `snowflake.cortex.cost` - Cost in USD (counter)

## Patched Files Summary

| File | Target Location | Description |
|------|----------------|-------------|
| `callback_handler_patched.py` | `opentelemetry/instrumentation/langchain/callback_handler.py` | Parent span linking, model extraction, provider propagation, Cortex Inference attributes |
| `span_emitter_patched.py` | `opentelemetry/util/genai/emitters/span.py` | Tool attributes, Snowflake Search & Analyst metadata extraction |
| `types_patched.py` | `opentelemetry/util/genai/types.py` | `parent_span` field for hierarchy |
| `attributes_patched.py` | `opentelemetry/util/genai/attributes.py` | Snowflake Cortex Search + Analyst + Inference attributes + pricing defaults |
| `instruments_patched.py` | `opentelemetry/util/genai/instruments.py` | Search score histogram + Cortex cost counters |
| `metrics_patched.py` | `opentelemetry/util/genai/emitters/metrics.py` | Search score metric recording with exemplar linking |
| `environment_variables_patched.py` | `opentelemetry/util/genai/environment_variables.py` | Snowflake Cortex pricing env var definitions |
| `config_patched.py` | `opentelemetry/util/genai/config.py` | Settings dataclass with pricing configuration |

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
- `gen_ai.request.model` - Actual model name (e.g., `claude-4-sonnet`)
- `gen_ai.response.model` - Response model name
- `gen_ai.response.id` - Unique request ID from Cortex Inference API
- `gen_ai.response.finish_reasons` - Array of stop reasons (e.g., `["stop"]`, `["tool_calls"]`)
- `gen_ai.provider.name` - Provider (e.g., `snowflake`)
- `snowflake.inference.guard_tokens` - Guard tokens consumed (if Cortex Guard enabled)

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

**Snowflake Cortex Analyst tool spans should also include:**
- `snowflake.database`
- `snowflake.schema`
- `snowflake.warehouse`
- `snowflake.cortex_analyst.request_id`
- `snowflake.cortex_analyst.semantic_model.name`
- `snowflake.cortex_analyst.semantic_model.type`
- `snowflake.cortex_analyst.sql`
- `snowflake.cortex_analyst.model_names`
- `snowflake.cortex_analyst.question_category`
- `snowflake.cortex_analyst.verified_query.name` (if VQR matched)
- `snowflake.cortex_analyst.verified_query.question` (if VQR matched)
- `snowflake.cortex_analyst.verified_query.sql` (if VQR matched)
- `snowflake.cortex_analyst.verified_query.verified_by` (if VQR matched)
- `snowflake.cortex_analyst.warnings_count`

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

### Cortex Search Tool Span

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
     -> snowflake.database: Str(HEALTHCARE_DB)
     -> snowflake.schema: Str(KNOWLEDGE_SCHEMA)
     -> snowflake.warehouse: Str(PAYERS_CC_WH)
     -> gen_ai.tool.type: Str(function)
     -> gen_ai.tool.call.arguments: Str({"query": "previous calls for Justin Tran"})
     -> gen_ai.tool.call.result: Str([Knowledge Search] Found 9 results...)
     -> snowflake.cortex_search.request_id: Str(e14a3c0c-c258-4267-a0d5-63075dbe178d)
     -> snowflake.cortex_search.top_score: Double(0.446)
     -> snowflake.cortex_search.result_count: Int(9)
     -> snowflake.cortex_search.sources: Str(transcripts)
     -> gen_ai.provider.name: Str(snowflake)
```

### Cortex Analyst Tool Span

```
Span #5
    Trace ID       : 1ee0b6f4f88c813637195748268cfeb3
    Parent ID      : 0446be4472da32a1
    ID             : 7a8b9c0d1e2f3g4h
    Name           : tool query_healthcare_database
    Kind           : Client
Attributes:
     -> gen_ai.tool.name: Str(query_healthcare_database)
     -> gen_ai.tool.call.id: Str(tooluse_abc123xyz)
     -> gen_ai.tool.type: Str(function)
     -> gen_ai.tool.call.arguments: Str({"query": "coverage info for member 578154695"})
     -> gen_ai.tool.call.result: Str([Database Query Results] Query result: {...})
     -> snowflake.database: Str(HEALTHCARE_DB)
     -> snowflake.schema: Str(PUBLIC)
     -> snowflake.warehouse: Str(PAYERS_CC_WH)
     -> snowflake.cortex_analyst.request_id: Str(1855de73-2d56-48dc-8707-8a38af05ad6a)
     -> snowflake.cortex_analyst.semantic_model.name: Str(@HEALTHCARE_DB.STAGING.SEMANTIC_MODELS/healthcare_semantic_model.yaml)
     -> snowflake.cortex_analyst.semantic_model.type: Str(FILE_ON_STAGE)
     -> snowflake.cortex_analyst.sql: Str(SELECT DISTINCT member_id, member_name... WHERE member_id = '578154695')
     -> snowflake.cortex_analyst.model_names: Str(['claude-4-sonnet'])
     -> snowflake.cortex_analyst.question_category: Str(CLEAR_SQL)
     -> snowflake.cortex_analyst.verified_query.name: Str(member_details)
     -> snowflake.cortex_analyst.verified_query.question: Str(Tell me about member 470154690)
     -> snowflake.cortex_analyst.verified_query.sql: Str(SELECT DISTINCT member_id... WHERE member_id = '470154690')
     -> snowflake.cortex_analyst.verified_query.verified_by: Str(system)
     -> snowflake.cortex_analyst.warnings_count: Int(0)
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
