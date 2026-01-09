# Span emitter (moved from generators/span_emitter.py)
from __future__ import annotations

import json
import re
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..attributes import (
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_AGENT_TOOLS,
    GEN_AI_AGENT_TYPE,
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_EMBEDDINGS_INPUT_TEXTS,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_ENCODING_FORMATS,
    GEN_AI_STEP_ASSIGNED_AGENT,
    GEN_AI_STEP_NAME,
    GEN_AI_STEP_OBJECTIVE,
    GEN_AI_STEP_SOURCE,
    GEN_AI_STEP_STATUS,
    GEN_AI_STEP_TYPE,
    GEN_AI_WORKFLOW_DESCRIPTION,
    GEN_AI_WORKFLOW_NAME,
    GEN_AI_WORKFLOW_TYPE,
    SERVER_ADDRESS,
    SERVER_PORT,
    SNOWFLAKE_CORTEX_ANALYST_MODEL_NAMES,
    SNOWFLAKE_CORTEX_ANALYST_QUESTION_CATEGORY,
    SNOWFLAKE_CORTEX_ANALYST_REQUEST_ID,
    SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_NAME,
    SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_TYPE,
    SNOWFLAKE_CORTEX_ANALYST_SQL,
    SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_NAME,
    SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_QUESTION,
    SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_SQL,
    SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_VERIFIED_BY,
    SNOWFLAKE_CORTEX_ANALYST_WARNINGS_COUNT,
    # Snowflake Cortex Search provider-specific attributes
    SNOWFLAKE_CORTEX_SEARCH_REQUEST_ID,
    SNOWFLAKE_CORTEX_SEARCH_RESULT_COUNT,
    SNOWFLAKE_CORTEX_SEARCH_SOURCES,
    SNOWFLAKE_CORTEX_SEARCH_TOP_SCORE,
    # Snowflake Cortex Analyst provider-specific attributes
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_WAREHOUSE,
)
from ..interfaces import EmitterMeta
from ..span_context import extract_span_context, store_span_context
from ..types import (
    AgentCreation,
    AgentInvocation,
    ContentCapturingMode,
    EmbeddingInvocation,
    Error,
    LLMInvocation,
    Step,
    ToolCall,
    Workflow,
)
from ..types import (
    GenAI as GenAIType,
)
from .utils import (
    _apply_function_definitions,
    _apply_llm_finish_semconv,
    _extract_system_instructions,
    _serialize_messages,
    filter_semconv_gen_ai_attributes,
)

_SPAN_ALLOWED_SUPPLEMENTAL_KEYS: tuple[str, ...] = (
    "gen_ai.framework",
    "gen_ai.request.id",
    # Standard GenAI semconv attributes (from response_metadata)
    "gen_ai.response.finish_reasons",
    # Snowflake-specific attributes (not in semconv)
    "snowflake.inference.guard_tokens",  # Cortex Guard token usage
    "snowflake.database",
    "snowflake.schema",
    "snowflake.warehouse",
)
_SPAN_BLOCKED_SUPPLEMENTAL_KEYS: set[str] = {"request_top_p", "ls_temperature"}


def _extract_snowflake_search_metadata(
    tool_response: str,
) -> dict[str, Any]:
    """
    Extract Snowflake Cortex Search metadata from tool response.

    Looks for patterns embedded in the tool response:
    - @top_cosine_similarity: 0.XXX
    - @scores: {cosine=X.XXX, text=X.XXX}
    - @sources: ["faqs", "policies", "transcripts"]
    - @request_ids: ["xxx", "yyy"]
    - Result count from "[Knowledge Search] Found N relevant result(s):"
    - Snowflake context (database, schema, warehouse) from Cortex Search Metadata section

    Returns a dict with extracted metadata for span attributes.
    """
    result: dict[str, Any] = {}

    # Normalize escaped newlines for consistent regex matching
    tool_response = tool_response.replace("\\n", "\n")

    # Extract top cosine similarity score
    score_match = re.search(r"@top_cosine_similarity:\s*([\d.]+)", tool_response)
    if score_match:
        try:
            result["top_score"] = float(score_match.group(1))
        except (ValueError, TypeError):
            pass

    # Fallback: try to extract from cosine= pattern in @scores
    if "top_score" not in result:
        cosine_match = re.search(r"cosine[=:]\s*([\d.]+)", tool_response)
        if cosine_match:
            try:
                result["top_score"] = float(cosine_match.group(1))
            except (ValueError, TypeError):
                pass

    # Extract sources list
    sources_match = re.search(r"@sources:\s*\[([^\]]+)\]", tool_response)
    if sources_match:
        sources_str = sources_match.group(1)
        # Parse list items, handling quotes
        sources = [s.strip().strip("'\"") for s in sources_str.split(",")]
        sources = [s for s in sources if s]  # Filter empty
        if sources:
            result["sources"] = sources

    # Extract request IDs
    request_match = re.search(r"@request_ids:\s*\[([^\]]+)\]", tool_response)
    if request_match:
        request_str = request_match.group(1)
        request_ids = [r.strip().strip("'\"") for r in request_str.split(",")]
        request_ids = [r for r in request_ids if r]
        if request_ids:
            # Use first request_id for span attribute (typically same for one search)
            result["request_id"] = request_ids[0]

    # Extract result count from "[Knowledge Search] Found N relevant result(s):"
    count_match = re.search(r"\[Knowledge Search\]\s*Found\s+(\d+)\s+relevant", tool_response)
    if count_match:
        try:
            result["result_count"] = int(count_match.group(1))
        except (ValueError, TypeError):
            pass

    # Extract Snowflake context from Cortex Search Metadata section
    # These are the same attributes as Cortex Analyst for consistency
    if "--- Cortex Search Metadata ---" in tool_response:
        # Helper to extract and clean identifier values (strips trailing quotes)
        def clean_identifier(pattern: str, text: str) -> str | None:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                # Clean up trailing quotes that might be captured from JSON/string boundaries
                return value.rstrip('"')
            return None

        # Extract database
        db = clean_identifier(r"@snowflake\.database:\s*(\S+)", tool_response)
        if db:
            result["database"] = db

        # Extract schema
        schema = clean_identifier(r"@snowflake\.schema:\s*(\S+)", tool_response)
        if schema:
            result["schema"] = schema

        # Extract warehouse
        warehouse = clean_identifier(r"@snowflake\.warehouse:\s*(\S+)", tool_response)
        if warehouse:
            result["warehouse"] = warehouse

    return result


def _extract_snowflake_analyst_metadata(
    tool_response: str,
) -> dict[str, Any]:
    """
    Extract Snowflake Cortex Analyst metadata from tool response.

    Looks for patterns embedded in the tool response with @cortex_analyst.* prefixes:
    - @snowflake.database: HEALTHCARE_DB
    - @snowflake.schema: PUBLIC
    - @snowflake.warehouse: WAREHOUSE_NAME
    - @cortex_analyst.semantic_model.name: model_path
    - @cortex_analyst.semantic_model.type: FILE_ON_STAGE|SEMANTIC_VIEW
    - @cortex_analyst.request_id: uuid
    - @cortex_analyst.sql: SQL_STATEMENT
    - @cortex_analyst.model_names: ['model1', 'model2']
    - @cortex_analyst.question_category: CLEAR_SQL|AMBIGUOUS|etc
    - @cortex_analyst.verified_query.name: query_name
    - @cortex_analyst.verified_query.question: original_question
    - @cortex_analyst.verified_query.sql: verified_sql
    - @cortex_analyst.verified_query.verified_by: user
    - @cortex_analyst.warnings_count: N

    Returns a dict with extracted metadata for span attributes.
    """
    result: dict[str, Any] = {}

    # Check if this is a Cortex Analyst response (has the metadata section)
    if "--- Cortex Analyst Metadata ---" not in tool_response:
        return result

    # Normalize escaped newlines to real newlines for consistent parsing
    # This handles cases where the string was JSON-serialized/deserialized
    normalized = tool_response.replace("\\n", "\n")

    # Helper to extract single-line value (stops at newline or end)
    def extract_value(pattern: str, text: str) -> str | None:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            # Clean up any trailing markers that might have been captured
            if value.endswith('"'):
                value = value.rstrip('"')
            return value
        return None

    # Extract Snowflake context - use \S+ for simple identifier values
    db = extract_value(r"@snowflake\.database:\s*(\S+)", normalized)
    if db:
        result["database"] = db

    schema = extract_value(r"@snowflake\.schema:\s*(\S+)", normalized)
    if schema:
        result["schema"] = schema

    warehouse = extract_value(r"@snowflake\.warehouse:\s*(\S+)", normalized)
    if warehouse:
        result["warehouse"] = warehouse

    # Extract semantic model info
    model_name = extract_value(r"@cortex_analyst\.semantic_model\.name:\s*([^\n]+)", normalized)
    if model_name:
        result["semantic_model_name"] = model_name

    model_type = extract_value(r"@cortex_analyst\.semantic_model\.type:\s*([^\n]+)", normalized)
    if model_type:
        result["semantic_model_type"] = model_type

    # Extract request ID
    request_id = extract_value(r"@cortex_analyst\.request_id:\s*([^\n]+)", normalized)
    if request_id:
        result["request_id"] = request_id

    # Extract SQL (may be multiline, capture until next @cortex_analyst. marker)
    sql_match = re.search(r"@cortex_analyst\.sql:\s*(.+?)(?=\n@cortex_analyst\.)", normalized, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
        # Truncate very long SQL for span attributes (8KB limit recommended)
        if len(sql) > 4096:
            sql = sql[:4096] + "..."
        result["sql"] = sql

    # Extract model names (list format)
    model_names_match = re.search(r"@cortex_analyst\.model_names:\s*\[([^\]]+)\]", normalized)
    if model_names_match:
        names_str = model_names_match.group(1)
        names = [n.strip().strip("'\"") for n in names_str.split(",")]
        names = [n for n in names if n]
        if names:
            result["model_names"] = ",".join(names)

    # Extract question category
    category = extract_value(r"@cortex_analyst\.question_category:\s*([^\n]+)", normalized)
    if category:
        result["question_category"] = category

    # Extract verified query info
    vq_name = extract_value(r"@cortex_analyst\.verified_query\.name:\s*([^\n]+)", normalized)
    if vq_name:
        result["verified_query_name"] = vq_name

    vq_question = extract_value(r"@cortex_analyst\.verified_query\.question:\s*([^\n]+)", normalized)
    if vq_question:
        result["verified_query_question"] = vq_question

    # Extract verified query SQL (may be multiline)
    vq_sql_match = re.search(r"@cortex_analyst\.verified_query\.sql:\s*(.+?)(?=\n@cortex_analyst\.)", normalized, re.DOTALL)
    if vq_sql_match:
        vq_sql = vq_sql_match.group(1).strip()
        if len(vq_sql) > 2048:
            vq_sql = vq_sql[:2048] + "..."
        result["verified_query_sql"] = vq_sql

    vq_verified_by = extract_value(r"@cortex_analyst\.verified_query\.verified_by:\s*([^\n]+)", normalized)
    if vq_verified_by:
        result["verified_query_verified_by"] = vq_verified_by

    # Extract warnings count
    warnings_match = re.search(r"@cortex_analyst\.warnings_count:\s*(\d+)", normalized)
    if warnings_match:
        try:
            result["warnings_count"] = int(warnings_match.group(1))
        except (ValueError, TypeError):
            pass

    return result


# --- Snowflake Metadata Attribute Helpers (DRY) ---

# Mapping of metadata keys to span attribute names for Snowflake context
_SNOWFLAKE_CONTEXT_ATTR_MAP: dict[str, str] = {
    "database": SNOWFLAKE_DATABASE,
    "schema": SNOWFLAKE_SCHEMA,
    "warehouse": SNOWFLAKE_WAREHOUSE,
}

# Mapping for Cortex Search specific attributes
_CORTEX_SEARCH_ATTR_MAP: dict[str, str] = {
    "request_id": SNOWFLAKE_CORTEX_SEARCH_REQUEST_ID,
    "top_score": SNOWFLAKE_CORTEX_SEARCH_TOP_SCORE,
    "result_count": SNOWFLAKE_CORTEX_SEARCH_RESULT_COUNT,
}

# Mapping for Cortex Analyst specific attributes
_CORTEX_ANALYST_ATTR_MAP: dict[str, str] = {
    "request_id": SNOWFLAKE_CORTEX_ANALYST_REQUEST_ID,
    "semantic_model_name": SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_NAME,
    "semantic_model_type": SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_TYPE,
    "sql": SNOWFLAKE_CORTEX_ANALYST_SQL,
    "model_names": SNOWFLAKE_CORTEX_ANALYST_MODEL_NAMES,
    "question_category": SNOWFLAKE_CORTEX_ANALYST_QUESTION_CATEGORY,
    "verified_query_name": SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_NAME,
    "verified_query_question": SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_QUESTION,
    "verified_query_sql": SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_SQL,
    "verified_query_verified_by": SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_VERIFIED_BY,
    "warnings_count": SNOWFLAKE_CORTEX_ANALYST_WARNINGS_COUNT,
}


def _apply_metadata_to_span(
    span: Span,
    metadata: dict[str, Any],
    attr_map: dict[str, str],
    *,
    join_lists: bool = False,
) -> None:
    """Apply metadata dictionary to span using attribute mapping.

    Args:
        span: The span to set attributes on.
        metadata: Dictionary of extracted metadata.
        attr_map: Mapping from metadata keys to span attribute names.
        join_lists: If True, join list values with comma before setting.
    """
    for meta_key, attr_name in attr_map.items():
        if meta_key in metadata:
            value = metadata[meta_key]
            if join_lists and isinstance(value, list):
                value = ",".join(str(v) for v in value)
            span.set_attribute(attr_name, value)


def _apply_cortex_search_metadata(span: Span, metadata: dict[str, Any]) -> None:
    """Apply Cortex Search metadata to span attributes."""
    # Apply shared Snowflake context attributes
    _apply_metadata_to_span(span, metadata, _SNOWFLAKE_CONTEXT_ATTR_MAP)
    # Apply Cortex Search specific attributes
    _apply_metadata_to_span(span, metadata, _CORTEX_SEARCH_ATTR_MAP)
    # Handle sources specially (needs to be joined)
    if "sources" in metadata:
        span.set_attribute(
            SNOWFLAKE_CORTEX_SEARCH_SOURCES,
            ",".join(metadata["sources"]),
        )


def _apply_cortex_analyst_metadata(span: Span, metadata: dict[str, Any]) -> None:
    """Apply Cortex Analyst metadata to span attributes."""
    # Apply shared Snowflake context attributes
    _apply_metadata_to_span(span, metadata, _SNOWFLAKE_CONTEXT_ATTR_MAP)
    # Apply Cortex Analyst specific attributes
    _apply_metadata_to_span(span, metadata, _CORTEX_ANALYST_ATTR_MAP)


def _sanitize_span_attribute_value(value: Any) -> Any | None:
    """Cast arbitrary invocation attribute values to OTEL-compatible types."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        sanitized_items: list[Any] = []
        for item in value:
            sanitized = _sanitize_span_attribute_value(item)
            if sanitized is None:
                continue
            if isinstance(sanitized, list):
                sanitized_items.append(str(sanitized))
            else:
                sanitized_items.append(sanitized)
        return sanitized_items
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=str)
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _apply_gen_ai_semconv_attributes(
    span: Span,
    attributes: dict[str, Any] | None,
) -> None:
    if not attributes:
        return
    for key, value in attributes.items():
        sanitized = _sanitize_span_attribute_value(value)
        if sanitized is None:
            continue
        try:
            span.set_attribute(key, sanitized)
        except Exception:  # pragma: no cover - defensive
            pass


def _apply_sampled_for_evaluation(
    span: Span,
    is_sampled: bool,
) -> None:
    span.set_attribute("gen_ai.evaluation.sampled", is_sampled)


class SpanEmitter(EmitterMeta):
    """Span-focused emitter supporting optional content capture.

    Original implementation migrated from generators/span_emitter.py. Additional telemetry
    (metrics, content events) are handled by separate emitters composed via CompositeEmitter.
    """

    role = "span"
    name = "semconv_span"

    def __init__(self, tracer: Tracer | None = None, capture_content: bool = False):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content
        self._content_mode = ContentCapturingMode.NO_CONTENT

    def set_capture_content(self, value: bool):  # pragma: no cover - trivial mutator
        self._capture_content = value

    def set_content_mode(self, mode: ContentCapturingMode) -> None:  # pragma: no cover - trivial mutator
        self._content_mode = mode

    def handles(self, obj: object) -> bool:
        return True

    # ---- helpers ---------------------------------------------------------
    def _end_span_with_cleanup(self, entity: GenAIType) -> None:
        """Clean up context token and end the span.

        This is a DRY helper to avoid repeating the same cleanup pattern
        in every _finish_* and _error_* method.
        """
        token = entity.context_token
        if token is not None and hasattr(token, "__exit__"):
            try:
                token.__exit__(None, None, None)  # type: ignore[misc]
            except Exception:
                pass
        if entity.span is not None:
            entity.span.end()

    def _apply_start_attrs(self, invocation: GenAIType):
        span = getattr(invocation, "span", None)
        if span is None:
            return
        semconv_attrs = dict(invocation.semantic_convention_attributes())
        if isinstance(invocation, ToolCall):
            enum_val = getattr(GenAI.GenAiOperationNameValues, "EXECUTE_TOOL", None)
            semconv_attrs[GenAI.GEN_AI_OPERATION_NAME] = enum_val.value if enum_val else "execute_tool"
            # Add tool-specific semantic convention attributes
            if invocation.name:
                span.set_attribute("gen_ai.tool.name", invocation.name)
            if invocation.id:
                span.set_attribute("gen_ai.tool.call.id", invocation.id)
            # Tool type defaults to "function" for LangChain tools
            span.set_attribute("gen_ai.tool.type", "function")
            # Tool arguments (from attributes dict if available)
            if self._capture_content:
                tool_args = invocation.attributes.get("tool.arguments")
                if tool_args:
                    span.set_attribute("gen_ai.tool.call.arguments", tool_args)
        elif isinstance(invocation, EmbeddingInvocation) or isinstance(invocation, LLMInvocation):
            semconv_attrs.setdefault(GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model)
        _apply_gen_ai_semconv_attributes(span, semconv_attrs)
        supplemental = getattr(invocation, "attributes", None)
        if supplemental:
            semconv_subset = filter_semconv_gen_ai_attributes(supplemental, extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS)
            if semconv_subset:
                _apply_gen_ai_semconv_attributes(span, semconv_subset)
            for key, value in supplemental.items():
                if key in (semconv_subset or {}):
                    continue
                if key in _SPAN_BLOCKED_SUPPLEMENTAL_KEYS:
                    continue
                if not key.startswith("custom_") and key not in _SPAN_ALLOWED_SUPPLEMENTAL_KEYS:
                    continue
                if key in span.attributes:  # type: ignore[attr-defined]
                    continue
                sanitized = _sanitize_span_attribute_value(value)
                if sanitized is None:
                    continue
                try:
                    span.set_attribute(key, sanitized)
                except Exception:  # pragma: no cover - defensive
                    pass
        provider = getattr(invocation, "provider", None)
        if provider:
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
        server_address = getattr(invocation, "server_address", None)
        if server_address:
            span.set_attribute(SERVER_ADDRESS, server_address)
        server_port = getattr(invocation, "server_port", None)
        if server_port:
            span.set_attribute(SERVER_PORT, server_port)
        # framework (named field)
        if isinstance(invocation, LLMInvocation) and invocation.framework:
            span.set_attribute("gen_ai.framework", invocation.framework)
        # function definitions (semantic conv derived from structured list)
        if isinstance(invocation, LLMInvocation):
            _apply_function_definitions(span, invocation.request_functions)
        # Agent context (already covered by semconv metadata on base fields)

    def _apply_finish_attrs(self, invocation: LLMInvocation | EmbeddingInvocation):
        span = getattr(invocation, "span", None)
        if span is None:
            return

        # Capture input messages and system instructions if enabled
        if self._capture_content and isinstance(invocation, LLMInvocation) and invocation.input_messages:
            # Extract and set system instructions separately
            system_instructions = _extract_system_instructions(invocation.input_messages)
            if system_instructions is not None:
                span.set_attribute(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions)

            # Serialize input messages (excluding system messages)
            serialized_in = _serialize_messages(invocation.input_messages, exclude_system=True)
            if serialized_in is not None:
                span.set_attribute(GEN_AI_INPUT_MESSAGES, serialized_in)

        # Finish-time semconv attributes (response + usage tokens + functions)
        if isinstance(invocation, LLMInvocation):
            _apply_llm_finish_semconv(span, invocation)
        _apply_gen_ai_semconv_attributes(span, invocation.semantic_convention_attributes())
        extra_attrs = filter_semconv_gen_ai_attributes(
            getattr(invocation, "attributes", None),
            extras=_SPAN_ALLOWED_SUPPLEMENTAL_KEYS,
        )
        if extra_attrs:
            _apply_gen_ai_semconv_attributes(span, extra_attrs)

        # Capture output messages if enabled
        if self._capture_content and isinstance(invocation, LLMInvocation) and invocation.output_messages:
            serialized = _serialize_messages(invocation.output_messages)
            if serialized is not None:
                span.set_attribute(GEN_AI_OUTPUT_MESSAGES, serialized)

    def _attach_span(
        self,
        invocation: GenAIType,
        span: Span,
        context_manager: Any,
    ) -> None:
        invocation.span = span  # type: ignore[assignment]
        invocation.context_token = context_manager  # type: ignore[assignment]
        store_span_context(invocation, extract_span_context(span))

    # ---- lifecycle -------------------------------------------------------
    def on_start(self, invocation: LLMInvocation | EmbeddingInvocation) -> None:  # type: ignore[override]
        # Handle new agentic types
        if isinstance(invocation, Workflow):
            self._start_workflow(invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._start_agent(invocation)
        elif isinstance(invocation, Step):
            self._start_step(invocation)
        # Handle existing types
        elif isinstance(invocation, ToolCall):
            span_name = f"tool {invocation.name}"
            parent_span = getattr(invocation, "parent_span", None)
            parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
            cm = self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                end_on_exit=False,
                context=parent_ctx,
            )
            span = cm.__enter__()
            self._attach_span(invocation, span, cm)
            self._apply_start_attrs(invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._start_embedding(invocation)
        else:
            # Use operation field for span name (defaults to "chat")
            operation = getattr(invocation, "operation", "chat")
            model_name = invocation.request_model
            span_name = f"{operation} {model_name}"
            parent_span = getattr(invocation, "parent_span", None)
            parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
            cm = self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                end_on_exit=False,
                context=parent_ctx,
            )
            span = cm.__enter__()
            self._attach_span(invocation, span, cm)
            self._apply_start_attrs(invocation)

    def on_end(self, invocation: LLMInvocation | EmbeddingInvocation) -> None:
        _apply_sampled_for_evaluation(invocation.span, invocation.sample_for_evaluation)  # type: ignore[override]
        if isinstance(invocation, Workflow):
            self._finish_workflow(invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._finish_agent(invocation)
        elif isinstance(invocation, Step):
            self._finish_step(invocation)
        elif isinstance(invocation, ToolCall):
            self._finish_tool_call(invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._finish_embedding(invocation)
        else:
            span = getattr(invocation, "span", None)
            if span is None:
                return
            self._apply_finish_attrs(invocation)
            token = getattr(invocation, "context_token", None)
            if token is not None and hasattr(token, "__exit__"):
                try:  # pragma: no cover
                    token.__exit__(None, None, None)  # type: ignore[misc]
                except Exception:  # pragma: no cover
                    pass
            span.end()

    def on_error(self, error: Error, invocation: LLMInvocation | EmbeddingInvocation) -> None:  # type: ignore[override]
        if isinstance(invocation, Workflow):
            self._error_workflow(error, invocation)
        elif isinstance(invocation, (AgentCreation, AgentInvocation)):
            self._error_agent(error, invocation)
        elif isinstance(invocation, Step):
            self._error_step(error, invocation)
        elif isinstance(invocation, ToolCall):
            self._error_tool_call(error, invocation)
        elif isinstance(invocation, EmbeddingInvocation):
            self._error_embedding(error, invocation)
        else:
            span = getattr(invocation, "span", None)
            if span is None:
                return
            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
            self._apply_finish_attrs(invocation)
            token = getattr(invocation, "context_token", None)
            if token is not None and hasattr(token, "__exit__"):
                try:  # pragma: no cover
                    token.__exit__(None, None, None)  # type: ignore[misc]
                except Exception:  # pragma: no cover
                    pass
            span.end()

    # ---- Workflow lifecycle ----------------------------------------------
    def _start_workflow(self, workflow: Workflow) -> None:
        """Start a workflow span."""
        span_name = f"workflow {workflow.name}"
        parent_span = getattr(workflow, "parent_span", None)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
        cm = self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            end_on_exit=False,
            context=parent_ctx,
        )
        span = cm.__enter__()
        self._attach_span(workflow, span, cm)

        # Set workflow attributes
        # TODO: Align to enum when semconvs is updated.
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "invoke_workflow")
        span.set_attribute(GEN_AI_WORKFLOW_NAME, workflow.name)
        if workflow.workflow_type:
            span.set_attribute(GEN_AI_WORKFLOW_TYPE, workflow.workflow_type)
        if workflow.description:
            span.set_attribute(GEN_AI_WORKFLOW_DESCRIPTION, workflow.description)
        if workflow.framework:
            span.set_attribute("gen_ai.framework", workflow.framework)
        if workflow.initial_input and self._capture_content:
            input_msg = {
                "role": "user",
                "parts": [{"type": "text", "content": workflow.initial_input}],
            }
            span.set_attribute("gen_ai.input.messages", json.dumps([input_msg]))
        _apply_gen_ai_semconv_attributes(span, workflow.semantic_convention_attributes())

    def _finish_workflow(self, workflow: Workflow) -> None:
        """Finish a workflow span."""
        span = workflow.span
        if span is None:
            return
        # Set final output if capture_content enabled
        if workflow.final_output and self._capture_content:
            output_msg = {
                "role": "assistant",
                "parts": [{"type": "text", "content": workflow.final_output}],
                "finish_reason": "stop",
            }
            span.set_attribute("gen_ai.output.messages", json.dumps([output_msg]))
        _apply_gen_ai_semconv_attributes(span, workflow.semantic_convention_attributes())
        self._end_span_with_cleanup(workflow)

    def _error_workflow(self, error: Error, workflow: Workflow) -> None:
        """Fail a workflow span with error status."""
        span = workflow.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
        _apply_gen_ai_semconv_attributes(span, workflow.semantic_convention_attributes())
        self._end_span_with_cleanup(workflow)

    # ---- Agent lifecycle -------------------------------------------------
    def _start_agent(self, agent: AgentCreation | AgentInvocation) -> None:
        """Start an agent span (create or invoke)."""
        # Span name per semantic conventions
        if agent.operation == "create_agent":
            span_name = f"create_agent {agent.name}"
        else:
            span_name = f"invoke_agent {agent.name}"

        parent_span = getattr(agent, "parent_span", None)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
        cm = self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            end_on_exit=False,
            context=parent_ctx,
        )
        span = cm.__enter__()
        self._attach_span(agent, span, cm)

        # Required attributes per semantic conventions
        # Set operation name based on agent operation (create or invoke)
        semconv_attrs = dict(agent.semantic_convention_attributes())
        semconv_attrs.setdefault(GEN_AI_AGENT_NAME, agent.name)
        semconv_attrs.setdefault(GEN_AI_AGENT_ID, str(agent.run_id))
        _apply_gen_ai_semconv_attributes(span, semconv_attrs)

        # Optional attributes
        if agent.agent_type:
            span.set_attribute(GEN_AI_AGENT_TYPE, agent.agent_type)
        if agent.description:
            span.set_attribute(GEN_AI_AGENT_DESCRIPTION, agent.description)
        if agent.framework:
            span.set_attribute("gen_ai.framework", agent.framework)
        if agent.tools:
            span.set_attribute(GEN_AI_AGENT_TOOLS, agent.tools)
        if agent.system_instructions and self._capture_content:
            system_parts = [{"type": "text", "content": agent.system_instructions}]
            span.set_attribute(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(system_parts))
        if isinstance(agent, AgentInvocation) and agent.input_context and self._capture_content:
            input_msg = {
                "role": "user",
                "parts": [{"type": "text", "content": agent.input_context}],
            }
            span.set_attribute("gen_ai.input.messages", json.dumps([input_msg]))
        _apply_gen_ai_semconv_attributes(span, agent.semantic_convention_attributes())

    def _finish_agent(self, agent: AgentCreation | AgentInvocation) -> None:
        """Finish an agent span."""
        span = agent.span
        if span is None:
            return
        # Set output result if capture_content enabled
        if isinstance(agent, AgentInvocation) and agent.output_result and self._capture_content:
            output_msg = {
                "role": "assistant",
                "parts": [{"type": "text", "content": agent.output_result}],
                "finish_reason": "stop",
            }
            span.set_attribute("gen_ai.output.messages", json.dumps([output_msg]))
        _apply_gen_ai_semconv_attributes(span, agent.semantic_convention_attributes())
        self._end_span_with_cleanup(agent)

    def _error_agent(self, error: Error, agent: AgentCreation | AgentInvocation) -> None:
        """Fail an agent span with error status."""
        span = agent.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
        _apply_gen_ai_semconv_attributes(span, agent.semantic_convention_attributes())
        self._end_span_with_cleanup(agent)

    # ---- Step lifecycle --------------------------------------------------
    def _start_step(self, step: Step) -> None:
        """Start a step span."""
        span_name = f"step {step.name}"
        parent_span = getattr(step, "parent_span", None)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
        cm = self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            end_on_exit=False,
            context=parent_ctx,
        )
        span = cm.__enter__()
        self._attach_span(step, span, cm)

        # Set step attributes
        span.set_attribute(GEN_AI_STEP_NAME, step.name)
        if step.step_type:
            span.set_attribute(GEN_AI_STEP_TYPE, step.step_type)
        if step.objective:
            span.set_attribute(GEN_AI_STEP_OBJECTIVE, step.objective)
        if step.source:
            span.set_attribute(GEN_AI_STEP_SOURCE, step.source)
        if step.assigned_agent:
            span.set_attribute(GEN_AI_STEP_ASSIGNED_AGENT, step.assigned_agent)
        if step.status:
            span.set_attribute(GEN_AI_STEP_STATUS, step.status)
        if step.input_data and self._capture_content:
            input_msg = {
                "role": "user",
                "parts": [{"type": "text", "content": step.input_data}],
            }
            span.set_attribute("gen_ai.input.messages", json.dumps([input_msg]))
        _apply_gen_ai_semconv_attributes(span, step.semantic_convention_attributes())

    def _finish_step(self, step: Step) -> None:
        """Finish a step span."""
        span = step.span
        if span is None:
            return
        # Set output data if capture_content enabled
        if step.output_data and self._capture_content:
            output_msg = {
                "role": "assistant",
                "parts": [{"type": "text", "content": step.output_data}],
                "finish_reason": "stop",
            }
            span.set_attribute("gen_ai.output.messages", json.dumps([output_msg]))
        # Update status if changed
        if step.status:
            span.set_attribute(GEN_AI_STEP_STATUS, step.status)
        _apply_gen_ai_semconv_attributes(span, step.semantic_convention_attributes())
        self._end_span_with_cleanup(step)

    def _finish_tool_call(self, tool: ToolCall) -> None:
        """Finish a tool call span with result attributes.

        Also extracts and applies Snowflake Cortex provider-specific
        attributes when the tool response contains search/analyst metadata.
        """
        span = tool.span
        if span is None:
            return

        tool_response = tool.attributes.get("tool.response")

        # Add tool result if capture_content enabled and response available
        if self._capture_content and tool_response:
            span.set_attribute("gen_ai.tool.call.result", tool_response)

        # Extract and apply Snowflake Cortex metadata if present
        if tool_response and isinstance(tool_response, str):
            # Apply Cortex Search metadata
            search_metadata = _extract_snowflake_search_metadata(tool_response)
            if search_metadata:
                _apply_cortex_search_metadata(span, search_metadata)

            # Apply Cortex Analyst metadata
            analyst_metadata = _extract_snowflake_analyst_metadata(tool_response)
            if analyst_metadata:
                _apply_cortex_analyst_metadata(span, analyst_metadata)

        # Apply any semconv attributes from the tool
        _apply_gen_ai_semconv_attributes(span, tool.semantic_convention_attributes())
        self._end_span_with_cleanup(tool)

    def _error_tool_call(self, error: Error, tool: ToolCall) -> None:
        """Fail a tool call span with error status."""
        span = tool.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
        _apply_gen_ai_semconv_attributes(span, tool.semantic_convention_attributes())
        self._end_span_with_cleanup(tool)

    def _error_step(self, error: Error, step: Step) -> None:
        """Fail a step span with error status."""
        span = step.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
        # Update status to failed
        span.set_attribute(GEN_AI_STEP_STATUS, "failed")
        _apply_gen_ai_semconv_attributes(span, step.semantic_convention_attributes())
        self._end_span_with_cleanup(step)

    # ---- Embedding lifecycle ---------------------------------------------
    def _start_embedding(self, embedding: EmbeddingInvocation) -> None:
        """Start an embedding span."""
        span_name = f"{embedding.operation_name} {embedding.request_model}"
        parent_span = getattr(embedding, "parent_span", None)
        parent_ctx = trace.set_span_in_context(parent_span) if parent_span is not None else None
        cm = self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            end_on_exit=False,
            context=parent_ctx,
        )
        span = cm.__enter__()
        self._attach_span(embedding, span, cm)
        self._apply_start_attrs(embedding)

        # Set embedding-specific start attributes
        if embedding.encoding_formats:
            span.set_attribute(GEN_AI_REQUEST_ENCODING_FORMATS, embedding.encoding_formats)
        if self._capture_content and embedding.input_texts:
            # Capture input texts as array attribute
            span.set_attribute(GEN_AI_EMBEDDINGS_INPUT_TEXTS, embedding.input_texts)

    def _finish_embedding(self, embedding: EmbeddingInvocation) -> None:
        """Finish an embedding span."""
        span = embedding.span
        if span is None:
            return
        # Apply finish-time semantic conventions
        if embedding.dimension_count:
            span.set_attribute(GEN_AI_EMBEDDINGS_DIMENSION_COUNT, embedding.dimension_count)
        if embedding.input_tokens is not None:
            span.set_attribute(GenAI.GEN_AI_USAGE_INPUT_TOKENS, embedding.input_tokens)
        self._end_span_with_cleanup(embedding)

    def _error_embedding(self, error: Error, embedding: EmbeddingInvocation) -> None:
        """Fail an embedding span with error status."""
        span = embedding.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)
        # Set error type from invocation if available
        if embedding.error_type:
            span.set_attribute(ErrorAttributes.ERROR_TYPE, embedding.error_type)
        self._end_span_with_cleanup(embedding)
