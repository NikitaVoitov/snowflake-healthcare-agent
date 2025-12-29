from __future__ import annotations

import re
from typing import Any

from opentelemetry import trace
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

from ..attributes import (
    SNOWFLAKE_CORTEX_SEARCH_SOURCES,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_WAREHOUSE,
)
from ..config import parse_env
from ..instruments import Instruments
from ..interfaces import EmitterMeta
from ..types import (
    AgentInvocation,
    EmbeddingInvocation,
    Error,
    LLMInvocation,
    ToolCall,
    Workflow,
)
from .utils import (
    _get_metric_attributes,
    _record_duration,
    _record_token_metrics,
)


def _extract_search_score_from_tool_response(
    tool_response: str | None,
) -> dict[str, Any] | None:
    """
    Extract search score metadata from tool response.

    Looks for patterns like:
    - @top_cosine_similarity: 0.XXX
    - @scores: {cosine=X.XXX, text=X.XXX}
    - @sources: ["faqs", "policies"]
    - @request_ids: ["xxx"]

    Returns a dict with extracted metadata or None.
    """
    if not tool_response:
        return None

    result: dict[str, Any] = {}

    # Try to extract top_cosine_similarity
    score_match = re.search(r"@top_cosine_similarity:\s*([\d.]+)", tool_response)
    if score_match:
        try:
            result["top_score"] = float(score_match.group(1))
        except (ValueError, TypeError):
            pass

    # Try to extract from @scores patterns
    if "top_score" not in result:
        scores_match = re.search(r"cosine[=:]?\s*([\d.]+)", tool_response)
        if scores_match:
            try:
                result["top_score"] = float(scores_match.group(1))
            except (ValueError, TypeError):
                pass

    # Try to extract sources
    sources_match = re.search(r"@sources:\s*\[([^\]]+)\]", tool_response)
    if sources_match:
        # Parse the list items
        sources_str = sources_match.group(1)
        sources = [s.strip().strip("'\"") for s in sources_str.split(",")]
        result["sources"] = sources

    # Try to extract request_ids
    request_match = re.search(r"@request_ids:\s*\[([^\]]+)\]", tool_response)
    if request_match:
        request_str = request_match.group(1)
        request_ids = [r.strip().strip("'\"") for r in request_str.split(",")]
        result["request_ids"] = request_ids

    return result if result else None


class MetricsEmitter(EmitterMeta):
    """Emits GenAI metrics (duration + token usage).

    Supports LLMInvocation, EmbeddingInvocation, ToolCall, Workflow, and Agent.
    """

    role = "metric"
    name = "semconv_metrics"

    def __init__(self, meter: Meter | None = None):
        _meter: Meter = meter or get_meter(__name__)
        instruments = Instruments(_meter)
        self._duration_histogram: Histogram = instruments.operation_duration_histogram
        self._token_histogram: Histogram = instruments.token_usage_histogram
        self._workflow_duration_histogram: Histogram = instruments.workflow_duration_histogram
        self._agent_duration_histogram: Histogram = instruments.agent_duration_histogram
        # New: Search score histogram for tool operations
        self._search_score_histogram: Histogram = instruments.tool_search_score_histogram
        # Snowflake Cortex cost tracking counters
        self._cortex_analyst_message_counter = instruments.cortex_analyst_message_counter
        self._cortex_search_query_counter = instruments.cortex_search_query_counter
        self._cortex_credits_counter = instruments.cortex_credits_counter
        self._cortex_cost_counter = instruments.cortex_cost_counter
        # Load pricing config from environment
        self._settings = parse_env()

    def on_start(self, obj: Any) -> None:  # no-op for metrics
        return None

    def on_end(self, obj: Any) -> None:
        if isinstance(obj, Workflow):
            self._record_workflow_metrics(obj)
            return
        if isinstance(obj, AgentInvocation):
            self._record_agent_metrics(obj)
            return
        # Step metrics removed

        if isinstance(obj, LLMInvocation):
            llm_invocation = obj
            metric_attrs = _get_metric_attributes(
                llm_invocation.request_model,
                llm_invocation.response_model_name,
                llm_invocation.operation,
                llm_invocation.provider,
                llm_invocation.framework,
                server_address=llm_invocation.server_address,
                server_port=llm_invocation.server_port,
            )
            # Add agent context if available
            if llm_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = llm_invocation.agent_name
            if llm_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = llm_invocation.agent_id

            _record_token_metrics(
                self._token_histogram,
                llm_invocation.input_tokens,
                llm_invocation.output_tokens,
                metric_attrs,
                span=getattr(llm_invocation, "span", None),
            )
            _record_duration(
                self._duration_histogram,
                llm_invocation,
                metric_attrs,
                span=getattr(llm_invocation, "span", None),
            )
            return
        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                tool_invocation.name,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
            # Add agent context if available
            if tool_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool_invocation.agent_name
            if tool_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool_invocation.agent_id

            _record_duration(
                self._duration_histogram,
                tool_invocation,
                metric_attrs,
                span=getattr(tool_invocation, "span", None),
            )

            # Record search score metric if available in tool response
            self._record_search_score_metric(tool_invocation)

            # Record Cortex Analyst cost metrics if applicable
            tool_response = tool_invocation.attributes.get("tool.response", "")
            if tool_response and "--- Cortex Analyst Metadata ---" in tool_response:
                context = None
                span = getattr(tool_invocation, "span", None)
                if span is not None:
                    try:
                        context = trace.set_span_in_context(span)
                    except (ValueError, RuntimeError):
                        context = None
                self._record_cortex_analyst_cost(tool_invocation, tool_response, context)

        if isinstance(obj, EmbeddingInvocation):
            embedding_invocation = obj
            metric_attrs = _get_metric_attributes(
                embedding_invocation.request_model,
                None,
                embedding_invocation.operation_name,
                embedding_invocation.provider,
                embedding_invocation.framework,
                server_address=embedding_invocation.server_address,
                server_port=embedding_invocation.server_port,
            )
            # Add agent context if available
            if embedding_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = embedding_invocation.agent_name
            if embedding_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = embedding_invocation.agent_id

            _record_duration(
                self._duration_histogram,
                embedding_invocation,
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )

    def on_error(self, error: Error, obj: Any) -> None:
        # Handle new agentic types
        if isinstance(obj, Workflow):
            self._record_workflow_metrics(obj)
            return
        if isinstance(obj, AgentInvocation):
            self._record_agent_metrics(obj)
            return
        # Step metrics removed

        # Handle existing types with agent context
        if isinstance(obj, LLMInvocation):
            llm_invocation = obj
            metric_attrs = _get_metric_attributes(
                llm_invocation.request_model,
                llm_invocation.response_model_name,
                llm_invocation.operation,
                llm_invocation.provider,
                llm_invocation.framework,
                server_address=llm_invocation.server_address,
                server_port=llm_invocation.server_port,
            )
            # Add agent context if available
            if llm_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = llm_invocation.agent_name
            if llm_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = llm_invocation.agent_id
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

            _record_duration(self._duration_histogram, llm_invocation, metric_attrs)
            return
        if isinstance(obj, ToolCall):
            tool_invocation = obj
            metric_attrs = _get_metric_attributes(
                tool_invocation.name,
                None,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
                tool_invocation.provider,
                tool_invocation.framework,
            )
            # Add agent context if available
            if tool_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool_invocation.agent_name
            if tool_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = tool_invocation.agent_id
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

            _record_duration(
                self._duration_histogram,
                tool_invocation,
                metric_attrs,
                span=getattr(tool_invocation, "span", None),
            )

        if isinstance(obj, EmbeddingInvocation):
            embedding_invocation = obj
            metric_attrs = _get_metric_attributes(
                embedding_invocation.request_model,
                None,
                embedding_invocation.operation_name,
                embedding_invocation.provider,
                embedding_invocation.framework,
                server_address=embedding_invocation.server_address,
                server_port=embedding_invocation.server_port,
            )
            # Add agent context if available
            if embedding_invocation.agent_name:
                metric_attrs[GenAI.GEN_AI_AGENT_NAME] = embedding_invocation.agent_name
            if embedding_invocation.agent_id:
                metric_attrs[GenAI.GEN_AI_AGENT_ID] = embedding_invocation.agent_id
            if getattr(error, "type", None) is not None:
                metric_attrs[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

            _record_duration(
                self._duration_histogram,
                embedding_invocation,
                metric_attrs,
                span=getattr(embedding_invocation, "span", None),
            )

    def handles(self, obj: Any) -> bool:
        return isinstance(
            obj,
            (
                LLMInvocation,
                ToolCall,
                Workflow,
                AgentInvocation,
                EmbeddingInvocation,
            ),
        )

    # Helper methods for new agentic types
    def _record_workflow_metrics(self, workflow: Workflow) -> None:
        """Record metrics for a workflow."""
        if workflow.end_time is None:
            return
        duration = workflow.end_time - workflow.start_time
        metric_attrs = {
            "gen_ai.workflow.name": workflow.name,
        }
        if workflow.workflow_type:
            metric_attrs["gen_ai.workflow.type"] = workflow.workflow_type
        if workflow.framework:
            metric_attrs["gen_ai.framework"] = workflow.framework

        context = None
        span = getattr(workflow, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        self._workflow_duration_histogram.record(duration, attributes=metric_attrs, context=context)

    def _record_agent_metrics(self, agent: AgentInvocation) -> None:
        """Record metrics for an agent operation."""
        if agent.end_time is None:
            return
        duration = agent.end_time - agent.start_time
        metric_attrs = {
            GenAI.GEN_AI_OPERATION_NAME: agent.operation,
            GenAI.GEN_AI_AGENT_NAME: agent.name,
            GenAI.GEN_AI_AGENT_ID: str(agent.run_id),
        }
        if agent.agent_type:
            metric_attrs["gen_ai.agent.type"] = agent.agent_type
        if agent.framework:
            metric_attrs["gen_ai.framework"] = agent.framework

        context = None
        span = getattr(agent, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                context = None

        self._agent_duration_histogram.record(duration, attributes=metric_attrs, context=context)

    def _record_search_score_metric(self, tool: ToolCall) -> None:
        """Record search score metric for search/retrieval tools.

        Extracts score information from the tool response and records
        a histogram metric with relevant dimensions.
        """
        tool_response = tool.attributes.get("tool.response")
        if not tool_response:
            return

        # Extract search metadata from response
        search_metadata = _extract_search_score_from_tool_response(tool_response)
        if not search_metadata or "top_score" not in search_metadata:
            return

        top_score = search_metadata["top_score"]

        # Build metric attributes with dimensions
        metric_attrs: dict[str, Any] = {
            "gen_ai.tool.name": tool.name,
        }

        # Add provider if available
        if tool.provider:
            metric_attrs[GenAI.GEN_AI_PROVIDER_NAME] = tool.provider

        # Add source if available (for Snowflake Cortex Search)
        sources = search_metadata.get("sources", [])
        if sources:
            # Use first source for dimension (or join if multiple)
            metric_attrs[SNOWFLAKE_CORTEX_SEARCH_SOURCES] = ",".join(sources)

        # Add agent context if available
        if tool.agent_name:
            metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool.agent_name

        # Get span context for metric
        context = None
        span = getattr(tool, "span", None)
        if span is not None:
            try:
                context = trace.set_span_in_context(span)
            except (ValueError, RuntimeError):  # pragma: no cover
                context = None

        # Record the score histogram
        self._search_score_histogram.record(top_score, attributes=metric_attrs, context=context)

        # Record Cortex Search cost metrics (if this is a Cortex Search tool)
        if "--- Cortex Search Metadata ---" in tool_response:
            self._record_cortex_search_cost(tool, tool_response, context)

    def _record_cortex_analyst_cost(
        self, tool: ToolCall, tool_response: str, context: Any
    ) -> None:
        """Record cost metrics for Snowflake Cortex Analyst tool calls.

        Cortex Analyst charges per message (0.067 credits/message by default).
        Each successful tool response = 1 message.
        """
        # Build metric attributes
        metric_attrs: dict[str, Any] = {
            "gen_ai.tool.name": tool.name,
            GenAI.GEN_AI_PROVIDER_NAME: "snowflake",
            "snowflake.service": "cortex_analyst",
        }

        # Extract database/warehouse from response for dimensions
        # Use word-boundary pattern to avoid capturing trailing metadata
        db_match = re.search(r"@snowflake\.database:\s*([A-Za-z0-9_]+)", tool_response)
        if db_match:
            metric_attrs[SNOWFLAKE_DATABASE] = db_match.group(1)

        wh_match = re.search(r"@snowflake\.warehouse:\s*([A-Za-z0-9_]+)", tool_response)
        if wh_match:
            metric_attrs[SNOWFLAKE_WAREHOUSE] = wh_match.group(1)

        if tool.agent_name:
            metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool.agent_name

        # Record message count (1 per successful response)
        self._cortex_analyst_message_counter.add(1, attributes=metric_attrs, context=context)

        # Calculate and record credits/cost
        credits = self._settings.snowflake_cortex_analyst_credits_per_message
        cost_usd = credits * self._settings.snowflake_credit_price_usd

        self._cortex_credits_counter.add(credits, attributes=metric_attrs, context=context)
        self._cortex_cost_counter.add(cost_usd, attributes=metric_attrs, context=context)

    def _record_cortex_search_cost(
        self, tool: ToolCall, tool_response: str, context: Any
    ) -> None:
        """Record cost metrics for Snowflake Cortex Search tool calls.

        Cortex Search charges per query (1.7 credits/1000 queries by default).
        Each successful tool response = 1 query.
        """
        # Build metric attributes
        metric_attrs: dict[str, Any] = {
            "gen_ai.tool.name": tool.name,
            GenAI.GEN_AI_PROVIDER_NAME: "snowflake",
            "snowflake.service": "cortex_search",
        }

        # Extract database/warehouse from response for dimensions
        # Use word-boundary pattern to avoid capturing trailing metadata
        db_match = re.search(r"@snowflake\.database:\s*([A-Za-z0-9_]+)", tool_response)
        if db_match:
            metric_attrs[SNOWFLAKE_DATABASE] = db_match.group(1)

        wh_match = re.search(r"@snowflake\.warehouse:\s*([A-Za-z0-9_]+)", tool_response)
        if wh_match:
            metric_attrs[SNOWFLAKE_WAREHOUSE] = wh_match.group(1)

        if tool.agent_name:
            metric_attrs[GenAI.GEN_AI_AGENT_NAME] = tool.agent_name

        # Record query count (1 per successful response)
        self._cortex_search_query_counter.add(1, attributes=metric_attrs, context=context)

        # Calculate and record credits/cost
        credits = self._settings.snowflake_cortex_search_credits_per_query
        cost_usd = credits * self._settings.snowflake_credit_price_usd

        self._cortex_credits_counter.add(credits, attributes=metric_attrs, context=context)
        self._cortex_cost_counter.add(cost_usd, attributes=metric_attrs, context=context)
