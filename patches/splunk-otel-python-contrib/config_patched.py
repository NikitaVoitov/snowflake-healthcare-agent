from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .attributes import (
    SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE_DEFAULT,
    SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES_DEFAULT,
    SNOWFLAKE_CREDIT_PRICE_USD_DEFAULT,
)

# Note: Using modern dict type hint instead of Dict from typing
from .emitters.spec import CategoryOverride
from .environment_variables import (
    OTEL_GENAI_EVALUATION_EVENT_LEGACY,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE,
    OTEL_SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE,
    OTEL_SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES,
    OTEL_SNOWFLAKE_CREDIT_PRICE_USD,
)
from .types import ContentCapturingMode
from .utils import get_content_capturing_mode

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Configuration for GenAI emitters derived from environment variables."""

    enable_span: bool
    enable_metrics: bool
    enable_content_events: bool
    extra_emitters: list[str]
    only_traceloop_compat: bool
    raw_tokens: list[str]
    capture_messages_mode: ContentCapturingMode
    capture_messages_override: bool
    legacy_capture_request: bool
    emit_legacy_evaluation_event: bool
    category_overrides: dict[str, CategoryOverride]
    evaluation_sample_rate: float = 1.0
    # Snowflake Cortex pricing configuration
    snowflake_cortex_analyst_credits_per_message: float = SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE_DEFAULT
    snowflake_cortex_search_credits_per_1000_queries: float = SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES_DEFAULT
    snowflake_credit_price_usd: float = SNOWFLAKE_CREDIT_PRICE_USD_DEFAULT

    @property
    def snowflake_cortex_search_credits_per_query(self) -> float:
        """Credits per individual Cortex Search query."""
        return self.snowflake_cortex_search_credits_per_1000_queries / 1000.0

    def estimate_cortex_analyst_cost_usd(self, message_count: int) -> float:
        """Estimate cost in USD for Cortex Analyst messages."""
        credits = message_count * self.snowflake_cortex_analyst_credits_per_message
        return credits * self.snowflake_credit_price_usd

    def estimate_cortex_search_cost_usd(self, query_count: int) -> float:
        """Estimate cost in USD for Cortex Search queries."""
        credits = query_count * self.snowflake_cortex_search_credits_per_query
        return credits * self.snowflake_credit_price_usd


def parse_env() -> Settings:
    """Parse emitter-related environment variables into structured settings."""

    raw_val = os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS, "span")
    tokens = [token.strip().lower() for token in raw_val.split(",") if token.strip()]
    if not tokens:
        tokens = ["span"]

    baseline_map = {
        "span": (True, False, False),
        "span_metric": (True, True, False),
        "span_metric_event": (True, True, True),
    }

    baseline = next((token for token in tokens if token in baseline_map), None)
    extra_emitters: list[str] = []
    only_traceloop_compat = False

    if baseline is None:
        if tokens == ["traceloop_compat"]:
            baseline = "span"
            extra_emitters = ["traceloop_compat"]
            only_traceloop_compat = True
        else:
            baseline = "span"
            extra_emitters = [token for token in tokens if token not in baseline_map]
    else:
        extra_emitters = [token for token in tokens if token != baseline]

    enable_span, enable_metrics, enable_content_events = baseline_map.get(baseline, (True, False, False))

    capture_messages_override = any(
        env is not None
        for env in (
            os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT),
            os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE),
        )
    )
    capture_mode = get_content_capturing_mode()

    # Legacy flag removed: always False now
    legacy_capture_request = False

    overrides: dict[str, CategoryOverride] = {}
    override_env_map = {
        "span": os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN, ""),
        "metrics": os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS_METRICS, ""),
        "content_events": os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS_CONTENT_EVENTS, ""),
        "evaluation": os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION, ""),
    }
    for category, raw in override_env_map.items():
        override = _parse_category_override(category, raw)
        if override is not None:
            overrides[category] = override

    legacy_event_flag = os.environ.get(OTEL_GENAI_EVALUATION_EVENT_LEGACY, "").strip()
    emit_legacy_event = legacy_event_flag.lower() in {"1", "true", "yes"}

    evaluation_sample_rate = os.environ.get(OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE)
    if evaluation_sample_rate is None or evaluation_sample_rate.strip() == "":
        evaluation_sample_rate = 1.0
    try:
        evaluation_sample_rate = float(evaluation_sample_rate)
    except ValueError:
        evaluation_sample_rate = 1.0
    if evaluation_sample_rate < 0.0:
        evaluation_sample_rate = 0.0
    if evaluation_sample_rate > 1.0:
        evaluation_sample_rate = 1.0

    # Parse Snowflake Cortex pricing configuration
    cortex_analyst_rate = _parse_float_env(
        OTEL_SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE,
        SNOWFLAKE_CORTEX_ANALYST_CREDITS_PER_MESSAGE_DEFAULT,
    )
    cortex_search_rate = _parse_float_env(
        OTEL_SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES,
        SNOWFLAKE_CORTEX_SEARCH_CREDITS_PER_1000_QUERIES_DEFAULT,
    )
    credit_price_usd = _parse_float_env(
        OTEL_SNOWFLAKE_CREDIT_PRICE_USD,
        SNOWFLAKE_CREDIT_PRICE_USD_DEFAULT,
    )

    return Settings(
        enable_span=enable_span,
        enable_metrics=enable_metrics,
        enable_content_events=enable_content_events,
        extra_emitters=extra_emitters,
        only_traceloop_compat=only_traceloop_compat,
        raw_tokens=tokens,
        capture_messages_mode=capture_mode,
        capture_messages_override=capture_messages_override,
        legacy_capture_request=legacy_capture_request,
        emit_legacy_evaluation_event=emit_legacy_event,
        category_overrides=overrides,
        evaluation_sample_rate=evaluation_sample_rate,
        snowflake_cortex_analyst_credits_per_message=cortex_analyst_rate,
        snowflake_cortex_search_credits_per_1000_queries=cortex_search_rate,
        snowflake_credit_price_usd=credit_price_usd,
    )


def _parse_float_env(env_var: str, default: float) -> float:
    """Parse a positive float from environment variable with fallback to default."""
    value = os.environ.get(env_var)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
        if parsed < 0:
            _logger.warning(
                "Negative value for %s: %s, using default %s",
                env_var,
                value,
                default,
            )
            return default
        return parsed
    except ValueError:
        _logger.warning("Invalid float for %s: %s, using default %s", env_var, value, default)
        return default


def _parse_category_override(category: str, raw: str) -> CategoryOverride | None:  # pragma: no cover - thin parsing
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    directive = None
    remainder = text
    if ":" in text:
        prefix, remainder = text.split(":", 1)
        directive = prefix.strip().lower()
    names = [name.strip() for name in remainder.split(",") if name.strip()]
    mode_map = {
        None: "append",
        "append": "append",
        "prepend": "prepend",
        "replace": "replace-category",
        "replace-category": "replace-category",
        "replace-same-name": "replace-same-name",
    }
    mode = mode_map.get(directive)
    if mode is None:
        if directive:
            _logger.warning(
                "Unknown emitter override directive '%s' for category '%s'",
                directive,
                category,
            )
        mode = "append"
    if mode != "replace-category" and not names:
        return None
    return CategoryOverride(mode=mode, emitter_names=tuple(names))


__all__ = ["Settings", "parse_env"]
