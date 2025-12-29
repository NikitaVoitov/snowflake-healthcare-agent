"""OpenTelemetry setup for Healthcare Agent.

This module provides programmatic OpenTelemetry initialization for use with
langgraph dev mode and other scenarios where zero-code instrumentation
via `opentelemetry-instrument` is not available.

Note: Histogram bucket boundaries for gen_ai.tool.search.score are pre-configured
in splunk-otel-python-contrib using explicit_bucket_boundaries_advisory, so no
manual View configuration is required here.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_opentelemetry() -> bool:
    """Initialize OpenTelemetry with OTLP exporters.

    Call this at application startup before any LangChain imports.
    Safe to call multiple times (idempotent).

    Returns:
        True if instrumentation was enabled, False otherwise.
    """
    # Check if instrumentation should be enabled
    if not os.getenv("OTEL_TRACES_EXPORTER"):
        logger.debug("OTel disabled: OTEL_TRACES_EXPORTER not set")
        return False

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Check if already configured
        current_provider = trace.get_tracer_provider()
        if current_provider.__class__.__name__ != "ProxyTracerProvider":
            logger.debug("OTel already configured")
            return True

        # Build resource from environment
        service_name = os.getenv("OTEL_SERVICE_NAME", "healthcare-agent")
        resource_attrs = {
            "service.name": service_name,
            "service.version": "0.2.0",
        }

        # Parse OTEL_RESOURCE_ATTRIBUTES if present
        env_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if env_attrs:
            for attr in env_attrs.split(","):
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    resource_attrs[key.strip()] = value.strip()

        resource = Resource.create(resource_attrs)

        # Setup TracerProvider
        # Increased timeout to 30s to reduce DEADLINE_EXCEEDED errors over WAN
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(timeout=30)))
        trace.set_tracer_provider(tracer_provider)

        # Setup MeterProvider
        # Note: Histogram bucket boundaries for metrics like gen_ai.tool.search.score
        # are pre-configured in splunk-otel-python-contrib using
        # explicit_bucket_boundaries_advisory - no Views needed here.
        # Increased timeout to 30s to reduce DEADLINE_EXCEEDED errors over WAN
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(timeout=30),  # 30 second timeout (default is 10s)
            export_interval_millis=60000,    # Export every 60s instead of 30s
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # Instrument LangChain
        LangChainInstrumentor().instrument()

        logger.info(
            "OpenTelemetry initialized: service=%s, endpoint=%s",
            service_name,
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "default"),
        )
        return True

    except ImportError as e:
        logger.warning("OTel packages not installed: %s", e)
        return False
    except Exception as e:
        logger.error("OTel setup failed: %s", e)
        return False
