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
import socket
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)


def test_endpoint_connectivity(endpoint: str, timeout: float = 5.0) -> dict:
    """Test connectivity to an OTLP endpoint.

    Args:
        endpoint: Full URL like http://host:port or just host:port
        timeout: Connection timeout in seconds

    Returns:
        Dict with test results for each port
    """
    results = {}

    # Parse the endpoint
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        port = parsed.port
    else:
        # Assume host:port format
        parts = endpoint.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else None

    if not host:
        return {"error": "Could not parse host from endpoint"}

    # Test ports to check
    ports_to_test = [4317, 4318]  # gRPC and HTTP
    if port and port not in ports_to_test:
        ports_to_test.append(port)

    for test_port in ports_to_test:
        port_name = {4317: "gRPC", 4318: "HTTP"}.get(test_port, str(test_port))
        try:
            # Try to resolve DNS first
            logger.info("[OTel] Testing %s:%d (%s)...", host, test_port, port_name)

            # DNS resolution
            try:
                ip_addresses = socket.gethostbyname_ex(host)[2]
                logger.info("[OTel] DNS resolved %s -> %s", host, ip_addresses)
            except socket.gaierror as e:
                results[test_port] = {"status": "DNS_FAILED", "error": str(e)}
                logger.error("[OTel] DNS resolution failed for %s: %s", host, e)
                continue

            # TCP connection test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                result = sock.connect_ex((host, test_port))
                if result == 0:
                    results[test_port] = {"status": "REACHABLE", "ip": ip_addresses[0]}
                    logger.info("[OTel] ✓ Port %d (%s) is REACHABLE", test_port, port_name)
                else:
                    results[test_port] = {"status": "UNREACHABLE", "error_code": result}
                    logger.warning("[OTel] ✗ Port %d (%s) is UNREACHABLE (code=%d)", test_port, port_name, result)
            finally:
                sock.close()

        except TimeoutError:
            results[test_port] = {"status": "TIMEOUT"}
            logger.warning("[OTel] ✗ Port %d (%s) connection TIMEOUT", test_port, port_name)
        except Exception as e:
            results[test_port] = {"status": "ERROR", "error": str(e)}
            logger.error("[OTel] ✗ Port %d (%s) test failed: %s", test_port, port_name, e)

    return results


class LoggingSpanProcessor:
    """Custom span processor that logs span lifecycle for debugging."""

    def __init__(self, name: str = "OTel"):
        self.name = name
        self._span_count = 0

    def on_start(self, span: "ReadableSpan", parent_context=None) -> None:  # noqa: ARG002
        """Called when a span starts."""
        self._span_count += 1
        logger.info(
            "[%s] Span started #%d: %s (trace_id=%s)",
            self.name,
            self._span_count,
            span.name,
            format(span.context.trace_id, "032x") if span.context else "none",
        )

    def on_end(self, span: "ReadableSpan") -> None:
        """Called when a span ends."""
        logger.info(
            "[%s] Span ended: %s (duration=%dms)",
            self.name,
            span.name,
            (span.end_time - span.start_time) // 1_000_000 if span.end_time and span.start_time else 0,
        )

    def shutdown(self) -> None:
        logger.info("[%s] Shutdown - total spans processed: %d", self.name, self._span_count)

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True


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
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
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

        # Determine endpoint - use HTTP (port 4318) instead of gRPC (4317)
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

        # Test connectivity to the endpoint before configuring
        if endpoint:
            logger.info("[OTel] Testing connectivity to OTLP endpoint...")
            connectivity_results = test_endpoint_connectivity(endpoint)
            logger.info("[OTel] Connectivity test results: %s", connectivity_results)

        # Convert gRPC port to HTTP port if needed
        if ":4317" in endpoint:
            http_endpoint = endpoint.replace(":4317", ":4318")
        else:
            http_endpoint = endpoint

        # Setup TracerProvider
        # Increased timeout to 30s to reduce DEADLINE_EXCEEDED errors over WAN
        tracer_provider = TracerProvider(resource=resource)

        # Add logging processor first (for debugging)
        tracer_provider.add_span_processor(LoggingSpanProcessor("Healthcare"))

        # Add OTLP HTTP exporter with batch processor
        # HTTP exporter uses /v1/traces endpoint automatically
        traces_endpoint = f"{http_endpoint}/v1/traces" if http_endpoint else None
        span_exporter = OTLPSpanExporter(endpoint=traces_endpoint, timeout=30)
        batch_processor = BatchSpanProcessor(
            span_exporter,
            max_queue_size=2048,
            schedule_delay_millis=5000,  # Export every 5 seconds
            max_export_batch_size=512,
        )
        tracer_provider.add_span_processor(batch_processor)
        trace.set_tracer_provider(tracer_provider)

        logger.info(
            "OTLP HTTP SpanExporter configured: endpoint=%s",
            traces_endpoint or "default",
        )

        # Setup MeterProvider
        # Note: Histogram bucket boundaries for metrics like gen_ai.tool.search.score
        # are pre-configured in splunk-otel-python-contrib using
        # explicit_bucket_boundaries_advisory - no Views needed here.
        # Increased timeout to 30s to reduce DEADLINE_EXCEEDED errors over WAN
        metrics_endpoint = f"{http_endpoint}/v1/metrics" if http_endpoint else None
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=metrics_endpoint, timeout=30),
            export_interval_millis=60000,  # Export every 60s instead of 30s
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # Instrument LangChain globally (before any LangChain imports)
        LangChainInstrumentor().instrument()
        # Note: FastAPI instrumented via instrument_app() in main.py after app creation

        logger.info(
            "OpenTelemetry initialized: service=%s, http_endpoint=%s",
            service_name,
            http_endpoint or "default",
        )

        # Create a test span to verify export pipeline works
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("otel_setup_test") as test_span:
            test_span.set_attribute("test.purpose", "verify_export")
            test_span.set_attribute("service.name", service_name)
            logger.info("Test span created: trace_id=%s", format(test_span.context.trace_id, "032x"))

        return True

    except ImportError as e:
        logger.warning("OTel packages not installed: %s", e)
        return False
    except Exception as e:
        logger.error("OTel setup failed: %s", e)
        return False
