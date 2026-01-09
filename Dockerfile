# =============================================================================
# Healthcare Multi-Agent Lab - SPCS Container
#
# Method 1: Minimal Base Image - gcr.io/distroless/python3-debian12 (53MB)
# Method 2: Multistage Build - builder + runtime stages  
# Method 3: Minimize Layers - combined RUN/COPY commands
# Method 4: Caching - dependency files copied BEFORE install
# Method 5: Dockerignore - comprehensive exclusions (.dockerignore)
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install ONLY SPCS dependencies (no main deps)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create venv and install SPCS production deps directly (avoids langchain-snowflake PyPI conflict)
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        "langgraph>=1.0.0" \
        "langchain>=1.0.0" \
        "langchain-core>=0.3.75" \
        "langgraph-checkpoint-snowflake>=0.1.0" \
        "snowflake-snowpark-python>=1.14.0" \
        "snowflake-sqlalchemy>=1.6.0" \
        "fastapi>=0.115.0" \
        "uvicorn>=0.32.0" \
        "aiohttp>=3.9.0" \
        "PyJWT>=2.8.0" \
        "cryptography>=43.0.0" \
        "pydantic>=2.9.0" \
        "pydantic-settings>=2.6.0" \
        "opentelemetry-api>=1.20.0" \
        "opentelemetry-sdk>=1.20.0" \
        "opentelemetry-exporter-otlp-proto-http>=1.20.0" \
        "opentelemetry-instrumentation-fastapi>=0.40b0"

# =============================================================================
# Install langchain-snowflake from patched SOURCE (not PyPI - has version conflicts)
# =============================================================================
COPY original_langchain_snwoflake_repo/langchain-snowflake/libs/snowflake ./langchain-snowflake/

RUN uv pip install --python /app/.venv/bin/python /app/langchain-snowflake

# =============================================================================
# Install splunk-otel-python-contrib from SOURCE (not PyPI!)
# Following official pattern: https://github.com/signalfx/splunk-otel-python-contrib
# =============================================================================
COPY splunk-otel-python-contrib/util/ ./splunk-otel/util/
COPY splunk-otel-python-contrib/instrumentation-genai/ ./splunk-otel/instrumentation-genai/

RUN uv pip install --python /app/.venv/bin/python /app/splunk-otel/util/opentelemetry-util-genai --no-deps && \
    uv pip install --python /app/.venv/bin/python /app/splunk-otel/instrumentation-genai/opentelemetry-instrumentation-langchain

# =============================================================================
# Apply ALL patches (langchain-snowflake + splunk-otel-python-contrib)
# =============================================================================
COPY patches/ ./patches/
RUN chmod +x patches/apply_patches.sh patches/splunk-otel-python-contrib/apply_patches.sh && \
    cd /app && ./patches/apply_patches.sh && \
    ./patches/splunk-otel-python-contrib/apply_patches.sh /app/.venv

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Google Distroless (METHOD 1: minimal, secure, 53MB base)
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/python3-debian12:latest

WORKDIR /app

# METHOD 3: Minimal COPY commands - packages already patched in builder
COPY --from=builder /app/.venv/lib/python3.11/site-packages /usr/lib/python3.11/site-packages
COPY src/ ./src/

# Environment (distroless runs as nonroot by default)
# PYTHONPATH required because distroless doesn't include site-packages in sys.path
ENV PYTHONPATH=/usr/lib/python3.11/site-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# CMD provides arguments to distroless ENTRYPOINT (python3 is already the entrypoint)
CMD ["-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
