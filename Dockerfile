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

# METHOD 4: Copy dependency files FIRST for better cache utilization
COPY pyproject.toml uv.lock ./

# Install ONLY spcs-production group (PEP 735 dependency-group)
# --only-group ensures we don't pull in uvloop, watchfiles, websockets, etc.
RUN uv sync --no-dev --only-group spcs-production --no-install-project && \
    # Strip unnecessary files in same layer (saves ~50MB)
    find /app/.venv/lib/python3.11/site-packages \( \
        -type d -name "__pycache__" -o \
        -type d -name "tests" -o \
        -type d -name "test" -o \
        -type d -name "testing" -o \
        -type d -name "*.dist-info" -o \
        -type d -name "*.egg-info" \
    \) -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv/lib/python3.11/site-packages \( \
        -type f -name "*.pyc" -o \
        -type f -name "*.pyo" -o \
        -type f -name "*.pyi" -o \
        -type f -name "*.md" -o \
        -type f -name "*.rst" -o \
        -type f -name "*.txt" -o \
        -type f -name "LICENSE*" -o \
        -type f -name "CHANGELOG*" \
    \) -delete 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python3.11/site-packages/setuptools* \
           /app/.venv/lib/python3.11/site-packages/pkg_resources* \
           /app/.venv/lib/python3.11/site-packages/_distutils_hack* 2>/dev/null || true

# Apply patches in builder (before copy to runtime)
COPY patches/langchain_snowflake_rest_client_patched.py \
    /app/.venv/lib/python3.11/site-packages/langchain_snowflake/_connection/rest_client.py
COPY patches/langchain_snowflake_tools_patched.py \
    /app/.venv/lib/python3.11/site-packages/langchain_snowflake/chat_models/tools.py
# NEW: Streaming patches for token-level streaming with tools
COPY patches/langchain_snowflake_streaming_patched.py \
    /app/.venv/lib/python3.11/site-packages/langchain_snowflake/chat_models/streaming.py
COPY patches/langchain_snowflake_base_patched.py \
    /app/.venv/lib/python3.11/site-packages/langchain_snowflake/chat_models/base.py

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
