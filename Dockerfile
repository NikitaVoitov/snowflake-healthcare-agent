# =============================================================================
# Healthcare Multi-Agent Lab - SPCS Container
# Multi-stage build using uv for fast, reproducible Python dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies with uv
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the local package - we copy src/ separately)
# --frozen = exact versions from lock file
# --no-dev = exclude dev dependencies  
# --extra spcs = include SPCS-specific deps
# --no-install-project = skip editable install of local package
RUN uv sync --frozen --no-dev --extra spcs --no-install-project

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ ./src/

# PATCH: Apply fixes for langchain-snowflake SPCS compatibility
# 1. rest_client.py: Fix _get_base_url() to use SNOWFLAKE_HOST env var (required for SPCS OAuth)
# 2. tools.py: Fix _parse_json_response() to avoid content duplication (issue #35)
COPY patches/langchain_snowflake_rest_client_patched.py /app/.venv/lib/python3.11/site-packages/langchain_snowflake/_connection/rest_client.py
COPY patches/langchain_snowflake_tools_patched.py /app/.venv/lib/python3.11/site-packages/langchain_snowflake/chat_models/tools.py

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
