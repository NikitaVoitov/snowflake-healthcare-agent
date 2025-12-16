"""Request logging middleware for Healthcare Multi-Agent API."""

import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request ID tracking and timing."""

    async def dispatch(self, request: Request, call_next):
        """Process request with logging and timing.

        Adds X-Request-ID header for request correlation and logs
        request duration.
        """
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Log request start
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} started"
        )

        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        # Log completion
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"completed in {duration_ms:.2f}ms with status {response.status_code}"
        )

        return response

