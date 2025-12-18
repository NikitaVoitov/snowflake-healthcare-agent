"""FastAPI application entry point for Healthcare Multi-Agent API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.dependencies import get_compiled_graph
from src.middleware.logging import RequestLoggingMiddleware
from src.models.responses import HealthResponse
from src.routers import agent_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Lifespan context manager for startup/shutdown.

    Startup: Pre-compile graph, warm caches, setup checkpointer tables, init Snowpark
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Healthcare Multi-Agent Service")
    logger.info(f"Snowflake account: {settings.snowflake_account}")
    logger.info(f"Database: {settings.snowflake_database}")

    # Initialize Snowpark session for Cortex tools
    from src.graphs.workflow import set_snowpark_session

    try:
        snowpark_session = _create_snowpark_session()
        set_snowpark_session(snowpark_session)
        logger.info("Snowpark session initialized for Cortex tools")
    except Exception as e:
        logger.warning(f"Snowpark session init failed (tools will use mocks): {e}")

    # Setup checkpointer tables (idempotent - creates if not exist)
    from src.dependencies import get_checkpointer

    checkpointer = get_checkpointer()
    if hasattr(checkpointer, "asetup"):
        try:
            await checkpointer.asetup()
            logger.info("Checkpointer tables verified/created")
        except Exception as e:
            logger.warning(f"Checkpointer setup warning: {e}")

    # Pre-compile graph to warm cache
    _ = get_compiled_graph()
    logger.info("Graph compiled and cached")

    yield

    # Shutdown
    logger.info("Shutting down Healthcare Multi-Agent Service")


def _create_snowpark_session():
    """Create Snowpark session based on environment (SPCS vs local)."""
    import os
    from pathlib import Path

    from snowflake.snowpark import Session

    from src.config import is_running_in_spcs

    if is_running_in_spcs():
        # SPCS uses OAuth token authentication
        token_path = Path("/snowflake/session/token")
        if not token_path.exists():
            raise ValueError("SPCS token file not found")

        token = token_path.read_text().strip()
        host = os.getenv("SNOWFLAKE_HOST", "")
        if not host:
            raise ValueError("SNOWFLAKE_HOST not set in SPCS")

        logger.info("Creating Snowpark session with SPCS OAuth token")
        return Session.builder.configs({
            "account": host.replace(".snowflakecomputing.com", ""),
            "host": host,
            "authenticator": "oauth",
            "token": token,
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
        }).create()
    else:
        # Local uses key-pair auth
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        if not settings.snowflake_private_key_path:
            raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH required for local Snowpark")

        with open(settings.snowflake_private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=(settings.snowflake_private_key_passphrase.encode()
                          if settings.snowflake_private_key_passphrase else None),
                backend=default_backend(),
            )

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        logger.info("Creating Snowpark session with key-pair auth")
        return Session.builder.configs({
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "private_key": private_key_bytes,
            "database": settings.snowflake_database,
            "warehouse": settings.snowflake_warehouse,
        }).create()


app = FastAPI(
    title="Healthcare Multi-Agent API",
    description=(
        "Async LangGraph multi-agent system for healthcare contact center. "
        "Provides intelligent routing to Cortex Analyst (member data) and "
        "Cortex Search (knowledge base) with parallel execution support."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Request logging middleware (must be added before CORS for full coverage)
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware (uses settings for production configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    """Catch and log unexpected errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check endpoint for container orchestration."""
    return HealthResponse(status="healthy", version="0.1.0")


# Include routers
app.include_router(agent_routes.router, prefix="/agents", tags=["agents"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

