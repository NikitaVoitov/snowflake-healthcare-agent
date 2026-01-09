"""FastAPI application entry point for Healthcare Multi-Agent API."""

import logging
import warnings
from contextlib import asynccontextmanager

# Configure logging FIRST so OTel setup logs are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress known warning from third-party langchain-snowflake package
# This is a harmless Pydantic field shadowing warning in their code
warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "ChatSnowflake" shadows an attribute',
    category=UserWarning,
    module="langchain_snowflake",
)

# Initialize OpenTelemetry BEFORE any LangChain imports
# This must be done early to instrument all LangChain components
from src.otel_setup import setup_opentelemetry

setup_opentelemetry()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.dependencies import get_compiled_graph
from src.middleware.logging import RequestLoggingMiddleware
from src.models.responses import HealthResponse
from src.routers import agent_routes
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Lifespan context manager for startup/shutdown.

    Startup: Pre-compile graph, warm caches, setup checkpointer tables, init Snowpark
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Healthcare Multi-Agent Service (ReAct Workflow)")
    logger.info(f"Snowflake account: {settings.snowflake_account}")
    logger.info(f"Database: {settings.snowflake_database}")

    # Initialize Snowpark session for Cortex tools AND ChatSnowflake
    # Uses resilient session wrapper for automatic reconnection on token expiration
    from src.graphs.react_workflow import set_snowpark_session
    from src.services.llm_service import set_session as set_llm_session
    from src.services.snowflake_session import set_global_session

    try:
        snowpark_session = _create_snowpark_session()
        set_snowpark_session(snowpark_session)
        set_llm_session(snowpark_session)  # Share session with ChatSnowflake
        set_global_session(snowpark_session)  # Register with resilient wrapper
        logger.info("Snowpark session initialized for Cortex tools (resilient wrapper enabled)")
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
    logger.info("ReAct graph compiled and cached")

    yield

    # Shutdown
    logger.info("Shutting down Healthcare Multi-Agent Service")


def _create_snowpark_session():
    """Create Snowpark session based on environment (SPCS vs local).
    
    SPCS: Uses token_file_path for automatic token refresh (not static token).
    Local: Uses key-pair authentication.
    
    The token_file_path approach is critical for SPCS - it tells the connector
    to read the token fresh from the file on each request, picking up the
    automatically-refreshed token that SPCS maintains at /snowflake/session/token.
    
    Reference: https://docs.snowflake.com/en/developer-guide/snowpark-container-services/additional-considerations-services-jobs
    """
    from src.config import is_running_in_spcs
    from src.services.snowflake_session import (
        create_local_snowpark_session,
        create_spcs_snowpark_session,
    )

    if is_running_in_spcs():
        logger.info("Creating Snowpark session for SPCS with token_file_path (auto-refresh enabled)")
        return create_spcs_snowpark_session()
    else:
        logger.info("Creating Snowpark session with key-pair auth (local)")
        return create_local_snowpark_session()


app = FastAPI(
    title="Healthcare Multi-Agent API",
    description=(
        "Async LangGraph ReAct multi-agent system for healthcare contact center. "
        "Uses intelligent reasoning loop with Cortex Analyst (member data) and "
        "Cortex Search (knowledge base) for comprehensive query handling."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Instrument FastAPI for OTel tracing (must be done after app creation)
# See: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumented for OpenTelemetry")
except ImportError:
    logger.debug("FastAPI OTel instrumentation not available")
except Exception as e:
    logger.warning(f"FastAPI OTel instrumentation failed: {e}")

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


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Basic health check endpoint for container orchestration.
    
    Returns a simple healthy/unhealthy status. For detailed Snowflake
    connectivity checks, use /health/snowflake instead.
    """
    return HealthResponse(status="healthy", version="0.2.0")


@app.get("/health/snowflake", tags=["health"])
async def health_snowflake() -> dict:
    """Detailed Snowflake connectivity health check.
    
    Validates:
    - Token file exists and is readable (SPCS only)
    - Session can be created with current credentials
    - Simple query executes successfully
    
    This endpoint is useful for diagnosing token expiration issues
    and verifying the SPCS OAuth token refresh is working.
    
    Returns:
        Detailed health status with individual check results.
    """
    from src.services.snowflake_session import check_snowflake_health
    
    health_result = await check_snowflake_health()
    
    # Add environment info
    health_result["environment"] = {
        "database": settings.snowflake_database,
        "warehouse": settings.snowflake_warehouse,
        "is_spcs": settings.is_spcs,
    }
    
    return health_result


# Include routers
app.include_router(agent_routes.router, prefix="/agents", tags=["agents"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
