"""FastAPI application entry point for Healthcare Multi-Agent API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.dependencies import get_compiled_graph
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

    Startup: Pre-compile graph, warm caches
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Healthcare Multi-Agent Service")
    logger.info(f"Snowflake account: {settings.snowflake_account}")
    logger.info(f"Database: {settings.snowflake_database}")

    # Pre-compile graph to warm cache
    _ = get_compiled_graph()
    logger.info("Graph compiled and cached")

    yield

    # Shutdown
    logger.info("Shutting down Healthcare Multi-Agent Service")


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
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

