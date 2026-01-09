"""Response models for Healthcare Multi-Agent API."""

from pydantic import Field

from src.models.base import BaseSchema


class AgentResponse(BaseSchema):
    """Response model for agent query endpoint."""

    output: str = Field(..., description="Final synthesized response")
    routing: str = Field(..., description="Routing decision: analyst | search | both")
    execution_id: str = Field(..., description="Unique execution identifier")
    checkpoint_id: str | None = Field(None, description="LangGraph checkpoint ID")
    analyst_results: dict | None = Field(None, description="Cortex Analyst results")
    search_results: list | None = Field(None, description="Cortex Search results")


class StreamEvent(BaseSchema):
    """Event model for streaming responses."""

    event_type: str = Field(
        ...,
        description="Event type: node_start | node_end | token | error | complete",
    )
    node: str | None = Field(None, description="Node name that generated this event")
    data: dict = Field(default_factory=dict, description="Event payload data")


class HealthResponse(BaseSchema):
    """Health check response."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="0.2.0", description="API version")


class SnowflakeHealthResponse(BaseSchema):
    """Detailed Snowflake connectivity health check response."""

    status: str = Field(..., description="Overall health status: healthy | unhealthy")
    is_spcs: bool = Field(..., description="Whether running in SPCS environment")
    checks: dict = Field(default_factory=dict, description="Individual check results")
    environment: dict | None = Field(None, description="Environment configuration")

