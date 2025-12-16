"""Request models for Healthcare Multi-Agent API."""

from pydantic import Field

from src.models.base import BaseSchema


class QueryRequest(BaseSchema):
    """Request model for agent query endpoint."""

    query: str = Field(..., min_length=1, max_length=5000, description="User query text")
    member_id: str | None = Field(
        None,
        pattern=r"^[A-Z]{3}\d+$",
        description="Optional member ID for member-specific queries",
    )
    tenant_id: str = Field(default="default", description="Tenant identifier for multi-tenancy")
    user_id: str = Field(default="anonymous", description="User identifier for tracking")


class StreamRequest(QueryRequest):
    """Request model for streaming agent endpoint."""

    include_intermediate: bool = Field(
        default=True,
        description="Include intermediate node outputs in stream",
    )

