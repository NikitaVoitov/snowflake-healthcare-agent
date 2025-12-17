"""Request models for Healthcare Multi-Agent API."""

from pydantic import Field, field_validator

from src.models.base import BaseSchema


class QueryRequest(BaseSchema):
    """Request model for agent query endpoint."""

    query: str = Field(..., min_length=1, max_length=5000, description="User query text")
    member_id: str | None = Field(
        None,
        description="Optional 9-digit member ID for member-specific queries",
    )
    tenant_id: str = Field(default="default", description="Tenant identifier for multi-tenancy")
    user_id: str = Field(default="anonymous", description="User identifier for tracking")
    thread_id: str | None = Field(
        None,
        description="Optional thread ID for conversation continuation. If not provided, a new thread is created.",
    )

    @field_validator("member_id", mode="before")
    @classmethod
    def validate_member_id(cls, v: str | None) -> str | None:
        """Convert empty string to None and validate 9-digit format."""
        if v is None or v == "":
            return None
        if not v.isdigit() or len(v) != 9:
            raise ValueError("member_id must be a 9-digit numeric string")
        return v


class StreamRequest(QueryRequest):
    """Request model for streaming agent endpoint."""

    include_intermediate: bool = Field(
        default=True,
        description="Include intermediate node outputs in stream",
    )
