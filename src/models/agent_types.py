"""Agent-specific types for Healthcare Multi-Agent workflow."""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class ErrorType(Enum):
    """Categorized error types for agent workflow."""

    VALIDATION = "validation"
    CORTEX_API = "cortex_api"
    TIMEOUT = "timeout"
    MEMBER_NOT_FOUND = "member_not_found"
    MAX_STEPS = "max_steps_exceeded"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetail:
    """Structured error information for error handling node."""

    error_type: ErrorType
    message: str
    node: str
    timestamp: str
    retriable: bool
    fallback_action: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "node": self.node,
            "timestamp": self.timestamp,
            "retriable": self.retriable,
            "fallback_action": self.fallback_action,
        }


class AnalystResultModel(BaseModel):
    """Validated result from Cortex Analyst agent."""

    member_id: str | None = Field(None, description="Member ID from query (None for aggregate queries)")
    claims: list = Field(default_factory=list, description="Member's claims data")
    coverage: dict = Field(default_factory=dict, description="Plan coverage details")
    raw_response: str | None = Field(None, description="Raw Cortex Analyst response")
    aggregate_result: dict | None = Field(None, description="Aggregate query result (counts, totals, etc.)")


class SearchResultModel(BaseModel):
    """Validated result from Cortex Search agent."""

    text: str = Field(..., max_length=10000, description="Document text content")
    score: float = Field(..., ge=0, le=1, description="Relevance score 0-1")
    source: str = Field(..., description="Document source: faqs | policies | transcripts")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
