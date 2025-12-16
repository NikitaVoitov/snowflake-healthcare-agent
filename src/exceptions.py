"""Custom exceptions for Healthcare Multi-Agent API."""


class HealthcareAgentError(Exception):
    """Base exception for healthcare agent errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentTimeoutError(HealthcareAgentError):
    """Raised when agent execution exceeds timeout."""

    pass


class AgentExecutionError(HealthcareAgentError):
    """Raised when agent execution fails."""

    pass


class MemberNotFoundError(HealthcareAgentError):
    """Raised when member ID is not found in database."""

    pass


class CortexServiceError(HealthcareAgentError):
    """Raised when Cortex service call fails."""

    pass

