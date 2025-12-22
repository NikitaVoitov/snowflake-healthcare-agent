"""Graphs package exports - Modern ReAct workflow.

Uses message-based state with ToolNode for automatic tool execution.
"""

from src.graphs.react_state import (
    ConversationTurn,
    ErrorHandlerOutput,
    FinalAnswerOutput,
    HealthcareAgentState,
    ModelOutput,
    ToolsOutput,
)

# Workflow functions are imported lazily to avoid circular import
# Use: from src.graphs.react_workflow import build_react_graph, ...

__all__ = [
    # State types
    "HealthcareAgentState",
    "ConversationTurn",
    "ModelOutput",
    "ToolsOutput",
    "FinalAnswerOutput",
    "ErrorHandlerOutput",
]
