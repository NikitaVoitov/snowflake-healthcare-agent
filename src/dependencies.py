"""FastAPI dependency injection for Healthcare Multi-Agent API.

Simplified to use ReAct workflow exclusively.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from src.config import Settings, settings
from src.graphs.react_workflow import HEALTHCARE_TOOLS, compile_react_graph
from src.services.checkpointer import create_checkpointer
from src.services.react_agent_service import AgentService

logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Singleton Settings instance.
    """
    return settings


@lru_cache
def get_checkpointer() -> BaseCheckpointSaver:
    """Create and cache checkpointer for the application.

    Returns:
        BaseCheckpointSaver implementation (CustomAsyncSnowflakeSaver or MemorySaver).
    """
    return create_checkpointer()


@lru_cache
def get_compiled_graph() -> Pregel:
    """Build and cache the compiled ReAct healthcare graph WITH checkpointer.

    Returns:
        Compiled ReAct StateGraph (Pregel instance) with checkpointing enabled.

    Note:
        Agent markers (tags) are critical for OTel trace visualization.
        The "agent" tag triggers invoke_agent span instead of workflow.
    """
    logger.info("Compiling ReAct healthcare workflow graph")
    checkpointer = get_checkpointer()
    graph = compile_react_graph(checkpointer=checkpointer)

    # CRITICAL: Apply agent markers for OTel instrumentation
    # The "agent" tag triggers invoke_agent span (not workflow) in traces
    # This enables Splunk O11y Agent Flow visualization
    return graph.with_config(
        {
            "run_name": "healthcare_agent",
            # Tags: Agent markers preserved across all invocations
            "tags": [
                "agent",  # Triggers invoke_agent span type
                "agent:healthcare_agent",  # Agent name (gen_ai.agent.name)
                "agent_type:react",  # Agent type (gen_ai.agent.type)
                "agent_description:Healthcare ReAct agent for member inquiries using Snowflake Cortex Analyst and Search",
                f"agent_tools:{','.join(t.name for t in HEALTHCARE_TOOLS)}",
            ],
            # Metadata: Additional context for traces
            "metadata": {
                "agent_name": "healthcare_agent",
                "agent_type": "react",
            },
        }
    )


def get_agent_service(
    graph: Annotated[Pregel, Depends(get_compiled_graph)],
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> AgentService:
    """Provide configured agent service.

    Args:
        graph: Compiled workflow graph (Pregel instance).
        app_settings: Application settings.

    Returns:
        Configured AgentService instance.
    """
    return AgentService(graph, app_settings)


# Type aliases for cleaner route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
GraphDep = Annotated[Pregel, Depends(get_compiled_graph)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
CheckpointerDep = Annotated[BaseCheckpointSaver, Depends(get_checkpointer)]
