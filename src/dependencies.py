"""FastAPI dependency injection for Healthcare Multi-Agent API."""

import logging
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends
from langgraph.pregel import Pregel

from src.config import Settings, settings
from src.graphs.workflow import create_healthcare_graph
from src.services.agent_service import AgentService
from src.services.checkpointer import create_checkpointer

logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Singleton Settings instance
    """
    return settings


@lru_cache
def get_compiled_graph() -> Pregel:
    """Build and cache the compiled healthcare graph.

    Returns:
        Compiled StateGraph (Pregel instance) ready for execution
    """
    logger.info("Compiling healthcare workflow graph")
    return create_healthcare_graph().compile()


async def get_checkpointer() -> Any:
    """Create async checkpointer for each request.

    Returns:
        AsyncSnowflakeSaver or MemorySaver fallback
    """
    return await create_checkpointer()


def get_agent_service(
    graph: Annotated[Pregel, Depends(get_compiled_graph)],
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> AgentService:
    """Provide configured agent service.

    Args:
        graph: Compiled workflow graph (Pregel instance)
        app_settings: Application settings

    Returns:
        Configured AgentService instance
    """
    return AgentService(graph, app_settings)


# Type aliases for cleaner route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
GraphDep = Annotated[Pregel, Depends(get_compiled_graph)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
CheckpointerDep = Annotated[Any, Depends(get_checkpointer)]

