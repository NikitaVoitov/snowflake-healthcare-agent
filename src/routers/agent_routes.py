"""Agent API routes with RORO (Receive Object, Return Object) pattern."""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.dependencies import AgentServiceDep, CheckpointerDep
from src.models.requests import QueryRequest, StreamRequest
from src.models.responses import AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=AgentResponse, status_code=200)
async def query_agent(
    request: QueryRequest,
    service: AgentServiceDep,
    checkpointer: CheckpointerDep,
) -> AgentResponse:
    """Execute healthcare agent with query (synchronous response).

    Analyzes the query, routes to appropriate agents (Analyst/Search/Both),
    and returns synthesized response.

    Args:
        request: Query request with user query and optional member_id
        service: Injected AgentService
        checkpointer: Injected checkpointer for state persistence

    Returns:
        AgentResponse with output, routing decision, and metadata

    Raises:
        HTTPException 400: Invalid query
        HTTPException 504: Agent execution timeout
        HTTPException 500: Internal server error
    """
    try:
        logger.info(f"Query request: {request.query[:50]}... member_id={request.member_id}")
        return await service.execute(request, checkpointer)
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TimeoutError as e:
        logger.error("Agent execution timed out")
        raise HTTPException(status_code=504, detail="Agent execution timed out") from e
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Agent execution failed") from e


@router.post("/stream", status_code=200)
async def stream_agent(
    request: StreamRequest,
    service: AgentServiceDep,
    checkpointer: CheckpointerDep,
) -> StreamingResponse:
    """Stream healthcare agent execution events (Server-Sent Events).

    Provides real-time updates as agent nodes execute, useful for
    displaying progress in UI.

    Args:
        request: Stream request with query and streaming options
        service: Injected AgentService
        checkpointer: Injected checkpointer

    Returns:
        StreamingResponse with SSE events for each node transition
    """

    async def event_generator():
        """Generate SSE events from agent stream."""
        try:
            async for event in service.stream(request, checkpointer):
                yield f"data: {json.dumps(event.model_dump(), default=str)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_event = {"event_type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_event)}\n\n"

    logger.info(f"Stream request: {request.query[:50]}...")
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

