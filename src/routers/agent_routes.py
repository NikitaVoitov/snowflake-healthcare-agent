"""Agent API routes with RORO (Receive Object, Return Object) pattern."""

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


class SnowflakeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Snowflake data types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

from src.dependencies import AgentServiceDep, CheckpointerDep
from src.models.requests import QueryRequest, StreamRequest
from src.models.responses import AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Snowflake Service Function Endpoint (Batch Format)
# =============================================================================
@router.post("/sf-query")
async def sf_query_agent(
    request: Request,
    service: AgentServiceDep,
    checkpointer: CheckpointerDep,
) -> JSONResponse:
    """Snowflake Service Function endpoint for SQL-based agent calls.

    Accepts Snowflake's batch format: {"data": [[row_index, query, member_id], ...]}
    Returns Snowflake's response format: {"data": [[row_index, result], ...]}

    This endpoint enables calling the agent via SQL:
        SELECT HEALTHCARE_AGENT_QUERY('What is my deductible?', 'ABC1001');
    """
    try:
        body = await request.json()
        data = body.get("data", [])

        if not data:
            return JSONResponse(content={"data": []})

        results: list[list[Any]] = []

        for row in data:
            row_index = row[0]
            query_text = row[1] if len(row) > 1 else ""
            member_id = row[2] if len(row) > 2 else None

            # Skip empty queries
            if not query_text or not query_text.strip():
                results.append([row_index, {"error": "Empty query"}])
                continue

            try:
                # Create QueryRequest and execute
                query_request = QueryRequest(
                    query=query_text,
                    member_id=member_id,
                    tenant_id="sf_function",
                    user_id="sql_caller",
                )
                response = await service.execute(query_request, checkpointer)
                results.append([row_index, response.model_dump()])
            except Exception as e:
                logger.error(f"Row {row_index} failed: {e}")
                results.append([row_index, {"error": str(e)}])

        # Use custom encoder for Snowflake date/Decimal types
        json_str = json.dumps({"data": results}, cls=SnowflakeJSONEncoder)
        return JSONResponse(content=json.loads(json_str))

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e
    except Exception as e:
        logger.error(f"SF query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


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

