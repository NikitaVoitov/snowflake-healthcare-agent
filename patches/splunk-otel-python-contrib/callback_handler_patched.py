"""Simplified LangChain callback handler (Phase 1).

Only maps callbacks to GenAI util types and delegates lifecycle to TelemetryHandler.
Complex logic removed (agent heuristics, child counting, prompt capture, events).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Step,
    Text,
    ToolCall,
    Workflow,
)
from opentelemetry.util.genai.types import (
    Error as GenAIError,
)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


def _serialize(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return str(obj)
        except (TypeError, ValueError):
            return None


def _resolve_agent_name(tags: list[str] | None, metadata: dict[str, Any] | None) -> str | None:
    if metadata:
        for key in ("agent_name", "gen_ai.agent.name", "agent"):
            value = metadata.get(key)
            if value:
                return _safe_str(value)
    if tags:
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag_value = tag.strip()
            lower_value = tag_value.lower()
            if lower_value.startswith("agent:") and len(tag_value) > 6:
                return _safe_str(tag_value.split(":", 1)[1])
            if lower_value.startswith("agent_") and len(tag_value) > 6:
                return _safe_str(tag_value.split("_", 1)[1])
            # Don't return "agent" itself as a name - it's just a marker tag
    return None


def _is_agent_root(tags: list[str] | None, metadata: dict[str, Any] | None) -> bool:
    if _resolve_agent_name(tags, metadata):
        return True
    if tags:
        for tag in tags:
            try:
                if "agent" in str(tag).lower():
                    return True
            except (TypeError, ValueError):
                continue
    if metadata and metadata.get("agent_name"):
        return True
    return False


def _extract_tool_details(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not metadata:
        return None

    tool_data: dict[str, Any] = {}
    nested = metadata.get("gen_ai.tool")
    if isinstance(nested, dict):
        tool_data.update(nested)

    detection_flag = bool(tool_data)
    for key, value in list(metadata.items()):
        if not isinstance(key, str):
            continue
        lower_key = key.lower()
        if lower_key.startswith("gen_ai.tool."):
            suffix = key.split(".", 2)[-1]
            tool_data[suffix] = value
            detection_flag = True
            continue
        if lower_key in {
            "tool_name",
            "tool_id",
            "tool_call_id",
            "tool_args",
            "tool_arguments",
            "tool_input",
            "tool_parameters",
        }:
            name_parts = lower_key.split("_", 1)
            suffix = name_parts[-1] if len(name_parts) > 1 else lower_key
            tool_data[suffix] = value
            detection_flag = True

    for hint_key in (
        "gen_ai.step.type",
        "step_type",
        "type",
        "run_type",
        "langchain_run_type",
    ):
        hint_val = metadata.get(hint_key)
        if isinstance(hint_val, str) and "tool" in hint_val.lower():
            detection_flag = True
            break

    if not detection_flag:
        return None

    name_value = tool_data.get("name") or metadata.get("gen_ai.step.name")
    if not name_value:
        return None

    arguments = tool_data.get("arguments")
    if arguments is None:
        for candidate in ("input", "args", "parameters"):
            if candidate in tool_data:
                arguments = tool_data[candidate]
                break

    tool_id = tool_data.get("id") or tool_data.get("call_id")
    if tool_id is not None:
        tool_id = _safe_str(tool_id)

    return {
        "name": _safe_str(name_value),
        "arguments": arguments,
        "id": tool_id,
    }


class LangchainCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        telemetry_handler: TelemetryHandler | None = None,
    ) -> None:
        super().__init__()
        self._handler = telemetry_handler

    def _resolve_parent_span(self, parent_run_id: UUID | None) -> Any | None:
        """Resolve parent_run_id to an actual OpenTelemetry Span from the handler's registry.

        This enables proper parent-child span linking in traces.
        """
        if parent_run_id is None:
            return None
        return self._handler.get_span_by_run_id(parent_run_id)

    def _find_nearest_agent(self, run_id: UUID | None) -> AgentInvocation | None:
        current = run_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._handler.get_entity(current)
            if isinstance(entity, AgentInvocation):
                return entity
            if entity is None:
                break
            current = getattr(entity, "parent_run_id", None)
        return None

    def _find_nearest_provider_context(self, run_id: UUID | None) -> str | None:
        """Find provider from nearest parent that has one (Agent, Workflow, or Step).

        This enables provider inheritance for tools even when no AgentInvocation exists
        in the hierarchy (e.g., single-workflow patterns).

        Args:
            run_id: The run_id to start searching from.

        Returns:
            The provider string if found, None otherwise.
        """
        current = run_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._handler.get_entity(current)
            if entity is None:
                break
            # Check if entity has provider set (works for Agent, Workflow, Step - all inherit from GenAI)
            provider = getattr(entity, "provider", None)
            if provider:
                return _safe_str(provider)
            current = getattr(entity, "parent_run_id", None)
        return None

    def _propagate_provider_to_root(self, run_id: UUID | None, provider: str) -> None:
        """Propagate provider to ALL ancestors up to the root.

        This enables provider inheritance for sibling branches (like LangGraph's
        separate model_node and tools_node) by setting provider on the common
        root workflow. Without this, tools in a different branch than the LLM
        won't be able to find the provider.

        Flow: LLM (provider detected) → model_node → workflow (set here)
              tool → tools_node → workflow (find here)

        Args:
            run_id: The starting run_id to walk up from.
            provider: The provider string to propagate.
        """
        if run_id is None or not provider:
            return
        current = run_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._handler.get_entity(current)
            if entity is None:
                break
            # Set provider on every ancestor that doesn't have one
            if not getattr(entity, "provider", None):
                entity.provider = provider
            current = getattr(entity, "parent_run_id", None)

    def _start_agent_invocation(
        self,
        *,
        name: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        attrs: dict[str, Any],
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None,
        agent_name: str | None,
    ) -> AgentInvocation:
        agent = AgentInvocation(
            name=name,
            run_id=run_id,
            attributes=attrs,
        )
        agent.input_context = _serialize(inputs)
        agent.agent_name = _safe_str(agent_name) if agent_name else name
        agent.parent_run_id = parent_run_id
        # Resolve parent_run_id to actual span for proper parent-child linking
        agent.parent_span = self._resolve_parent_span(parent_run_id)
        agent.framework = "langchain"
        if metadata:
            if metadata.get("agent_type"):
                agent.agent_type = _safe_str(metadata["agent_type"])
            if metadata.get("model_name"):
                agent.model = _safe_str(metadata["model_name"])
            if metadata.get("system"):
                agent.system = _safe_str(metadata["system"])
        self._handler.start_agent(agent)
        return agent

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "chain")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        agent_name_hint = _resolve_agent_name(tags, metadata)
        if parent_run_id is None:
            if _is_agent_root(tags, metadata):
                self._start_agent_invocation(
                    name=name,
                    run_id=run_id,
                    parent_run_id=None,
                    attrs=attrs,
                    inputs=inputs,
                    metadata=metadata,
                    agent_name=agent_name_hint,
                )
            else:
                wf = Workflow(name=name, run_id=run_id, attributes=attrs)
                wf.initial_input = _serialize(inputs)
                # Extract workflow metadata for Agent Flow visualization in Splunk O11y
                # These attributes enable the visual workflow graph in trace views
                # Note: LangGraph dev server replaces custom metadata with its own,
                # but merges tags. We detect LangGraph from its metadata keys.
                if metadata:
                    # workflow_type enables Agent Flow visualization (e.g., "graph", "sequential")
                    if metadata.get("workflow_type"):
                        wf.workflow_type = _safe_str(metadata["workflow_type"])
                    elif metadata.get("langgraph_version") or metadata.get("graph_id"):
                        # LangGraph detected from its internal metadata - set type to "graph"
                        wf.workflow_type = "graph"
                    # ls_description is LangSmith convention, also check description
                    desc = metadata.get("ls_description") or metadata.get("description")
                    if desc:
                        wf.description = _safe_str(desc)
                    # framework helps identify the orchestration library
                    if metadata.get("framework"):
                        wf.framework = _safe_str(metadata["framework"])
                    elif metadata.get("langgraph_version"):
                        # LangGraph detected - set framework
                        wf.framework = "langgraph"
                # Fallback: detect framework and description from tags
                # Tags are preserved by LangGraph dev server (unlike metadata)
                if tags:
                    if not wf.framework and "langgraph" in tags:
                        wf.framework = "langgraph"
                        if not wf.workflow_type:
                            wf.workflow_type = "graph"
                    # Support description via tag: "description:My workflow description"
                    if not wf.description:
                        for tag in tags:
                            if isinstance(tag, str) and tag.startswith("description:"):
                                wf.description = tag[len("description:"):]
                                break
                # Generate description from graph_id if still not set
                if not wf.description and metadata and metadata.get("graph_id"):
                    graph_id = _safe_str(metadata["graph_id"])
                    wf.description = f"LangGraph workflow: {graph_id}"
                # No parent for root workflow (parent_run_id is None here)
                self._handler.start_workflow(wf)
            return
        else:
            context_agent = self._find_nearest_agent(parent_run_id)
            context_agent_name = _safe_str(context_agent.agent_name or context_agent.name) if context_agent else None
            if agent_name_hint:
                hint_normalized = agent_name_hint.lower()
                context_normalized = context_agent_name.lower() if context_agent_name else None
                if context_normalized != hint_normalized:
                    self._start_agent_invocation(
                        name=name,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        attrs=attrs,
                        inputs=inputs,
                        metadata=metadata,
                        agent_name=agent_name_hint,
                    )
                    return
            tool_info = _extract_tool_details(metadata)
            if tool_info is not None:
                existing = self._handler.get_entity(run_id)
                if isinstance(existing, ToolCall):
                    tool = existing
                    if context_agent is not None:
                        agent_name_value = context_agent.agent_name or context_agent.name
                        if not getattr(tool, "agent_name", None):
                            tool.agent_name = _safe_str(agent_name_value)
                        if not getattr(tool, "agent_id", None):
                            tool.agent_id = str(context_agent.run_id)
                else:
                    # Filter out tool-specific metadata from attributes
                    # since they're stored in dedicated fields
                    tool_attrs = {k: v for k, v in attrs.items() if not (isinstance(k, str) and k.lower().startswith("gen_ai.tool."))}
                    arguments = tool_info.get("arguments")
                    if arguments is None:
                        arguments = inputs
                    tool = ToolCall(
                        name=tool_info.get("name", name),
                        id=tool_info.get("id"),
                        arguments=arguments,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        attributes=tool_attrs,
                    )
                    tool.framework = "langchain"
                    # Resolve parent_run_id to actual span for proper parent-child linking
                    tool.parent_span = self._resolve_parent_span(parent_run_id)
                    if context_agent is not None and context_agent_name is not None:
                        tool.agent_name = context_agent_name
                        tool.agent_id = str(context_agent.run_id)
                    self._handler.start_tool_call(tool)
                if inputs is not None and getattr(tool, "arguments", None) is None:
                    tool.arguments = inputs
                if getattr(tool, "arguments", None) is not None:
                    serialized_args = _serialize(tool.arguments)
                    if serialized_args is not None:
                        tool.attributes.setdefault("tool.arguments", serialized_args)
            else:
                step = Step(
                    name=name,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    step_type="chain",
                    attributes=attrs,
                )
                # Resolve parent_run_id to actual span for proper parent-child linking
                step.parent_span = self._resolve_parent_span(parent_run_id)
                if context_agent is not None:
                    if context_agent_name is not None:
                        step.agent_name = context_agent_name
                    step.agent_id = str(context_agent.run_id)
                step.input_data = _serialize(inputs)
                self._handler.start_step(step)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_kwargs: Any,
    ) -> None:
        entity = self._handler.get_entity(run_id)
        if entity is None:
            return
        if isinstance(entity, Workflow):
            entity.final_output = _serialize(outputs)
            self._handler.stop_workflow(entity)
        elif isinstance(entity, AgentInvocation):
            entity.output_result = _serialize(outputs)
            self._handler.stop_agent(entity)
        elif isinstance(entity, Step):
            entity.output_data = _serialize(outputs)
            self._handler.stop_step(entity)
        elif isinstance(entity, ToolCall):
            serialized = _serialize(outputs)
            if serialized is not None:
                entity.attributes.setdefault("tool.response", serialized)
            self._handler.stop_tool_call(entity)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        invocation_params = extra.get("invocation_params", {})
        # Priority: invocation_params.model_name > invocation_params.model > metadata.model_name > metadata.ls_model_name
        # Note: Some providers use "model" (e.g., ChatSnowflake) while others use "model_name" (e.g., ChatOpenAI)
        model_source = (
            invocation_params.get("model_name")
            or invocation_params.get("model")  # ChatSnowflake and similar providers use "model"
            or (metadata.get("model_name") if metadata else None)
            or (metadata.get("ls_model_name") if metadata else None)  # LangSmith params
        )
        request_model = _safe_str(model_source or "unknown_model")
        input_messages: list[InputMessage] = []
        for batch in messages:
            for m in batch:
                content = getattr(m, "content", "")
                input_messages.append(InputMessage(role="user", parts=[Text(content=_safe_str(content))]))

        # Build attributes from metadata and invocation_params
        attrs: dict[str, Any] = {}

        # Process metadata - skip ls_* fields as they're extracted to dedicated fields
        if metadata:
            for key, value in metadata.items():
                if isinstance(key, str) and not key.startswith("ls_"):
                    attrs[key] = value

        # Add tags
        if tags:
            attrs["tags"] = [str(t) for t in tags]

        # Process invocation_params - add with request_ prefix
        if invocation_params:
            # Standard params get request_ prefix
            for key in (
                "top_p",
                "seed",
                "temperature",
                "frequency_penalty",
                "presence_penalty",
            ):
                if key in invocation_params:
                    attrs[f"request_{key}"] = invocation_params[key]

            # Handle nested model_kwargs
            if "model_kwargs" in invocation_params:
                attrs["model_kwargs"] = invocation_params["model_kwargs"]
                # Also check for max_tokens in model_kwargs
                mk = invocation_params["model_kwargs"]
                if isinstance(mk, dict) and "max_tokens" in mk:
                    attrs["request_max_tokens"] = mk["max_tokens"]

        # Handle max_tokens from metadata.ls_max_tokens
        if metadata and "ls_max_tokens" in metadata and "request_max_tokens" not in attrs:
            attrs["request_max_tokens"] = metadata["ls_max_tokens"]

        # Add callback info from serialized
        if payload.get("name"):
            attrs["callback.name"] = payload["name"]
        if payload.get("id"):
            attrs["callback.id"] = payload["id"]

        # Set provider from ls_provider in metadata
        provider = None
        if metadata and "ls_provider" in metadata:
            provider = _safe_str(metadata["ls_provider"])

        inv = LLMInvocation(
            request_model=request_model,
            input_messages=input_messages,
            attributes=attrs,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        # Resolve parent_run_id to actual span for proper parent-child linking
        inv.parent_span = self._resolve_parent_span(parent_run_id)
        if provider:
            inv.provider = provider
        if parent_run_id is not None:
            context_agent = self._find_nearest_agent(parent_run_id)
            if context_agent is not None:
                agent_name_value = context_agent.agent_name or context_agent.name
                inv.agent_name = _safe_str(agent_name_value)
                inv.agent_id = str(context_agent.run_id)
            # Propagate provider to ALL ancestors up to root workflow
            # This enables provider inheritance for tools in sibling branches
            # (e.g., LangGraph's separate model_node and tools_node steps)
            if provider:
                self._propagate_provider_to_root(parent_run_id, provider)
        self._handler.start_llm(inv)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        message_batches = [[HumanMessage(content=p)] for p in prompts]
        self.on_chat_model_start(
            serialized=serialized,
            messages=message_batches,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **extra,
        )
        inv = self._handler.get_entity(run_id)
        if isinstance(inv, LLMInvocation):
            inv.operation = "generate_text"

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_kwargs: Any,
    ) -> None:
        inv = self._handler.get_entity(run_id)
        if not isinstance(inv, LLMInvocation):
            return
        generations = getattr(response, "generations", [])
        content = None
        if generations and generations[0] and generations[0][0].message:
            content = getattr(generations[0][0].message, "content", None)
        if content is not None:
            finish_reason = generations[0][0].generation_info.get("finish_reason") if generations[0][0].generation_info else None
            if finish_reason == "tool_calls":
                inv.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=["ToolCall"],
                        finish_reason=finish_reason or "tool_calls",
                    )
                ]
            else:
                inv.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=_safe_str(content))],
                        finish_reason=finish_reason or "stop",
                    )
                ]
        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        inv.input_tokens = usage.get("prompt_tokens")
        inv.output_tokens = usage.get("completion_tokens")

        # Extract response model and metadata from response_metadata
        # Uses standard GenAI semantic conventions where available:
        # - gen_ai.response.id (standard semconv)
        # - gen_ai.response.finish_reasons (standard semconv, array)
        # - snowflake.inference.guard_tokens (Snowflake-specific)
        if generations:
            for generation_list in generations:
                for generation in generation_list:
                    if hasattr(generation, "message") and hasattr(generation.message, "response_metadata") and generation.message.response_metadata:
                        response_meta = generation.message.response_metadata

                        # Extract model name
                        if not inv.response_model_name:
                            model_name = response_meta.get("model_name")
                            if model_name:
                                inv.response_model_name = _safe_str(model_name)

                        # Extract response ID (standard semconv: gen_ai.response.id)
                        # langchain-snowflake stores this as "snowflake_request_id"
                        if response_meta.get("snowflake_request_id") and not inv.response_id:
                            inv.response_id = _safe_str(response_meta["snowflake_request_id"])

                        # Extract finish reason (standard semconv: gen_ai.response.finish_reasons)
                        # Stored as array per semconv spec
                        if response_meta.get("finish_reason"):
                            inv.attributes["gen_ai.response.finish_reasons"] = [_safe_str(response_meta["finish_reason"])]

                        # Snowflake-specific: Guard tokens consumed by Cortex Guard
                        if response_meta.get("snowflake_guard_tokens") is not None:
                            inv.attributes["snowflake.inference.guard_tokens"] = int(response_meta["snowflake_guard_tokens"])
                        break
                if inv.response_model_name:
                    break

        self._handler.stop_llm(inv)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "tool")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        context_agent = self._find_nearest_agent(parent_run_id) if parent_run_id is not None else None
        context_agent_name = _safe_str(context_agent.agent_name or context_agent.name) if context_agent else None

        # Extract tool_call_id from multiple possible locations:
        # 1. extra.tool_call_id - LangGraph ToolNode passes it here
        # 2. metadata.tool_call_id - some frameworks pass it in metadata
        # 3. inputs dict with tool_call_id key
        # 4. Fall back to serialized.id (tool's unique identifier, not tool_call_id)
        tool_call_id = (
            extra.get("tool_call_id")
            or (metadata.get("tool_call_id") if metadata else None)
            or (inputs.get("tool_call_id") if isinstance(inputs, dict) else None)
        )
        if tool_call_id is not None:
            tool_call_id = _safe_str(tool_call_id)

        # Legacy: serialized.id is the tool's identifier (e.g., ["langchain", "tools", "search"])
        # This is NOT the tool_call_id from the LLM response
        id_source = payload.get("id") or extra.get("id")
        if tool_call_id is None and id_source is not None:
            # Only use serialized.id as fallback if it looks like a tool_call_id
            # (i.e., a string, not a list of module path components)
            if isinstance(id_source, str) and not id_source.startswith("["):
                tool_call_id = _safe_str(id_source)

        arguments: Any = inputs if inputs is not None else input_str
        existing = self._handler.get_entity(run_id)
        if isinstance(existing, ToolCall):
            if arguments is not None:
                existing.arguments = arguments
            if attrs:
                existing.attributes.update(attrs)
            if context_agent is not None:
                if not getattr(existing, "agent_name", None) and context_agent_name is not None:
                    existing.agent_name = context_agent_name
                if not getattr(existing, "agent_id", None):
                    existing.agent_id = str(context_agent.run_id)
            # Inherit provider from ANY parent context (Agent, Workflow, or Step)
            # This works for both multi-agent and single-workflow patterns
            if not getattr(existing, "provider", None):
                inherited_provider = self._find_nearest_provider_context(parent_run_id)
                if inherited_provider:
                    existing.provider = inherited_provider
            # Update tool_call_id if we found it and it's not set
            if tool_call_id and not existing.id:
                existing.id = tool_call_id
            if existing.framework is None:
                existing.framework = "langchain"
            return
        tool = ToolCall(
            name=name,
            id=tool_call_id,
            arguments=arguments,
            run_id=run_id,
            parent_run_id=parent_run_id,
            attributes=attrs,
        )
        tool.framework = "langchain"
        # Resolve parent_run_id to actual span for proper parent-child linking
        tool.parent_span = self._resolve_parent_span(parent_run_id)
        if context_agent is not None:
            if context_agent_name is not None:
                tool.agent_name = context_agent_name
            tool.agent_id = str(context_agent.run_id)
        inherited_provider = self._find_nearest_provider_context(parent_run_id)
        if inherited_provider:
            tool.provider = inherited_provider
        if not tool.provider and metadata:
            provider_hint = metadata.get("ls_provider") or metadata.get("provider")
            if provider_hint:
                tool.provider = _safe_str(provider_hint)
        if arguments is not None:
            serialized_args = _serialize(arguments)
            if serialized_args is not None:
                tool.attributes.setdefault("tool.arguments", serialized_args)
        if inputs is None and input_str:
            tool.attributes.setdefault("tool.input_str", _safe_str(input_str))
        self._handler.start_tool_call(tool)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_kwargs: Any,
    ) -> None:
        tool = self._handler.get_entity(run_id)
        if not isinstance(tool, ToolCall):
            return
        serialized = _serialize(output)
        if serialized is not None:
            tool.attributes.setdefault("tool.response", serialized)
        self._handler.stop_tool_call(tool)

    def _fail(self, run_id: UUID, error: BaseException) -> None:
        self._handler.fail_by_run_id(run_id, GenAIError(message=str(error), type=type(error)))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)
