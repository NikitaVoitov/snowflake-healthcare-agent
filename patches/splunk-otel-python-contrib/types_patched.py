# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from contextvars import Token
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any, Literal, Union

# Note: Using modern dict/list type hints instead of Dict/List from typing
from uuid import UUID, uuid4

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span, SpanContext

# Backward compatibility: older semconv builds may miss new GEN_AI attributes
if not hasattr(GenAIAttributes, "GEN_AI_PROVIDER_NAME"):
    GenAIAttributes.GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
# Import security attribute from centralized attributes module
from opentelemetry.util.genai.attributes import GEN_AI_SECURITY_EVENT_ID
from opentelemetry.util.types import AttributeValue

ContextToken = Token  # simple alias; avoid TypeAlias warning tools


class ContentCapturingMode(Enum):
    # Do not capture content (default).
    NO_CONTENT = 0
    # Only capture content in spans.
    SPAN_ONLY = 1
    # Only capture content in events.
    EVENT_ONLY = 2
    # Capture content in both spans and events.
    SPAN_AND_EVENT = 3


def _new_input_messages() -> list["InputMessage"]:  # quotes for forward ref
    return []


def _new_output_messages() -> list["OutputMessage"]:  # quotes for forward ref
    return []


def _new_str_any_dict() -> dict[str, Any]:
    return {}


@dataclass(kw_only=True)
class GenAI:
    """Base type for all GenAI telemetry entities."""

    context_token: ContextToken | None = None
    span: Span | None = None
    span_context: SpanContext | None = None
    # Parent span for proper parent-child trace linking
    parent_span: Span | None = None
    trace_id: int | None = None
    span_id: int | None = None
    trace_flags: int | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    provider: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_PROVIDER_NAME},
    )
    framework: str | None = None
    attributes: dict[str, Any] = field(default_factory=_new_str_any_dict)
    run_id: UUID = field(default_factory=uuid4)
    parent_run_id: UUID | None = None
    agent_name: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_NAME},
    )
    agent_id: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_ID},
    )
    system: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_SYSTEM},
    )
    conversation_id: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_CONVERSATION_ID},
    )
    data_source_id: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_DATA_SOURCE_ID},
    )
    sample_for_evaluation: bool | None = field(default=True)

    def semantic_convention_attributes(self) -> dict[str, Any]:
        """Return semantic convention attributes defined on this dataclass."""

        result: dict[str, Any] = {}
        for data_field in dataclass_fields(self):
            semconv_key = data_field.metadata.get("semconv")
            if not semconv_key:
                continue
            value = getattr(self, data_field.name)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            result[semconv_key] = value
        return result


@dataclass()
class ToolCall(GenAI):
    """Represents a single tool call invocation (Phase 4)."""

    arguments: Any
    name: str
    id: str | None
    type: Literal["tool_call"] = "tool_call"


@dataclass()
class ToolCallResponse:
    response: Any
    id: str | None
    type: Literal["tool_call_response"] = "tool_call_response"


FinishReason = Literal["content_filter", "error", "length", "stop", "tool_calls"]


@dataclass()
class Text:
    content: str
    type: Literal["text"] = "text"


MessagePart = Union[Text, "ToolCall", ToolCallResponse, Any]


@dataclass()
class InputMessage:
    role: str
    parts: list[MessagePart]


@dataclass()
class OutputMessage:
    role: str
    parts: list[MessagePart]
    finish_reason: str | FinishReason


@dataclass
class LLMInvocation(GenAI):
    """Represents a single large language model invocation.

    Only fields tagged with ``metadata["semconv"]`` are emitted as
    semantic-convention attributes by the span emitters. Additional fields are
    util-only helpers or inputs to alternative span flavors (e.g. Traceloop).
    """

    request_model: str = field(metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL})
    server_address: str | None = field(
        default=None,
        metadata={"semconv": ServerAttributes.SERVER_ADDRESS},
    )
    server_port: int | None = field(
        default=None,
        metadata={"semconv": ServerAttributes.SERVER_PORT},
    )
    input_messages: list[InputMessage] = field(default_factory=_new_input_messages)
    # Traceloop compatibility relies on enumerating these lists into prefixed attributes.
    output_messages: list[OutputMessage] = field(default_factory=_new_output_messages)
    operation: str = field(
        default=GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    response_model_name: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_MODEL},
    )
    response_id: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_ID},
    )
    input_tokens: AttributeValue | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS},
    )
    output_tokens: AttributeValue | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS},
    )
    # Structured function/tool definitions for semantic convention emission
    request_functions: list[dict[str, Any]] = field(default_factory=list)
    request_temperature: float | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE},
    )
    request_top_p: float | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TOP_P},
    )
    request_top_k: int | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_TOP_K},
    )
    request_frequency_penalty: float | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY},
    )
    request_presence_penalty: float | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY},
    )
    request_stop_sequences: list[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES},
    )
    request_max_tokens: int | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS},
    )
    request_choice_count: int | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT},
    )
    request_seed: int | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_SEED},
    )
    request_encoding_formats: list[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS},
    )
    output_type: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_OUTPUT_TYPE},
    )
    response_finish_reasons: list[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS},
    )
    request_service_tier: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER},
    )
    response_service_tier: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER},
    )
    response_system_fingerprint: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT},
    )
    # Security inspection attribute (Cisco AI Defense)
    security_event_id: str | None = field(
        default=None,
        metadata={"semconv": GEN_AI_SECURITY_EVENT_ID},
    )


@dataclass
class Error:
    message: str
    type: type[BaseException]


@dataclass
class EvaluationResult:
    """Represents the outcome of a single evaluation metric.

    Additional fields (e.g., judge model, threshold) can be added without
    breaking callers that rely only on the current contract.
    """

    metric_name: str
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    error: Error | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingInvocation(GenAI):
    """Represents a single embedding model invocation."""

    operation_name: str = field(
        default=GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    request_model: str = field(
        default="",
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL},
    )
    input_texts: list[str] = field(default_factory=list)
    dimension_count: int | None = None
    server_port: int | None = None
    server_address: str | None = None
    input_tokens: int | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS},
    )
    encoding_formats: list[str] = field(
        default_factory=list,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS},
    )
    error_type: str | None = None


@dataclass
class Workflow(GenAI):
    """Represents a workflow orchestrating multiple agents and steps.

    A workflow is the top-level orchestration unit in agentic AI systems,
    coordinating agents and steps to achieve a complex goal. Workflows are optional
    and typically used in multi-agent or multi-step scenarios.

    Attributes:
        name: Identifier for the workflow (e.g., "customer_support_pipeline")
        workflow_type: Type of orchestration (e.g., "sequential", "parallel", "graph", "dynamic")
        description: Human-readable description of the workflow's purpose
        framework: Framework implementing the workflow (e.g., "langgraph", "crewai", "autogen")
        initial_input: User's initial query/request that triggered the workflow
        final_output: Final response/result produced by the workflow
        attributes: Additional custom attributes for workflow-specific metadata
        start_time: Timestamp when workflow started
        end_time: Timestamp when workflow completed
        span: OpenTelemetry span associated with this workflow
        context_token: Context token for span management
        run_id: Unique identifier for this workflow execution
        parent_run_id: Optional parent workflow/trace identifier
    """

    name: str
    workflow_type: str | None = None  # sequential, parallel, graph, dynamic
    description: str | None = None
    initial_input: str | None = None  # User's initial query/request
    final_output: str | None = None  # Final response/result


@dataclass
class _BaseAgent(GenAI):
    """Shared fields for agent lifecycle phases."""

    name: str
    agent_type: str | None = None  # researcher, planner, executor, critic, etc.
    description: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_AGENT_DESCRIPTION},
    )
    model: str | None = field(
        default=None,
        metadata={"semconv": GenAIAttributes.GEN_AI_REQUEST_MODEL},
    )  # primary model if applicable
    tools: list[str] = field(default_factory=list)  # available tool names
    system_instructions: str | None = None  # System prompt/instructions


@dataclass
class AgentCreation(_BaseAgent):
    """Represents agent creation/initialisation."""

    operation: Literal["create_agent"] = field(
        init=False,
        default="create_agent",
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    input_context: str | None = None  # optional initial context


@dataclass
class AgentInvocation(_BaseAgent):
    """Represents agent execution (`invoke_agent`)."""

    operation: Literal["invoke_agent"] = field(
        init=False,
        default="invoke_agent",
        metadata={"semconv": GenAIAttributes.GEN_AI_OPERATION_NAME},
    )
    input_context: str | None = None  # Input for invoke operations
    output_result: str | None = None  # Output for invoke operations


@dataclass
class Step(GenAI):
    """Represents a discrete unit of work in an agentic AI system.

    Steps can be orchestrated at the workflow level (assigned to agents) or
    decomposed internally by agents during execution. This design supports both
    scenarios through flexible parent relationships.
    """

    name: str
    objective: str | None = None  # what the step aims to achieve
    step_type: str | None = None  # planning, execution, reflection, tool_use, etc.
    source: Literal["workflow", "agent"] | None = None  # where step originated
    assigned_agent: str | None = None  # for workflow-assigned steps
    status: str | None = None  # pending, in_progress, completed, failed
    description: str | None = None
    input_data: str | None = None  # Input data/context for the step
    output_data: str | None = None  # Output data/result from the step


__all__ = [
    # Type aliases
    "ContextToken",
    "FinishReason",
    "MessagePart",
    # Enums
    "ContentCapturingMode",
    # Data types
    "Text",
    "InputMessage",
    "OutputMessage",
    "ToolCall",
    "ToolCallResponse",
    "GenAI",
    "LLMInvocation",
    "EmbeddingInvocation",
    "Error",
    "EvaluationResult",
    # Agentic AI types
    "Workflow",
    "AgentCreation",
    "AgentInvocation",
    "Step",
    # Security semconv constant (Cisco AI Defense) - re-exported from attributes
    "GEN_AI_SECURITY_EVENT_ID",
]
