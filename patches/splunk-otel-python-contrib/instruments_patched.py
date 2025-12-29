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

from opentelemetry.metrics import Histogram, Meter

# ============================================================================
# Histogram Bucket Boundaries (pre-configured for common use cases)
# ============================================================================
# These follow the same pattern as opentelemetry-instrumentation-openai-v2
# using explicit_bucket_boundaries_advisory for proper distribution analysis.

# Cosine similarity scores (0.0 - 1.0 range)
# Used for semantic search quality metrics
_GEN_AI_TOOL_SEARCH_SCORE_BUCKETS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


class Instruments:
    """
    Manages OpenTelemetry metrics instruments for GenAI telemetry.
    """

    def __init__(self, meter: Meter):
        self.operation_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI client operations",
        )
        self.token_usage_histogram: Histogram = meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        )
        # Agentic AI metrics
        self.workflow_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.workflow.duration",
            unit="s",
            description="Duration of GenAI workflows",
        )
        self.agent_duration_histogram: Histogram = meter.create_histogram(
            name="gen_ai.agent.duration",
            unit="s",
            description="Duration of agent operations",
        )
        # ================================================================
        # Tool Search Score Histogram
        # ================================================================
        # Records the top similarity score from search/retrieval tools.
        # This is useful for monitoring search quality over time.
        # Dimensions:
        #   - gen_ai.tool.name: The tool that performed the search
        #   - gen_ai.provider.name: The provider (e.g., "snowflake")
        #   - snowflake.cortex_search.source: The source searched (if applicable)
        #
        # Bucket boundaries are pre-configured for cosine similarity scores (0-1)
        # so users don't need to manually configure Views.
        self.tool_search_score_histogram: Histogram = meter.create_histogram(
            name="gen_ai.tool.search.score",
            unit="{score}",
            description="Top similarity score from search/retrieval tool operations",
            explicit_bucket_boundaries_advisory=_GEN_AI_TOOL_SEARCH_SCORE_BUCKETS,
        )
