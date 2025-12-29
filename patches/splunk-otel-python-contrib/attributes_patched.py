"""
Centralized constants for GenAI telemetry attribute names.
This module replaces inline string literals for span & event attributes.
"""

# Semantic attribute names for core GenAI spans/events
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_COMPLETION_PREFIX = "gen_ai.completion"

# Additional semantic attribute constants
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
GEN_AI_EVALUATION_ATTRIBUTES_PREFIX = "gen_ai.evaluation.attributes."

# Agent attributes (from semantic conventions)
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_AGENT_TOOLS = "gen_ai.agent.tools"
GEN_AI_AGENT_TYPE = "gen_ai.agent.type"
GEN_AI_AGENT_SYSTEM_INSTRUCTIONS = "gen_ai.agent.system_instructions"
GEN_AI_AGENT_INPUT_CONTEXT = "gen_ai.agent.input_context"
GEN_AI_AGENT_OUTPUT_RESULT = "gen_ai.agent.output_result"

# Workflow attributes (not in semantic conventions)
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_WORKFLOW_TYPE = "gen_ai.workflow.type"
GEN_AI_WORKFLOW_DESCRIPTION = "gen_ai.workflow.description"
GEN_AI_WORKFLOW_INITIAL_INPUT = "gen_ai.workflow.initial_input"
GEN_AI_WORKFLOW_FINAL_OUTPUT = "gen_ai.workflow.final_output"

# Step attributes (not in semantic conventions)
GEN_AI_STEP_NAME = "gen_ai.step.name"
GEN_AI_STEP_TYPE = "gen_ai.step.type"
GEN_AI_STEP_OBJECTIVE = "gen_ai.step.objective"
GEN_AI_STEP_SOURCE = "gen_ai.step.source"
GEN_AI_STEP_ASSIGNED_AGENT = "gen_ai.step.assigned_agent"
GEN_AI_STEP_STATUS = "gen_ai.step.status"
GEN_AI_STEP_INPUT_DATA = "gen_ai.step.input_data"
GEN_AI_STEP_OUTPUT_DATA = "gen_ai.step.output_data"

# Embedding attributes
GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
GEN_AI_EMBEDDINGS_INPUT_TEXTS = "gen_ai.embeddings.input.texts"
GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"

# Server attributes (from semantic conventions)
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"

# Security attributes (Cisco AI Defense)
GEN_AI_SECURITY_EVENT_ID = "gen_ai.security.event_id"

# ============================================================================
# Snowflake Connection Context Attributes (Shared)
# ============================================================================
# These attributes capture the Snowflake connection context and are shared
# between Cortex Search and Cortex Analyst for consistent observability.

SNOWFLAKE_DATABASE = "snowflake.database"
SNOWFLAKE_SCHEMA = "snowflake.schema"
SNOWFLAKE_WAREHOUSE = "snowflake.warehouse"

# ============================================================================
# Snowflake Cortex Search Provider-Specific Attributes
# ============================================================================
# These attributes capture metadata specific to Snowflake Cortex Search
# service responses, enabling detailed observability for search operations.

# Request identifier from Snowflake Cortex Search REST API
# Value is extracted from the _snowflake_request_id field in the API response
SNOWFLAKE_CORTEX_SEARCH_REQUEST_ID = "snowflake.cortex_search.request_id"

# Top cosine similarity score from the search results
# Useful for monitoring search quality and relevance thresholds
SNOWFLAKE_CORTEX_SEARCH_TOP_SCORE = "snowflake.cortex_search.top_score"

# Number of results returned from the search
# Helps track search efficiency and result coverage
SNOWFLAKE_CORTEX_SEARCH_RESULT_COUNT = "snowflake.cortex_search.result_count"

# Sources (services) that were searched
# Examples: "faqs", "policies", "transcripts"
SNOWFLAKE_CORTEX_SEARCH_SOURCES = "snowflake.cortex_search.sources"

# ============================================================================
# Snowflake Cortex Analyst Provider-Specific Attributes
# ============================================================================
# These attributes capture metadata specific to Snowflake Cortex Analyst
# (NL→SQL) service responses, enabling detailed observability for analytics.

# Request identifier from Snowflake Cortex Analyst REST API
SNOWFLAKE_CORTEX_ANALYST_REQUEST_ID = "snowflake.cortex_analyst.request_id"

# Semantic model information
SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_NAME = "snowflake.cortex_analyst.semantic_model.name"
SNOWFLAKE_CORTEX_ANALYST_SEMANTIC_MODEL_TYPE = "snowflake.cortex_analyst.semantic_model.type"

# Generated SQL query (if capture_content enabled)
SNOWFLAKE_CORTEX_ANALYST_SQL = "snowflake.cortex_analyst.sql"

# LLM models used for the query (comma-separated)
SNOWFLAKE_CORTEX_ANALYST_MODEL_NAMES = "snowflake.cortex_analyst.model_names"

# How the question was classified (CLEAR_SQL, AMBIGUOUS, etc.)
SNOWFLAKE_CORTEX_ANALYST_QUESTION_CATEGORY = "snowflake.cortex_analyst.question_category"

# Verified Query Repository (VQR) match information
SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_NAME = "snowflake.cortex_analyst.verified_query.name"
SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_QUESTION = "snowflake.cortex_analyst.verified_query.question"
SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_SQL = "snowflake.cortex_analyst.verified_query.sql"
SNOWFLAKE_CORTEX_ANALYST_VERIFIED_QUERY_VERIFIED_BY = "snowflake.cortex_analyst.verified_query.verified_by"

# Number of warnings returned by Cortex Analyst
SNOWFLAKE_CORTEX_ANALYST_WARNINGS_COUNT = "snowflake.cortex_analyst.warnings_count"

# ============================================================================
# Snowflake Cortex Inference Provider-Specific Attributes
# ============================================================================
# Most LLM completion metadata uses standard GenAI semconv attributes:
# - gen_ai.response.id (request ID from API)
# - gen_ai.response.finish_reasons (why completion stopped)
# - gen_ai.response.model (model that generated response)
#
# Only Snowflake-specific features that have no semconv equivalent are here:

# Guard tokens consumed by Cortex Guard (content safety) if enabled
# Useful for tracking cost and usage of Snowflake's safety features
SNOWFLAKE_INFERENCE_GUARD_TOKENS = "snowflake.inference.guard_tokens"
