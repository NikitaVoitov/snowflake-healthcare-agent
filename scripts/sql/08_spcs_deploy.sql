-- =============================================================================
-- Healthcare Multi-Agent Lab - SPCS Container Deployment
-- Phase 7: Deploy FastAPI container to Snowpark Container Services
-- 
-- IMPORTANT: This is the WORKING version tested and deployed successfully.
-- Current version: v1.0.84 (Dec 2024) - Token streaming patches, ToolCallChunk support
--
-- Key learnings:
--   1. CREATE OR REPLACE SERVICE is NOT supported - must DROP then CREATE
--   2. Probes with httpGet require specific format (removed for simplicity)
--   3. SPCS uses OAuth token auth via /snowflake/session/token - no credentials needed
--   4. Always use versioned image tags (not :latest) for reliable deploys
--   5. Container name is healthcare-agent (singular), not healthcare-agents
--   6. Module-level Snowpark session (not ContextVar) for async propagation
--   7. USE_REACT_WORKFLOW=true enables ReAct reasoning loop with conversation memory
--   8. ChatSnowflake from langchain-snowflake for native Cortex integration
--   9. claude-3-5-sonnet model required for tool calling support
--  10. Fully async architecture - no blocking calls in event loop
--  11. Distroless images need CMD as args only (ENTRYPOINT=python3 already set)
--  12. langchain-snowflake patches required for SPCS OAuth (SNOWFLAKE_HOST)
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

-- -----------------------------------------------------------------------------
-- Step 1: Verify Image Repository exists
-- -----------------------------------------------------------------------------
-- The image repository should already exist from 05_compute_resources.sql
SHOW IMAGE REPOSITORIES IN SCHEMA STAGING;
-- Expected: HEALTHCARE_IMAGES with URL like:
-- cisco-splunkincubation.registry.snowflakecomputing.com/healthcare_db/staging/healthcare_images

-- -----------------------------------------------------------------------------
-- Step 2: Verify External Access Integration exists
-- -----------------------------------------------------------------------------
-- Should already exist from 05_compute_resources.sql
SHOW EXTERNAL ACCESS INTEGRATIONS;
-- Expected: HEALTHCARE_EXTERNAL_ACCESS

-- -----------------------------------------------------------------------------
-- Step 3: Drop existing service if present (required - no REPLACE support)
-- -----------------------------------------------------------------------------
DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;

-- -----------------------------------------------------------------------------
-- Step 4: Create the SPCS Service
-- -----------------------------------------------------------------------------
-- WORKING version - Token streaming patches + ToolCallChunk support (v1.0.84)
CREATE SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE
    IN COMPUTE POOL AGENTS_POOL
    EXTERNAL_ACCESS_INTEGRATIONS = (HEALTHCARE_EXTERNAL_ACCESS)
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    AUTO_SUSPEND_SECS = 0
    COMMENT = 'Healthcare ReAct Agent v1.0.84 - Token streaming patches, ToolCallChunk support'
    FROM SPECIFICATION $$
spec:
  containers:
    - name: healthcare-agent
      image: /healthcare_db/staging/healthcare_images/healthcare-agent:1.0.84
      env:
        # Only database config needed - SPCS handles auth via OAuth token
        SNOWFLAKE_DATABASE: HEALTHCARE_DB
        SNOWFLAKE_WAREHOUSE: PAYERS_CC_WH
        # Agent configuration
        MAX_AGENT_STEPS: "5"
        AGENT_TIMEOUT_SECONDS: "60"
        # Cortex LLM - claude-3-5-sonnet for tool calling support
        CORTEX_LLM_MODEL: "claude-3-5-sonnet"
        # Enable ReAct workflow (Reasoning + Acting with conversation memory)
        USE_REACT_WORKFLOW: "true"
        # Enable LLM streaming (requires langchain-snowflake patches)
        ENABLE_LLM_STREAMING: "true"
      resources:
        requests:
          memory: 2Gi
          cpu: 1
        limits:
          memory: 4Gi
          cpu: 2
  endpoints:
    - name: query-api
      port: 8000
      public: false
$$;

-- -----------------------------------------------------------------------------
-- Step 5: Verify Service Status
-- -----------------------------------------------------------------------------
-- Check service is created
SHOW SERVICES IN COMPUTE POOL AGENTS_POOL;

-- Wait ~60 seconds for container to start, then check status
-- Expected: status=READY, message=Running
SELECT SYSTEM$GET_SERVICE_STATUS('STAGING.HEALTHCARE_AGENTS_SERVICE');

-- -----------------------------------------------------------------------------
-- Step 6: View Service Logs (for debugging)
-- -----------------------------------------------------------------------------
-- Get container logs - useful for troubleshooting startup issues
CALL SYSTEM$GET_SERVICE_LOGS('STAGING.HEALTHCARE_AGENTS_SERVICE', 0, 'healthcare-agent');

-- Expected successful startup logs:
-- INFO: Started server process [1]
-- INFO: Waiting for application startup.
-- INFO: Compiling healthcare workflow graph
-- INFO: Application startup complete.
-- INFO: Uvicorn running on http://0.0.0.0:8000

-- -----------------------------------------------------------------------------
-- Step 7: Get Service Endpoint URL
-- -----------------------------------------------------------------------------
SHOW ENDPOINTS IN SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE;
-- Note: ingress_url will be NULL for public=false endpoints
-- Internal services are accessed via service name within SPCS/Streamlit

-- -----------------------------------------------------------------------------
-- Step 8: Grant Access to Service (for Streamlit)
-- -----------------------------------------------------------------------------
GRANT USAGE ON SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE TO ROLE PUBLIC;

-- -----------------------------------------------------------------------------
-- Step 9: Create Service Function for SQL Invocation
-- -----------------------------------------------------------------------------
-- IMPORTANT: Endpoint name must match the actual endpoint defined in spec (query-api)
-- This allows calling the agent from SQL/Streamlit without direct HTTP access
DROP FUNCTION IF EXISTS STAGING.HEALTHCARE_AGENT_QUERY(VARCHAR, VARCHAR, VARCHAR);

CREATE OR REPLACE FUNCTION STAGING.HEALTHCARE_AGENT_QUERY(
    query VARCHAR,
    member_id VARCHAR DEFAULT NULL,
    execution_id VARCHAR DEFAULT 'default'
)
RETURNS VARIANT
SERVICE = STAGING.HEALTHCARE_AGENTS_SERVICE
ENDPOINT = 'query-api'
AS '/agents/sf-query';

-- Grant execute on the function
GRANT USAGE ON FUNCTION STAGING.HEALTHCARE_AGENT_QUERY(VARCHAR, VARCHAR, VARCHAR) TO ROLE PUBLIC;

-- -----------------------------------------------------------------------------
-- Docker Build & Push Commands (run from local terminal)
-- -----------------------------------------------------------------------------
/*
# VERSION: Update this for each deployment
export VERSION=1.0.84
export REGISTRY=cisco-splunkincubation.registry.snowflakecomputing.com

# 1. Build for linux/amd64 (required for SPCS)
docker buildx build --platform linux/amd64 -t healthcare-agent:$VERSION .

# 2. Login to Snowflake registry (requires jwt connection profile)
cd /Users/nvoitov/Documents/Splunk/Snowflake/healthcare
export $(grep -E '^(SNOWFLAKE_|PRIVATE_KEY)' .env | xargs)
export PRIVATE_KEY_PASSPHRASE="$SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"
snow spcs image-registry login -c jwt

# 3. Tag for Snowflake registry
docker tag healthcare-agent:$VERSION \
  $REGISTRY/healthcare_db/staging/healthcare_images/healthcare-agent:$VERSION

# 4. Push to registry
docker push \
  $REGISTRY/healthcare_db/staging/healthcare_images/healthcare-agent:$VERSION
*/

-- -----------------------------------------------------------------------------
-- Test Commands (run after service is READY)
-- -----------------------------------------------------------------------------
/*
-- Test via service function (recommended)
SELECT STAGING.HEALTHCARE_AGENT_QUERY(
    'What are the claims for this member?',
    '700180341',  -- Valid test member ID
    'default'
);

-- Test general question (no member_id)
SELECT STAGING.HEALTHCARE_AGENT_QUERY(
    'How do I file an appeal?',
    NULL,
    'default'
);

-- Direct endpoint test (alternative)
SELECT HEALTHCARE_DB.STAGING.HEALTHCARE_AGENTS_SERVICE!QUERY_API(
    '/health',
    'GET'
);
*/

-- -----------------------------------------------------------------------------
-- Valid Test Member IDs (from CALL_CENTER_MEMBER_DENORMALIZED)
-- -----------------------------------------------------------------------------
/*
-- Use these for testing - DO NOT use placeholder IDs like 100000001
-- 700180341 - Christina Frey, Premium Health Plan, PPO
-- 149061173 - Test member with claims
-- 470154690 - Angel Underwood, Standard Health Plan, EPO

-- Verify member exists before testing:
SELECT DISTINCT member_id, name, plan_name, plan_type 
FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED 
WHERE member_id IN ('700180341', '149061173', '470154690');
*/

-- -----------------------------------------------------------------------------
-- Service Management Commands
-- -----------------------------------------------------------------------------
/*
-- Check service status
SELECT SYSTEM$GET_SERVICE_STATUS('STAGING.HEALTHCARE_AGENTS_SERVICE');

-- View service logs (for debugging)
CALL SYSTEM$GET_SERVICE_LOGS('STAGING.HEALTHCARE_AGENTS_SERVICE', 0, 'healthcare-agent');

-- Suspend service (stops billing)
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE SUSPEND;

-- Resume service
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE RESUME;

-- Drop service completely
DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;

-- Update to new image version (must drop and recreate)
-- 1. Update VERSION variable in docker commands above
-- 2. Build and push new image
-- 3. Drop existing service: DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;
-- 4. Run CREATE SERVICE with new image tag in spec
*/
