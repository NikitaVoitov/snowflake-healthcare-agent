-- =============================================================================
-- Healthcare Multi-Agent Lab - SPCS Container Deployment
-- Phase 7: Deploy FastAPI container to Snowpark Container Services
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

-- -----------------------------------------------------------------------------
-- Step 1: Create Image Repository (if not exists)
-- -----------------------------------------------------------------------------
CREATE IMAGE REPOSITORY IF NOT EXISTS STAGING.HEALTHCARE_IMAGES
    COMMENT = 'Docker images for Healthcare Multi-Agent Lab containers';

-- Get the repository URL for docker push
SHOW IMAGE REPOSITORIES IN SCHEMA STAGING;
-- Use the repository_url value for: docker push <repository_url>/healthcare-agents:latest

-- -----------------------------------------------------------------------------
-- Step 2: Create Network Rule for External Access (Cortex/LLM APIs)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE NETWORK RULE STAGING.CORTEX_EGRESS_RULE
    TYPE = 'HOST_PORT'
    MODE = 'EGRESS'
    VALUE_LIST = (
        'api.snowflake.com:443',
        '*.snowflakecomputing.com:443'
    )
    COMMENT = 'Allow outbound access to Snowflake Cortex APIs';

-- -----------------------------------------------------------------------------
-- Step 3: Create External Access Integration
-- -----------------------------------------------------------------------------
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CORTEX_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (STAGING.CORTEX_EGRESS_RULE)
    ENABLED = TRUE
    COMMENT = 'External access for Healthcare Agent to call Cortex APIs';

-- -----------------------------------------------------------------------------
-- Step 4: Create Service Specification
-- -----------------------------------------------------------------------------
-- Note: Replace <IMAGE_REPO_URL> with actual repository URL from Step 1

CREATE OR REPLACE SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE
    IN COMPUTE POOL AGENTS_POOL
    EXTERNAL_ACCESS_INTEGRATIONS = (CORTEX_ACCESS_INTEGRATION)
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    AUTO_SUSPEND_SECS = 300
    COMMENT = 'Healthcare Multi-Agent FastAPI Service'
    FROM SPECIFICATION $$
spec:
  containers:
    - name: healthcare-agent
      image: /healthcare_db/staging/healthcare_images/healthcare-agents:latest
      env:
        # Snowflake connection (uses service identity in SPCS)
        SNOWFLAKE_ACCOUNT: ${SNOWFLAKE_ACCOUNT}
        SNOWFLAKE_DATABASE: HEALTHCARE_DB
        SNOWFLAKE_WAREHOUSE: PAYERS_CC_WH
        SNOWFLAKE_ROLE: ACCOUNTADMIN
        # Agent configuration
        MAX_AGENT_STEPS: "3"
        AGENT_TIMEOUT_SECONDS: "30"
        # Cortex services
        CORTEX_ANALYST_SEMANTIC_MODEL: "@STAGING.MEMBER_SEMANTIC_MODEL"
        CORTEX_SEARCH_SERVICES: "FAQS_SEARCH,POLICIES_SEARCH,TRANSCRIPTS_SEARCH"
      resources:
        requests:
          memory: 2Gi
          cpu: "1"
        limits:
          memory: 4Gi
          cpu: "2"
      readinessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 10
        periodSeconds: 10
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 30
        periodSeconds: 30
  endpoints:
    - name: query-api
      port: 8000
      public: false
$$;

-- -----------------------------------------------------------------------------
-- Step 5: Verify Service Status
-- -----------------------------------------------------------------------------
-- Check service status
SHOW SERVICES IN COMPUTE POOL AGENTS_POOL;

-- Get detailed service status
SELECT SYSTEM$GET_SERVICE_STATUS('STAGING.HEALTHCARE_AGENTS_SERVICE');

-- Get service logs (for debugging)
SELECT SYSTEM$GET_SERVICE_LOGS('STAGING.HEALTHCARE_AGENTS_SERVICE', 'healthcare-agent', 100);

-- -----------------------------------------------------------------------------
-- Step 6: Get Service Endpoint URL
-- -----------------------------------------------------------------------------
SHOW ENDPOINTS IN SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE;
-- Use the ingress_url for Streamlit to connect

-- -----------------------------------------------------------------------------
-- Step 7: Grant Access to Service
-- -----------------------------------------------------------------------------
GRANT USAGE ON SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE TO ROLE PUBLIC;

-- -----------------------------------------------------------------------------
-- Test Commands (run from SQL worksheet after deployment)
-- -----------------------------------------------------------------------------
/*
-- Test health endpoint
SELECT SYSTEM$CALL_SERVICE(
    'STAGING.HEALTHCARE_AGENTS_SERVICE',
    'query-api',
    '/health',
    'GET'
);

-- Test query endpoint
SELECT SYSTEM$CALL_SERVICE(
    'STAGING.HEALTHCARE_AGENTS_SERVICE',
    'query-api', 
    '/agents/query',
    'POST',
    '{"query": "What are my claims?", "memberId": "ABC1001"}'
);
*/

-- -----------------------------------------------------------------------------
-- Cleanup Commands (if needed)
-- -----------------------------------------------------------------------------
/*
-- Suspend service
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE SUSPEND;

-- Resume service  
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE RESUME;

-- Drop service
DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;
*/
