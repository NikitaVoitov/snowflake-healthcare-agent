-- =============================================================================
-- Healthcare Multi-Agent Lab - SPCS Container Deployment
-- Phase 7: Deploy FastAPI container to Snowpark Container Services
-- 
-- IMPORTANT: This is the WORKING version tested and deployed successfully.
-- Key learnings:
--   1. CREATE OR REPLACE SERVICE is NOT supported - must DROP then CREATE
--   2. Probes with httpGet require specific format (removed for simplicity)
--   3. SPCS uses OAuth token auth - no credentials needed in env vars
--   4. Always use versioned image tags (not :latest) for reliable deploys
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
-- <account>.registry.snowflakecomputing.com/healthcare_db/staging/healthcare_images

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
-- WORKING version - tested and deployed successfully
CREATE SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE
    IN COMPUTE POOL AGENTS_POOL
    EXTERNAL_ACCESS_INTEGRATIONS = (HEALTHCARE_EXTERNAL_ACCESS)
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    AUTO_SUSPEND_SECS = 300
    COMMENT = 'Healthcare Multi-Agent FastAPI Service v1.0.1'
    FROM SPECIFICATION $$
spec:
  containers:
    - name: healthcare-agent
      image: /healthcare_db/staging/healthcare_images/healthcare-agents:v1.0.1
      env:
        # Only database config needed - SPCS handles auth via service identity
        SNOWFLAKE_DATABASE: HEALTHCARE_DB
        SNOWFLAKE_WAREHOUSE: PAYERS_CC_WH
        # Agent configuration
        MAX_AGENT_STEPS: "3"
        AGENT_TIMEOUT_SECONDS: "30"
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
-- Docker Build & Push Commands (run from local terminal)
-- -----------------------------------------------------------------------------
/*
# 1. Build for linux/amd64 (required for SPCS)
docker buildx build --platform linux/amd64 -t healthcare-agents:v1.0.1 .

# 2. Login to Snowflake registry
snow spcs image-registry login -c jwt

# 3. Tag for Snowflake registry
docker tag healthcare-agents:v1.0.1 \
  <account>.registry.snowflakecomputing.com/healthcare_db/staging/healthcare_images/healthcare-agents:v1.0.1

# 4. Push to registry
docker push \
  <account>.registry.snowflakecomputing.com/healthcare_db/staging/healthcare_images/healthcare-agents:v1.0.1
*/

-- -----------------------------------------------------------------------------
-- Test Commands (run after service is READY)
-- -----------------------------------------------------------------------------
/*
-- Test health endpoint via service function
SELECT HEALTHCARE_DB.STAGING.HEALTHCARE_AGENTS_SERVICE!QUERY_API('/health', 'GET');

-- Test query endpoint
SELECT HEALTHCARE_DB.STAGING.HEALTHCARE_AGENTS_SERVICE!QUERY_API(
    '/agents/query',
    'POST',
    '{"query": "What are my claims?", "memberId": "ABC1001"}'
);
*/

-- -----------------------------------------------------------------------------
-- Service Management Commands
-- -----------------------------------------------------------------------------
/*
-- Suspend service (stops billing)
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE SUSPEND;

-- Resume service
ALTER SERVICE STAGING.HEALTHCARE_AGENTS_SERVICE RESUME;

-- Drop service completely
DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;

-- Update to new image version (must drop and recreate)
DROP SERVICE IF EXISTS STAGING.HEALTHCARE_AGENTS_SERVICE;
-- Then run CREATE SERVICE with new image tag
*/
