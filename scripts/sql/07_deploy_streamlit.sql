-- =============================================================================
-- Healthcare ReAct Agent - Streamlit App Deployment
-- Deploy Streamlit App with Container Runtime (SPCS) or Warehouse Runtime
-- 
-- Container Runtime: Uses SPCS compute pool, internal DNS access to agent service
-- Warehouse Runtime: Uses virtual warehouse, service function for agent access
--
-- Version: 1.0.82
-- Last Updated: December 25, 2024
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

-- =============================================================================
-- OPTION A: CONTAINER RUNTIME (Recommended for SSE Streaming)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Step 1: Create Stage for Streamlit Files (Container Runtime)
-- -----------------------------------------------------------------------------
CREATE STAGE IF NOT EXISTS STAGING.HEALTHCARE_STREAMLIT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Streamlit Container Runtime application files';

-- -----------------------------------------------------------------------------
-- Step 2: Create PyPI External Access Integration
-- Required for Container Runtime to install Python packages from PyPI
-- -----------------------------------------------------------------------------
CREATE OR REPLACE NETWORK RULE STAGING.PYPI_NETWORK_RULE
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('pypi.org', 'pypi.python.org', 'pythonhosted.org', 'files.pythonhosted.org');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION PYPI_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (STAGING.PYPI_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'PyPI access for Streamlit Container Runtime';

-- -----------------------------------------------------------------------------
-- Step 3: Upload Streamlit App Files (run from CLI)
-- -----------------------------------------------------------------------------
-- Run these commands from terminal:
/*
cd /path/to/healthcare

# Upload app.py and pyproject.toml to stage root
snow sql -c jwt -q "PUT file://streamlit/app.py @HEALTHCARE_DB.STAGING.HEALTHCARE_STREAMLIT_STAGE OVERWRITE=TRUE AUTO_COMPRESS=FALSE;"
snow sql -c jwt -q "PUT file://streamlit/pyproject.toml @HEALTHCARE_DB.STAGING.HEALTHCARE_STREAMLIT_STAGE OVERWRITE=TRUE AUTO_COMPRESS=FALSE;"
*/

-- Verify files uploaded
LIST @STAGING.HEALTHCARE_STREAMLIT_STAGE;

-- -----------------------------------------------------------------------------
-- Step 4: Create Streamlit App with Container Runtime
-- Uses SPCS compute pool for execution with internal DNS access
-- -----------------------------------------------------------------------------
CREATE OR REPLACE STREAMLIT STAGING.HEALTHCARE_ASSISTANT
    FROM @STAGING.HEALTHCARE_STREAMLIT_STAGE
    MAIN_FILE = 'app.py'
    QUERY_WAREHOUSE = PAYERS_CC_WH
    RUNTIME_NAME = 'SYSTEM$ST_CONTAINER_RUNTIME_PY3_11'
    COMPUTE_POOL = AGENTS_POOL
    EXTERNAL_ACCESS_INTEGRATIONS = (PYPI_ACCESS_INTEGRATION)
    COMMENT = 'Healthcare Contact Center Assistant v1.0.82 - Container Runtime with SSE streaming';

-- -----------------------------------------------------------------------------
-- Step 5: Grant Permissions
-- -----------------------------------------------------------------------------
-- Grant USAGE to specific roles
GRANT USAGE ON STREAMLIT STAGING.HEALTHCARE_ASSISTANT TO ROLE PUBLIC;

-- Future grants for internal Streamlit services (prevents permission errors on logs/overview)
GRANT OPERATE ON FUTURE SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;
GRANT MONITOR ON FUTURE SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;

-- Grant on all existing services (includes internal Streamlit container services)
GRANT OPERATE ON ALL SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;
GRANT MONITOR ON ALL SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;

-- -----------------------------------------------------------------------------
-- Step 6: Verify Deployment
-- -----------------------------------------------------------------------------
SHOW STREAMLITS IN SCHEMA STAGING;
DESCRIBE STREAMLIT STAGING.HEALTHCARE_ASSISTANT;

-- Check runtime type
SELECT 
    STREAMLIT_NAME,
    ROOT_LOCATION,
    QUERY_WAREHOUSE,
    COMMENT
FROM INFORMATION_SCHEMA.STREAMLITS
WHERE STREAMLIT_SCHEMA = 'STAGING';


-- =============================================================================
-- OPTION B: WAREHOUSE RUNTIME (Legacy - No Streaming Support)
-- =============================================================================
/*
-- Use this if you don't need SSE streaming or don't have a compute pool

-- Create legacy stage
CREATE STAGE IF NOT EXISTS STAGING.STREAMLIT_APPS
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Streamlit Warehouse Runtime files';

-- Upload files
-- snow sql -c jwt -q "PUT file://streamlit/app.py @HEALTHCARE_DB.STAGING.STREAMLIT_APPS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"

-- Create with ROOT_LOCATION (legacy syntax)
CREATE OR REPLACE STREAMLIT STAGING.HEALTHCARE_ASSISTANT
    ROOT_LOCATION = '@HEALTHCARE_DB.STAGING.STREAMLIT_APPS'
    MAIN_FILE = 'app.py'
    QUERY_WAREHOUSE = 'PAYERS_CC_WH'
    COMMENT = 'Healthcare Contact Center Assistant - Warehouse Runtime';

GRANT USAGE ON STREAMLIT STAGING.HEALTHCARE_ASSISTANT TO ROLE PUBLIC;
*/


-- =============================================================================
-- Troubleshooting Commands
-- =============================================================================
/*
-- Check stage contents
LIST @STAGING.HEALTHCARE_STREAMLIT_STAGE;

-- Check service status (Container Runtime creates internal service)
SHOW SERVICES IN SCHEMA STAGING LIKE 'STPLATSTREAMLIT%';

-- Get logs from internal Streamlit service (requires MONITOR privilege)
-- Replace STPLATSTREAMLIT... with actual service name from SHOW SERVICES
SELECT SYSTEM$GET_SERVICE_LOGS('HEALTHCARE_DB.STAGING.STPLATSTREAMLIT...', 0, 'stplatstreamlit...', 50);

-- Re-upload files
PUT file://streamlit/app.py @HEALTHCARE_DB.STAGING.HEALTHCARE_STREAMLIT_STAGE OVERWRITE=TRUE AUTO_COMPRESS=FALSE;
PUT file://streamlit/pyproject.toml @HEALTHCARE_DB.STAGING.HEALTHCARE_STREAMLIT_STAGE OVERWRITE=TRUE AUTO_COMPRESS=FALSE;

-- Refresh stage
ALTER STAGE STAGING.HEALTHCARE_STREAMLIT_STAGE REFRESH;

-- Drop and recreate if issues
DROP STREAMLIT IF EXISTS STAGING.HEALTHCARE_ASSISTANT;

-- Check EAI status
SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI%';

-- Grant permissions on new internal service after recreation
-- (Run this if you see "Insufficient privileges" on logs/overview)
GRANT OPERATE ON ALL SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;
GRANT MONITOR ON ALL SERVICES IN SCHEMA STAGING TO ROLE ACCOUNTADMIN;
*/


-- =============================================================================
-- Container Runtime vs Warehouse Runtime Comparison
-- =============================================================================
/*
| Feature                  | Container Runtime         | Warehouse Runtime        |
|--------------------------|---------------------------|--------------------------|
| Compute                  | SPCS Compute Pool         | Virtual Warehouse        |
| Instance                 | Shared across viewers     | Personal per viewer      |
| Network                  | Internal DNS to SPCS      | External only            |
| SSE Streaming            | ✅ Direct HTTP to agent   | ❌ Service function only |
| Dependencies             | pyproject.toml            | environment.yml          |
| Startup Time             | Slower (container build)  | Faster                   |
| Cost                     | Compute pool credits      | Warehouse credits        |
| Syntax                   | FROM @stage               | ROOT_LOCATION = @stage   |
*/
