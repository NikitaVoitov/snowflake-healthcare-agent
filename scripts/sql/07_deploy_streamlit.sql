-- =============================================================================
-- Healthcare ReAct Agent - Streamlit App Deployment
-- Deploy Native Streamlit App to Snowsight
-- 
-- The app uses the service function STAGING.HEALTHCARE_AGENT_QUERY() to
-- communicate with the SPCS ReAct agent container. No external access needed.
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

-- -----------------------------------------------------------------------------
-- Step 1: Create Stage for Streamlit Files
-- -----------------------------------------------------------------------------
CREATE STAGE IF NOT EXISTS STAGING.STREAMLIT_APPS
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Streamlit application files';

-- -----------------------------------------------------------------------------
-- Step 2: Upload Streamlit App Files (run from CLI)
-- -----------------------------------------------------------------------------
-- Run these commands from terminal:
-- snow sql -c jwt --query "PUT file://streamlit/app.py @HEALTHCARE_DB.STAGING.STREAMLIT_APPS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
-- snow sql -c jwt --query "PUT file://streamlit/environment.yml @HEALTHCARE_DB.STAGING.STREAMLIT_APPS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"

-- Verify files uploaded
LIST @STAGING.STREAMLIT_APPS;

-- -----------------------------------------------------------------------------
-- Step 3: Create Streamlit App
-- -----------------------------------------------------------------------------
CREATE OR REPLACE STREAMLIT STAGING.HEALTHCARE_ASSISTANT
    ROOT_LOCATION = '@HEALTHCARE_DB.STAGING.STREAMLIT_APPS'
    MAIN_FILE = 'app.py'
    QUERY_WAREHOUSE = 'PAYERS_CC_WH'
    COMMENT = 'Healthcare Contact Center ReAct Agent Assistant v1.0.47';

-- -----------------------------------------------------------------------------
-- Step 4: Grant Access
-- -----------------------------------------------------------------------------
-- Grant to specific roles as needed
GRANT USAGE ON STREAMLIT STAGING.HEALTHCARE_ASSISTANT TO ROLE PUBLIC;

-- -----------------------------------------------------------------------------
-- Step 5: Verify Deployment
-- -----------------------------------------------------------------------------
SHOW STREAMLITS IN SCHEMA STAGING;

-- Get Streamlit URL
DESCRIBE STREAMLIT STAGING.HEALTHCARE_ASSISTANT;

-- -----------------------------------------------------------------------------
-- Troubleshooting Commands
-- -----------------------------------------------------------------------------
/*
-- Check stage contents
LIST @STAGING.STREAMLIT_APPS;

-- Re-upload files if needed
PUT file://streamlit/app.py @HEALTHCARE_DB.STAGING.STREAMLIT_APPS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://streamlit/environment.yml @HEALTHCARE_DB.STAGING.STREAMLIT_APPS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Refresh stage
ALTER STAGE STAGING.STREAMLIT_APPS REFRESH;

-- Drop and recreate if issues
DROP STREAMLIT IF EXISTS STAGING.HEALTHCARE_ASSISTANT;
*/

