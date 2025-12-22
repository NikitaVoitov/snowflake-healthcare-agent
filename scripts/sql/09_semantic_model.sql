-- =============================================================================
-- Phase 9: Semantic Model Deployment for Cortex Analyst
-- =============================================================================
-- This script creates the stage for semantic model YAML files and provides
-- instructions for uploading the healthcare_semantic_model.yaml file.
--
-- Prerequisites:
--   - HEALTHCARE_DB database exists
--   - STAGING schema exists
--   - ACCOUNTADMIN or equivalent role with stage creation permissions
--
-- Usage:
--   1. Run this script to create the stage
--   2. Upload the semantic model using Snow CLI (see comments below)
--   3. Verify the upload with LIST command
-- =============================================================================

USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

-- -----------------------------------------------------------------------------
-- SECTION 1: Create Stage for Semantic Models
-- -----------------------------------------------------------------------------

-- Create an internal stage for semantic model YAML files
-- Using SSE encryption (Snowflake Server-Side Encryption)
CREATE STAGE IF NOT EXISTS STAGING.SEMANTIC_MODELS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stage for Cortex Analyst semantic model YAML files';

-- Grant usage to roles that need to access the semantic model
GRANT USAGE ON STAGE STAGING.SEMANTIC_MODELS TO ROLE ACCOUNTADMIN;

-- If you have a service role for the healthcare agent, grant access:
-- GRANT READ ON STAGE STAGING.SEMANTIC_MODELS TO ROLE HEALTHCARE_SERVICE_ROLE;

-- -----------------------------------------------------------------------------
-- SECTION 2: Upload Semantic Model (Run from Local Terminal)
-- -----------------------------------------------------------------------------
/*
After running the stage creation above, upload the semantic model YAML file
using Snow CLI from your local terminal:

# Option 1: Using Snow CLI (recommended)
cd /Users/nvoitov/Documents/Splunk/Snowflake/healthcare
snow sql -c jwt --query "PUT file://scripts/sql/semantic_models/healthcare_semantic_model.yaml @HEALTHCARE_DB.STAGING.SEMANTIC_MODELS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"

# Option 2: Using Snowflake Web UI (Snowsight)
# 1. Navigate to Data > Databases > HEALTHCARE_DB > STAGING > Stages
# 2. Click on SEMANTIC_MODELS stage
# 3. Click "Upload Files" and select healthcare_semantic_model.yaml

# After uploading, refresh the directory table:
snow sql -c jwt --query "ALTER STAGE HEALTHCARE_DB.STAGING.SEMANTIC_MODELS REFRESH"
*/

-- -----------------------------------------------------------------------------
-- SECTION 3: Verification Commands
-- -----------------------------------------------------------------------------

-- List files in the semantic models stage
LIST @STAGING.SEMANTIC_MODELS;

-- Check the stage properties
DESCRIBE STAGE STAGING.SEMANTIC_MODELS;

-- Query directory table to see file metadata
SELECT * FROM DIRECTORY(@STAGING.SEMANTIC_MODELS);

-- -----------------------------------------------------------------------------
-- SECTION 4: Test Cortex Analyst API (Optional)
-- -----------------------------------------------------------------------------
/*
-- Test the semantic model by calling Cortex Analyst directly via SQL
-- Note: This requires the Cortex Analyst API to be available in your region

-- Example: Using SYSTEM$CORTEX_ANALYST_MESSAGE (if available)
-- SELECT SYSTEM$CORTEX_ANALYST_MESSAGE(
--     '@HEALTHCARE_DB.STAGING.SEMANTIC_MODELS/healthcare_semantic_model.yaml',
--     'How many members do we have in total?'
-- );
*/

-- -----------------------------------------------------------------------------
-- SECTION 5: Cleanup (if needed)
-- -----------------------------------------------------------------------------
/*
-- Remove a specific file from the stage
-- REMOVE @STAGING.SEMANTIC_MODELS/healthcare_semantic_model.yaml;

-- Drop the entire stage (WARNING: deletes all files)
-- DROP STAGE IF EXISTS STAGING.SEMANTIC_MODELS;
*/

-- Show confirmation
SELECT 'Semantic model stage created successfully' AS STATUS;
