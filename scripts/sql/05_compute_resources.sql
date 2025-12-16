-- Phase 2: Compute Resources Setup
-- Creates compute pool for SPCS agent containers and warehouses for AI workloads
-- Target: HEALTHCARE_DB infrastructure

USE DATABASE HEALTHCARE_DB;

--------------------------------------------------------------------------------
-- SECTION 1: COMPUTE POOL FOR SPCS AGENTS
-- Used by the LangGraph multi-agent container service
-- CPU_X64_M provides balanced compute for agent orchestration
--------------------------------------------------------------------------------

CREATE COMPUTE POOL IF NOT EXISTS AGENTS_POOL
  MIN_NODES = 1
  MAX_NODES = 3
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300
  COMMENT = 'Compute pool for Healthcare Multi-Agent LangGraph containers';

--------------------------------------------------------------------------------
-- SECTION 2: TRANSCRIPTION WAREHOUSE
-- Dedicated warehouse for AI_TRANSCRIBE audio processing
-- Medium size for efficient parallel transcription of multiple files
--------------------------------------------------------------------------------

CREATE WAREHOUSE IF NOT EXISTS TRANSCRIPTION_WH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  COMMENT = 'Dedicated warehouse for AI_TRANSCRIBE audio processing';

--------------------------------------------------------------------------------
-- SECTION 3: GPU COMPUTE POOL (Optional - for advanced ML workloads)
-- Use for custom ML models or advanced Cortex functions requiring GPU
-- NOTE: GPU pools incur higher costs - enable only when needed
--------------------------------------------------------------------------------

-- Uncomment to create GPU pool for ML inference:
/*
CREATE COMPUTE POOL IF NOT EXISTS HEALTHCARE_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300
  COMMENT = 'GPU compute pool for ML inference workloads';
*/

--------------------------------------------------------------------------------
-- SECTION 4: IMAGE REPOSITORY FOR SPCS
-- Repository to store Docker images for the multi-agent container
--------------------------------------------------------------------------------

CREATE IMAGE REPOSITORY IF NOT EXISTS STAGING.HEALTHCARE_IMAGES
  COMMENT = 'Image repository for Healthcare Multi-Agent containers';

--------------------------------------------------------------------------------
-- SECTION 5: NETWORK RULES FOR EXTERNAL ACCESS (if needed)
-- Enable outbound network access for the SPCS container
--------------------------------------------------------------------------------

-- Network rule for general HTTPS access
CREATE OR REPLACE NETWORK RULE STAGING.HEALTHCARE_EGRESS_RULE
  TYPE = HOST_PORT
  MODE = EGRESS
  VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80')
  COMMENT = 'Allow outbound HTTPS/HTTP for Healthcare agents';

-- External access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION HEALTHCARE_EXTERNAL_ACCESS
  ALLOWED_NETWORK_RULES = (STAGING.HEALTHCARE_EGRESS_RULE)
  ENABLED = TRUE
  COMMENT = 'External access for Healthcare Multi-Agent containers';

--------------------------------------------------------------------------------
-- SECTION 6: VALIDATION QUERIES
--------------------------------------------------------------------------------

-- Check compute pools
SHOW COMPUTE POOLS;

-- Check warehouses
SHOW WAREHOUSES LIKE '%_WH';

-- Check image repositories
SHOW IMAGE REPOSITORIES IN SCHEMA STAGING;

-- Describe compute pool details
-- DESCRIBE COMPUTE POOL AGENTS_POOL;

