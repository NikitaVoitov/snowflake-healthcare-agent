-- Phase 2: Cortex Search Services Setup
-- Creates search services for FAQs, Policies, and Call Transcripts
-- These services enable semantic search across knowledge base content
-- Target: HEALTHCARE_DB.KNOWLEDGE_SCHEMA

USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

--------------------------------------------------------------------------------
-- SECTION 1: CORTEX SEARCH FOR FAQs
-- Enables semantic search over parsed FAQ documents from Enterprise_NXT PDFs
-- Search column: answer (contains full FAQ text extracted from PDFs)
-- Filter attribute: category
--------------------------------------------------------------------------------

CREATE OR REPLACE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.FAQS_SEARCH
  ON answer
  ATTRIBUTES category
  WAREHOUSE = PAYERS_CC_WH
  TARGET_LAG = '1 hour'
  COMMENT = 'Semantic search over FAQ documents for healthcare contact center'
  AS (
    SELECT 
      faq_id, 
      question,  -- filename/title
      answer,    -- full text content (search column)
      category,
      updated_date
    FROM KNOWLEDGE_SCHEMA.FAQS
  );

--------------------------------------------------------------------------------
-- SECTION 2: CORTEX SEARCH FOR POLICIES
-- Enables semantic search over policy documents
-- Search column: content
-- Filter attribute: policy_name
-- NOTE: POLICIES table is currently empty - service will activate when data is loaded
--------------------------------------------------------------------------------

CREATE OR REPLACE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.POLICIES_SEARCH
  ON content
  ATTRIBUTES policy_name
  WAREHOUSE = PAYERS_CC_WH
  TARGET_LAG = '1 hour'
  COMMENT = 'Semantic search over policy documents for healthcare contact center'
  AS (
    SELECT 
      policy_id, 
      policy_name, 
      content, 
      version,
      effective_date
    FROM KNOWLEDGE_SCHEMA.POLICIES
  );

--------------------------------------------------------------------------------
-- SECTION 3: CORTEX SEARCH FOR CALL TRANSCRIPTS
-- Enables semantic search over transcribed call recordings
-- Search column: transcript_text
-- Filter attributes: member_id (for member-specific searches)
-- NOTE: Transcripts are populated by the AI_TRANSCRIBE scheduled task
--------------------------------------------------------------------------------

CREATE OR REPLACE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.TRANSCRIPTS_SEARCH
  ON transcript_text
  ATTRIBUTES member_id
  WAREHOUSE = PAYERS_CC_WH
  TARGET_LAG = '1 hour'
  COMMENT = 'Semantic search over call transcripts for healthcare contact center'
  AS (
    SELECT 
      transcript_id, 
      audio_id,
      member_id, 
      call_date, 
      transcript_text, 
      summary
    FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS
  );

--------------------------------------------------------------------------------
-- SECTION 4: VALIDATION QUERIES
--------------------------------------------------------------------------------

-- List all Cortex Search services in KNOWLEDGE_SCHEMA
SHOW CORTEX SEARCH SERVICES IN SCHEMA KNOWLEDGE_SCHEMA;

-- Check service details
-- DESCRIBE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.FAQS_SEARCH;
-- DESCRIBE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.POLICIES_SEARCH;
-- DESCRIBE CORTEX SEARCH SERVICE KNOWLEDGE_SCHEMA.TRANSCRIPTS_SEARCH;

--------------------------------------------------------------------------------
-- SECTION 5: SAMPLE SEARCH QUERIES (for testing)
--------------------------------------------------------------------------------

/*
-- Search FAQs for enrollment information
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'KNOWLEDGE_SCHEMA.FAQS_SEARCH',
  '{
    "query": "How do I enroll employees in a health plan?",
    "columns": ["faq_id", "question", "answer"],
    "limit": 3
  }'
);

-- Search Transcripts for specific member
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'KNOWLEDGE_SCHEMA.TRANSCRIPTS_SEARCH',
  '{
    "query": "claim status inquiry",
    "columns": ["transcript_id", "member_id", "summary"],
    "filter": {"member_id": "MEM123"},
    "limit": 5
  }'
);
*/

