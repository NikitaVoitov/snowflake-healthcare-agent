-- Phase 2: Load foundational data from notebooks/data assets
-- Source: notebooks/data/ (CSV, PDF, MP3 files)
-- Target: HEALTHCARE_DB tables and stages
-- 
-- EXECUTED SUCCESSFULLY: 2025-12-16
-- Results:
--   - CALL_CENTER_MEMBER_DENORMALIZED: 632 rows
--   - MEMBERS: 242 rows (distinct from denormalized)
--   - CLAIMS: 632 rows
--   - COVERAGE: 242 rows (distinct plans)
--   - CALLER_INTENT_TRAIN: 1000 rows
--   - CALLER_INTENT_PREDICT: 242 rows
--   - FAQS: 4 rows (parsed from PDFs)
--   - AUDIO_FILES: 32 rows (registered for transcription)

USE DATABASE HEALTHCARE_DB;
USE WAREHOUSE PAYERS_CC_WH;

--------------------------------------------------------------------------------
-- SECTION 1: FILE FORMATS
--------------------------------------------------------------------------------

-- CSV format for structured data
CREATE OR REPLACE FILE FORMAT STAGING.CSV_FORMAT
  TYPE = 'CSV'
  FIELD_DELIMITER = ','
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF = ('', 'NULL', 'null')
  EMPTY_FIELD_AS_NULL = TRUE;

--------------------------------------------------------------------------------
-- SECTION 2: STAGES WITH SERVER-SIDE ENCRYPTION (required for PARSE_DOCUMENT)
--------------------------------------------------------------------------------

-- Stage for CSV, PDF files (SSE required for AI_PARSE_DOCUMENT)
CREATE OR REPLACE STAGE STAGING.RAW_DATA_SSE
DIRECTORY = (ENABLE = TRUE)
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for audio files (SSE for AI_TRANSCRIBE)
CREATE OR REPLACE STAGE STAGING.AUDIO_STAGE_SSE
DIRECTORY = (ENABLE = TRUE)
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Enable directory on original stages for metadata
ALTER STAGE STAGING.RAW_DATA SET DIRECTORY = (ENABLE = TRUE);
ALTER STAGE STAGING.AUDIO_STAGE SET DIRECTORY = (ENABLE = TRUE);

--------------------------------------------------------------------------------
-- SECTION 3: PUT FILES TO STAGES (run via Snow CLI from project root)
--------------------------------------------------------------------------------
/*
-- CSV data files
snow sql -c jwt --query "PUT file://notebooks/data/DATA_PRODUCT/CALL_CENTER_MEMBER_DENORMALIZED.csv @HEALTHCARE_DB.STAGING.RAW_DATA/DATA_PRODUCT/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
snow sql -c jwt --query "PUT file://notebooks/data/CALLER_INTENT/CALLER_INTENT_TRAIN_DATASET.csv @HEALTHCARE_DB.STAGING.RAW_DATA/CALLER_INTENT/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
snow sql -c jwt --query "PUT file://notebooks/data/CALLER_INTENT/CALLER_INTENT_PREDICT_DATASET.csv @HEALTHCARE_DB.STAGING.RAW_DATA/CALLER_INTENT/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"

-- PDF FAQ files (use SSE stage for PARSE_DOCUMENT)
snow sql -c jwt --query "PUT file://notebooks/data/FAQS/Enterprise_NXT_FAQ1.pdf @HEALTHCARE_DB.STAGING.RAW_DATA_SSE/FAQS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
snow sql -c jwt --query "PUT file://notebooks/data/FAQS/Enterprise_NXT_FAQ2.pdf @HEALTHCARE_DB.STAGING.RAW_DATA_SSE/FAQS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
snow sql -c jwt --query "PUT file://notebooks/data/FAQS/Enterprise_NXT_FAQ3.pdf @HEALTHCARE_DB.STAGING.RAW_DATA_SSE/FAQS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
snow sql -c jwt --query "PUT file://notebooks/data/FAQS/Enterprise_NXT_FAQ4.pdf @HEALTHCARE_DB.STAGING.RAW_DATA_SSE/FAQS/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"

-- MP3 audio files (use SSE stage for AI_TRANSCRIBE)
for f in notebooks/data/CALL_RECORDINGS/call_script_*.mp3; do
  snow sql -c jwt --query "PUT file://$f @HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
done

-- Refresh directory tables after PUT
snow sql -c jwt --query "ALTER STAGE HEALTHCARE_DB.STAGING.RAW_DATA REFRESH"
snow sql -c jwt --query "ALTER STAGE HEALTHCARE_DB.STAGING.RAW_DATA_SSE REFRESH"
snow sql -c jwt --query "ALTER STAGE HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE REFRESH"
*/

--------------------------------------------------------------------------------
-- SECTION 4: DENORMALIZED MEMBER TABLE (matches original QuickStart)
--------------------------------------------------------------------------------

CREATE OR REPLACE TABLE MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED (
    MEMBER_ID VARCHAR(20),
    NAME VARCHAR(100),
    DOB DATE,
    GENDER VARCHAR(10),
    ADDRESS VARCHAR(500),
    MEMBER_PHONE VARCHAR(20),
    PLAN_ID VARCHAR(20),
    PLAN_NAME VARCHAR(100),
    CVG_START_DATE DATE,
    CVG_END_DATE DATE,
    PCP VARCHAR(100),
    PCP_PHONE VARCHAR(20),
    PLAN_TYPE VARCHAR(20),
    PREMIUM DECIMAL(10,2),
    SMOKER_IND BOOLEAN,
    LIFESTYLE_INFO VARCHAR(100),
    CHRONIC_CONDITION VARCHAR(100),
    GRIEVANCE_ID VARCHAR(20),
    GRIEVANCE_DATE DATE,
    GRIEVANCE_TYPE VARCHAR(100),
    GRIEVANCE_STATUS VARCHAR(50),
    GRIEVANCE_RESOLUTION_DATE DATE,
    CLAIM_ID VARCHAR(20),
    CLAIM_SERVICE_FROM_DATE DATE,
    CLAIM_PROVIDER VARCHAR(200),
    CLAIM_SERVICE VARCHAR(100),
    CLAIM_BILL_AMT DECIMAL(10,2),
    CLAIM_ALLOW_AMT DECIMAL(10,2),
    CLAIM_COPAY_AMT DECIMAL(10,2),
    CLAIM_COINSURANCE_AMT DECIMAL(10,2),
    CLAIM_DEDUCTIBLE_AMT DECIMAL(10,2),
    CLAIM_PAID_AMT DECIMAL(10,2),
    CLAIM_STATUS VARCHAR(20),
    CLAIM_PAID_DATE DATE,
    CLAIM_SERVICE_TO_DATE DATE,
    CLAIM_SUBMISSION_DATE DATE
);

COPY INTO MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
FROM @STAGING.RAW_DATA/DATA_PRODUCT/CALL_CENTER_MEMBER_DENORMALIZED.csv
FILE_FORMAT = (FORMAT_NAME = STAGING.CSV_FORMAT)
ON_ERROR = 'CONTINUE'
FORCE = TRUE;

--------------------------------------------------------------------------------
-- SECTION 5: NORMALIZED MEMBER TABLES (extracted from denormalized)
--------------------------------------------------------------------------------

-- MEMBERS: distinct members from denormalized table
TRUNCATE TABLE IF EXISTS MEMBER_SCHEMA.MEMBERS;

INSERT INTO MEMBER_SCHEMA.MEMBERS (member_id, dob, name, plan_id, status, address, phone)
SELECT DISTINCT
    MEMBER_ID,
    DOB,
    NAME,
    PLAN_ID,
    CASE 
        WHEN CVG_END_DATE >= CURRENT_DATE() THEN 'ACTIVE'
        ELSE 'INACTIVE'
    END AS status,
    ADDRESS,
    MEMBER_PHONE
FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
WHERE MEMBER_ID IS NOT NULL;

-- COVERAGE: distinct plans from denormalized table
TRUNCATE TABLE IF EXISTS MEMBER_SCHEMA.COVERAGE;

INSERT INTO MEMBER_SCHEMA.COVERAGE (plan_id, plan_name, deductible, copay_office, copay_er)
SELECT DISTINCT
    PLAN_ID,
    PLAN_NAME,
    AVG(CLAIM_DEDUCTIBLE_AMT) AS deductible,
    AVG(CLAIM_COPAY_AMT) AS copay_office,
    AVG(CLAIM_COPAY_AMT) * 2 AS copay_er
FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
WHERE PLAN_ID IS NOT NULL
GROUP BY PLAN_ID, PLAN_NAME;

-- CLAIMS: all claims from denormalized table
TRUNCATE TABLE IF EXISTS MEMBER_SCHEMA.CLAIMS;

INSERT INTO MEMBER_SCHEMA.CLAIMS (claim_id, member_id, claim_date, service_type, amount, status)
SELECT DISTINCT
    CLAIM_ID,
    MEMBER_ID,
    CLAIM_SERVICE_FROM_DATE AS claim_date,
    CLAIM_SERVICE AS service_type,
    CLAIM_BILL_AMT AS amount,
    CLAIM_STATUS AS status
FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
WHERE CLAIM_ID IS NOT NULL;

--------------------------------------------------------------------------------
-- SECTION 6: CALLER INTENT TABLES (for intent classification model)
--------------------------------------------------------------------------------

CREATE OR REPLACE TABLE KNOWLEDGE_SCHEMA.CALLER_INTENT_TRAIN_DATASET (
    MEMBER_ID VARCHAR(20),
    RECENT_ENROLLMENT_EVENT_IND BOOLEAN,
    PCP_CHANGE_IND BOOLEAN,
    ACTIVE_CM_PROGRAM_IND BOOLEAN,
    CHRONIC_CONDITION_IND BOOLEAN,
    ACTIVE_GRIEVANCE_IND BOOLEAN,
    ACTIVE_CLAIM_IND BOOLEAN,
    POTENTIAL_CALLER_INTENT_CATEGORY VARCHAR(100)
);

COPY INTO KNOWLEDGE_SCHEMA.CALLER_INTENT_TRAIN_DATASET
FROM @STAGING.RAW_DATA/CALLER_INTENT/CALLER_INTENT_TRAIN_DATASET.csv
FILE_FORMAT = (FORMAT_NAME = STAGING.CSV_FORMAT)
ON_ERROR = 'CONTINUE'
FORCE = TRUE;

CREATE OR REPLACE TABLE KNOWLEDGE_SCHEMA.CALLER_INTENT_PREDICT_DATASET (
    MEMBER_ID VARCHAR(20),
    RECENT_ENROLLMENT_EVENT_IND BOOLEAN,
    PCP_CHANGE_IND BOOLEAN,
    ACTIVE_CM_PROGRAM_IND BOOLEAN,
    CHRONIC_CONDITION_IND BOOLEAN,
    ACTIVE_GRIEVANCE_IND BOOLEAN,
    ACTIVE_CLAIM_IND BOOLEAN
);

COPY INTO KNOWLEDGE_SCHEMA.CALLER_INTENT_PREDICT_DATASET
FROM @STAGING.RAW_DATA/CALLER_INTENT/CALLER_INTENT_PREDICT_DATASET.csv
FILE_FORMAT = (FORMAT_NAME = STAGING.CSV_FORMAT)
ON_ERROR = 'CONTINUE'
FORCE = TRUE;

--------------------------------------------------------------------------------
-- SECTION 7: FAQ TABLE (parsed from PDFs using PARSE_DOCUMENT)
-- NOTE: Requires SSE-encrypted stage for PARSE_DOCUMENT to work
--------------------------------------------------------------------------------

-- Create staging table for raw PDF content
CREATE OR REPLACE TABLE KNOWLEDGE_SCHEMA.FAQ_RAW_CONTENT (
    file_name VARCHAR(200),
    file_path VARCHAR(500),
    parsed_content VARIANT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parse PDFs and extract text content (use SSE stage)
INSERT INTO KNOWLEDGE_SCHEMA.FAQ_RAW_CONTENT (file_name, file_path, parsed_content)
SELECT 
    RELATIVE_PATH AS file_name,
    '@STAGING.RAW_DATA_SSE/' || RELATIVE_PATH AS file_path,
    SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
        '@STAGING.RAW_DATA_SSE',
        RELATIVE_PATH,
        {'mode': 'OCR'}
    ) AS parsed_content
FROM DIRECTORY(@STAGING.RAW_DATA_SSE)
WHERE RELATIVE_PATH LIKE 'FAQS/%.pdf';

-- Populate FAQS table from parsed content
TRUNCATE TABLE IF EXISTS KNOWLEDGE_SCHEMA.FAQS;

INSERT INTO KNOWLEDGE_SCHEMA.FAQS (faq_id, question, answer, category, updated_date)
SELECT 
    'FAQ_' || ROW_NUMBER() OVER (ORDER BY file_name) AS faq_id,
    REPLACE(REPLACE(file_name, 'FAQS/', ''), '.pdf', '') AS question,
    parsed_content:content::TEXT AS answer,
    'Enterprise_NXT' AS category,
    CURRENT_DATE() AS updated_date
FROM KNOWLEDGE_SCHEMA.FAQ_RAW_CONTENT
WHERE parsed_content:content IS NOT NULL;

--------------------------------------------------------------------------------
-- SECTION 8: AUDIO FILES METADATA (register MP3s for transcription)
-- NOTE: Requires SSE-encrypted stage for AI_TRANSCRIBE to work
--------------------------------------------------------------------------------

TRUNCATE TABLE IF EXISTS KNOWLEDGE_SCHEMA.AUDIO_FILES;

INSERT INTO KNOWLEDGE_SCHEMA.AUDIO_FILES (audio_id, call_recording_path, uploaded_date)
SELECT 
    REPLACE(REPLACE(RELATIVE_PATH, 'call_script_', 'AUDIO_'), '.mp3', '') AS audio_id,
    '@STAGING.AUDIO_STAGE_SSE/' || RELATIVE_PATH AS call_recording_path,
    CURRENT_DATE() AS uploaded_date
FROM DIRECTORY(@STAGING.AUDIO_STAGE_SSE)
WHERE RELATIVE_PATH LIKE '%.mp3';

--------------------------------------------------------------------------------
-- SECTION 9: VALIDATION QUERIES
--------------------------------------------------------------------------------

-- Verify denormalized table load
SELECT 'CALL_CENTER_MEMBER_DENORMALIZED' AS table_name, COUNT(*) AS row_count 
FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED;

-- Verify normalized tables
SELECT 'MEMBERS' AS table_name, COUNT(*) AS row_count FROM MEMBER_SCHEMA.MEMBERS
UNION ALL
SELECT 'CLAIMS' AS table_name, COUNT(*) AS row_count FROM MEMBER_SCHEMA.CLAIMS
UNION ALL
SELECT 'COVERAGE' AS table_name, COUNT(*) AS row_count FROM MEMBER_SCHEMA.COVERAGE;

-- Verify caller intent tables
SELECT 'CALLER_INTENT_TRAIN' AS table_name, COUNT(*) AS row_count 
FROM KNOWLEDGE_SCHEMA.CALLER_INTENT_TRAIN_DATASET
UNION ALL
SELECT 'CALLER_INTENT_PREDICT' AS table_name, COUNT(*) AS row_count 
FROM KNOWLEDGE_SCHEMA.CALLER_INTENT_PREDICT_DATASET;

-- Verify FAQ parsing
SELECT 'FAQS' AS table_name, COUNT(*) AS row_count FROM KNOWLEDGE_SCHEMA.FAQS;

-- Verify audio files registration
SELECT 'AUDIO_FILES' AS table_name, COUNT(*) AS row_count FROM KNOWLEDGE_SCHEMA.AUDIO_FILES;

-- Full validation summary
SELECT 'CALL_CENTER_MEMBER_DENORMALIZED' AS table_name, COUNT(*) AS row_count FROM MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
UNION ALL SELECT 'MEMBERS', COUNT(*) FROM MEMBER_SCHEMA.MEMBERS
UNION ALL SELECT 'CLAIMS', COUNT(*) FROM MEMBER_SCHEMA.CLAIMS
UNION ALL SELECT 'COVERAGE', COUNT(*) FROM MEMBER_SCHEMA.COVERAGE
UNION ALL SELECT 'CALLER_INTENT_TRAIN', COUNT(*) FROM KNOWLEDGE_SCHEMA.CALLER_INTENT_TRAIN_DATASET
UNION ALL SELECT 'CALLER_INTENT_PREDICT', COUNT(*) FROM KNOWLEDGE_SCHEMA.CALLER_INTENT_PREDICT_DATASET
UNION ALL SELECT 'FAQS', COUNT(*) FROM KNOWLEDGE_SCHEMA.FAQS
UNION ALL SELECT 'AUDIO_FILES', COUNT(*) FROM KNOWLEDGE_SCHEMA.AUDIO_FILES;
