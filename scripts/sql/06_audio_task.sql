-- Phase 3: Audio Transcription Pipeline
-- Creates scheduled task to transcribe MP3 files using AI_TRANSCRIBE
-- Generates summaries and sentiment analysis using Cortex LLM functions
-- Target: HEALTHCARE_DB.KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS

USE DATABASE HEALTHCARE_DB;

--------------------------------------------------------------------------------
-- SECTION 1: CREATE DEDICATED WAREHOUSE FOR TRANSCRIPTION
-- Medium warehouse for AI_TRANSCRIBE workloads (can process multiple files)
--------------------------------------------------------------------------------

CREATE WAREHOUSE IF NOT EXISTS TRANSCRIPTION_WH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  COMMENT = 'Dedicated warehouse for AI_TRANSCRIBE audio processing';

--------------------------------------------------------------------------------
-- SECTION 2: CREATE SCHEDULED TASK FOR DAILY TRANSCRIPTION
-- Runs daily at 2 AM UTC to process new audio files
-- Uses speaker diarization for call center recordings
--------------------------------------------------------------------------------

CREATE OR REPLACE TASK KNOWLEDGE_SCHEMA.AUDIO_TRANSCRIPTION_DAILY
  WAREHOUSE = TRANSCRIPTION_WH
  SCHEDULE = 'USING CRON 0 2 * * * UTC'
  COMMENT = 'Daily task to transcribe new audio files and generate summaries'
AS
  -- Process audio files that haven't been transcribed yet
  INSERT INTO KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS 
    (transcript_id, audio_id, member_id, call_date, transcript_text, summary, sentiment)
  WITH new_audio AS (
    SELECT 
      af.audio_id,
      af.member_id,
      af.call_date,
      -- Extract just the filename from the full stage path for TO_FILE
      -- call_recording_path format: @HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE/call_script_X.mp3
      SPLIT_PART(af.call_recording_path, '/', -1) AS filename,
      AI_TRANSCRIBE(
        TO_FILE('@HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE', SPLIT_PART(af.call_recording_path, '/', -1)),
        {'timestamp_granularity': 'speaker'}
      ) AS raw_transcript
    FROM KNOWLEDGE_SCHEMA.AUDIO_FILES af
    WHERE af.audio_id NOT IN (
      SELECT ct.audio_id 
      FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS ct
      WHERE ct.audio_id IS NOT NULL
    )
  )
  SELECT 
    'TRANS_' || audio_id AS transcript_id,
    audio_id,
    member_id,
    COALESCE(call_date, CURRENT_DATE()) AS call_date,
    -- Extract text from JSON response
    raw_transcript:text::TEXT AS transcript_text,
    -- Generate summary using Cortex LLM
    SNOWFLAKE.CORTEX.SUMMARIZE(raw_transcript:text::TEXT) AS summary,
    -- Analyze sentiment
    SNOWFLAKE.CORTEX.SENTIMENT(raw_transcript:text::TEXT) AS sentiment
  FROM new_audio
  WHERE raw_transcript:text IS NOT NULL;

-- Resume the task to enable scheduled execution
ALTER TASK KNOWLEDGE_SCHEMA.AUDIO_TRANSCRIPTION_DAILY RESUME;

--------------------------------------------------------------------------------
-- SECTION 3: MANUAL EXECUTION PROCEDURE
-- Use this to run transcription immediately instead of waiting for schedule
--------------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE KNOWLEDGE_SCHEMA.TRANSCRIBE_NEW_AUDIO()
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
  INSERT INTO KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS 
    (transcript_id, audio_id, member_id, call_date, transcript_text, summary, sentiment)
  WITH new_audio AS (
    SELECT 
      af.audio_id,
      af.member_id,
      af.call_date,
      SPLIT_PART(af.call_recording_path, '/', -1) AS filename,
      AI_TRANSCRIBE(
        TO_FILE('@HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE', SPLIT_PART(af.call_recording_path, '/', -1)),
        {'timestamp_granularity': 'speaker'}
      ) AS raw_transcript
    FROM KNOWLEDGE_SCHEMA.AUDIO_FILES af
    WHERE af.audio_id NOT IN (
      SELECT ct.audio_id 
      FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS ct
      WHERE ct.audio_id IS NOT NULL
    )
  )
  SELECT 
    'TRANS_' || audio_id AS transcript_id,
    audio_id,
    member_id,
    COALESCE(call_date, CURRENT_DATE()) AS call_date,
    raw_transcript:text::TEXT AS transcript_text,
    SNOWFLAKE.CORTEX.SUMMARIZE(raw_transcript:text::TEXT) AS summary,
    SNOWFLAKE.CORTEX.SENTIMENT(raw_transcript:text::TEXT) AS sentiment
  FROM new_audio
  WHERE raw_transcript:text IS NOT NULL;
  
  RETURN 'Transcription complete. Processed ' || SQLROWCOUNT || ' audio files.';
END;
$$;

--------------------------------------------------------------------------------
-- SECTION 4: SINGLE FILE TRANSCRIPTION (for testing)
-- Transcribe a single audio file by audio_id
--------------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE KNOWLEDGE_SCHEMA.TRANSCRIBE_SINGLE_AUDIO(p_audio_id VARCHAR)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
DECLARE
  v_filename VARCHAR;
  v_result VARIANT;
BEGIN
  -- Get the filename from AUDIO_FILES
  SELECT SPLIT_PART(call_recording_path, '/', -1) 
  INTO v_filename
  FROM KNOWLEDGE_SCHEMA.AUDIO_FILES 
  WHERE audio_id = :p_audio_id;
  
  IF (v_filename IS NULL) THEN
    RETURN OBJECT_CONSTRUCT('error', 'Audio file not found: ' || :p_audio_id);
  END IF;
  
  -- Transcribe the file
  SELECT AI_TRANSCRIBE(
    TO_FILE('@HEALTHCARE_DB.STAGING.AUDIO_STAGE_SSE', :v_filename),
    {'timestamp_granularity': 'speaker'}
  ) INTO v_result;
  
  RETURN v_result;
END;
$$;

--------------------------------------------------------------------------------
-- SECTION 5: TASK MONITORING QUERIES
--------------------------------------------------------------------------------

-- Check task status
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) 
-- WHERE NAME = 'AUDIO_TRANSCRIPTION_DAILY' ORDER BY SCHEDULED_TIME DESC LIMIT 10;

-- Check pending audio files (not yet transcribed)
-- SELECT COUNT(*) AS pending_count FROM KNOWLEDGE_SCHEMA.AUDIO_FILES af
-- WHERE af.audio_id NOT IN (SELECT audio_id FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS WHERE audio_id IS NOT NULL);

-- Check transcription progress
-- SELECT 
--   (SELECT COUNT(*) FROM KNOWLEDGE_SCHEMA.AUDIO_FILES) AS total_audio_files,
--   (SELECT COUNT(*) FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS) AS transcribed_files,
--   (SELECT COUNT(*) FROM KNOWLEDGE_SCHEMA.AUDIO_FILES WHERE audio_id NOT IN 
--     (SELECT audio_id FROM KNOWLEDGE_SCHEMA.CALL_TRANSCRIPTS WHERE audio_id IS NOT NULL)) AS pending_files;

--------------------------------------------------------------------------------
-- SECTION 6: EXECUTE TASK MANUALLY (for immediate transcription)
-- Uncomment to run transcription now instead of waiting for scheduled time
--------------------------------------------------------------------------------

-- EXECUTE TASK KNOWLEDGE_SCHEMA.AUDIO_TRANSCRIPTION_DAILY;

-- Or use the stored procedure:
-- CALL KNOWLEDGE_SCHEMA.TRANSCRIBE_NEW_AUDIO();

-- Test single file transcription:
-- CALL KNOWLEDGE_SCHEMA.TRANSCRIBE_SINGLE_AUDIO('AUDIO_1');

