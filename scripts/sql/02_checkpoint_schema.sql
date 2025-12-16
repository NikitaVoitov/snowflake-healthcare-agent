-- Phase 1: Checkpoint schema for LangGraph AsyncSnowflakeSaver

USE DATABASE HEALTHCARE_DB;
USE SCHEMA CHECKPOINT_SCHEMA;

CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINTS (
  checkpoint_ns TEXT,
  checkpoint_id TEXT,
  parent_checkpoint_id TEXT,
  type_qualifier TEXT,
  checkpoint OBJECT,
  metadata OBJECT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINT_BLOBS (
  checkpoint_ns TEXT,
  checkpoint_id TEXT,
  channel TEXT,
  blob VARIANT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (checkpoint_ns, checkpoint_id, channel)
);

CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINT_WRITES (
  checkpoint_ns TEXT,
  checkpoint_id TEXT,
  task_id TEXT,
  channel TEXT,
  version INT,
  value VARIANT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (checkpoint_ns, checkpoint_id, task_id, channel)
);

CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINT_MIGRATIONS (
  checkpoint_ns TEXT,
  migration_id TEXT,
  version INT,
  description TEXT,
  migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (checkpoint_ns, migration_id)
);
