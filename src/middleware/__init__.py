"""Middleware package for Healthcare Agent.

This package contains:
- healthcare_prompts: Dynamic prompt generation middleware for LangChain v1
- logging: Request/response logging middleware for FastAPI
"""

from src.middleware.healthcare_prompts import (
    create_healthcare_middleware,
    healthcare_prompt,
)

__all__ = ["healthcare_prompt", "create_healthcare_middleware"]
