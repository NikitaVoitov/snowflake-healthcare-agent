"""Base Pydantic schema with shared configuration."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base schema with camelCase serialization and validation defaults."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
        from_attributes=True,
    )

