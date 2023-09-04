try:
    from pydantic.v1 import BaseModel, Field, validate_arguments
except ImportError:
    from pydantic import BaseModel, Field, validate_arguments

__all__ = ["BaseModel", "Field", "validate_arguments"]
