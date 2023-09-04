try:
    from pydantic.v1 import BaseModel, validate_arguments
except ImportError:
    from pydantic import BaseModel, validate_arguments

__all__ = ["BaseModel", "validate_arguments"]
