"""Shared exception type for the model_definition package."""
from __future__ import annotations


class ModelValidationError(ValueError):
    """Raised when a ModelSpec or LayerSpec is invalid.

    Messages are written to be understandable when shown directly to a
    future UI user, not just to a developer reading a traceback.
    """
