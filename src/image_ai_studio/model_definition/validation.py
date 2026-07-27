"""Top-level validation entry point for a ModelSpec.

Two validation layers already run before this module is involved:

* Each ``LayerSpec`` dataclass validates its own parameters (kernel_size
  > 0, etc.) in ``__post_init__`` (see specs.py) -- this fires even when
  the spec is built from JSON.
* ``shape_inference.infer_model_shapes`` validates that each layer's
  output shape is a valid input for the next layer, and that no
  computed dimension collapses to zero or below.

``validate_model_spec`` is the single call ``builder.build_model`` makes
before constructing any ``torch.nn.Module``. A future UI can also call
it directly -- e.g. to show inline shape-validation errors while a
model is being edited, without needing torch installed at all, since
this module (like specs.py and shape_inference.py) never imports torch.
"""
from __future__ import annotations

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import LayerShapeInfo, infer_model_shapes
from image_ai_studio.model_definition.specs import ModelSpec

__all__ = ["ModelValidationError", "validate_model_spec"]


def validate_model_spec(model_spec: ModelSpec) -> list[LayerShapeInfo]:
    """Validate a ModelSpec end-to-end and return its per-layer shape trace.

    Raises ModelValidationError on the first problem found.
    """
    return infer_model_shapes(model_spec)
