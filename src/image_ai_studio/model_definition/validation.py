"""ModelSpec 검증 진입점.

- 파라미터 검증: 각 LayerSpec __post_init__ (specs.py)
- shape 연결 검증: shape_inference.infer_model_shapes
- validate_model_spec: 위 둘을 묶은 단일 진입점, builder.build_model에서 호출
- torch 미의존 (향후 UI 프로세스에서 torch 없이도 사용 가능)
"""
from __future__ import annotations

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import LayerShapeInfo, infer_model_shapes
from image_ai_studio.model_definition.specs import ModelSpec

__all__ = ["ModelValidationError", "validate_model_spec"]


def validate_model_spec(model_spec: ModelSpec) -> list[LayerShapeInfo]:
    """ModelSpec 검증 후 shape trace 반환. 실패 시 ModelValidationError."""
    return infer_model_shapes(model_spec)
