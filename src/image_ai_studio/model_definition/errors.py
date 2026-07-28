"""model_definition 공용 예외 타입."""
from __future__ import annotations


class ModelValidationError(ValueError):
    """ModelSpec/LayerSpec 검증 실패 시 발생. 메시지는 UI 노출 가능한 수준으로 작성."""
