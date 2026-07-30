"""TrainingHistory JSON 저장/로드. 표준 json만 사용, 새 외부 의존성 없음.

checkpoint.py는 model state_dict(텐서) 저장/로드 책임만 유지하고, 이
파일은 TrainingHistory(순수 float/int 메트릭) 저장/로드만 담당한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from image_ai_studio.training.loop import TrainingHistory


def save_training_history(history: TrainingHistory, path: str | Path) -> None:
    """TrainingHistory를 JSON 파일로 저장. 상위 디렉터리 자동 생성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")


def load_training_history(path: str | Path) -> TrainingHistory:
    """save_training_history()로 저장한 JSON을 TrainingHistory로 복원."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainingHistory(**data)
