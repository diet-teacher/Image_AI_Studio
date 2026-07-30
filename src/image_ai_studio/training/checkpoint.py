"""state_dict 저장/재로드만 지원 (optimizer state/epoch 등 full checkpoint는 Phase 4A 범위 밖)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_state_dict(model: nn.Module, path: str | Path) -> None:
    """model.state_dict()를 파일로 저장. 상위 디렉터리 자동 생성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict(model: nn.Module, path: str | Path, *, map_location: str = "cpu") -> nn.Module:
    """저장된 state_dict를 model에 로드 (in-place) 후 그대로 반환."""
    path = Path(path)
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return model
