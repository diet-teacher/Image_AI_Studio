"""TrainingHistory JSON 저장/로드 (history.py) round-trip 테스트."""
from __future__ import annotations

import json
from pathlib import Path

from image_ai_studio.training.history import load_training_history, save_training_history
from image_ai_studio.training.loop import TrainingHistory


def _example_history() -> TrainingHistory:
    return TrainingHistory(
        train_losses=[1.0, 0.5, 0.25],
        val_losses=[1.1, 0.6, 0.7],
        val_accuracies=[0.2, 0.6, 0.55],
        best_epoch=2,
        best_val_loss=0.6,
    )


def test_save_and_load_history_round_trips(tmp_path: Path) -> None:
    original = _example_history()
    path = tmp_path / "history.json"
    save_training_history(original, path)
    assert path.exists()

    restored = load_training_history(path)
    assert restored == original


def test_save_history_creates_parent_directories(tmp_path: Path) -> None:
    history = _example_history()
    nested_path = tmp_path / "nested" / "dir" / "history.json"
    save_training_history(history, nested_path)
    assert nested_path.exists()


def test_history_json_uses_expected_keys_and_1_indexed_best_epoch(tmp_path: Path) -> None:
    history = _example_history()
    path = tmp_path / "history.json"
    save_training_history(history, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "train_losses",
        "val_losses",
        "val_accuracies",
        "best_epoch",
        "best_val_loss",
    }
    assert data["best_epoch"] == 2  # 1-indexed, 0이 아님


def test_history_with_default_none_best_fields_round_trips(tmp_path: Path) -> None:
    """best_epoch/best_val_loss는 run_training()이 항상 채우지만, 기본값
    None으로 만든 TrainingHistory도 JSON round-trip이 안전한지 확인."""
    history = TrainingHistory(train_losses=[], val_losses=[], val_accuracies=[])
    path = tmp_path / "history.json"
    save_training_history(history, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["best_epoch"] is None
    assert data["best_val_loss"] is None

    restored = load_training_history(path)
    assert restored == history
