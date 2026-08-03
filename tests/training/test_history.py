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
        "stopped_early",
        "stopped_by_user",
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


# -- Phase 4E: stopped_early ---------------------------------------------------


def test_stopped_early_true_round_trips(tmp_path: Path) -> None:
    history = TrainingHistory(
        train_losses=[1.0, 0.9], val_losses=[1.0, 1.0], val_accuracies=[0.1, 0.1], stopped_early=True
    )
    path = tmp_path / "history.json"
    save_training_history(history, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stopped_early"] is True

    restored = load_training_history(path)
    assert restored == history
    assert restored.stopped_early is True


def test_stopped_early_false_round_trips(tmp_path: Path) -> None:
    history = _example_history()
    assert history.stopped_early is False  # 기본값
    path = tmp_path / "history.json"
    save_training_history(history, path)

    restored = load_training_history(path)
    assert restored.stopped_early is False


def test_load_training_history_defaults_stopped_early_when_key_is_missing(tmp_path: Path) -> None:
    """Phase 4D까지 저장된 과거 형식 history.json에는 stopped_early 키가
    없다 -- load_training_history()가 TrainingHistory(**data)로 복원하므로,
    누락된 키는 dataclass 기본값(False)으로 자동 채워져야 한다 (하위 호환)."""
    legacy_data = {
        "train_losses": [1.0, 0.5],
        "val_losses": [1.1, 0.6],
        "val_accuracies": [0.2, 0.6],
        "best_epoch": 2,
        "best_val_loss": 0.6,
        # "stopped_early" 키 없음 (Phase 4D까지의 실제 저장 형식)
    }
    path = tmp_path / "legacy_history.json"
    path.write_text(json.dumps(legacy_data), encoding="utf-8")

    restored = load_training_history(path)

    assert restored.stopped_early is False
    assert restored.best_epoch == 2


# -- Phase 4I: stopped_by_user --------------------------------------------------


def test_stopped_by_user_true_round_trips(tmp_path: Path) -> None:
    history = TrainingHistory(
        train_losses=[1.0, 0.9], val_losses=[1.0, 1.0], val_accuracies=[0.1, 0.1], stopped_by_user=True
    )
    path = tmp_path / "history.json"
    save_training_history(history, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stopped_by_user"] is True

    restored = load_training_history(path)
    assert restored == history
    assert restored.stopped_by_user is True


def test_load_training_history_defaults_stopped_by_user_when_key_is_missing(tmp_path: Path) -> None:
    """Phase 4H까지 저장된 history.json에는 stopped_by_user 키가 없다 --
    load_training_history()가 TrainingHistory(**data)로 복원하므로, 누락된
    키는 dataclass 기본값(False)으로 자동 채워져야 한다 (하위 호환,
    stopped_early와 동일한 메커니즘)."""
    legacy_data = {
        "train_losses": [1.0, 0.5],
        "val_losses": [1.1, 0.6],
        "val_accuracies": [0.2, 0.6],
        "best_epoch": 2,
        "best_val_loss": 0.6,
        "stopped_early": False,
        # "stopped_by_user" 키 없음 (Phase 4H까지의 실제 저장 형식)
    }
    path = tmp_path / "legacy_history.json"
    path.write_text(json.dumps(legacy_data), encoding="utf-8")

    restored = load_training_history(path)

    assert restored.stopped_by_user is False
    assert restored.best_epoch == 2
