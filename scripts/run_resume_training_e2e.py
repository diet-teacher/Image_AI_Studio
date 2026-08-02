#!/usr/bin/env python
"""Phase 4F checkpoint/resume 엔드투엔드 파이프라인. 이 스크립트의 책임은
딱 하나, "epoch 경계에서 저장한 checkpoint로부터 학습을 재개했을 때, 중단
없이 연속 실행한 학습과 실제로 같은 결과가 나오는가"를 실제로 실행해서
증명하는 것이다.

    (a) 연속 실행:      5 epoch를 한 번에 실행
    (b) 분할 + resume:  3 epoch 실행 -> full checkpoint 저장
                        -> (새 프로세스를 흉내내어 model/DataLoader/
                           generator를 전부 새로 만들고) checkpoint 로드
                        -> DataLoader generator state 복원
                        -> CPU RNG state 복원
                        -> 2 epoch resume

(a)와 (b)의 다음 항목이 전부 일치하는지 비교한다: model parameters,
optimizer state, scheduler state, TrainingHistory, best_state_dict,
best_epoch, best_val_loss, epochs_without_improvement.

synthetic dataset(training/dataset.py, 네트워크 불필요)과 Dropout이 포함된
기존 모델(examples/models/phase4_training_model.json -- run_training_e2e.py
가 이미 쓰고 있는 모델)을 사용한다. Dropout은 DataLoader의 shuffle
generator가 아니라 전역 CPU RNG에 의존하므로(tests/training/test_loop.py의
동일 테스트에서 실측 확인), 이 스크립트가 실제로 통과하려면 CPU RNG state
복원이 빠지면 안 된다 -- 즉 이 스크립트 자체가 그 필요성의 증거다.

이 스크립트는 TorchScript export나 C++ parity를 다시 수행하지 않는다 --
그건 이미 Phase 0/4A~4E의 다른 E2E들이 충분히 검증하고 있고, 이 스크립트의
책임은 resume exactness 하나뿐이다 (단일 책임 유지).

- model_definition/*, export/*, C++ 러너, training/dataset.py,
  training/history.py 전부 변경 없음
- training/loop.py, training/checkpoint.py는 Phase 4F에서 확장됨 (신규
  TrainingResumeState/save_training_checkpoint/load_training_checkpoint 등)
- 기존 scripts/run_training_e2e.py / run_real_training_e2e.py /
  run_imagefolder_training_e2e.py는 회귀 앵커로 이번 Phase에서도 수정하지
  않았다

사용법:

    python scripts/run_resume_training_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.training.checkpoint import (
    load_training_checkpoint,
    require_compatible_resume_config,
    save_training_checkpoint,
)
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import TrainingHistory, TrainingResumeState, run_training

SEED = 20260801
MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase4_training_model.json"
CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "training" / "phase4f_resume_checkpoint.pt"
TRAIN_SIZE = 32
VAL_SIZE = 16

EPOCHS_CONTINUOUS = 5
EPOCHS_FIRST = 3
EPOCHS_RESUME = 2  # EPOCHS_FIRST + EPOCHS_RESUME == EPOCHS_CONTINUOUS

# Dropout(전역 CPU RNG)과 BatchNorm(배치 순서에 의존)을 모두 포함한 조합으로
# resume 계약을 검증하기 위해, optimizer/scheduler도 기본값(Adam, scheduler
# 없음)이 아니라 SGD + ReduceLROnPlateau를 사용한다 (더 많은 상태 종류를
# 실제로 왕복시켜 봄).
TRAINING_CONFIG_KWARGS = dict(
    batch_size=8,
    learning_rate=1e-2,
    optimizer="sgd",
    momentum=0.9,
    lr_scheduler="plateau",
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=1,
)


def make_loaders(num_classes: int, input_shape: tuple[int, int, int]) -> tuple[DataLoader, DataLoader, torch.Generator]:
    train_dataset, val_dataset = make_train_val_datasets(
        input_shape, num_classes, seed=SEED, train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG_KWARGS["batch_size"],
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAINING_CONFIG_KWARGS["batch_size"], shuffle=False, num_workers=0
    )
    return train_loader, val_loader, generator


def assert_deep_equal(a: object, b: object, path: str = "value") -> None:
    """optimizer/scheduler state_dict처럼 텐서와 스칼라가 섞인 중첩
    dict/list를 재귀적으로 정확히 비교한다."""
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f"tensor mismatch at {path}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"dict keys mismatch at {path}: {a.keys()} != {b.keys()}")
        for key in a:
            assert_deep_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"length mismatch at {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            assert_deep_equal(x, y, f"{path}[{index}]")
    else:
        if a != b:
            raise AssertionError(f"value mismatch at {path}: {a!r} != {b!r}")


def main() -> int:
    print("Phase 4F Resume E2E")
    print(f"Model JSON: {MODEL_JSON}")

    print("ModelSpec:")
    try:
        model_spec = load_model_spec(MODEL_JSON)
        shape_trace = validate_model_spec(model_spec)
        final_shape = shape_trace[-1].output_shape
        print(f"  PASS ({len(model_spec.layers)} layers, final shape {final_shape})")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4F E2E: FAIL")
        return 1
    num_classes = final_shape[0]

    config_continuous = TrainingConfig(epochs=EPOCHS_CONTINUOUS, **TRAINING_CONFIG_KWARGS)
    config_first = TrainingConfig(epochs=EPOCHS_FIRST, **TRAINING_CONFIG_KWARGS)
    config_resume = TrainingConfig(epochs=EPOCHS_RESUME, **TRAINING_CONFIG_KWARGS)

    # -- (a) 연속 실행 --------------------------------------------------------
    print(f"\n(a) Continuous run: {EPOCHS_CONTINUOUS} epochs")
    torch.manual_seed(SEED)
    model_a = build_model(model_spec)
    train_loader_a, val_loader_a, _ = make_loaders(num_classes, model_spec.input_shape)
    torch.manual_seed(SEED)  # Dropout이 소비할 CPU RNG를 model 생성과 분리해 고정
    result_a = run_training(model_a, train_loader_a, val_loader_a, config_continuous)
    for epoch, (tl, vl) in enumerate(zip(result_a.history.train_losses, result_a.history.val_losses), start=1):
        print(f"  epoch {epoch}: train_loss={tl:.6f} val_loss={vl:.6f}")
    print(f"  best_epoch={result_a.history.best_epoch} best_val_loss={result_a.history.best_val_loss:.6f}")

    # -- (b) 분할 + resume ------------------------------------------------------
    print(f"\n(b) Split run: {EPOCHS_FIRST} epochs, then checkpoint + resume {EPOCHS_RESUME} epochs")
    torch.manual_seed(SEED)
    model_b = build_model(model_spec)
    train_loader_b, val_loader_b, generator_b = make_loaders(num_classes, model_spec.input_shape)
    torch.manual_seed(SEED)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, config_first)
    for epoch, (tl, vl) in enumerate(zip(result_b1.history.train_losses, result_b1.history.val_losses), start=1):
        print(f"  epoch {epoch}: train_loss={tl:.6f} val_loss={vl:.6f}")

    print("Checkpoint save:")
    save_training_checkpoint(
        CHECKPOINT_PATH,
        model=model_b,
        training_result=result_b1,
        training_config=config_first,
        loader_generator_state=generator_b.get_state(),
        cpu_rng_state=torch.get_rng_state(),
    )
    print(f"  saved to {CHECKPOINT_PATH}")

    print("Checkpoint load (new model/DataLoader/generator, simulating a new process):")
    try:
        payload = load_training_checkpoint(CHECKPOINT_PATH)
        # 이 호출은 조기 실패(fail fast)를 위한 것일 뿐이다 -- 실제 계약은
        # run_training()이 resume_state를 받을 때 항상 스스로 강제한다
        # (아래에서 이 호출을 지워도 run_training()이 여전히 막는다).
        require_compatible_resume_config(payload["training_config"], config_resume)
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4F E2E: FAIL")
        return 1
    print("  PASS")

    model_b2 = build_model(model_spec)
    model_b2.load_state_dict(payload["model_state_dict"])
    train_dataset2, val_dataset2 = make_train_val_datasets(
        model_spec.input_shape, num_classes, seed=SEED, train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(payload["loader_generator_state"])
    train_loader_b2 = DataLoader(
        train_dataset2,
        batch_size=TRAINING_CONFIG_KWARGS["batch_size"],
        shuffle=True,
        generator=restored_generator,
        drop_last=True,
        num_workers=0,
    )
    val_loader_b2 = DataLoader(
        val_dataset2, batch_size=TRAINING_CONFIG_KWARGS["batch_size"], shuffle=False, num_workers=0
    )
    torch.set_rng_state(payload["cpu_rng_state"])

    resume_state = TrainingResumeState(
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=payload["scheduler_state_dict"],
        history=TrainingHistory(**payload["history"]),
        epochs_without_improvement=payload["epochs_without_improvement"],
        best_state_dict=payload["best_state_dict"],
        training_config=payload["training_config"],
    )
    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, config_resume, resume_state=resume_state
    )
    # result_b2.history는 이전 3 epoch + 새 2 epoch를 이어붙인 전체(5개) 기록이다
    # -- 여기서는 이번에 새로 실행된 뒤쪽 2개만 잘라서 보여준다.
    resumed_train_losses = result_b2.history.train_losses[EPOCHS_FIRST:]
    resumed_val_losses = result_b2.history.val_losses[EPOCHS_FIRST:]
    for epoch, (tl, vl) in enumerate(
        zip(resumed_train_losses, resumed_val_losses), start=EPOCHS_FIRST + 1
    ):
        print(f"  epoch {epoch}: train_loss={tl:.6f} val_loss={vl:.6f}")
    print(f"  best_epoch={result_b2.history.best_epoch} best_val_loss={result_b2.history.best_val_loss:.6f}")

    # -- 비교 -------------------------------------------------------------------
    print("\nExact resume comparison (continuous vs split+resume):")
    checks: list[tuple[str, bool]] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            checks.append((name, True))
        except AssertionError as exc:
            checks.append((name, False))
            print(f"  {name}: FAIL ({exc})")

    check("model state", lambda: [assert_deep_equal(t, model_b2.state_dict()[n], n) for n, t in model_a.state_dict().items()])
    check(
        "optimizer state",
        lambda: assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict, "optimizer"),
    )
    check(
        "scheduler state",
        lambda: assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict, "scheduler"),
    )
    check("history.train_losses", lambda: assert_deep_equal(result_a.history.train_losses, result_b2.history.train_losses))
    check("history.val_losses", lambda: assert_deep_equal(result_a.history.val_losses, result_b2.history.val_losses))
    check(
        "history.val_accuracies",
        lambda: assert_deep_equal(result_a.history.val_accuracies, result_b2.history.val_accuracies),
    )
    check(
        "best_state_dict",
        lambda: [assert_deep_equal(t, result_b2.best_state_dict[n], n) for n, t in result_a.best_state_dict.items()],
    )
    check("best_epoch", lambda: assert_deep_equal(result_a.history.best_epoch, result_b2.history.best_epoch))
    check("best_val_loss", lambda: assert_deep_equal(result_a.history.best_val_loss, result_b2.history.best_val_loss))
    check(
        "epochs_without_improvement",
        lambda: assert_deep_equal(result_a.epochs_without_improvement, result_b2.epochs_without_improvement),
    )

    for name, ok in checks:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    overall_ok = all(ok for _, ok in checks)
    print(f"\nPHASE 4F E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
