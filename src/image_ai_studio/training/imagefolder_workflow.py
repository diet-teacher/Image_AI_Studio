"""ImageFolder 학습 orchestration (Phase 4H). `scripts/train_imagefolder.py`
(production CLI)와 `scripts/run_imagefolder_training_e2e.py`(회귀 검증
E2E) 둘 다 이 모듈의 `run_imagefolder_training_workflow()`만 호출한다 --
"학습 본질" 로직(ModelSpec/dataset 검증, model build/resume, 학습 실행,
checkpoint/history/best model/class mapping/test 결과 저장, TorchScript
export)은 여기 한 곳에만 있고, 두 호출자는 이걸 향해서만 의존한다(스크립트
끼리 서로 import하지 않음 -- docs/phase4h_production_training_cli_design.md
§4/§12).

이 모듈은 E2E 전용 로직(loss-decreased 게이트, class mapping/best model
reload 자체 검증, C++ parity)을 전혀 모른다 -- 그건 호출자(E2E)가 반환된
`ImageFolderWorkflowResult`를 받은 뒤 자기 책임으로 수행한다. 마찬가지로
`image_ai_studio.tools.run_and_compare`(C++ runner)는 이 모듈에서 아예
import하지 않는다 -- production CLI 경로가 실수로도 C++ 빌드/CUDA 가용성에
의존하지 않도록 하기 위함.

예외는 삼키거나 재포장하지 않는다 -- ModelValidationError/
TrainingConfigError/ValueError/OSError를 그대로 전파한다(전부 이미
ValueError의 서브클래스). TorchScript export 실패만 예외로 승격한다
(TorchScriptExporter.export()는 예외 대신 metadata.json의 status 필드로
실패를 표현하므로, 워크플로우의 "성공하면 Result, 실패하면 예외"라는
단일 출력 계약에 맞춘다).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.specs import ModelSpec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.training.checkpoint import (
    load_training_checkpoint,
    save_state_dict,
    save_training_checkpoint,
)
from image_ai_studio.training.config import TrainingConfig, require_compatible_resume_config
from image_ai_studio.training.history import save_training_history
from image_ai_studio.training.imagefolder_resume import (
    build_imagefolder_resume_metadata,
    load_imagefolder_resume_metadata,
    metadata_path_for_checkpoint,
    require_compatible_imagefolder_resume_metadata,
    save_imagefolder_resume_metadata,
)
from image_ai_studio.training.loop import (
    CheckpointHook,
    EpochCheckpointView,
    ShouldStopCallback,
    TrainingHistory,
    TrainingProgressCallback,
    TrainingResult,
    TrainingResumeState,
    evaluate_classification_metrics,
    run_training,
)
from image_ai_studio.training.metrics import ClassificationMetrics
from image_ai_studio.training.torchvision_dataset import (
    ImageFolderSplits,
    make_imagefolder_datasets,
    require_matching_num_classes,
    save_class_mapping,
)

SEED = 20260730

_TORCHSCRIPT_MODEL_FILENAME = "model.ts"
_TORCHSCRIPT_METADATA_FILENAME = "model_metadata.json"


@dataclass
class ImageFolderWorkflowRequest:
    """워크플로우 호출에 필요한 전부. `training_config`는 이미 검증된
    `TrainingConfig` 인스턴스를 그대로 받는다(호출자가 CLI argparse에서
    조립하든, E2E가 고정 상수로 조립하든 이 dataclass는 신경 쓰지 않는다).

    checkpoint_every(Phase 4J, docs/phase4j_epoch_checkpoint_design.md
    §6/§11)는 global epoch이 이 값의 배수가 될 때마다 `checkpoint_out`을
    자동으로 갱신한다. `None`(기본값)이면 학습 도중 자동 저장을 하지
    않고, 기존과 동일하게 학습 종료 시 최종 저장만 수행한다.
    `checkpoint_every`를 켜려면 `checkpoint_out`이 함께 있어야 한다."""

    model_json_path: Path
    dataset_root: Path
    training_config: TrainingConfig
    output_dir: Path
    resume_from: Path | None = None
    checkpoint_out: Path | None = None
    export_torchscript: bool = True
    seed: int = SEED
    checkpoint_every: int | None = None


@dataclass
class ImageFolderWorkflowResult:
    """저장된 파일 경로와 학습 결과 지표만 담는다 -- 살아있는 nn.Module/
    텐서 객체는 담지 않는다(호출자가 필요하면 저장된 파일에서 다시
    읽으면 된다). export/checkpoint를 하지 않았으면 해당 경로는 None."""

    history: TrainingHistory
    test_loss: float
    test_accuracy: float
    best_model_state_dict_path: Path
    training_history_path: Path
    class_mapping_path: Path
    test_result_path: Path
    checkpoint_path: Path | None
    checkpoint_metadata_path: Path | None
    torchscript_model_path: Path | None
    torchscript_metadata_path: Path | None
    # Phase 4O: 최종 test 평가의 confusion matrix/macro precision/recall/F1
    # (test-only, validation/TrainingHistory/checkpoint는 무수정). 기본값
    # None은 "test 평가가 생략될 수 있다"는 뜻이 아니라, 이 dataclass를
    # 이 필드 없이 직접 생성하던 기존 코드(테스트의 manual/fake
    # constructor 호출)와의 생성자 하위호환을 위한 것이다 -- 마지막 필드로
    # 둬야 그 앞의 필드들이 여전히 기본값 없이 위치/키워드 인자로 채워질
    # 수 있다. `run_imagefolder_training_workflow()`가 정상 완료해 반환하는
    # production 결과의 test_metrics는 항상 실제 ClassificationMetrics다.
    test_metrics: ClassificationMetrics | None = None


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_resume(
    request: ImageFolderWorkflowRequest,
    model_spec: ModelSpec,
    splits: ImageFolderSplits,
) -> tuple[nn.Module, torch.Generator, TrainingResumeState | None, torch.Tensor | None]:
    """`request.resume_from`이 None이면 (신규 model, 신규 generator, None,
    None)을 반환한다. 있으면 metadata 로드/검증 -> model build+load ->
    generator 복원 -> ResumeState 조립 -> config 검증까지 전부 수행한 뒤
    (model, restored_generator, resume_state, payload["cpu_rng_state"])를
    반환한다.

    **이 함수는 전역 CPU RNG를 절대 건드리지 않는다.** 네 번째 반환값
    (cpu_rng_state)은 호출자가 DataLoader 생성을 전부 마친 뒤,
    run_training() 호출 바로 직전에 torch.set_rng_state()로 직접 적용해야
    한다 -- 이 함수 안에서 미리 복원하면, 함수가 반환된 뒤 호출자가 하는
    DataLoader 생성이 복원 시점과 run_training() 사이에 끼어들게 되어
    "RNG 복원은 항상 마지막"이라는 불변조건이 함수 경계 때문에 깨진다."""
    if request.resume_from is None:
        _set_seed(request.seed)
        model = build_model(model_spec)
        loader_generator = torch.Generator().manual_seed(request.seed)
        return model, loader_generator, None, None

    saved_metadata = load_imagefolder_resume_metadata(metadata_path_for_checkpoint(request.resume_from))
    payload = load_training_checkpoint(request.resume_from)

    current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
    require_compatible_imagefolder_resume_metadata(saved_metadata, current_metadata)

    _set_seed(request.seed)
    model = build_model(model_spec)
    # payload["best_state_dict"]가 아니라 model_state_dict를 쓴다 --
    # best_state_dict를 쓰면 "최고 성능 epoch"에서 재개하게 되어 resume
    # 시작점 계약(마지막으로 완료된 epoch에서 이어간다)을 깬다.
    model.load_state_dict(payload["model_state_dict"])

    restored_generator = torch.Generator()
    restored_generator.set_state(payload["loader_generator_state"])

    resume_state = TrainingResumeState(
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=payload["scheduler_state_dict"],
        history=TrainingHistory(**payload["history"]),
        epochs_without_improvement=payload["epochs_without_improvement"],
        best_state_dict=payload["best_state_dict"],
        training_config=payload["training_config"],
    )
    require_compatible_resume_config(resume_state.training_config, request.training_config)

    return model, restored_generator, resume_state, payload["cpu_rng_state"]


def _validate_checkpoint_every(value: int | None) -> None:
    """checkpoint_every 유효성 검증(Phase 4J, §6-2/§11-2). `config.py`의
    private `_require_positive_int()`는 재사용하지 않는다 -- 이 모듈
    자체의 validator로 둔다."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint_every must be an integer or None, got {value!r}")
    if value < 1:
        raise ValueError(f"checkpoint_every must be at least 1, got {value!r}")


def _normalized_path(path: str | Path) -> str:
    """두 경로가 같은 파일을 가리키는지 비교하기 위한 정규화(Phase 4J,
    §11-2). Path.resolve()로 상대/절대 표기 차이를 없애고,
    os.path.normcase()로 Windows의 대소문자 비구분 파일시스템에서의
    오탐/누락을 줄인다(POSIX에서 normcase는 no-op)."""
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _is_in_place_resume(request: ImageFolderWorkflowRequest) -> bool:
    """resume_from과 checkpoint_out이 정확히 같은 파일을 가리키면(Phase
    4J, §6-4/§11-2) True -- 이 경우에만 기존 checkpoint_out 경로를
    갱신하는 것이 허용된다. §7-3의 metadata_ready 초기값도 이 함수
    하나를 그대로 재사용한다."""
    if request.resume_from is None or request.checkpoint_out is None:
        return False
    return _normalized_path(request.resume_from) == _normalized_path(request.checkpoint_out)


def _validate_checkpoint_output_paths(request: ImageFolderWorkflowRequest) -> None:
    """출력 경로 재사용 정책(Phase 4J, §6-5): in-place resume(resume_from
    == checkpoint_out)만 기존 checkpoint_out 경로를 갱신할 수 있다.
    그 외(fresh 또는 다른 경로로의 resume)는 checkpoint_out과 그
    metadata sidecar가 완전히 비어있는 새 경로여야 한다 -- 기존 파일이
    있으면 학습을 시작하기 전에 거부한다(기존 파일을 지우거나 바꾸지
    않는다)."""
    if request.checkpoint_out is None:
        return
    if _is_in_place_resume(request):
        return

    checkpoint_path = Path(request.checkpoint_out)
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    if checkpoint_path.exists():
        raise ValueError(
            f"{checkpoint_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path. To continue training this exact checkpoint, "
            "pass it as both --resume-from and --checkpoint-out."
        )
    if metadata_path.exists():
        raise ValueError(
            f"{metadata_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path."
        )


def _make_checkpoint_hook(
    request: ImageFolderWorkflowRequest,
    ensure_checkpoint_metadata: Callable[[], None],
) -> CheckpointHook:
    """global epoch 기준 cadence로 동작하는 checkpoint_hook을 만든다
    (Phase 4J, §11-3). `model_spec`/`splits`는 다시 캡처하지 않는다 --
    `ensure_checkpoint_metadata`가 이미 그것들을 캡처했으므로 이 hook은
    그 함수 하나만 공유해서 쓴다."""

    def hook(view: EpochCheckpointView) -> None:
        global_epoch = len(view.history.train_losses)
        if global_epoch % request.checkpoint_every != 0:
            return  # non-scheduled epoch -- state_dict()/RNG 조회를 전혀 하지 않는다

        if view.loader_generator is None:
            raise ValueError(
                "auto checkpoint requires an explicit DataLoader generator for exact "
                "resume, but loader_generator is None"
            )

        ensure_checkpoint_metadata()  # §7-3, checkpoint보다 먼저

        training_result = TrainingResult(
            history=view.history,
            best_state_dict=view.best_state_dict,
            optimizer_state_dict=view.optimizer.state_dict(),
            scheduler_state_dict=(view.scheduler.state_dict() if view.scheduler is not None else None),
            epochs_without_improvement=view.epochs_without_improvement,
        )
        save_training_checkpoint(  # 원자적(§7-2)
            request.checkpoint_out,
            model=view.model,
            training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=view.loader_generator.get_state(),
            cpu_rng_state=torch.get_rng_state(),
        )

    return hook


def run_imagefolder_training_workflow(
    request: ImageFolderWorkflowRequest,
    *,
    progress_callback: TrainingProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> ImageFolderWorkflowResult:
    _validate_checkpoint_every(request.checkpoint_every)
    if request.checkpoint_every is not None and request.checkpoint_out is None:
        raise ValueError("checkpoint_every requires checkpoint_out to be set")
    _validate_checkpoint_output_paths(request)

    model_spec = load_model_spec(request.model_json_path)
    shape_trace = validate_model_spec(model_spec)
    final_shape = shape_trace[-1].output_shape

    splits = make_imagefolder_datasets(model_spec.input_shape, root=request.dataset_root)
    require_matching_num_classes(len(splits.classes), final_shape)

    # metadata_ready/ensure_checkpoint_metadata는 이 workflow 호출 하나당
    # 정확히 한 번 만들어지는 closure 상태다 -- scheduled checkpoint_hook과
    # 아래의 학습 종료 후 최종 저장이 이 하나를 함께 공유해서, metadata
    # sidecar를 이번 실행 동안 최대 한 번만 쓴다(Phase 4J, §7-3/§11-3).
    # in-place resume은 _prepare_resume()이 이미 metadata를 로드/검증했으므로
    # True로 시작해 절대 다시 쓰지 않는다.
    metadata_ready = _is_in_place_resume(request)

    def ensure_checkpoint_metadata() -> None:
        nonlocal metadata_ready
        if metadata_ready:
            return
        metadata_path = metadata_path_for_checkpoint(request.checkpoint_out)
        current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
        save_imagefolder_resume_metadata(current_metadata, metadata_path)  # 원자적(§7-2)
        metadata_ready = True

    model, loader_generator, resume_state, cpu_rng_state = _prepare_resume(request, model_spec, splits)

    batch_size = request.training_config.batch_size
    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=0)

    # DataLoader 생성이 전부 끝난 뒤, 다른 RNG 소비 작업 없이 즉시
    # run_training()을 호출한다 -- fresh 경로에서는 cpu_rng_state가 None이라
    # 이 블록이 아무 일도 하지 않는다.
    if cpu_rng_state is not None:
        torch.set_rng_state(cpu_rng_state)

    checkpoint_hook = (
        _make_checkpoint_hook(request, ensure_checkpoint_metadata) if request.checkpoint_every is not None else None
    )
    training_result = run_training(
        model, train_loader, val_loader, request.training_config, device="cpu", resume_state=resume_state,
        progress_callback=progress_callback, should_stop=should_stop, checkpoint_hook=checkpoint_hook,
    )
    # checkpoint 저장에 쓸 RNG snapshot -- 이후 코드(TorchScript export의
    # set_seed() 등)가 전역 RNG를 다시 바꾸기 전에, 학습이 실제로 끝난
    # 시점의 상태를 독립적인 snapshot으로 캡처해 둔다.
    cpu_rng_state_after = torch.get_rng_state().clone()
    loader_generator_state_after = loader_generator.get_state().clone()
    history = training_result.history

    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_history_path = output_dir / "training_history.json"
    save_training_history(history, training_history_path)

    class_mapping_path = output_dir / "class_mapping.json"
    save_class_mapping(splits.classes, splits.class_to_idx, class_mapping_path)

    # checkpoint는 `model`(현재/마지막 epoch 가중치)이 아직 어떤 방식으로도
    # best 가중치로 대체되지 않은 이 시점에 저장한다 -- 아래 best_model
    # 생성(별도 인스턴스) 이전에 두어야, best_state_dict를 현재 모델로
    # 착각해서 저장하는 버그가 애초에 발생할 수 없다. 학습 도중
    # checkpoint_hook이 이미 몇 번 저장했더라도, 이 최종 저장은 항상
    # 실행된다(Phase 4J, §6-4) -- should_stop() 평가 이후의 정확한
    # stopped_by_user 값을 반영하는 저장은 이 최종 저장뿐이기 때문이다
    # (마지막 epoch이 scheduled epoch였다면 같은 global epoch이 두 번
    # 저장되는 것은 의도된 동작이다, §9-4).
    checkpoint_path: Path | None = None
    checkpoint_metadata_path: Path | None = None
    if request.checkpoint_out is not None:
        checkpoint_path = request.checkpoint_out
        checkpoint_metadata_path = metadata_path_for_checkpoint(checkpoint_path)
        ensure_checkpoint_metadata()  # §7-3 -- 이미 준비됐으면(scheduled 저장이 있었으면) 아무 것도 안 함
        save_training_checkpoint(  # 원자적(§7-2)
            checkpoint_path,
            model=model,
            training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=loader_generator_state_after,
            cpu_rng_state=cpu_rng_state_after,
        )

    # run_training()은 best_state_dict를 메모리로만 반환한다 -- 여기서 새
    # model에 로드한 뒤 저장한다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    best_model_state_dict_path = output_dir / "best_model_state_dict.pt"
    save_state_dict(best_model, best_model_state_dict_path)

    # class_mapping.json(위에서 저장)의 classes 순서가 confusion_matrix/
    # per_class_recall의 class index 순서와 동일하다는 계약을 위해, 여기서
    # 쓰는 num_classes도 그 순서를 만드는 len(splits.classes)를 그대로
    # 재사용한다(295행의 require_matching_num_classes()가 이미 이 값과
    # model 출력 차원의 일치를 검증했으므로 여기서 다시 검증하지 않는다).
    test_loss, test_accuracy, test_metrics = evaluate_classification_metrics(
        best_model, test_loader, num_classes=len(splits.classes), device="cpu"
    )
    test_result_path = output_dir / "test_result.json"
    test_result_path.write_text(
        json.dumps(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "classification_metrics": asdict(test_metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torchscript_model_path: Path | None = None
    torchscript_metadata_path: Path | None = None
    ts_model_path = output_dir / _TORCHSCRIPT_MODEL_FILENAME
    ts_metadata_path = output_dir / _TORCHSCRIPT_METADATA_FILENAME
    if request.export_torchscript:
        _set_seed(request.seed)
        example_input = torch.randn(1, *model_spec.input_shape, dtype=torch.float32)
        TorchScriptExporter().export(
            best_model,
            example_input,
            ts_model_path,
            ts_metadata_path,
            model_name=model_spec.name,
            state_dict_path=best_model_state_dict_path,
        )
        export_metadata = json.loads(ts_metadata_path.read_text())
        if export_metadata["status"] != "PASS":
            raise ValueError(f"TorchScript export failed: {export_metadata.get('error_log')}")
        torchscript_model_path = ts_model_path
        torchscript_metadata_path = ts_metadata_path
    else:
        # 이전 실행이 같은 output_dir에 남긴 TorchScript 산출물을 제거한다
        # -- 지우지 않으면 사용자가 이번 실행 결과로 착각할 수 있다.
        # 워크플로우가 고정 이름으로 관리하는 이 두 파일만 지우고, output_dir의
        # 다른 파일은 건드리지 않는다. 삭제 실패(권한 등)는 감싸지 않고
        # 그대로 전파한다.
        ts_model_path.unlink(missing_ok=True)
        ts_metadata_path.unlink(missing_ok=True)

    return ImageFolderWorkflowResult(
        history=history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        best_model_state_dict_path=best_model_state_dict_path,
        training_history_path=training_history_path,
        class_mapping_path=class_mapping_path,
        test_result_path=test_result_path,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        torchscript_model_path=torchscript_model_path,
        torchscript_metadata_path=torchscript_metadata_path,
        test_metrics=test_metrics,
    )
