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
from dataclasses import dataclass
from pathlib import Path

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
from image_ai_studio.training.loop import TrainingHistory, TrainingResumeState, evaluate, run_training
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
    조립하든, E2E가 고정 상수로 조립하든 이 dataclass는 신경 쓰지 않는다)."""

    model_json_path: Path
    dataset_root: Path
    training_config: TrainingConfig
    output_dir: Path
    resume_from: Path | None = None
    checkpoint_out: Path | None = None
    export_torchscript: bool = True
    seed: int = SEED


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


def run_imagefolder_training_workflow(request: ImageFolderWorkflowRequest) -> ImageFolderWorkflowResult:
    model_spec = load_model_spec(request.model_json_path)
    shape_trace = validate_model_spec(model_spec)
    final_shape = shape_trace[-1].output_shape

    splits = make_imagefolder_datasets(model_spec.input_shape, root=request.dataset_root)
    require_matching_num_classes(len(splits.classes), final_shape)

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

    training_result = run_training(
        model, train_loader, val_loader, request.training_config, device="cpu", resume_state=resume_state
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
    # 착각해서 저장하는 버그가 애초에 발생할 수 없다.
    checkpoint_path: Path | None = None
    checkpoint_metadata_path: Path | None = None
    if request.checkpoint_out is not None:
        checkpoint_path = request.checkpoint_out
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=loader_generator_state_after,
            cpu_rng_state=cpu_rng_state_after,
        )
        checkpoint_metadata_path = metadata_path_for_checkpoint(checkpoint_path)
        save_imagefolder_resume_metadata(
            build_imagefolder_resume_metadata(model_spec, splits), checkpoint_metadata_path
        )

    # run_training()은 best_state_dict를 메모리로만 반환한다 -- 여기서 새
    # model에 로드한 뒤 저장한다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    best_model_state_dict_path = output_dir / "best_model_state_dict.pt"
    save_state_dict(best_model, best_model_state_dict_path)

    test_loss, test_accuracy = evaluate(best_model, test_loader, device="cpu")
    test_result_path = output_dir / "test_result.json"
    test_result_path.write_text(
        json.dumps({"test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2), encoding="utf-8"
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
    )
