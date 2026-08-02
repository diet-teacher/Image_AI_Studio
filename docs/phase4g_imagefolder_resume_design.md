# Phase 4G: ImageFolder Resume CLI Integration — 설계안

**상태: 구현 및 검증 완료.** 이 문서는 Phase 4F(core checkpoint/resume)를
`scripts/run_imagefolder_training_e2e.py`에 실제로 연결하기 위한 상세 설계와,
그 설계를 그대로 구현한 뒤 검증한 결과를 담는다.

**2026-08-02 개정 1 (사용자 리뷰 반영, 설계 단계)**: (1) resume 시 RNG 복원
순서 수정 — 잘못된 checkpoint/config로 실패할 때 전역 CPU RNG를 먼저 바꾸지
않도록 재정렬 (§3-2), (2) metadata 검증 순서 수정 — dataset을 만들기 전에는
계산 불가능한 검증을 요구하던 모순 제거 (§3-2), (3) dataset fingerprint를
줄바꿈 문자열 해시에서 canonical JSON 해시(class_index 포함)로 변경 (§6),
(4) `main(argv)`를 실제로 두 번 호출하는 CLI 배선 통합 테스트 추가 (§11-4),
(5) metadata 경로 유도를 `metadata_path_for_checkpoint()` 단일 함수로 통일
(§7-8).

**2026-08-02 개정 2 (사용자 리뷰 반영, 구현 완료 후)**: (6) ImageFolder
exact-resume 통합 테스트(§11-2)에 `lr_scheduler="plateau"`를 추가하고
optimizer/scheduler state_dict까지 재귀 비교(`_assert_deep_equal`)하도록
보강 — 이전 버전은 테스트 설명에 "optimizer/scheduler state"를 검증한다고
적어 놓고 실제로는 model/history만 비교하고 있었음, (7) CLI의 resume
checkpoint 로드 구간이 `ValueError`뿐 아니라 `OSError`도 잡도록 수정 —
metadata sidecar는 있는데 checkpoint(.pt) 파일만 없어지면
`load_training_checkpoint()`의 `torch.load()`가 `FileNotFoundError`를 그대로
올려 traceback이 노출되는 문제가 있었음 (§9), (8) 스크립트의 출력 문구를
`PHASE 4D E2E` -> `IMAGEFOLDER TRAINING E2E`로 통일 — Phase 4D 이후
4E/4G까지 계속 확장된 스크립트인데 문구만 Phase 4D에 고정되어 있었음.

최종 검증 결과 (모두 이 저장소에서 직접 실행 확인):

* `tests/training/` + `tests/scripts/`: **166 passed**
* 전체 `pytest`: **323 passed**
* 기존 `run_training_e2e.py`(Phase 4A/4B), `run_real_training_e2e.py`
  (Phase 4C), `run_imagefolder_training_e2e.py` 기본 실행(신규 플래그
  없음, Phase 4D/4E), `run_resume_training_e2e.py`(Phase 4F) 전부 기존과
  동일한 수치로 PASS (회귀 없음)
* `run_imagefolder_training_e2e.py --epochs 3 --checkpoint-out ...` 후
  `--epochs 2 --resume-from ... --checkpoint-out ...`(동일 경로로 덮어쓰기
  포함)로 실제 CIFAR-10 ImageFolder fixture를 학습한 결과가 연속 5 epoch
  실행과 모든 epoch의 train/val loss, val_acc, best_epoch, best_val_loss,
  test_loss/test_accuracy에서 정확히 일치
* TorchScript export 및 C++ CPU/CUDA parity 회귀 없음(모든 실행에서
  PASS 유지)

---

## 1. 저장소 분석 결과 (추정 없이 실제 코드 기준)

`scripts/run_imagefolder_training_e2e.py`(398줄, Phase 4D 신설/Phase 4E 확장)를
기준으로 실제 실행 흐름을 추적한 결과:

| 질문 | 실제 위치 |
|---|---|
| ImageFolder train/val/test 로더 생성 | `make_imagefolder_datasets()` 호출: `run_imagefolder_training_e2e.py:207` (구현은 `torchvision_dataset.py:178-216`) |
| DataLoader용 shuffle generator 생성 | `run_imagefolder_training_e2e.py:232` — `loader_generator = torch.Generator().manual_seed(SEED)`. **train_loader에만 쓰인다** (`:233-242`). val/test는 `shuffle=False`라 generator가 없다 (`:243, 246`). |
| class_to_idx 검증 | `torchvision_dataset.py:_require_matching_classes()` — train/val/test 세 `ImageFolder.class_to_idx`가 완전히 같은지만 확인. **검증 결과를 파일로 저장하지는 않는다** (지금은 매 실행마다 다시 계산). |
| best state_dict 저장 + export 흐름 | `:275-346` — `best_model = build_model(model_spec)` (새 인스턴스) → `best_model.load_state_dict(training_result.best_state_dict)` → `save_state_dict(best_model, ...)` → reload sanity check → `TorchScriptExporter().export(...)` |
| C++ parity 실행 | `:362-390` — `find_runner_binary()` / `run_case()` |
| TrainingConfig CLI 인자 조립 | `parse_args()` (`:126-149`) + `main()` (`:161-172`) |
| checkpoint 저장에 필요한 model/result/generator/RNG 접근 가능 지점 | `run_training()` 호출 직후(`:249`) — `model`(로컬 변수, 현재/마지막 epoch 가중치), `training_result`(optimizer/scheduler/history/best_state_dict/epochs_without_improvement), `loader_generator`(로컬 변수, `.get_state()` 가능), `torch.get_rng_state()`(전역, 즉시 호출 가능) 전부 `main()`의 같은 스코프에서 접근 가능. |

**중요한 발견 — 현재 CLI는 `--epochs`를 아예 노출하지 않는다.** `epochs`/`batch_size`/
`learning_rate`는 `parse_args()`에 없고 `main()`에서 `DEFAULT_EPOCHS=5`,
`DEFAULT_BATCH_SIZE=8`, `DEFAULT_LEARNING_RATE=1e-3`로 하드코딩된다(`:100-102`,
`:163-165`). resume의 "`--epochs N`은 추가 epoch 수"라는 Phase 4F 계약을 CLI에서
쓰려면 **`--epochs`를 새로 CLI 인자로 추가해야 한다** — 이건 기존 동작 확장이
아니라 지금 없는 기능을 새로 여는 것이다. `batch_size`/`learning_rate`는
`RESUME_CONFIG_FIELDS`에 포함되지만 현재 CLI로 바꿀 수 없으므로 resume 시
자동으로(비교 대상이 항상 같은 값이라) 항상 일치한다 — 이번 Phase에서 같이
노출할 필요는 없다(§15에서 다시 다룸).

**호의적인 발견 — `model`은 best_model 로딩으로 절대 오염되지 않는다.**
`:275`의 `best_model = build_model(model_spec)`는 **완전히 새로운 인스턴스**이고,
`run_training()`이 반환한 뒤의 `model`(`:249`에서 쓰인 바로 그 객체)은 이후
어디에서도 다시 `load_state_dict()`되지 않는다. 즉 checkpoint 저장 코드가
`model`(현재 가중치)을 참조하는 한, §10에서 사용자가 우려한 "best_state_dict를
현재 모델로 착각해서 저장" 버그는 **스크립트의 기존 구조상 애초에 발생할 수
없다** — checkpoint 저장 호출을 `best_model` 생성(`:275`) **이전**에 두기만
하면 된다.

`torchvision_dataset.py`(244줄) 확인 결과: `ImageFolderSplits.train/val/test`는
제네릭 `Dataset`이 아니라 **`ImageFolder` 타입 그대로** 노출된다(`:125-127`).
따라서 `.samples`(`[(절대경로, class_idx), ...]`)와 `.root`에 캐스팅 없이 직접
접근 가능하다. 이 세션에서 임시 fixture로 직접 실행해 확인한 결과 `.samples`는
`[('...\\cat\\a.png', 0), ('...\\dog\\b.png', 1)]` 형태이고, torchvision
0.27.1의 `DatasetFolder.make_dataset()` 소스(`inspect.getsource()`로 확인)는

```python
for target_class in sorted(class_to_idx.keys()):
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):
            ...
```

로 **클래스명 → 디렉터리 → 파일명 순으로 이미 정렬**되어 있다. 즉
`ImageFolder.samples`는 매 실행/머신에서 항상 같은 순서로 나온다 — 상대경로
해시를 만들 때 별도 정렬 로직 없이도 재현 가능하지만(§6), 방어적으로 정렬을
한 번 더 걸어도 비용이 거의 없으므로 그렇게 설계한다(torchvision 내부 구현이
바뀌어도 안전).

`build_transform()`(`:50-61`)은 `Resize`/`ToTensor`/`Normalize`뿐이고 augmentation이
없다 — train/val/test 전부 완전히 결정론적이라 exact-resume 통합 테스트에서
`torch.equal()` 수준의 정확한 동등성 비교가 가능하다(§11-2).

---

## 2. Phase 4G 목표/범위

`run_imagefolder_training_e2e.py`에 다음 CLI 인자를 추가한다:

- `--epochs N` (신규 — 현재 CLI에 없음, 반드시 추가해야 resume 워크플로우가 성립)
- `--resume-from PATH` (신규)
- `--checkpoint-out PATH` (신규)

`--epochs`는 Phase 4F 계약을 그대로 따른다: resume 시에도 "이번 실행에서 추가로
실행할 epoch 수"이지 "총 목표 epoch"가 아니다(`run_training()`이 이미 이렇게
구현되어 있음, `loop.py:248-253`). CLI 문서/도움말에도 이 의미를 명시한다.

```
# 새로 학습 + checkpoint 저장
python scripts/run_imagefolder_training_e2e.py --dataset-root ... --model-json ... --epochs 3 --checkpoint-out artifacts/training/foo_checkpoint.pt

# 이어서 2 epoch 더
python scripts/run_imagefolder_training_e2e.py --dataset-root ... --model-json ... --epochs 2 --resume-from artifacts/training/foo_checkpoint.pt --checkpoint-out artifacts/training/foo_checkpoint.pt
```

---

## 3. 실행 흐름 설계

### 3-1. 신규 학습 경로 (--resume-from 없음)

기존 흐름(`:152-394`)을 그대로 유지하되, 다음 두 가지만 삽입한다:

1. `run_training()` 호출(`:249`) **직후**, 다른 코드가 실행되기 전에
   `cpu_rng_state = torch.get_rng_state().clone()`와
   `loader_generator_state = loader_generator.get_state().clone()`를 캡처한다.
   **`.clone()`이 필요하다** — `get_rng_state()`/`Generator.get_state()`는
   호출 시점의 상태를 담은 새 텐서를 반환하긴 하지만, 이후 저장 시점까지의
   구간에 다른 코드가 실수로 같은 참조를 공유하거나 in-place로 건드릴 가능성을
   원천 차단하기 위해 독립적인 snapshot으로 명시적으로 분리해 둔다(호출자가
   들고 있는 값이 이후 어떤 이유로도 몰래 바뀌지 않는다는 계약을 코드
   레벨에서 보장). **타이밍도 중요하다** — `:310`에 있는 두 번째 `set_seed()`
   호출(example_input 재현용)이 전역 RNG를 다시 초기화하므로, 그 이전에
   캡처해야 "학습이 실제로 끝난 시점"의 RNG 상태를 정확히 담는다.
2. `--checkpoint-out`이 주어졌으면, `best_model` 생성(`:275`) **이전** 지점에
   `save_training_checkpoint(checkpoint_out, model=model, training_result=training_result, training_config=training_config, loader_generator_state=loader_generator_state, cpu_rng_state=cpu_rng_state)`와 함께 ImageFolder resume metadata(§6-7)를 저장한다.

이 위치 선택 이유: `model`이 아직 어떤 방식으로도 best 가중치로 대체되지 않은
시점이고(§1의 호의적 발견), 이후의 best_model/export/parity 코드 경로는 전혀
건드리지 않는다.

### 3-2. Resume 경로 (--resume-from 있음)

**이전 버전은 "dataset 생성 전에 metadata를 전부 검증한다"고 서술했지만
실제로는 불가능하다** — split 크기/class_to_idx/파일 목록 해시는 **현재
ImageFolder splits가 있어야만** 계산할 수 있다. 아래는 그 모순을 없앤 순서다.
`scripts/run_resume_training_e2e.py`(Phase 4F 레퍼런스)가 증명한 RNG 복원
순서(모델 생성 → ... → RNG 복원은 항상 맨 마지막)는 그대로 유지한다.

1. ModelSpec 로드/검증 (기존 신규 학습 경로와 동일, `load_model_spec()` + `validate_model_spec()`).
2. saved metadata 로드(`load_imagefolder_resume_metadata(metadata_path_for_checkpoint(resume_from))`)와 checkpoint payload 로드(`load_training_checkpoint(resume_from)`). **여기서 실패하면(파일 없음/구조 오류) 이후 어떤 RNG/모델 작업도 시작되지 않는다.** CLI는 이 두 호출을 `except (ValueError, OSError)`로 감싼다 — `load_imagefolder_resume_metadata()`는 파일이 없으면 항상 `ValueError`를 내지만, `load_training_checkpoint()`는 존재 확인 없이 곧바로 `torch.load()`를 호출하므로(예: metadata sidecar는 남아 있는데 checkpoint(.pt) 파일만 지워진 경우) `FileNotFoundError`(`OSError`)가 그대로 올라올 수 있다 — `ValueError`만 잡으면 이 경우 traceback이 그대로 노출된다.
3. `make_imagefolder_datasets()`로 **현재** splits 생성 — 결정론적이라 RNG/시드와 무관하게 지금 실행해도 안전.
4. `build_imagefolder_resume_metadata(model_spec, splits)`로 **현재** metadata 생성.
5. `require_compatible_imagefolder_resume_metadata(saved, current)` — saved metadata와 방금 만든 current metadata를 비교한다. **model/DataLoader를 만들기 전에** 여기서 끝낸다(ModelSpec/dataset이 불일치하면 그 이후 어떤 것도 만들 이유가 없음).
   - *선택적 최적화(필수 아님)*: `saved.model_spec_hash != hash_model_spec(model_spec)`만 2단계 직후 먼저 비교해 dataset을 읽기 전에 더 빨리 실패시킬 수도 있다 — 이번 설계에서는 필수로 두지 않는다(dataset 읽기 자체가 비싸지 않고, 검증 로직을 두 곳에 나누면 유지보수 부담만 커짐).
6. model/DataLoader 준비 — `set_seed()` → `model = build_model(model_spec)` → `model.load_state_dict(payload["model_state_dict"])` (**`payload["best_state_dict"]`가 아니라 `model_state_dict`** — best_state_dict를 쓰면 "최고 성능 epoch"에서 재개하게 되어 resume 시작점 계약을 깬다) → `restored_generator = torch.Generator(); restored_generator.set_state(payload["loader_generator_state"])` → `train_loader`(생성자 인자로 `restored_generator`) / `val_loader` / `test_loader` 생성.
7. `TrainingResumeState` 조립(`history=TrainingHistory(**payload["history"])` 등, `run_resume_training_e2e.py`와 동일 패턴) 및 그 자체의 `__post_init__` 검증(stopped_early 등) 통과 확인.
8. `require_compatible_resume_config(resume_state.training_config, training_config)` **조기 호출** — `run_training()` 내부에서도 항상 강제되지만, 여기서 먼저 호출해 fail-fast한다. **이 시점까지 전역 CPU RNG는 아직 건드리지 않았다** — 1~8단계 중 무엇이 실패하든 `torch.get_rng_state()`는 이 흐름에 들어오기 이전과 동일하게 남는다.
9. `torch.set_rng_state(payload["cpu_rng_state"])` — **가장 마지막**, 이후 다른 RNG 소비 작업 없이 곧바로 10번으로 진행.
10. 즉시 `run_training(model, train_loader, val_loader, training_config, device="cpu", resume_state=resume_state)` 호출.

사용자가 명시한 위험 요소별 확인 결과:

- **잘못된 checkpoint/config 때문에 실패하는 경우 전역 CPU RNG를 먼저 바꾸지
  않아야 함** — 위 순서대로면 metadata 불일치(5단계)나 config 불일치(8단계)로
  실패할 때 `torch.set_rng_state()`(9단계)가 아직 호출되지 않은 상태라 전역
  RNG가 오염되지 않는다. 이 스크립트는 실패 후 같은 프로세스에서 다른 작업을
  계속하지 않지만(즉시 `return 1`), 이 원칙을 지켜두면 향후 같은 코드를
  재사용할 때도 안전하다.
- **모델 생성 전에 CPU RNG를 복원하면 안 됨** — `build_model()`이 내부적으로
  `nn.init`으로 전역 RNG를 소비하므로(가중치는 곧 `load_state_dict()`로
  덮어써지지만 RNG 소비 자체는 막을 수 없음), `set_seed()` → `build_model()`
  → ... → `set_rng_state()`(맨 마지막)를 유지한다. 6단계와 9단계가 이 순서를
  지킨다.
- **dataset/transform 생성 전 RNG 복원** — `ImageFolder`/`build_transform()`은
  랜덤 연산이 없으므로(§1) 3단계를 1~2단계 뒤, 6단계(모델 생성) 앞 어디에
  두어도 정확성에 영향이 없다. 이번 설계는 metadata 계산을 위해 3단계를 일찍
  당겼을 뿐이다.
- **임의의 랜덤 텐서 생성 전 RNG 복원** — CIFAR-10 경로의 `random_split()`과
  달리 ImageFolder 경로에는 RNG를 소비하는 코드가 없다. 확인됨, 문제 없음.
- **generator 상태 복원이 DataLoader 생성보다 먼저** — 6단계 내부에서
  `restored_generator.set_state()` 후 `DataLoader(generator=restored_generator)`
  순서로 보장.
- **best_state_dict를 resume 시작점으로 착각** — 6단계에서 명시적으로
  `payload["model_state_dict"]`를 사용하도록 설계.

---

## 4. Dataset/Model 호환성 메타데이터 설계 — 후보 A vs B

**결론: 후보 A(별도 메타데이터 JSON 파일)를 채택한다.**

근거:

- `checkpoint.py`는 현재 **완전히 dataset-agnostic**하다 — `torchvision_dataset.py`를
  import하지 않고, `run_resume_training_e2e.py`(synthetic dataset)와 향후
  ImageFolder 경로 양쪽에서 그대로 재사용된다. ImageFolder 전용 필드
  (class_to_idx, 파일 경로 해시 등)를 payload에 넣으면 이 dataset-agnostic한
  경계가 깨지고, synthetic dataset 쪽 resume 흐름에도 불필요한 필드가 강제된다.
- `CHECKPOINT_FORMAT_VERSION=1`을 올리면 `load_training_checkpoint()`의 구조
  검증이 버전별로 분기해야 하고, 사용자가 명시적으로 "가볍게 건드리지 말 것"을
  요청했다. 별도 JSON은 이 위험을 완전히 피한다.
- 프로젝트에는 이미 "학습 산출물 + 그 옆의 JSON 사이드카"라는 확립된 패턴이
  있다: `save_training_history()`(history JSON), `save_class_mapping()`(class
  mapping JSON). ImageFolder resume metadata JSON은 같은 패턴의 세 번째
  사이드카일 뿐이라 구조적으로 자연스럽다.
- 단점(두 파일이 따로 놀 위험)은 경로 자동 유도(§8)로 상쇄한다 — 사용자가
  독립적으로 두 경로를 지정할 수 없게 만들어 항상 짝으로 관리된다.

---

## 5. ModelSpec fingerprint 설계

**결론: `model_definition/serialization.py`의 기존 `model_spec_to_dict()`를
그대로 재사용해 canonical dict를 얻고, `json.dumps(..., sort_keys=True, separators=(",", ":"))` 후 SHA-256 해시한다.**

`model_spec_to_dict()`(`serialization.py:100-106`)와 `_layer_to_dict()`
(`:47-63`)을 직접 확인한 결과:

- 반환 dict는 `{"name": str, "input_shape": list[int], "layers": [...]}`뿐이고,
  각 layer는 `{"type": type_name, **asdict(layer)}` (BranchSpec만 재귀 처리).
  **파일 경로/절대경로 관련 필드가 전혀 없다** — model-json 파일 경로가 달라져도
  해시에 영향 없음(원하는 동작과 일치: 파일을 옮기거나 이름을 바꿔도 내용이
  같으면 같은 해시).
- 키 순서: dict 리터럴이라 삽입 순서가 고정이고 `asdict()`도 dataclass 필드
  선언 순서를 유지하므로 이미 결정론적이지만, `sort_keys=True`로 한 번 더
  보장한다(추가 비용 거의 없음, 방어적).
- float 필드(예: `DropoutSpec.p`)는 Python `json` 모듈이 `repr(float)`
  기반의 최단 왕복 표현을 쓰므로 같은 Python 프로세스/버전 내에서는 안정적이다
  — 이 해시는 크로스 플랫폼/크로스 언어 비교용이 아니라 "같은 리포지토리에서
  저장 시점과 재개 시점의 ModelSpec이 같은가"만 확인하면 되므로 이 정도
  안정성으로 충분하다.
- 기존 `model_spec_to_dict()`를 그대로 쓰므로 `model_definition/serialization.py`는
  **수정하지 않는다** — 새 해시 helper 함수 하나만 추가하면 된다.

**어느 모듈이 이 helper를 소유해야 하는가**: `model_definition/serialization.py`에는
두지 않는다 — 그 파일의 책임은 "ModelSpec ↔ JSON 변환"이지 "해시"가 아니고,
이 해시는 ImageFolder resume 호환성 검사라는 좁은 용도로만 쓰인다. §7에서 정한
신규 모듈(`training/imagefolder_resume.py`)에 `hash_model_spec(model_spec) -> str`로 둔다 — `model_spec_to_dict`를 import해서 쓸 뿐 재구현하지 않는다.

---

## 6. ImageFolder dataset identity 검증 범위

class_to_idx + split별 샘플 수 + split별 **파일 목록의 canonical JSON 해시**로
한정한다(이미지 내용 해시 없음, 데이터셋 루트 절대경로 일치 요구 없음).

**줄바꿈으로 이어붙인 문자열 대신 canonical JSON을 해싱한다** — 리스트 원소가
단순 문자열이 아니라 `{path, class_index}` 구조를 갖도록 해서, 클래스 배정
정보를 "상대경로의 첫 구성요소가 클래스명"이라는 암묵적 관례에만 의존하지
않고 fingerprint에 명시적으로 포함한다:

```python
entries = [
    {
        "path": Path(path).relative_to(dataset.root).as_posix(),
        "class_index": class_index,
    }
    for path, class_index in dataset.samples
]
entries.sort(key=lambda item: (item["path"], item["class_index"]))

canonical = json.dumps(
    entries,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

(§1에서 확인한 `ImageFolder.samples`/`.root`를 그대로 사용, train/val/test
각 split마다 독립적으로 계산.)

- **경로는 `.as_posix()`로 `/` 정규화** — 사용자 요구사항(다른 머신/디렉터리로
  옮겨도 resume 가능)의 자연스러운 연장으로, Windows에서 만든 메타데이터를
  나중에 다른 OS에서 확인해도 경로 구분자 차이로 오탐하지 않는다.
- **`class_index`를 fingerprint에 명시적으로 포함** — 상대경로의 첫 구성요소가
  클래스명이라는 `ImageFolder` 구조상의 관례에만 암묵적으로 의존하지 않는다.
  class_to_idx는 metadata의 별도 필드로도 독립 비교되지만(§7), 그건 "클래스
  이름 집합/index 배정이 같은가"이고 이 해시는 "그 배정 하에서 실제 파일
  목록이 같은가"라 서로 다른 질문을 검증한다.
- **`entries.sort()`** — torchvision이 이미 정렬된 순서로 `.samples`를 주지만
  (§1), canonical JSON을 생성하는 이 함수 자체가 입력 순서에 의존하지 않도록
  방어적으로 명시 정렬한다.
- `json.dumps(..., sort_keys=True, separators=(",", ":"))`로 키 순서/공백을
  고정 — ModelSpec 해시(§5)와 동일한 canonicalization 방식이라 두 fingerprint
  로직의 스타일이 통일된다.
- **명시적 한계(문서화 필요)**: 같은 경로/클래스에 있는 파일의 **내용만**
  바뀌면(덮어쓰기) 탐지하지 못한다. 파일 크기/개수/이름/클래스 배정이 바뀌는
  경우만 탐지 대상이다 — 이미지 전체 내용 해싱은 비용 문제로 명시적으로
  제외한다.

---

## 7. 메타데이터 저장/로드 API 설계

**모듈 위치: `src/image_ai_studio/training/imagefolder_resume.py` (신규).**
`scripts/` 내부 private helper가 아니라 `src/` 하위 모듈로 두는 이유:

- §11-1의 메타데이터 단위 테스트가 필요한데, 이 프로젝트의 테스트는
  `tests/training/*.py`가 전부 `src.image_ai_studio.training.*`를 import하는
  구조이고 `scripts/`를 직접 단위 테스트하는 선례가 없다. 테스트 가능성이
  모듈 위치를 결정한다.
- ModelSpec 해시(§5)가 `model_definition.serialization`을 import해야 하는데,
  이런 조합 로직은 "ImageFolder resume 호환성"이라는 독립된 개념적 단위이지
  스크립트의 부속물이 아니다.
- 의존 방향 확인: 이 신규 모듈은 `model_definition.serialization`과
  `torchvision_dataset`(`ImageFolder`/`ImageFolderSplits` 타입)만 import하면
  되고, `config.py`/`loop.py`/`checkpoint.py`는 이 모듈을 import하지 않는다 —
  순환 의존 위험 없음(Phase 4F 리뷰에서 확립된 계층 규칙과 충돌하지 않음).

API (사용자가 제시한 후보를 그대로 채택 — 이름이 이미 프로젝트의
`require_compatible_resume_config` 명명 관례와 일치):

```python
@dataclass
class ImageFolderResumeMetadata:
    metadata_version: int
    model_spec_hash: str
    class_to_idx: dict[str, int]
    train_size: int
    val_size: int
    test_size: int
    train_files_hash: str
    val_files_hash: str
    test_files_hash: str

def hash_model_spec(model_spec: ModelSpec) -> str: ...
def build_imagefolder_resume_metadata(model_spec: ModelSpec, splits: ImageFolderSplits) -> ImageFolderResumeMetadata: ...
def save_imagefolder_resume_metadata(metadata: ImageFolderResumeMetadata, path: str | Path) -> None: ...
def load_imagefolder_resume_metadata(path: str | Path) -> ImageFolderResumeMetadata: ...
def require_compatible_imagefolder_resume_metadata(
    saved: ImageFolderResumeMetadata, current: ImageFolderResumeMetadata
) -> None: ...  # 불일치 시 어떤 필드가 왜 다른지 구체적으로 보여주는 ValueError


def metadata_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """checkpoint 파일 경로로부터 metadata sidecar 경로를 유도한다 (예:
    checkpoint.pt -> checkpoint.pt.meta.json). 저장/로드 양쪽이 반드시 이
    함수 하나만 거치도록 해서 경로 유도 규칙이 두 곳에서 따로 구현되어
    어긋나는 것을 막는다."""
    path = Path(checkpoint_path)
    return path.parent / f"{path.name}.meta.json"
```

**`metadata_path_for_checkpoint()`가 경로 유도의 유일한 소유자다** — CLI
(`run_imagefolder_training_e2e.py`)는 checkpoint 저장/로드 양쪽에서 이 함수만
호출하고, 자체적으로 `f"{path}.meta.json"` 같은 문자열 조합을 별도로 만들지
않는다. §8의 유도 규칙 설명은 이 함수의 동작을 서술한 것이지 별도 구현이
아니다.

`METADATA_FORMAT_VERSION = 1`을 이 모듈에 별도로 둔다 — `CHECKPOINT_FORMAT_VERSION`과
독립적인 버전 공간(메타데이터 포맷과 checkpoint 포맷은 서로 다른 이유로
바뀔 수 있음).

---

## 8. CLI 옵션 및 정책

- `--resume-from PATH`, `--checkpoint-out PATH`, `--epochs N`(신규, §2).
- **메타데이터 경로는 항상 자동 유도한다** (별도 `--resume-metadata`/
  `--checkpoint-metadata-out` 플래그를 두지 않음 — 두 파일이 항상 짝으로
  움직여야 하므로 독립 지정을 허용하면 §4에서 언급한 "따로 노는 위험"이 다시
  생긴다). 유도 규칙은 §7에서 정의한 `metadata_path_for_checkpoint()` 하나로
  통일한다(저장/로드 양쪽이 이 함수만 호출):

  ```python
  def metadata_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
      path = Path(checkpoint_path)
      return path.parent / f"{path.name}.meta.json"
  ```

  `Path.with_suffix()`는 쓰지 않는다 — `with_suffix()`는 **마지막 확장자를
  교체**하므로 `checkpoint.pt`처럼 점이 하나뿐인 이름에서도 의미가 불분명해질
  수 있고, 파일명에 점이 여러 개 있는 경우(`foo.v2.pt`) 잘못된 부분을 자를
  위험이 있다. `path.name + ".meta.json"`는 기존 파일명을 절대 건드리지 않고
  항상 뒤에만 붙이므로 충돌 여지가 없다. 확인해야 할 경로 케이스(§11-1 단위
  테스트 대상): `checkpoint.pt` → `checkpoint.pt.meta.json`, `foo.v2.pt` →
  `foo.v2.pt.meta.json`(점이 여러 개여도 안전), `checkpoint`(확장자 없음) →
  `checkpoint.meta.json`.

- 4가지 조합:

  | resume-from | checkpoint-out | 동작 |
  |---|---|---|
  | 없음 | 없음 | 기존 동작과 완전히 동일 (하위 호환) |
  | 없음 | 있음 | 신규 학습, 종료 후 checkpoint + metadata 저장 |
  | 있음 | 없음 | resume 실행, metadata 필수 검증, **새 checkpoint는 저장하지 않음** — CLI 출력에 "이번 실행 결과는 저장되지 않음"을 명확히 표시 |
  | 있음 | 있음 | resume 실행 + 종료 후 새 checkpoint/metadata 저장 (반복 resume의 일반적 형태) |

- 정책: `--resume-from` 지정 시 대응하는 metadata 파일 로드는 **항상 필수**다
  (없으면 명확한 `ValueError`: "checkpoint는 있는데 metadata sidecar가
  없다 — ImageFolder resume은 `<checkpoint>.meta.json`이 함께 있어야
  한다"). `--checkpoint-out` 지정 시 metadata 저장도 항상 자동으로 같이
  일어난다(선택 불가).
- **`--resume-from`과 `--checkpoint-out`이 같은 경로인 경우도 정상 지원한다**
  (반복 resume의 일반적인 사용 패턴 — 같은 파일을 계속 덮어쓰며 이어서
  학습). 다만 **checkpoint(.pt)와 metadata(.meta.json)는 독립된 두 개의
  `torch.save()`/`json.dumps()` 호출로 저장되므로, 저장 도중 프로세스가
  중단되면(정전, 강제 종료 등) 한쪽만 갱신되고 다른 쪽은 이전 상태로 남을 수
  있다.** 이번 Phase에서는 두 파일을 하나의 atomic 연산으로 묶는 것까지는
  구현하지 않는다(범위 밖, §12) — 이 한계는 README에도 명시한다.

---

## 9. stopped_early CLI 정책

Phase 4F 계약(조회는 허용, resume 실행은 거부)을 그대로 둔다 — 새로 감싸거나
더 이른 지점에서 중복 검사하지 않는다:

- `TrainingResumeState.__post_init__`과 `run_training()` 둘 다 이미
  구체적이고 실행 가능한 조언이 담긴 `ValueError`를 낸다(`loop.py:192-198`,
  `:278-285`). CLI는 이 예외를 기존 스크립트의 다른 검증 단계들과 동일한
  스타일로만 감싼다 — `try/except ValueError`로 잡아 `print(f"  FAIL: {exc}")` 후
  `return 1` (`ModelValidationError`/`TrainingConfigError` 처리와 동일 패턴,
  `:198-201`, `:173-176`). 메시지를 재구성하거나 더 일찍 별도로 재검사하지
  않는다 — 로직이 두 곳에 존재하면 나중에 서로 어긋날 위험만 생긴다.
- 이번 실행 자체가 early stopping으로 멈추고 `--checkpoint-out`이 주어진
  경우: **checkpoint는 그대로 저장한다** (`save_training_checkpoint()`/
  `load_training_checkpoint()`가 이미 `stopped_early=True` 파일의 저장/조회를
  허용함, `checkpoint.py:16-21`). CLI는 저장 직후 다음과 같은 안내를 출력한다:
  `"Checkpoint saved (stopped_early=True) -- weights/history can still be loaded, but this checkpoint cannot be used to resume training further."` README에도 동일 내용을 기술한다.

---

## 10. best model/export/parity 흐름 보존 + 버그 회피

§1에서 확인했듯 기존 스크립트는 `model`(현재 가중치)과 `best_model`(별도
인스턴스, best 가중치)을 이미 구조적으로 분리하고 있다. checkpoint 저장
호출을 `best_model = build_model(model_spec)`(`:275`) **이전**에 두면(§3-1),
"best_state_dict를 model에 로드한 뒤 그 model로 checkpoint를 저장" 버그는
설계상 발생할 수 없다 — 별도 model 인스턴스나 snapshot 복사가 필요 없다
(기존 코드가 이미 그 구조를 갖고 있음).

이후 코드(best_model 생성 → state_dict 저장 → class mapping → test 평가 →
TorchScript export → C++ parity, `:275-390`)는 **한 줄도 수정하지 않는다** —
checkpoint 저장은 그 앞에 새 블록으로만 삽입된다.

---

## 11. 테스트 설계

### 11-1. 메타데이터 단위 테스트 (`tests/training/test_imagefolder_resume.py`, 신규)

`tmp_path` + PIL fixture(기존 `test_torchvision_dataset.py` 패턴 재사용)로:

- round-trip: save → load → 필드 전부 동일
- `hash_model_spec()`: 같은 ModelSpec 두 번 호출 → 같은 해시; 레이어 파라미터
  하나만 바꾼 ModelSpec → 다른 해시
- `require_compatible_imagefolder_resume_metadata()`: class_to_idx 차이,
  split 크기 차이, 파일 해시 차이(예: tmp fixture에서 파일 하나 rename),
  `metadata_version` 불일치, 필수 필드 누락(malformed) 각각에서 명확한
  `ValueError` 발생 확인 — 전수 조합이 아니라 계약의 핵심만.
- `metadata_path_for_checkpoint()`: `checkpoint.pt`/`foo.v2.pt`/`checkpoint`
  (확장자 없음) 세 케이스가 각각 `checkpoint.pt.meta.json`/
  `foo.v2.pt.meta.json`/`checkpoint.meta.json`로 유도되는지 확인.

### 11-2. ImageFolder exact-resume 통합 테스트

같은 파일 또는 인접 파일에서, `tmp_path`에 작은 실제 ImageFolder(2 클래스 x
소수 이미지)를 만들고, Dropout이 포함된 작은 ModelSpec(§1에서 확인한 대로
`build_transform()`이 augmentation 없이 완전히 결정론적이므로, CPU RNG
복원이 실제로 검증되려면 Dropout처럼 RNG를 소비하는 레이어가 모델에 있어야
함 — `run_resume_training_e2e.py`와 동일한 이유)로 다음을 비교한다:

- 연속 4 epoch 실행 vs (2 epoch → checkpoint 저장 → resume → 2 epoch 추가)
- `model.state_dict()`, optimizer/scheduler state, 전체 `history`,
  `best_state_dict`, `epochs_without_improvement`가 `torch.equal()` 기준
  정확히 일치. scheduler state를 실제로 검증하려면 `lr_scheduler="plateau"`를
  켠 `TrainingConfig`를 써야 하고(끄면 양쪽 다 `None`이라 비교가 공허해짐),
  optimizer/scheduler state_dict는 텐서와 스칼라가 섞인 중첩 dict라 재귀
  비교 헬퍼(`_assert_deep_equal`, `test_loop.py`의 동일 헬퍼와 같은 패턴)가
  필요함 — 구현 시 최초 버전이 이를 누락했다가 리뷰로 보강됨.

이 테스트는 CLI를 subprocess로 실행하지 않고, `imagefolder_resume.py`와
`checkpoint.py`/`loop.py`의 함수를 직접 호출한다 — `test_loop.py`의 기존
resume-exactness 테스트와 같은 레벨(빠르고, CLI 인자 파싱과는 관심사가
분리됨).

### 11-3. CLI/parser 테스트 (인자 파싱만)

`parse_args(argv=None)`는 이미 `argv`를 직접 받는 함수라(`:126`), **parser
factory로 리팩터링할 필요 없이** 그대로 테스트 가능하다 — 예:
`parse_args(["--resume-from", "foo.pt"])`. `scripts/`를 테스트에서 import하는
선례가 없으므로, 이 스크립트를 `importlib`로 로드하는 방식(스크립트 상단의
`sys.path.insert(0, str(REPO_ROOT / "src"))`가 부작용이지만 여러 번 실행돼도
멱등이라 안전)을 소규모로 검증한 뒤 신규 테스트 파일
`tests/scripts/test_run_imagefolder_training_e2e_args.py`에서 사용한다. 새
플래그(`--epochs`/`--resume-from`/`--checkpoint-out`)의 기본값/파싱만 확인
— 학습 자체는 실행하지 않는다.

### 11-4. CLI 배선 통합 테스트 (실제 `main(argv)` 실행)

**신규.** metadata 단위 테스트(§11-1), 함수 수준 exact-resume 테스트(§11-2),
parser 테스트(§11-3)만으로는 `main(argv)` 내부의 실제 배선(§3-1/§3-2에서
설계한 순서가 스크립트 코드로 정확히 옮겨졌는지, `argparse` 값이 올바른
변수로 전달되는지, 함수 호출 인자 이름이 맞는지)까지는 검증하지 못한다.
`tests/scripts/test_run_imagefolder_training_e2e_resume_cli.py`(신규)에서
실제 `main(argv)`를 두 번 호출한다:

```text
1회차: main(["--model-json", tiny_model_json, "--dataset-root", ds_root,
             "--epochs", "2", "--checkpoint-out", str(ckpt1)])
2회차: main(["--model-json", tiny_model_json, "--dataset-root", ds_root,
             "--epochs", "1", "--resume-from", str(ckpt1),
             "--checkpoint-out", str(ckpt2)])
```

fixture: `tmp_path`에 클래스 2개 x 소수 이미지짜리 실제 ImageFolder
train/val/test 폴더(PIL로 생성, 기존 `test_torchvision_dataset.py` 패턴
재사용)와, 그 dataset에 맞는 아주 작은 ModelSpec JSON. `find_runner_binary`/
`run_case`(C++ TorchScript runner 호출부)는 monkeypatch로 대체해 실제 빌드된
바이너리 없이도 테스트가 통과하도록 한다 — C++ parity 자체의 정확성은 이
테스트의 책임이 아니다(기존 E2E 스크립트가 이미 커버). TorchScript export는
순수 Python(`torch.jit`)이라 monkeypatch 없이 그대로 실행한다.

확인 항목:

- 두 호출 모두 반환값 `0`(PASS).
- `ckpt1`/`metadata_path_for_checkpoint(ckpt1)`, `ckpt2`/
  `metadata_path_for_checkpoint(ckpt2)` 네 파일이 모두 생성됨 — 경로 유도
  규칙이 저장 시점에 실제로 적용됨을 확인.
- 2회차 실행이 `ckpt1`과 그 metadata를 실제로 로드해서 사용했는지 —
  `load_training_checkpoint(ckpt2)["history"]`의 `train_losses` 길이가
  **3**(1회차 2 epoch + 2회차 1 epoch)인지로 간접 확인. 이건 `--epochs`가
  resume 시 "총 epoch"가 아니라 "추가 epoch 수"로 실제 CLI 경로에서도
  동작함을 증명하는 핵심 assertion이다.
- exact tensor 동등성(가중치 bit-identical 여부)은 이 테스트의 책임이
  아니다 — 그건 §11-2의 함수 수준 통합 테스트가 이미 담당하며, CLI 테스트는
  "배선이 맞는가"만 확인해 책임을 분리한다.

### 11-5. 회귀

반드시 재실행하고 기존과 동일하게 통과해야 하는 것: 기존 ImageFolder E2E
기본 실행(신규 플래그 없이), 기존 synthetic/real E2E, Phase 4F resume E2E
(`run_resume_training_e2e.py`), `tests/training/` 전체, 전체 `pytest`, 기존
TorchScript/C++ parity.

---

## 12. 제외 목록 확인

다음은 이번 Phase 범위에 포함하지 않는다 (사용자 목록 그대로 확인):
epoch마다 자동 checkpoint, 콜백, latest/best checkpoint 로테이션, N개
checkpoint 보존, SIGINT/Ctrl+C 자동 저장, batch-level resume, CUDA RNG
resume, multi-worker DataLoader resume, distributed checkpoint, random
augmentation exact-resume 지원, 전체 이미지 내용 해싱, GUI 연동, 다른 dataset
타입용 resume CLI, config override.

---

## 13. 파일 변경 계획

**수정**: `scripts/run_imagefolder_training_e2e.py` (CLI 플래그 + resume/checkpoint
배선), `README.md` (플래그/정책/예시 문서화).

**신규**: `src/image_ai_studio/training/imagefolder_resume.py`,
`tests/training/test_imagefolder_resume.py`,
`tests/scripts/test_run_imagefolder_training_e2e_args.py`(§11-3),
`tests/scripts/test_run_imagefolder_training_e2e_resume_cli.py`(§11-4, 신규),
`docs/phase4g_imagefolder_resume_design.md`(본 문서).

**변경하지 않음** (분석 결과 필요성을 찾지 못함 — 기존 API로 전부 충분):
`config.py`, `loop.py`, `checkpoint.py`, `history.py`, `dataset.py`,
`torchvision_dataset.py`, `model_definition/*`, `export/*`, `parity/*`,
C++ 코드, 다른 기존 E2E 스크립트. `run_training()`의 `resume_state` 매개변수,
`save_training_checkpoint()`/`load_training_checkpoint()`, `model_spec_to_dict()`,
`ImageFolder.samples`/`.root`가 이미 필요한 모든 것을 제공한다.

---

## 14. 구현 순서 (작은 단계)

1. `imagefolder_resume.py` 구현 (dataclass + `metadata_path_for_checkpoint()` + hash helper + save/load/require_compatible) — 순수 라이브러리 코드, CLI 미배선.
2. §11-1 단위 테스트 작성/통과(경로 유도 3케이스 포함).
3. `parse_args()`에 `--epochs`/`--resume-from`/`--checkpoint-out` 추가, **신규 학습 + checkpoint 저장 경로만** 배선(resume은 아직). 기존 기본 E2E가 그대로 통과하는지 먼저 확인(회귀), `--checkpoint-out` 지정 시 `checkpoint.pt` + `checkpoint.pt.meta.json` 생성 확인.
4. resume 경로 배선 — §3-2의 수정된 순서(metadata 검증 → model/DataLoader 준비 → ResumeState/config 검증 → CPU RNG 복원 → `run_training()`)를 그대로 코드로 옮긴다.
5. §11-2 exact-resume 통합 테스트(함수 수준, CLI 아님).
6. §11-3 CLI/parser 테스트(인자 파싱만).
7. §11-4 CLI 배선 통합 테스트(`main(argv)` 2회 호출, checkpoint→resume→checkpoint 체인 검증) — 4단계 배선이 실제로 맞물려 동작하는지 이 시점에 처음 증명된다.
8. stopped_early CLI 동작 수동 검증(작은 규모로 `--early-stopping-patience 1` + `--checkpoint-out` 후 resume 시도 → 거부 메시지 확인) — 가능하면 소규모 자동 테스트로도 추가.
9. README 갱신.
10. 전체 회귀(모든 E2E 스크립트 + 전체 pytest) + 본 문서의 최종 테스트 수치 갱신.

---

## 15. 미정 사항 + 추천

1. **`--batch-size`/`--learning-rate`도 CLI로 노출할지** — 추천: **아니오,
   범위 밖**. 현재 하드코딩값이라 resume 비교에서 항상 자동으로 일치하고,
   추가하면 "resume을 CLI에 연결한다"는 이번 Phase 목표를 넘어서는 기능
   확장이 된다.
2. **resume-from만 있고 checkpoint-out이 없는 조합을 허용할지(강제로 막을지)** —
   추천: **허용**. 사용자의 조합 매트릭스 요구사항 자체가 4가지 조합을 모두
   정의하라는 것이었고, 상태를 남기지 않고 한 번 더 이어서 돌려보는 것도
   정당한 사용 사례다.
3. **상대경로 해시에 `/` 정규화(`as_posix()`)를 넣을지** — **결정 완료, §6에
   반영됨**(canonical JSON fingerprint의 `path` 필드가 이미 `.as_posix()`를
   사용). 비용이 거의 없고 이식성을 실질적으로 높인다는 판단은 그대로 유지.
4. **`METADATA_FORMAT_VERSION`을 `CHECKPOINT_FORMAT_VERSION`과 별도로 둘지** —
   추천: **별도로 둔다**. 두 포맷은 서로 다른 이유로 바뀔 수 있는 독립적인
   개념이라 하나로 묶으면 향후 마이그레이션 시 불필요한 결합이 생긴다.

이 설계는 §1의 실제 코드 분석에 근거하며, 사용자가 제시한 파일 후보/모듈
후보를 그대로 채택했다(분석 결과 더 나은 구조가 필요하다는 근거를 찾지
못함). 구현은 §14 순서대로 완료됐고, 문서 상단의 최종 검증 결과와 개정
2(리뷰 반영)가 그 결과를 반영한다.
