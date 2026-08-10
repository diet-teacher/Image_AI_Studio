# Phase 4O: Test Classification Metrics — 설계안

## 1. 목적

ImageFolder 학습의 최종 test 평가에 confusion matrix, per-class recall,
macro precision/recall/F1을 추가해, 단일 scalar(`test_loss`/
`test_accuracy`)만으로는 보이지 않는 클래스 불균형/오분류 패턴을
진단할 수 있게 한다.

범위는 **test-only**로 좁힌다: 학습 루프의 validation 경로(`evaluate()`,
`TrainingHistory`, checkpoint/resume, `ReduceLROnPlateau`, early
stopping, best-model 선택)는 전부 무수정으로 유지하고, 최종 test 평가
1회에서만 상세 지표를 계산해 `test_result.json`에 추가한다.

## 2. 기존 구조 (재확인)

* `loop.py::evaluate(model, loader, device="cpu") -> (loss, accuracy)` --
  `nn.CrossEntropyLoss()`(항상 unsmoothed) + argmax accuracy,
  sample-weighted 평균. `run_training()`의 validation 경로와
  `imagefolder_workflow.py`의 최종 test 평가가 기존에 이 함수를
  공유했다.
* `evaluate()`를 `image_ai_studio.training.loop.evaluate` 경로로
  monkeypatch하는 테스트가 `test_loop.py`(21곳) + `test_imagefolder_workflow.py`
  (2곳) + `test_checkpoint.py`(1곳) = 24곳 있고, 전부
  `lambda model, loader, device="cpu": ...` 고정 시그니처를 쓴다.
* `imagefolder_workflow.py`의 최종 test 평가는 `evaluate(best_model,
  test_loader, device="cpu")` 단일 호출이었고, 그 결과를 그대로
  `test_result.json = {"test_loss": ..., "test_accuracy": ...}`로
  저장했다. 이 호출이 test set에 대한 유일한 forward pass였다.
* `class_to_idx`/`classes`는 `torchvision_dataset.py`/
  `imagefolder_workflow.py`(ImageFolder 전용 계층)에만 존재한다 --
  generic training core(`loop.py`)는 class 이름 개념 자체가 없다.
* `pyproject.toml`의 `dependencies = ["torch>=2.4", "numpy"]` -- sklearn은
  설치/선언 어디에도 없음(직접 `import sklearn` 확인).

## 3. `evaluate()` API 설계 (evaluate() 무수정 확정)

evaluate()의 반환값을 확장하거나(Method A) 새 dataclass로 교체하는
대신(Method C), `evaluate()`는 완전히 무수정으로 남기고 별도 함수
`evaluate_classification_metrics()`를 추가한다(Method B).

이유: `evaluate()`는 학습 루프(매 epoch, best-model 선택/early
stopping/scheduler)가 의존하는 안정된 계약 -- `(loss, accuracy)`
2-tuple, 고정 시그니처 -- 을 갖고 있고, 최종 test 진단은 다른 목적(다른
소비자, 다른 빈도)의 별도 관심사다. 24곳의 monkeypatch가 이 계약에
의존한다는 사실은 "테스트가 깨지니까 피한다"는 이유가 아니라, 그
계약이 실제로 안정적이고 널리 신뢰되고 있다는 방증으로만 참고했다.

## 4. `evaluate_classification_metrics()` 구조

```python
def evaluate_classification_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> tuple[float, float, ClassificationMetrics]:
    ...
```

`loop.py`에 추가한다. 한 번의 DataLoader 순회 안에서:

* `evaluate()`와 정확히 같은 방식(unsmoothed `nn.CrossEntropyLoss()`,
  argmax accuracy, sample-weighted 평균)으로 loss/accuracy를 계산
* confusion matrix를 배치 단위로 누적(`torch.bincount(labels *
  num_classes + predictions, minlength=num_classes**2).reshape(...)`)

를 함께 수행해 반환한다. `imagefolder_workflow.py`의 최종 test 평가는
`evaluate()` 대신 이 함수 하나만 호출한다 -- 같은 test 데이터셋을
`evaluate()`로 한 번, 이 함수로 또 한 번 두 번 순회하지 않는다.

empty loader 정책은 `evaluate()`와 동일하게 `ValueError`다(zero-division
으로 조용히 0 loss/accuracy를 반환하지 않는다).

confusion matrix는 evaluation 동안 `device` 위에서 배치 단위로
누적한다 -- GPU 평가 시에도 배치마다 CPU로 옮기는 동기화 없이 텐서
상에서만 더한다. evaluation이 끝난 뒤 `[num_classes, num_classes]`
matrix 하나를 `compute_classification_metrics()`에 넘기면, 그 함수
**진입 시 한 번만** `.cpu()`로 옮기고 그 이후 class별 파생 지표 계산은
전부 그 CPU tensor에서 수행한다(§5). class 수만큼 반복되는
`.item()` 호출을 GPU tensor에 대고 직접 하면 호출마다
device-to-host synchronization이 발생하므로, 그 반복 전에 먼저
tensor 전체를 한 번 옮겨야 이 계약이 실제로 지켜진다.

`num_classes`는 명시적 인자로 받는다. `imagefolder_workflow.py`는
`len(splits.classes)`(기존에 `require_matching_num_classes()`가 이미
model 출력 차원과의 일치를 검증한 값)를 그대로 넘긴다 -- 중복 검증을
추가하지 않는다. `num_classes <= 0`이면 confusion matrix 생성 자체가
의미가 없으므로 `evaluate_classification_metrics()` 진입 시 바로
`ValueError`를 낸다(positive-integer 최소 계약만, bool 처리 등
TrainingConfig 수준의 검증은 하지 않는다).

`loss`/`accuracy` 계산 방식은 `evaluate()`와 정확히 맞추되, `evaluate()`
내부를 refactor해서 공유 helper로 뽑아내는 것은 하지 않았다 -- 코드
중복(약 10줄)이 생기지만, Phase 4O 하나 때문에 기존 안정 API인
`evaluate()`의 내부 구조를 바꾸는 리스크를 지지 않는다.

## 5. `ClassificationMetrics` / `metrics.py`

신규 모듈 `src/image_ai_studio/training/metrics.py`:

```python
@dataclass
class ClassificationMetrics:
    confusion_matrix: list[list[int]]
    per_class_recall: list[float]
    macro_precision: float
    macro_recall: float
    macro_f1: float


def compute_classification_metrics(confusion_matrix: torch.Tensor) -> ClassificationMetrics:
    ...
```

loss/accuracy는 이 dataclass에 없다 -- `evaluate_classification_metrics()`
의 반환값이 `(loss, accuracy, ClassificationMetrics)`로 이미 loss/accuracy
를 별도로 반환하므로, `ClassificationMetrics`는 confusion matrix 기반
상세 지표만 담는다(중복 필드 없음). class 이름도 담지 않는다(§7).

이 모듈은 모델 forward/DataLoader 순회를 전혀 모른다 -- 이미 누적된
confusion matrix tensor 하나로부터 순수하게 파생 지표만 계산하는
함수만 둔다(모델/DataLoader 없이 정수 tensor만으로 단위 테스트 가능).

`compute_classification_metrics()`는 입력 tensor를 받으면 가장 먼저
`.detach().cpu()`로 한 번 옮긴 뒤(§4), 그 CPU tensor로 `diagonal()`/
`sum(dim=1)`/`sum(dim=0)`과 class별 `.item()` 반복을 전부 수행한다.
반환하는 `confusion_matrix`도 이 CPU tensor의 `.tolist()`다.

입력 shape에 대한 최소 방어도 이 함수 진입 시 수행한다: 2차원 tensor가
아니거나, square가 아니거나, `num_classes == 0`(빈 matrix)이면
`ValueError`를 낸다. dtype 강제, 음수 값 검증, "확인된 batch 수와
matrix 합이 일치하는가" 같은 넓은 범위의 입력 검증은 하지 않는다 --
shape/non-empty 계약만 최소로 방어한다.

새로운 `ClassificationEvaluationResult` 같은 상위 dataclass는 만들지
않는다 -- `(loss, accuracy, ClassificationMetrics)` tuple 반환이 기존
`evaluate()`의 `(loss, accuracy)` 계약과 자연스럽게 이어지고, 5개 필드
묶음 이상으로 abstraction을 늘릴 필요가 없다.

## 6. Confusion matrix 컨벤션

고정: `confusion_matrix[true_idx][predicted_idx]` -- row=실제 클래스,
column=예측 클래스. shape `[num_classes, num_classes]`. 내부 누적은
`torch.long` tensor, `ClassificationMetrics.confusion_matrix`로 노출할
때만 `list[list[int]]`로 변환한다(JSON 직렬화 대상이므로).

## 7. Class 순서 / class mapping 계약

`ClassificationMetrics`는 class 이름을 모른다 -- generic training core에는
class 이름 개념이 없다(Synthetic-training API 경로는 애초에 class 이름이
없음). confusion matrix/`per_class_recall`의 index 순서는 이미 저장되는
`class_mapping.json`의 `classes` 배열 순서와 동일하다는 계약으로
대신한다. `test_result.json`에 class 이름을 중복으로 넣지 않는다 --
class mapping의 source of truth는 `class_mapping.json` 하나로 유지한다.

## 8. Metric 정의

각 class `i`에 대해 `TP_i = cm[i][i]`, `true_count_i = sum(cm[i][:])`,
`pred_count_i = sum(cm[:, i])`.

* `recall_i = TP_i / true_count_i if true_count_i > 0 else 0.0` --
  외부 노출 이름은 `per_class_recall`. `per_class_accuracy`라는 이름은
  쓰지 않는다 -- class별 accuracy는 정의상 recall과 동일하기 때문에
  `per_class_accuracy`라는 이름은 전체 accuracy와 혼동을 유발할 수 있다.
* `precision_i = TP_i / pred_count_i if pred_count_i > 0 else 0.0`
* `f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if
  (precision_i + recall_i) > 0 else 0.0`
* `macro_precision = mean(precision_i)`, `macro_recall = mean(recall_i)`,
  `macro_f1 = mean(f1_i)` -- **`macro_f1`은 class별 F1을 먼저 구한 뒤
  평균한 값이지, `harmonic_mean(macro_precision, macro_recall)`이
  아니다.** 이 구분은 `tests/training/test_metrics.py`의
  `test_macro_f1_is_mean_of_per_class_f1_not_harmonic_mean_of_macros`로
  고정했다.

## 9. zero-division 정책

분모가 0이면 해당 class metric은 `0.0`이다(NaN/스킵/에러 대신). 이유:
(1) NaN은 표준 JSON에 없어 직렬화 시 문제, (2) 에러는 "특정 class에
예측/정답이 전혀 없으면 test 평가 자체가 실패한다"는 지나치게 강한
제약, (3) 스킵은 macro 평균의 분모가 클래스마다 달라져 혼란, (4) `0.0`은
"이 class에 대해 측정 불가/전혀 맞추지 못함"으로 사용자가 직관적으로
해석 가능하고 macro 평균 분모도 항상 `num_classes`로 고정된다.

**`0.0`이 항상 "모델이 그 class를 전부 틀렸다"는 뜻은 아니다** -- 특히
test set에 해당 class의 true sample 자체가 없으면 `recall=0.0`은
"측정할 sample이 없어 정책상 0.0을 기록했다"는 뜻이다. 사용자는 함께
저장되는 confusion matrix의 해당 row/column 합이 0인지 봐서 이 두
경우를 구별할 수 있다. 이번 Phase에서 별도 support/count field는
추가하지 않는다.

## 10. `ImageFolderWorkflowResult` 하위 호환

repository 전체에서 `ImageFolderWorkflowResult(` 생성 호출을 grep한
결과, production 1곳(`imagefolder_workflow.py` 자체의 반환문)과
`tests/scripts/test_train_imagefolder_cli.py`의 fake workflow 함수
8곳(CLI 테스트가 workflow를 monkeypatch하기 위해 만드는 fake 결과)이
발견됐다. 이 8곳은 `test_metrics`를 전혀 모르는 채로 이 dataclass를
직접 생성한다.

`test_metrics: ClassificationMetrics`를 required field로 추가하면
이 8곳이 전부 깨진다. 대신:

```python
test_metrics: ClassificationMetrics | None = None
```

를 dataclass의 **마지막 필드**로(기본값 없는 필드 뒤에 기본값 있는
필드가 오는 dataclass 순서 제약을 만족시키기 위해) 추가했다. 계약:

* `run_imagefolder_training_workflow()`가 정상 완료해 반환하는 production
  결과의 `test_metrics`는 항상 실제 `ClassificationMetrics`다(`None`이
  아니다) -- 최종 test 평가가 항상 수행되기 때문이다.
* `None`은 "production에서 test 평가가 생략될 수 있다"는 의미가 아니라
  순수하게 constructor 하위호환을 위한 기본값이다.
* 기존 8곳의 fake constructor 호출과 실제 production 호출 둘 다
  실행해 확인했다(`tests/scripts/test_train_imagefolder_cli.py` 39개,
  `tests/training/test_imagefolder_workflow.py`의 신규
  `test_production_result_has_classification_metrics` 등 전부 통과).

## 11. `test_result.json` 스키마

```json
{
  "test_loss": 0.42,
  "test_accuracy": 0.88,
  "classification_metrics": {
    "confusion_matrix": [[45, 5], [7, 43]],
    "per_class_recall": [0.9, 0.86],
    "macro_precision": 0.87,
    "macro_recall": 0.88,
    "macro_f1": 0.875
  }
}
```

기존 top-level key(`test_loss`, `test_accuracy`)는 이름/의미 변경 없이
그대로 유지한다(additive). `classification_metrics`는 `ClassificationMetrics`
를 `dataclasses.asdict()`로 변환한 것과 동일 구조다. schema version
필드는 이번 Phase에서 추가하지 않는다(순수 additive 변경이라 소비자가
키 존재 여부로 구버전/신버전을 구분할 수 있음).

## 12. CLI 출력 정책

`scripts/train_imagefolder.py`의 stdout은 **변경하지 않는다** -- 기존
`Test: loss=... accuracy=...` 한 줄을 그대로 유지하고 macro F1 등을
추가로 찍지 않는다. 클래스 수가 많으면 confusion matrix 전체를 stdout에
찍는 것은 나쁜 UX이고, 상세 지표는 이미 파일로 나가는 `test_result.json`
에서 확인할 수 있다. 이 정책 덕분에 `scripts/train_imagefolder.py`는
Phase 4O에서 **무수정**이다.

## 13. TrainingHistory / checkpoint / config 무영향

`TrainingHistory`, `checkpoint.py`, `config.py`는 이번 Phase에서 전혀
수정하지 않는다. `RESUME_CONFIG_FIELDS`, `RESUME_CONFIG_LEGACY_DEFAULTS`,
checkpoint format version, exact resume 전부 무영향이다. 근거:

* metric 계산은 학습이 끝난 뒤(`run_training()` 완료 후)
  `imagefolder_workflow.py`가 별도로 수행하는 최종 test 평가이므로,
  checkpoint 저장 시점에는 애초에 test metric이 존재하지 않는다.
* metric 계산은 `evaluate()`와 마찬가지로 `model.eval()` +
  `torch.inference_mode()` 하에서 순수 forward pass만 수행하므로
  gradient도, RNG 소비(eval 모드에서는 dropout 등 확률적 연산이
  비활성)도 없다 -- exact-resume이 의존하는 RNG 상태에 전혀 영향 없다.

## 14. 테스트 전략

* `tests/training/test_metrics.py`(신규, 12개): 손계산 confusion matrix
  예시(`true=[0,0,1,1,2,2]`, `pred=[0,1,1,1,0,2]`)로 confusion
  matrix/recall/precision/F1을 고정, perfect prediction, 한 class가
  전혀 predicted되지 않음, 한 class가 target에 없음, true/pred 양쪽에
  모두 없는 class, `num_classes` > 실제 등장 class 수, 정수 보존,
  `macro_f1`이 harmonic mean이 아니라 class별 F1 평균이라는 구분,
  그리고 안정화 라운드에서 추가한 입력 shape 계약 테스트 3개(non-2D
  거부, non-square 거부, `0 x 0` empty 거부).
* `tests/training/test_loop.py`(신규 8개): `evaluate_classification_metrics()`
  의 loss/accuracy가 기존 `evaluate()`와 일치, accuracy와 confusion
  matrix diagonal-sum invariant, 반환 shape, 전 지표 finite(NaN/inf
  없음), model parameter 불변, empty loader 정책이 `evaluate()`와 동일,
  그리고 안정화 라운드에서 추가한 `num_classes<=0`(0, -1 parametrize)
  거부 테스트 2개. `evaluate()`의 기존 테스트를 복제하지 않고 새 함수에
  필요한 차이점만 검증했다.
* `tests/training/test_imagefolder_workflow.py`(신규 3개): production
  결과의 `test_metrics is not None` + shape + `test_result.json`
  nested 스키마(기존 key 유지 + 정수 confusion matrix 보존), class
  index 순서가 실제 `class_mapping.json`과 일치함을 class별 test
  sample 개수를 비대칭(cat=6, dog=2)으로 만들어 관찰 가능한 방식으로
  검증(shape만 보고 순서가 맞다고 주장하지 않음),
  `ImageFolderWorkflowResult` constructor 하위 호환(`test_metrics`
  생략 시 `None`).
* 전체 회귀: 535 -> 558 tests collected(23개 신규: metrics 12 + loop 8 +
  workflow 3), 558 passed.

## 15. E2E 회귀

새 Phase 4O 전용 E2E 스크립트는 만들지 않았다. 기존
`scripts/run_imagefolder_training_e2e.py`의 class mapping 재검증 직후에
"Classification metrics check" 게이트를 추가해, `test_result.json`의
`classification_metrics`가 실제 production artifact로 기록되는지와
accuracy-confusion matrix diagonal 일치 invariant를 end-to-end로
확인했다. 기존 4개 E2E(`run_phase1_e2e.py`, `run_training_e2e.py`,
`run_real_training_e2e.py`, `run_resume_training_e2e.py`) + 이번에
확장한 `run_imagefolder_training_e2e.py`를 전부 재실행해 PASS를
확인했다. export/C++ inference 로직은 수정하지 않았으므로 별도
parity 검증은 추가하지 않았고, 기존 E2E 안의 parity 비교는 그대로
실행된다(전부 PASS).

## 16. 제외 범위 (재확인)

validation epoch별 상세 metric, `TrainingHistory`의 metric 필드,
metric 기반 early stopping, metric 기반 scheduler, class weight,
BCE/focal loss, per-class precision/F1 노출, micro/weighted average,
sklearn dependency, ROC-AUC/PR-AUC, top-k accuracy, specificity, CLI
macro F1 출력, GPU/device 노출, checkpoint 변경, config 변경, 기존
`evaluate()` API 변경, 새로운 E2E 스크립트.
