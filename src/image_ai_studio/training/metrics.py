"""Confusion matrix 기반 classification metric 계산 (Phase 4O). 이 모듈은
모델 forward/DataLoader 순회를 전혀 모른다 -- 이미 누적된 confusion matrix
tensor 하나로부터 순수하게 파생 지표만 계산한다(모델/DataLoader 의존이
없어 단위 테스트가 model/DataLoader 없이 정수 tensor만으로 가능하다).
confusion matrix를 실제로 배치별로 누적하는 forward pass는
`loop.py`의 `evaluate_classification_metrics()`가 담당한다.

confusion matrix convention(고정): `confusion_matrix[true_idx][predicted_idx]`
-- row=실제 클래스, column=예측 클래스. sklearn 등 흔한 관례와 동일하게
맞췄다.

zero-division 정책(고정): 분모가 0인 class metric은 0.0으로 처리한다
(NaN/스킵/에러 대신). 예를 들어 어떤 class가 test set의 true label에
전혀 없으면 그 class의 recall은 0.0이 된다 -- 이는 "모델이 그 class를
전부 틀렸다"는 뜻이 아니라 "측정할 sample이 없어 정책상 0.0을 기록했다"는
뜻이다. 사용자는 함께 저장되는 confusion_matrix의 해당 row/column 합이
0인지 봐서 이 두 경우를 구별할 수 있다(이 모듈은 별도 support/count
필드를 추가하지 않는다).

macro_f1은 harmonic_mean(macro_precision, macro_recall)이 아니다 --
class별 f1을 먼저 구한 뒤 그 값들을 평균한다."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationMetrics:
    """confusion matrix와 그로부터 파생된 macro 지표만 담는다 (loss/accuracy는
    담지 않는다 -- 그건 evaluate()/evaluate_classification_metrics()의 기존
    (loss, accuracy) 반환값이 이미 책임진다, 이 dataclass와 별개 관심사).
    class 이름은 담지 않는다 -- confusion_matrix/per_class_recall의 index
    순서가 어떤 클래스를 가리키는지는 이 dataclass의 관심사가 아니고,
    ImageFolder 워크플로우가 이미 갖고 있는 class_mapping.json의 classes
    순서와 동일하다는 계약으로 대신한다(이 모듈은 class 이름 개념 자체를
    모른다 -- generic training core에는 class 이름이 없다)."""

    confusion_matrix: list[list[int]]
    per_class_recall: list[float]
    macro_precision: float
    macro_recall: float
    macro_f1: float


def compute_classification_metrics(confusion_matrix: torch.Tensor) -> ClassificationMetrics:
    """누적된 `[num_classes, num_classes]` confusion matrix(row=true,
    column=predicted) tensor로부터 per-class recall과 macro
    precision/recall/f1을 계산한다. 입력 tensor는 수정하지 않는다.

    입력이 GPU tensor일 수 있으므로(evaluate_classification_metrics()가
    evaluation device 위에서 배치 단위로 누적한 결과), 진입 시 **한 번만**
    `.cpu()`로 옮긴 뒤 그 CPU tensor로 모든 class별 계산을 수행한다 --
    `[num_classes, num_classes]`는 항상 작은 고정 크기이므로 이 전체
    이동 자체는 매 배치가 아니라 evaluation 1회당 1번뿐이다. 이렇게 하지
    않고 GPU tensor에 대해 class마다 `.item()`을 반복하면 호출마다
    device-to-host synchronization이 발생해 배치 단위로는 device에
    누적하고 CPU 변환은 마지막에 한 번만 한다는 계약(evaluate_classification_metrics()
    쪽)이 여기서 사실상 무너진다.

    shape 계약(위반 시 ValueError, dtype/음수/기타 값 검증은 하지 않음):
    2차원, square, `num_classes > 0`."""
    if confusion_matrix.dim() != 2:
        raise ValueError(
            f"confusion_matrix must be a 2D [num_classes, num_classes] tensor, "
            f"got {confusion_matrix.dim()}D (shape={tuple(confusion_matrix.shape)})"
        )
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(f"confusion_matrix must be square, got shape={tuple(confusion_matrix.shape)}")
    if confusion_matrix.shape[0] == 0:
        raise ValueError("confusion_matrix must not be empty (num_classes must be > 0)")

    matrix = confusion_matrix.detach().cpu()
    num_classes = matrix.shape[0]
    true_positives = matrix.diagonal()
    true_counts = matrix.sum(dim=1)
    pred_counts = matrix.sum(dim=0)

    per_class_recall: list[float] = []
    per_class_precision: list[float] = []
    per_class_f1: list[float] = []
    for i in range(num_classes):
        tp = true_positives[i].item()
        true_count = true_counts[i].item()
        pred_count = pred_counts[i].item()

        recall = tp / true_count if true_count > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_recall.append(recall)
        per_class_precision.append(precision)
        per_class_f1.append(f1)

    macro_precision = sum(per_class_precision) / num_classes
    macro_recall = sum(per_class_recall) / num_classes
    macro_f1 = sum(per_class_f1) / num_classes

    return ClassificationMetrics(
        confusion_matrix=matrix.tolist(),
        per_class_recall=per_class_recall,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )
