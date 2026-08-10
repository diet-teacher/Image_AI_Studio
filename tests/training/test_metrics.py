"""metrics.py 순수 함수 테스트 (Phase 4O). 모델/DataLoader 없이 confusion
matrix tensor만으로 검증한다 -- forward pass를 도는 부분(evaluate_classification_metrics())의
테스트는 test_loop.py에 있다.
"""
from __future__ import annotations

import math

import pytest
import torch

from image_ai_studio.training.metrics import ClassificationMetrics, compute_classification_metrics


def _confusion_matrix(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def test_hand_computed_confusion_matrix_example() -> None:
    """true=[0,0,1,1,2,2], pred=[0,1,1,1,2,0] 조합 대신 사용자가 지정한
    true=[0,0,1,1,2,2], pred=[0,1,1,1,0,2]로 손계산 confusion matrix를
    고정한다 (row=true, column=predicted).

    true: 0 0 1 1 2 2
    pred: 0 1 1 1 0 2

    class0: true idx(0,1) -> pred(0,1)  => cm[0][0]+=1, cm[0][1]+=1
    class1: true idx(2,3) -> pred(1,1)  => cm[1][1]+=2
    class2: true idx(4,5) -> pred(0,2)  => cm[2][0]+=1, cm[2][2]+=1
    """
    cm = _confusion_matrix(
        [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert metrics.confusion_matrix == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # recall_i = TP_i / true_count_i
    assert metrics.per_class_recall == pytest.approx([1 / 2, 2 / 2, 1 / 2])

    # precision_i = TP_i / pred_count_i (pred_count: class0=2, class1=3, class2=1)
    expected_precision = [1 / 2, 2 / 3, 1 / 1]
    expected_recall = [1 / 2, 2 / 2, 1 / 2]
    expected_f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(expected_precision, expected_recall)
    ]

    assert metrics.macro_precision == pytest.approx(sum(expected_precision) / 3)
    assert metrics.macro_recall == pytest.approx(sum(expected_recall) / 3)
    assert metrics.macro_f1 == pytest.approx(sum(expected_f1) / 3)


def test_perfect_prediction_gives_all_metrics_one() -> None:
    cm = _confusion_matrix(
        [
            [5, 0, 0],
            [0, 3, 0],
            [0, 0, 2],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert metrics.per_class_recall == pytest.approx([1.0, 1.0, 1.0])
    assert metrics.macro_precision == pytest.approx(1.0)
    assert metrics.macro_recall == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)


def test_class_never_predicted_gets_zero_precision() -> None:
    """class 1이 true label로는 존재하지만 한 번도 predicted되지 않음
    (column 1이 전부 0) -> precision_1 분모=0 -> 0.0."""
    cm = _confusion_matrix(
        [
            [3, 0, 0],
            [2, 0, 0],
            [0, 0, 2],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert metrics.per_class_recall == pytest.approx([1.0, 0.0, 1.0])  # class1 recall: 0/2=0.0
    # precision: class0 = 3/5, class1 = 0/0(zero-division -> 0.0), class2 = 2/2
    assert metrics.macro_precision == pytest.approx((3 / 5 + 0.0 + 1.0) / 3)


def test_class_absent_from_targets_gets_zero_recall() -> None:
    """class 2가 target(row)에 전혀 없음(row 2가 전부 0) -> recall_2
    분모=0 -> 0.0. class 2가 predicted로는 등장할 수 있어 precision은
    0이 아닐 수 있다."""
    cm = _confusion_matrix(
        [
            [3, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert metrics.per_class_recall == pytest.approx([3 / 4, 1.0, 0.0])


def test_class_completely_absent_from_both_true_and_pred_gets_zero_everywhere() -> None:
    """class 2가 true/pred 양쪽 모두에 전혀 없음(row/column 둘 다 0) ->
    recall/precision/f1 전부 0.0, 그리고 에러 없이 계산된다(num_classes가
    실제 등장 class 수보다 큰 경우와 동일한 상황)."""
    cm = _confusion_matrix(
        [
            [4, 1, 0],
            [0, 3, 0],
            [0, 0, 0],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert metrics.per_class_recall[2] == pytest.approx(0.0)
    assert math.isfinite(metrics.macro_precision)
    assert math.isfinite(metrics.macro_recall)
    assert math.isfinite(metrics.macro_f1)


def test_num_classes_larger_than_observed_classes() -> None:
    """4x4 confusion matrix인데 class 3은 true/pred 어디에도 등장하지 않는
    경우 -- shape는 그대로 4x4이고 class 3의 지표는 전부 0.0."""
    cm = _confusion_matrix(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ]
    )

    metrics = compute_classification_metrics(cm)

    assert len(metrics.confusion_matrix) == 4
    assert len(metrics.per_class_recall) == 4
    assert metrics.per_class_recall[3] == pytest.approx(0.0)


def test_integer_confusion_matrix_values_preserved_as_int() -> None:
    """confusion_matrix가 float로 뭉개지지 않고 정수로 보존되는지 확인
    (JSON 직렬화 시 int로 남아야 함)."""
    cm = _confusion_matrix([[7, 2], [1, 9]])

    metrics = compute_classification_metrics(cm)

    for row in metrics.confusion_matrix:
        for value in row:
            assert isinstance(value, int)
    assert metrics.confusion_matrix == [[7, 2], [1, 9]]


def test_macro_f1_is_mean_of_per_class_f1_not_harmonic_mean_of_macros() -> None:
    """macro_f1 = mean(f1_i)이지, harmonic_mean(macro_precision, macro_recall)이
    아니다 -- 두 값이 실제로 다른 confusion matrix로 이 구분을 고정한다."""
    cm = _confusion_matrix(
        [
            [8, 2],
            [0, 5],
        ]
    )

    metrics = compute_classification_metrics(cm)

    precision = [8 / 8, 5 / 7]
    recall = [8 / 10, 5 / 5]
    per_class_f1 = [2 * p * r / (p + r) for p, r in zip(precision, recall)]
    mean_of_f1 = sum(per_class_f1) / 2

    macro_precision = sum(precision) / 2
    macro_recall = sum(recall) / 2
    harmonic_mean_of_macros = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    assert metrics.macro_f1 == pytest.approx(mean_of_f1)
    assert metrics.macro_f1 != pytest.approx(harmonic_mean_of_macros)


def test_classification_metrics_is_dataclass_with_expected_fields() -> None:
    metrics = ClassificationMetrics(
        confusion_matrix=[[1, 0], [0, 1]],
        per_class_recall=[1.0, 1.0],
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
    )
    assert metrics.confusion_matrix == [[1, 0], [0, 1]]


# -- 입력 shape 계약 (dtype/음수/개수 일관성 등은 검증하지 않는 최소 방어) --


def test_rejects_non_2d_confusion_matrix() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_classification_metrics(torch.zeros(3))


def test_rejects_non_square_confusion_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        compute_classification_metrics(torch.zeros(2, 3, dtype=torch.long))


def test_rejects_empty_confusion_matrix() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_classification_metrics(torch.zeros(0, 0, dtype=torch.long))
