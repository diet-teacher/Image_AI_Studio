"""shape inference 테스트: 레이어별 shape 계산 + 연결 검증."""
from __future__ import annotations

import pytest

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import format_shape_trace, infer_model_shapes
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    BranchSpec,
    Conv2dSpec,
    DropoutSpec,
    FlattenSpec,
    IdentitySpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
)


def test_conv2d_same_padding_preserves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 224, 224),
        layers=[Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].input_shape == (3, 224, 224)
    assert trace[0].output_shape == (32, 224, 224)
    assert trace[0].inferred == {"in_channels": 3}


def test_conv2d_stride_2_halves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 32, 32),
        layers=[Conv2dSpec(out_channels=8, kernel_size=3, stride=2, padding=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (8, 16, 16)


def test_max_pool2d_default_stride_equals_kernel_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 224, 224),
        layers=[MaxPool2dSpec(kernel_size=2)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 112, 112)


def test_adaptive_avg_pool2d_shape() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 55, 55),
        layers=[AdaptiveAvgPool2dSpec(output_size=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 1, 1)


def test_flatten_collapses_to_1d() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 112, 112),
        layers=[FlattenSpec()],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32 * 112 * 112,)


def test_linear_infers_in_features_from_previous_layer() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 112, 112),
        layers=[FlattenSpec(), LinearSpec(out_features=10)],
    )
    trace = infer_model_shapes(spec)
    flatten_info, linear_info = trace
    assert flatten_info.output_shape == (401408,)
    assert linear_info.input_shape == (401408,)
    assert linear_info.inferred == {"in_features": 401408}
    assert linear_info.output_shape == (10,)


@pytest.mark.parametrize("layer", [ReLUSpec(), DropoutSpec(p=0.3)])
def test_activation_and_dropout_preserve_shape(layer) -> None:
    spec = ModelSpec(name="m", input_shape=(3, 8, 8), layers=[layer])
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (3, 8, 8)


def test_batch_norm2d_preserves_shape_and_infers_num_features() -> None:
    spec = ModelSpec(name="m", input_shape=(16, 8, 8), layers=[BatchNorm2dSpec()])
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (16, 8, 8)
    assert trace[0].inferred == {"num_features": 16}


def test_full_example_pipeline_matches_expected_shapes() -> None:
    """Phase 1 스펙 예시 그대로: Input -> Conv2d -> MaxPool2d -> Flatten -> Linear."""
    spec = ModelSpec(
        name="example_model",
        input_shape=(3, 224, 224),
        layers=[
            Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1),
            MaxPool2dSpec(kernel_size=2, stride=2),
            FlattenSpec(),
            LinearSpec(out_features=10),
        ],
    )
    trace = infer_model_shapes(spec)
    shapes = [(info.input_shape, info.output_shape) for info in trace]
    assert shapes == [
        ((3, 224, 224), (32, 224, 224)),
        ((32, 224, 224), (32, 112, 112)),
        ((32, 112, 112), (401408,)),
        ((401408,), (10,)),
    ]


def test_conv2d_after_flatten_raises_clear_validation_error() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[FlattenSpec(), Conv2dSpec(out_channels=4, kernel_size=3)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 1 \(Conv2d\).*3D"):
        infer_model_shapes(spec)


def test_linear_before_flatten_raises_clear_validation_error() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[LinearSpec(out_features=10)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 0 \(Linear\)"):
        infer_model_shapes(spec)


def test_conv2d_directly_into_linear_without_flatten_raises() -> None:
    """Conv2d -> Linear 직결 금지 (자동 flatten 없음, Flatten 명시 필요)."""
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[Conv2dSpec(out_channels=4, kernel_size=3, padding=1), LinearSpec(out_features=10)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 1 \(Linear\).*1D"):
        infer_model_shapes(spec)


def test_pooling_that_collapses_spatial_size_to_zero_raises() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 4, 4),
        layers=[MaxPool2dSpec(kernel_size=5, stride=1, padding=0)],
    )
    with pytest.raises(ModelValidationError, match="MaxPool2d"):
        infer_model_shapes(spec)


def test_conv2d_that_collapses_spatial_size_to_zero_raises() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 2, 2),
        layers=[Conv2dSpec(out_channels=4, kernel_size=5, stride=1, padding=0)],
    )
    with pytest.raises(ModelValidationError, match="Conv2d"):
        infer_model_shapes(spec)


def test_format_shape_trace_renders_layer_name_and_shapes() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 224, 224),
        layers=[Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1)],
    )
    trace = infer_model_shapes(spec)
    text = format_shape_trace(trace)
    assert text == "Conv2d\n[3, 224, 224] -> [32, 224, 224]"


# -- ResidualBlockSpec --------------------------------------------------------


def test_residual_block_stride_1_preserves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(16, 32, 32),
        layers=[ResidualBlockSpec(out_channels=32)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 32, 32)
    assert trace[0].inferred == {"in_channels": 16}


def test_residual_block_stride_2_halves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(16, 32, 32),
        layers=[ResidualBlockSpec(out_channels=32, stride=2)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 16, 16)


def test_residual_block_after_flatten_raises_clear_validation_error() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[FlattenSpec(), ResidualBlockSpec(out_channels=8)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 1 \(ResidualBlock\).*3D"):
        infer_model_shapes(spec)


def test_residual_block_matches_actual_pytorch_module_on_odd_spatial_size() -> None:
    """홀수 spatial size(floor 반올림이 실제로 영향을 주는 경우)에서도
    shape_inference 예측과 실제 ResidualBlock forward 결과가 일치하는지 확인."""
    import torch

    from image_ai_studio.models.residual_block import ResidualBlock

    in_channels, out_channels, stride, h, w = 4, 6, 2, 7, 7
    spec = ModelSpec(
        name="m",
        input_shape=(in_channels, h, w),
        layers=[ResidualBlockSpec(out_channels=out_channels, stride=stride)],
    )
    predicted_shape = infer_model_shapes(spec)[0].output_shape

    block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride).eval()
    with torch.inference_mode():
        actual_output = block(torch.randn(1, in_channels, h, w))

    assert predicted_shape == tuple(actual_output.shape[1:])


# -- IdentitySpec --------------------------------------------------------------


def test_identity_preserves_shape() -> None:
    spec = ModelSpec(name="m", input_shape=(3, 8, 8), layers=[IdentitySpec()])
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (3, 8, 8)


# -- BranchSpec: merge="add" ---------------------------------------------------


def test_branch_add_requires_matching_shapes_and_preserves_them() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(8, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1)],
                    [IdentitySpec()],
                ],
                merge="add",
            )
        ],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (8, 8, 8)
    assert trace[0].inferred == {}


def test_branch_add_rejects_mismatched_branch_shapes() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(8, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1)],
                    [Conv2dSpec(out_channels=8, kernel_size=3, stride=2, padding=1)],
                ],
                merge="add",
            )
        ],
    )
    with pytest.raises(ModelValidationError, match=r'merge="add"'):
        infer_model_shapes(spec)


def test_branch_after_flatten_reuses_add_on_1d_shapes() -> None:
    """BranchSpec은 3D로 제한하지 않는다 -- 각 branch가 자기 sub-layer의 rank
    요구사항을 그대로 강제하므로(_require_rank), 1D 입력에서도 add가 성립하면
    허용된다."""
    spec = ModelSpec(
        name="m",
        input_shape=(3, 4, 4),
        layers=[
            FlattenSpec(),
            BranchSpec(branches=[[IdentitySpec()], [IdentitySpec()]], merge="add"),
        ],
    )
    trace = infer_model_shapes(spec)
    assert trace[-1].output_shape == (48,)


# -- BranchSpec: merge="concat" (channel-only) ---------------------------------


def test_branch_concat_sums_channels_and_keeps_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(4, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=4, kernel_size=3, stride=1, padding=1)],
                    [MaxPool2dSpec(kernel_size=1, stride=1, padding=0)],
                ],
                merge="concat",
            )
        ],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (8, 8, 8)


def test_branch_concat_rejects_mismatched_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=4, kernel_size=3, stride=1, padding=1)],
                    [Conv2dSpec(out_channels=4, kernel_size=3, stride=2, padding=1)],
                ],
                merge="concat",
            )
        ],
    )
    with pytest.raises(ModelValidationError, match=r'merge="concat".*\(height, width\)'):
        infer_model_shapes(spec)


def test_branch_concat_rejects_non_3d_branch_output() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [FlattenSpec()],
                    [Conv2dSpec(out_channels=4, kernel_size=3, stride=1, padding=1)],
                ],
                merge="concat",
            )
        ],
    )
    with pytest.raises(ModelValidationError, match=r'merge="concat".*3D'):
        infer_model_shapes(spec)


def test_branch_concat_matches_actual_branch_block_on_odd_spatial_size() -> None:
    """shape_inference의 concat 예측이 실제 BranchBlock(torch.cat(dim=1))
    forward 결과와 일치하는지 홀수 spatial size로 교차 검증."""
    import torch

    from image_ai_studio.models.branch_block import BranchBlock

    in_channels, h, w = 4, 7, 7
    spec = ModelSpec(
        name="m",
        input_shape=(in_channels, h, w),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=6, kernel_size=3, stride=2, padding=1)],
                    [Conv2dSpec(out_channels=3, kernel_size=3, stride=2, padding=1)],
                ],
                merge="concat",
            )
        ],
    )
    predicted_shape = infer_model_shapes(spec)[0].output_shape

    block = BranchBlock(
        branches=[
            torch.nn.Conv2d(in_channels, 6, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=2, padding=1),
        ],
        merge="concat",
    ).eval()
    with torch.inference_mode():
        actual_output = block(torch.randn(1, in_channels, h, w))

    assert predicted_shape == tuple(actual_output.shape[1:])
