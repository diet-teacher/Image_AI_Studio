"""Phase 1~3 모델(BatchNorm/Dropout/ResidualBlockSpec/BranchSpec)의 실제 학습 동작 검증.

Phase 0~3의 unit test는 전부 model.eval()로만 forward를 검증했다 -- train()
모드에서 BatchNorm running stats가 갱신되는지, Dropout이 train/eval에 따라
달라지는지, ResidualBlockSpec/BranchSpec 조합에서 backward가 실제로 흐르는지는
여기서 처음 검증한다.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    BranchSpec,
    Conv2dSpec,
    DropoutSpec,
    FlattenSpec,
    IdentitySpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
)
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import train_one_epoch


def _train_loader(spec: ModelSpec, num_classes: int, seed: int, batch_size: int = 8) -> DataLoader:
    train_dataset, _ = make_train_val_datasets(
        spec.input_shape, num_classes, seed=seed, train_size=16, val_size=4
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)


def _assert_all_params_have_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient after backward"


def test_batch_norm_running_stats_update_after_training() -> None:
    torch.manual_seed(0)
    spec = ModelSpec(
        name="bn_model",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            BatchNorm2dSpec(),
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )
    model = build_model(spec)
    bn = model[1]
    assert isinstance(bn, torch.nn.BatchNorm2d)

    running_mean_before = bn.running_mean.clone()
    running_var_before = bn.running_var.clone()

    loader = _train_loader(spec, num_classes=4, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_one_epoch(model, loader, optimizer)

    assert not torch.equal(running_mean_before, bn.running_mean)
    assert not torch.equal(running_var_before, bn.running_var)


def test_dropout_switches_between_train_and_eval() -> None:
    spec = ModelSpec(
        name="dropout_model",
        input_shape=(3, 8, 8),
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            DropoutSpec(p=0.5),
            LinearSpec(out_features=4),
        ],
    )
    model = build_model(spec)
    dropout = model[3]
    assert isinstance(dropout, torch.nn.Dropout)

    example = torch.randn(1, 3, 8, 8)

    model.train()
    assert dropout.training is True
    torch.manual_seed(1)
    out_a = model(example)
    torch.manual_seed(1)
    out_b = model(example)
    assert torch.allclose(out_a, out_b)  # train 모드도 같은 seed면 재현 가능해야 함

    model.eval()
    assert dropout.training is False
    with torch.inference_mode():
        out_c = model(example)
        out_d = model(example)
    assert torch.allclose(out_c, out_d)  # eval 모드는 seed와 무관하게 항상 결정적(드롭 없음)


def test_residual_block_backward_produces_gradients_and_updates_parameters() -> None:
    torch.manual_seed(0)
    spec = ModelSpec(
        name="residual_model",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            ResidualBlockSpec(out_channels=8, stride=1),  # in=4 != out=8 -> projection shortcut 경로
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )
    model = build_model(spec)
    loader = _train_loader(spec, num_classes=4, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = copy.deepcopy(model.state_dict())
    train_one_epoch(model, loader, optimizer)

    _assert_all_params_have_gradients(model)
    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_branch_with_identity_backward_produces_gradients_and_updates_parameters() -> None:
    """merge='add'의 한쪽 branch가 파라미터 없는 IdentitySpec이어도 다른 branch로
    gradient가 정상적으로 흐르는지 확인 (가장 까다로운 경계 케이스)."""
    torch.manual_seed(0)
    spec = ModelSpec(
        name="branch_model",
        input_shape=(4, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=4, kernel_size=3, padding=1), BatchNorm2dSpec()],
                    [IdentitySpec()],
                ],
                merge="add",
            ),
            ReLUSpec(),
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )
    model = build_model(spec)
    loader = _train_loader(spec, num_classes=4, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = copy.deepcopy(model.state_dict())
    train_one_epoch(model, loader, optimizer)

    _assert_all_params_have_gradients(model)
    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)
