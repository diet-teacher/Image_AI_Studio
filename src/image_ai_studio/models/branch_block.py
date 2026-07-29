import torch
from torch import Tensor, nn


class BranchBlock(nn.Module):
    """입력 하나를 N개 병렬 branch에 통과시킨 뒤 하나로 합치는 블록.

    branch 개수와 내부 구성은 고정되지 않고 생성 시점에 주어진다 (ResidualBlock의
    shortcut처럼 각 branch도 nn.Module 하나일 뿐). forward는 데이터값에 의존하는
    분기가 없어 torch.jit.trace와 호환된다. merge="concat"은 channel(dim=1)
    방향으로만 합친다 (Phase 3는 concat_dim을 노출하지 않음).
    """

    def __init__(self, branches: list[nn.Module], merge: str) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge

    def forward(self, x: Tensor) -> Tensor:
        outputs = [branch(x) for branch in self.branches]
        if self.merge == "add":
            result = outputs[0]
            for output in outputs[1:]:
                result = result + output
            return result
        if self.merge == "concat":
            return torch.cat(outputs, dim=1)
        raise ValueError(f"unsupported merge: {self.merge!r}")
