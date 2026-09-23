"""Phase 14 CP1: standalone ordered/form-based Qt ``ModelDesigner``.

Creates, loads, edits, validates, and atomically saves the existing
canonical ``ModelSpec`` JSON format (``model_definition/serialization.py``)
without requiring the user to hand-edit JSON. This widget never
re-implements parameter validation or shape-inference rules -- it only maps
typed widgets onto the exact JSON layer-dict shape produced by
``model_spec_to_dict`` / consumed by ``model_spec_from_dict``, and always
routes structural/parameter validation through ``model_spec_from_dict`` and
shape validation through ``validate_model_spec``.

Layer editing is ordered/form-based (add, remove, move up, move down, and a
per-layer-type parameter form) -- no drag-and-drop graph editing.
``BranchSpec`` branches are edited the same way, one non-nested
``LayerListPanel`` per branch; the "branch" layer type is simply not offered
inside a branch's own Add-layer combo, and ``BranchSpec.__post_init__``'s
existing nested-branch rejection remains the backstop.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import (
    load_model_spec,
    model_spec_from_dict,
    model_spec_to_dict,
    save_model_spec,
)
from image_ai_studio.model_definition.shape_inference import format_shape_trace
from image_ai_studio.model_definition.specs import ModelSpec
from image_ai_studio.model_definition.validation import validate_model_spec

_ERROR_MAX_CHARS = 200


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    return text.splitlines()[0][:_ERROR_MAX_CHARS]


def _format_int_text(value: int) -> str:
    return str(int(value))


def _parse_int_text(text: str, *, minimum: int) -> int:
    """Text -> exact Python ``int`` with no upper bound (unlike QSpinBox,
    which is backed by a 32-bit signed C++ int and cannot hold a canonical
    value like ``2147483648`` at all -- not even transiently)."""
    stripped = text.strip()
    if not stripped:
        raise ModelValidationError("value must not be empty")
    try:
        value = int(stripped)
    except ValueError:
        raise ModelValidationError(f"'{stripped}' is not a valid integer") from None
    if value < minimum:
        raise ModelValidationError(f"value must be >= {minimum}, got {value}")
    return value


def _format_float_text(value: float) -> str:
    # repr() is the shortest decimal string that round-trips back to the
    # exact same float -- unlike a fixed-decimals QDoubleSpinBox, it never
    # rounds a small canonical value (e.g. eps=1e-12) down to 0.0.
    return repr(float(value))


def _raw_or_formatted_int(value: object) -> str:
    """Render canonical ints exactly and preserve invalid user text verbatim."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _format_int_text(value)
    return value if isinstance(value, str) else str(value)


def _raw_or_formatted_float(value: object) -> str:
    """Render canonical floats exactly and preserve invalid user text verbatim."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_float_text(float(value))
    return value if isinstance(value, str) else str(value)


def _parse_float_text(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        raise ModelValidationError("value must not be empty")
    try:
        value = float(stripped)
    except ValueError:
        raise ModelValidationError(f"'{stripped}' is not a valid number") from None
    if not math.isfinite(value):
        raise ModelValidationError(f"'{stripped}' must be a finite number (no NaN/Infinity)")
    return value


@dataclass(frozen=True)
class _Field:
    """레이어 파라미터 하나에 대응하는 위젯 설명. ``kind``는 위젯 종류만
    결정한다 -- 값 자체의 정당성 검증은 항상 specs.py의 dataclass
    ``__post_init__``이 담당한다 (여기서는 재구현하지 않음)."""

    name: str
    kind: str  # "int" | "int_or_none" | "float" | "bool"
    default: object
    minimum: float = 0


_BRANCH_TYPE = "branch"

# All canonical integer and float parameters use text controls: Qt spin boxes
# impose representation limits that are narrower than the existing specs.
# The field metadata carries only an integer lower bound needed while typing;
# complete domain validation remains in the canonical spec dataclasses.

# serialization.py의 _LAYER_REGISTRY와 1:1 대응하는 JSON type 이름 목록
# (branch 제외 -- branch는 구조가 달라 별도 처리).
_LAYER_FIELDS: dict[str, tuple[_Field, ...]] = {
    "conv2d": (
        _Field("out_channels", "int", 32, 1),
        _Field("kernel_size", "int", 3, 1),
        _Field("stride", "int", 1, 1),
        _Field("padding", "int", 0, 0),
    ),
    "batch_norm2d": (
        _Field("eps", "float", 1e-5),
        _Field("momentum", "float", 0.1),
    ),
    "relu": (
        _Field("inplace", "bool", False),
    ),
    "max_pool2d": (
        _Field("kernel_size", "int", 2, 1),
        _Field("stride", "int_or_none", None, 1),
        _Field("padding", "int", 0, 0),
    ),
    "adaptive_avg_pool2d": (
        _Field("output_size", "int", 1, 1),
    ),
    "flatten": (),
    "linear": (
        _Field("out_features", "int", 128, 1),
        _Field("bias", "bool", True),
    ),
    "dropout": (
        _Field("p", "float", 0.5),
    ),
    "identity": (),
    "residual_block": (
        _Field("out_channels", "int", 64, 1),
        _Field("stride", "int", 1, 1),
    ),
}

_LAYER_TYPE_NAMES: tuple[str, ...] = tuple(sorted((*_LAYER_FIELDS.keys(), _BRANCH_TYPE)))


def _addable_type_names(allow_branch: bool) -> tuple[str, ...]:
    if allow_branch:
        return _LAYER_TYPE_NAMES
    return tuple(name for name in _LAYER_TYPE_NAMES if name != _BRANCH_TYPE)


def _default_layer_dict(type_name: str) -> dict:
    """새 레이어의 결정론적 기본값 -- 항상 그 자체로 유효한 파라미터."""
    if type_name == _BRANCH_TYPE:
        return {
            "type": _BRANCH_TYPE,
            "merge": "add",
            "branches": [
                [_default_layer_dict("identity")],
                [_default_layer_dict("identity")],
            ],
        }
    fields = _LAYER_FIELDS[type_name]
    return {"type": type_name, **{field.name: field.default for field in fields}}


def _starter_model_dict() -> dict:
    """New 클릭 시 시작하는 결정론적으로 유효한 모델 정의 (사용자가 그대로
    검사/수정 가능). 3x32x32 입력 -> Conv2d -> ReLU -> Flatten -> Linear(10)."""
    return {
        "name": "new_model",
        "input_shape": [3, 32, 32],
        "layers": [
            {"type": "conv2d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "relu", "inplace": False},
            {"type": "flatten"},
            {"type": "linear", "out_features": 10, "bias": True},
        ],
    }


def _layer_summary(layer: dict) -> str:
    type_name = layer.get("type", "?")
    if type_name == _BRANCH_TYPE:
        branch_count = len(layer.get("branches", []))
        return f"{type_name} (merge={layer.get('merge', 'add')}, branches={branch_count})"
    fields = _LAYER_FIELDS.get(type_name, ())
    parts = [f"{field.name}={layer.get(field.name, field.default)}" for field in fields]
    return f"{type_name} ({', '.join(parts)})" if parts else type_name


def _verify_layer_stageable(layer: dict) -> None:
    """Prove every canonical numeric value can round-trip through its text
    control before any live widget is touched."""
    type_name = layer.get("type")
    if type_name == _BRANCH_TYPE:
        for branch in layer.get("branches", []):
            for nested in branch:
                _verify_layer_stageable(nested)
        return
    for field in _LAYER_FIELDS.get(type_name, ()):
        value = layer.get(field.name, field.default)
        if field.kind == "float":
            _parse_float_text(_format_float_text(value))
        elif field.kind == "int":
            _parse_int_text(_format_int_text(value), minimum=int(field.minimum))
        elif field.kind == "int_or_none" and value is not None:
            _parse_int_text(_format_int_text(value), minimum=int(field.minimum))


def _verify_model_stageable(data: dict) -> None:
    """Widget-independent proof that ``data`` (``model_spec_to_dict`` shape)
    can be fully represented by the editor's text-based controls -- input
    shape ints and every layer's numeric fields -- before any live widget is
    mutated."""
    for value in data["input_shape"]:
        _parse_int_text(_format_int_text(value), minimum=1)
    for layer in data["layers"]:
        _verify_layer_stageable(layer)


class _OptionalIntWidget(QWidget):
    """``int | None`` 필드(예: MaxPool2d.stride) 전용 위젯. "Auto" 체크박스가
    켜지면 값은 None(=torch 기본 동작 위임), 꺼지면 스핀박스 값을 쓴다."""

    def __init__(self, field: _Field, layer: dict, on_changed: Callable[[], None]) -> None:
        super().__init__()
        self._field = field
        self._layer = layer
        self._on_changed = on_changed

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox("Auto")
        self._spin = _BigIntField(minimum=int(field.minimum))

        current = layer.get(field.name, field.default)
        is_auto = current is None
        self._checkbox.setChecked(is_auto)
        if current is None:
            self._spin.setValue(int(field.minimum))
        else:
            self._spin.setText(_raw_or_formatted_int(current))
        self._spin.setEnabled(not is_auto)

        self._checkbox.toggled.connect(self._sync)
        self._spin.textChanged.connect(self._sync)
        row.addWidget(self._checkbox)
        row.addWidget(self._spin)

    def _sync(self, *_args: object) -> None:
        is_auto = self._checkbox.isChecked()
        self._spin.setEnabled(not is_auto)
        if is_auto:
            self._layer[self._field.name] = None
        else:
            try:
                self._layer[self._field.name] = self._spin.value()
            except ModelValidationError:
                self._layer[self._field.name] = self._spin.text()
        self._on_changed()


def _build_field_widget(field: _Field, layer: dict, on_changed: Callable[[], None]) -> QWidget:
    if field.kind == "bool":
        checkbox = QCheckBox()
        checkbox.setChecked(bool(layer.get(field.name, field.default)))

        def _on_toggled(checked: bool, name: str = field.name) -> None:
            layer[name] = checked
            on_changed()

        checkbox.toggled.connect(_on_toggled)
        return checkbox

    if field.kind == "int_or_none":
        return _OptionalIntWidget(field, layer, on_changed)

    if field.kind == "int":
        spin = _BigIntField(minimum=int(field.minimum))
        spin.setText(_raw_or_formatted_int(layer.get(field.name, field.default)))

        def _on_value_changed(text: str, name: str = field.name) -> None:
            try:
                layer[name] = _parse_int_text(text, minimum=int(field.minimum))
            except ModelValidationError:
                layer[name] = text
            on_changed()

        spin.textChanged.connect(_on_value_changed)
        return spin

    # "float": a fixed-decimals QDoubleSpinBox cannot represent a very small
    # canonical value (e.g. BatchNorm2d eps=1e-12) without rounding it down to
    # 0.0, so this is a plain text field with explicit Python float parsing
    # instead. Domain bounds (eps > 0, momentum in (0.0, 1.0], ...) are left
    # entirely to specs.py's own dataclass validation -- this widget only
    # rejects empty/malformed/non-finite text.
    edit = QLineEdit()
    edit.setText(_raw_or_formatted_float(layer.get(field.name, field.default)))

    def _on_text_changed(text: str, name: str = field.name) -> None:
        try:
            value = _parse_float_text(text)
        except ModelValidationError:
            # Visible text is authoritative: keep invalid input in the layer
            # mapping so build/save cannot silently reuse a stale valid value.
            layer[name] = text
        else:
            layer[name] = value
        on_changed()

    edit.textChanged.connect(_on_text_changed)
    return edit


def _build_simple_param_form(layer: dict, on_changed: Callable[[], None]) -> QWidget:
    type_name = layer.get("type", "")
    fields = _LAYER_FIELDS.get(type_name, ())
    group = QGroupBox(f"{type_name} parameters")
    form = QFormLayout(group)
    if not fields:
        form.addRow(QLabel("(no parameters)"))
    for field in fields:
        form.addRow(f"{field.name}:", _build_field_widget(field, layer, on_changed))
    return group


class LayerListPanel(QWidget):
    """레이어(또는 branch 하나) 순서 리스트: add/remove/move-up/move-down +
    선택된 레이어의 타입별 파라미터 폼. ``self.layers``는 호출자가 넘긴
    리스트 객체 그대로를 in-place로 조작한다 (재할당하지 않음) -- 그래야
    ``BranchSpec.branches[i]``처럼 부모가 들고 있는 리스트 참조가 그대로
    유지된다. ``allow_branch=False``이면 Add 콤보에 "branch"가 나타나지
    않는다 (중첩 BranchSpec을 UI 레벨에서부터 막음; dataclass의 기존 금지는
    그대로 최종 방어선으로 남아 있다)."""

    changed = Signal()

    def __init__(self, layers: list[dict], *, allow_branch: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layers = layers
        self._allow_branch = allow_branch
        self._param_widget: QWidget | None = None
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_selection_changed)
        outer.addWidget(self._list_widget)

        controls_row = QHBoxLayout()
        self._type_combo = QComboBox()
        for type_name in _addable_type_names(self._allow_branch):
            self._type_combo.addItem(type_name)
        controls_row.addWidget(self._type_combo)

        self._add_button = QPushButton("Add")
        self._add_button.clicked.connect(self._on_add_clicked)
        controls_row.addWidget(self._add_button)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self._on_remove_clicked)
        controls_row.addWidget(self._remove_button)

        self._up_button = QPushButton("Move Up")
        self._up_button.clicked.connect(self._on_move_up_clicked)
        controls_row.addWidget(self._up_button)

        self._down_button = QPushButton("Move Down")
        self._down_button.clicked.connect(self._on_move_down_clicked)
        controls_row.addWidget(self._down_button)
        controls_row.addStretch(1)
        outer.addLayout(controls_row)

        self._param_layout = QVBoxLayout()
        outer.addLayout(self._param_layout)

    # -- state -------------------------------------------------------------

    def set_layers(self, layers: list[dict]) -> None:
        """리스트 객체 자체를 교체한다 (load/new에서 사용). 선택은
        초기화되고, 새 리스트 내용에 맞춰 파라미터 패널도 다시 그려진다."""
        self.layers = layers
        self._refresh_list()

    def current_index(self) -> int:
        return self._list_widget.currentRow()

    def select_row(self, row: int) -> None:
        """``self.layers``의 내용은 이미 확정된 상태에서 선택 행만 맞춘다
        (예: 실패한 population 이후 이전 상태 복원 시 선택 행까지 그대로
        되돌리기 위함)."""
        if 0 <= row < self._list_widget.count():
            self._list_widget.setCurrentRow(row)
        else:
            self._list_widget.setCurrentRow(-1)
            self._rebuild_param_form()

    def capture_selection_state(self) -> dict:
        """Capture selections by stable editor-tree position, not widget id."""
        state: dict = {"selected_row": self.current_index()}
        if isinstance(self._param_widget, _BranchEditor):
            state["branches"] = self._param_widget.capture_selection_state()
        return state

    def restore_selection_state(self, state: dict) -> None:
        """Restore this panel and any currently supported branch children."""
        self.select_row(int(state.get("selected_row", -1)))
        branches = state.get("branches")
        if isinstance(self._param_widget, _BranchEditor) and isinstance(branches, list):
            self._param_widget.restore_selection_state(branches)

    def _refresh_list(self, *, select_row: int | None = None) -> None:
        previous_blocked = self._list_widget.blockSignals(True)
        try:
            self._list_widget.clear()
            for layer in self.layers:
                self._list_widget.addItem(_layer_summary(layer))
        finally:
            self._list_widget.blockSignals(previous_blocked)
        if select_row is not None and 0 <= select_row < self._list_widget.count():
            self._list_widget.setCurrentRow(select_row)
        else:
            self._rebuild_param_form()

    def _rebuild_param_form(self) -> None:
        if self._param_widget is not None:
            self._param_layout.removeWidget(self._param_widget)
            self._param_widget.deleteLater()
            self._param_widget = None
        row = self.current_index()
        if row < 0 or row >= len(self.layers):
            return
        layer = self.layers[row]
        widget = _build_layer_param_widget(layer, self._on_param_edited)
        self._param_widget = widget
        self._param_layout.addWidget(widget)

    # -- slots ---------------------------------------------------------------

    def _on_selection_changed(self, _row: int) -> None:
        self._rebuild_param_form()

    def _on_add_clicked(self) -> None:
        type_name = self._type_combo.currentText()
        if not type_name:
            return
        self.layers.append(_default_layer_dict(type_name))
        self._refresh_list(select_row=len(self.layers) - 1)
        self.changed.emit()

    def _on_remove_clicked(self) -> None:
        row = self.current_index()
        if row < 0:
            return
        del self.layers[row]
        new_row = min(row, len(self.layers) - 1)
        self._refresh_list(select_row=new_row if new_row >= 0 else None)
        self.changed.emit()

    def _on_move_up_clicked(self) -> None:
        row = self.current_index()
        if row <= 0:
            return
        self.layers[row - 1], self.layers[row] = self.layers[row], self.layers[row - 1]
        self._refresh_list(select_row=row - 1)
        self.changed.emit()

    def _on_move_down_clicked(self) -> None:
        row = self.current_index()
        if row < 0 or row >= len(self.layers) - 1:
            return
        self.layers[row + 1], self.layers[row] = self.layers[row], self.layers[row + 1]
        self._refresh_list(select_row=row + 1)
        self.changed.emit()

    def _on_param_edited(self) -> None:
        row = self.current_index()
        if 0 <= row < len(self.layers):
            item = self._list_widget.item(row)
            if item is not None:
                item.setText(_layer_summary(self.layers[row]))
        self.changed.emit()


class _BranchEditor(QWidget):
    """BranchSpec 전용 파라미터 폼: merge 선택 + branch별 non-nested
    ``LayerListPanel`` + Add/Remove Branch. BranchSpec은 최소 2개 branch가
    필요하므로(specs.py) Remove Branch는 2개 이하로는 내려가지 않게 막는다
    (그 이상 세밀한 검증은 여기서 하지 않고 Validate/Save가 기존
    ``validate_model_spec``/dataclass 검증에 위임한다)."""

    def __init__(self, layer: dict, on_changed: Callable[[], None]) -> None:
        super().__init__()
        self._layer = layer
        self._on_changed = on_changed
        self._branch_panels: list[LayerListPanel] = []
        layer.setdefault("merge", "add")
        layer.setdefault("branches", [])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        merge_row = QHBoxLayout()
        merge_row.addWidget(QLabel("merge:"))
        self._merge_combo = QComboBox()
        self._merge_combo.addItems(["add", "concat"])
        self._merge_combo.setCurrentText(layer.get("merge", "add"))
        self._merge_combo.currentTextChanged.connect(self._on_merge_changed)
        merge_row.addWidget(self._merge_combo)
        merge_row.addStretch(1)
        outer.addLayout(merge_row)

        self._branches_layout = QVBoxLayout()
        outer.addLayout(self._branches_layout)

        add_branch_row = QHBoxLayout()
        self._add_branch_button = QPushButton("Add Branch")
        self._add_branch_button.clicked.connect(self._on_add_branch)
        add_branch_row.addWidget(self._add_branch_button)
        add_branch_row.addStretch(1)
        outer.addLayout(add_branch_row)

        self._rebuild_branches()

    def _on_merge_changed(self, text: str) -> None:
        self._layer["merge"] = text
        self._on_changed()

    def _on_add_branch(self) -> None:
        self._layer["branches"].append([_default_layer_dict("identity")])
        self._rebuild_branches()
        self._on_changed()

    def _on_remove_branch(self, branch_index: int) -> None:
        branches = self._layer["branches"]
        if len(branches) <= 2:
            return
        del branches[branch_index]
        self._rebuild_branches()
        self._on_changed()

    def _rebuild_branches(self) -> None:
        while self._branches_layout.count():
            item = self._branches_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._branch_panels = []
        branches = self._layer["branches"]
        for index, branch in enumerate(branches):
            group = QGroupBox(f"Branch {index}")
            box = QVBoxLayout(group)
            panel = LayerListPanel(branch, allow_branch=False)
            panel.changed.connect(self._on_changed)
            self._branch_panels.append(panel)
            box.addWidget(panel)

            remove_button = QPushButton("Remove Branch")
            remove_button.setEnabled(len(branches) > 2)
            remove_button.clicked.connect(lambda _checked=False, i=index: self._on_remove_branch(i))
            box.addWidget(remove_button)

            self._branches_layout.addWidget(group)

    def capture_selection_state(self) -> list[dict]:
        return [panel.capture_selection_state() for panel in self._branch_panels]

    def restore_selection_state(self, states: list[dict]) -> None:
        for panel, state in zip(self._branch_panels, states):
            panel.restore_selection_state(state)


def _build_layer_param_widget(layer: dict, on_changed: Callable[[], None]) -> QWidget:
    if layer.get("type") == _BRANCH_TYPE:
        return _BranchEditor(layer, on_changed)
    return _build_simple_param_form(layer, on_changed)


class _BigIntField(QWidget):
    """Text-based positive-integer field for model-level ``input_shape``
    values. Exposes the same ``value()``/``setValue()`` contract a QSpinBox
    would, but -- unlike QSpinBox, which is backed by a 32-bit signed C++
    int and cannot even transiently hold a canonical value like
    2147483648 -- this widget represents the value as plain text and parses
    it with Python's arbitrary-precision ``int()``, so it never clamps,
    overflows, or silently corrupts a value above Qt's 32-bit ceiling."""

    def __init__(self, *, minimum: int = 1) -> None:
        super().__init__()
        self._minimum = minimum
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.setText(_format_int_text(minimum))
        layout.addWidget(self._edit)

    def value(self) -> int:
        """Parse the current text into an exact Python int. Raises
        ``ModelValidationError`` (bounded, actionable) for empty, malformed,
        non-integer, or non-positive text -- never clamps or rounds."""
        return _parse_int_text(self._edit.text(), minimum=self._minimum)

    def setValue(self, value: int) -> None:
        self._edit.setText(_format_int_text(value))

    def text(self) -> str:
        return self._edit.text()

    def setText(self, text: str) -> None:
        self._edit.setText(text)

    @property
    def textChanged(self):
        """Expose the embedded QLineEdit signal for field wiring."""
        return self._edit.textChanged


class ModelDesigner(QWidget):
    """standalone ordered/form-based ModelSpec 에디터 (Phase 14 CP1).

    Model Name/Input Shape 입력 + ``LayerListPanel``로 구성되며, New/Load/
    Save/Validate와 Save for Training 명시적 액션을 제공한다. 모델 구조를 재구성하는
    유일한 진입점은 ``_build_model_spec()``(현재 위젯 상태 -> JSON dict ->
    ``model_spec_from_dict``)이며, shape 연결 검증은 항상
    ``validate_model_spec``을 그대로 호출한다 -- 이 위젯은 두 검증 중
    어느 것도 재구현하지 않는다."""

    # Emitted only after an explicit "Save for Training" action passes both
    # canonical validation and the canonical atomic save. Carries the
    # normalized absolute path of the just-saved file. A cancelled dialog or
    # any validation/save failure never emits -- the designer itself never
    # touches TrainingController/QtTrainingWorker; wiring this to the real
    # training input is entirely MainWindow's responsibility.
    model_saved_for_training = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._load_from_dict(_starter_model_dict())

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        form.addRow("Model Name:", self._name_edit)

        shape_row = QHBoxLayout()
        # ModelSpec.input_shape only requires positive ints (specs.py) with
        # no upper bound at all -- a QSpinBox (32-bit signed C++ int) cannot
        # represent a canonical value like 2147483648 even transiently, so
        # each dimension is a text-based _BigIntField instead.
        self._channels_spin = _BigIntField(minimum=1)
        self._height_spin = _BigIntField(minimum=1)
        self._width_spin = _BigIntField(minimum=1)
        for field_widget in (self._channels_spin, self._height_spin, self._width_spin):
            shape_row.addWidget(field_widget)
        form.addRow("Input Shape (C, H, W):", shape_row)
        layout.addLayout(form)

        actions_row = QHBoxLayout()
        self._new_button = QPushButton("New")
        self._new_button.clicked.connect(self._on_new_clicked)
        actions_row.addWidget(self._new_button)

        self._load_button = QPushButton("Load...")
        self._load_button.clicked.connect(self._on_load_clicked)
        actions_row.addWidget(self._load_button)

        self._save_button = QPushButton("Save...")
        self._save_button.clicked.connect(self._on_save_clicked)
        actions_row.addWidget(self._save_button)

        self._validate_button = QPushButton("Validate")
        self._validate_button.clicked.connect(self._on_validate_clicked)
        actions_row.addWidget(self._validate_button)

        self._use_for_training_button = QPushButton("Save for Training...")
        self._use_for_training_button.clicked.connect(self._on_use_for_training_clicked)
        actions_row.addWidget(self._use_for_training_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self._layer_panel = LayerListPanel([], allow_branch=True)
        layout.addWidget(self._layer_panel)

        self._status_label = QLabel("Idle")
        layout.addWidget(self._status_label)

        self._shape_trace_label = QLabel("")
        self._shape_trace_label.setWordWrap(True)
        layout.addWidget(self._shape_trace_label)

    # -- state -------------------------------------------------------------------

    def _capture_state(self) -> dict:
        """Snapshot every prior user-editable value (name, all three
        input-shape values, the full layer list/order/types/parameters/
        branches, and the current selection) so a failed load -- at any
        stage, including an unexpected failure mid-population -- can restore
        it exactly."""
        return {
            "name": self._name_edit.text(),
            "channels": self._channels_spin.text(),
            "height": self._height_spin.text(),
            "width": self._width_spin.text(),
            "layers": copy.deepcopy(self._layer_panel.layers),
            "selection": self._layer_panel.capture_selection_state(),
        }

    def _restore_state(self, state: dict) -> None:
        self._name_edit.setText(state["name"])
        self._channels_spin.setText(state["channels"])
        self._height_spin.setText(state["height"])
        self._width_spin.setText(state["width"])
        self._layer_panel.set_layers(state["layers"])
        self._layer_panel.restore_selection_state(state["selection"])

    def _load_from_dict(self, data: dict) -> None:
        """에디터 위젯 전체를 ``data``(model_spec_to_dict 형식)로 채운다.
        호출 전에 이미 ``_verify_model_stageable``로 표현 가능함이 증명된
        데이터만 넘겨야 한다. commit(위젯 population) 도중 예기치 못한
        실패가 나면 -- 이름이 이미 바뀐 뒤라도 -- 호출 전 상태를 완전히
        복원한 뒤 예외를 다시 던져 호출자가 일관되게 "Load failed"로
        보고할 수 있게 한다."""
        previous = self._capture_state()
        try:
            self._name_edit.setText(data["name"])
            channels, height, width = data["input_shape"]
            self._channels_spin.setValue(channels)
            self._height_spin.setValue(height)
            self._width_spin.setValue(width)
            self._layer_panel.set_layers(data["layers"])
            self._shape_trace_label.setText("")
        except Exception:
            self._restore_state(previous)
            raise

    def _build_model_spec(self) -> ModelSpec:
        """현재 위젯 상태 -> JSON dict -> 기존 ``model_spec_from_dict``.
        구조/파라미터 검증(누락 필드, 알 수 없는 타입, 각 dataclass
        ``__post_init__``)은 전부 그쪽에 위임한다."""
        data = {
            "name": self._name_edit.text(),
            "input_shape": [
                self._channels_spin.value(),
                self._height_spin.value(),
                self._width_spin.value(),
            ],
            "layers": self._layer_panel.layers,
        }
        return model_spec_from_dict(data)

    # -- actions -------------------------------------------------------------------

    def _on_new_clicked(self) -> None:
        self._load_from_dict(_starter_model_dict())
        self._status_label.setText("New model created")

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", filter="JSON Files (*.json)")
        if not path:
            return
        self._load_from_path(path)

    def _load_from_path(self, path: str | Path) -> None:
        """Transactional load: read -> deserialize (``load_model_spec``) ->
        validate shape-connectivity (``validate_model_spec``) -> stage (prove
        every model/layer value, including nested non-Branch branch layers,
        is representable by this editor's controls via
        ``_verify_model_stageable``) -- only after all of that succeeds does
        this touch a single live widget. A structurally valid but
        shape-disconnected file (e.g. a Linear layer straight after a
        Conv2d), or a value that could not be staged, would otherwise load
        "successfully" and silently replace/corrupt the last valid editor
        state. Every failure mode -- JSON/structural, parameter validation,
        shape-connectivity, staging, or an unexpected failure once
        population itself has begun -- reports the same concise bounded
        error and leaves the prior editor state (name, input shape, layer
        order/types/parameters/branches, and selection) completely intact."""
        try:
            model_spec = load_model_spec(path)
            validate_model_spec(model_spec)
            data = model_spec_to_dict(model_spec)
            _verify_model_stageable(data)
        except (ModelValidationError, OSError) as exc:
            self._status_label.setText(f"Load failed: {_first_line(exc)}")
            return
        try:
            self._load_from_dict(data)
        except Exception as exc:
            self._status_label.setText(f"Load failed: {_first_line(exc)}")
            return
        self._status_label.setText(f"Loaded: {Path(path).name}")

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", filter="JSON Files (*.json)")
        if not path:
            return
        self._save_to_path(path)

    def _save_to_path(self, path: str | Path) -> bool:
        """유효한 ModelSpec일 때만 저장한다: ``_build_model_spec()``과
        ``validate_model_spec()``이 먼저 성공해야 (atomic writer인)
        ``save_model_spec``이 호출되므로, 검증 실패 시 기존 목적지
        파일은 절대 건드리지 않는다. atomic save까지 모두 성공하면
        ``True``, 어느 단계든 실패하면 (bounded 에러를 status에 표시한 뒤)
        ``False``를 돌려준다 -- 호출자가 후속 동작(예: 학습용 경로 emit)을
        성공했을 때만 하도록 하기 위함이다."""
        try:
            model_spec = self._build_model_spec()
            validate_model_spec(model_spec)
        except ModelValidationError as exc:
            self._status_label.setText(f"Save failed: {_first_line(exc)}")
            return False
        try:
            save_model_spec(model_spec, path)
        except OSError as exc:
            self._status_label.setText(f"Save failed: {_first_line(exc)}")
            return False
        self._status_label.setText(f"Saved: {path}")
        return True

    def _on_use_for_training_clicked(self) -> None:
        """명시적 "이 모델을 저장하고 학습에 사용" 액션. Save...와 동일한
        canonical 검증 + atomic save를 거치고, **모든 단계가 성공했을
        때만** 방금 저장된 파일의 정규화된 절대 경로를
        ``model_saved_for_training``로 emit한다. 파일 대화상자 취소나
        검증/저장 실패는 학습 경로에 대해 완전한 no-op이며 (아무것도
        emit하지 않음), dirty/invalid 상태를 몰래 저장하지 않는다."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model for Training", filter="JSON Files (*.json)"
        )
        if not path:
            return
        if not self._save_to_path(path):
            return
        normalized = str(Path(path).resolve())
        self._status_label.setText(f"Saved for training: {normalized}")
        self.model_saved_for_training.emit(normalized)

    def _on_validate_clicked(self) -> None:
        try:
            model_spec = self._build_model_spec()
            trace = validate_model_spec(model_spec)
        except ModelValidationError as exc:
            self._shape_trace_label.setText(f"Validation failed: {_first_line(exc)}")
            self._status_label.setText("Invalid")
            return
        self._shape_trace_label.setText(format_shape_trace(trace))
        self._status_label.setText("Valid")
