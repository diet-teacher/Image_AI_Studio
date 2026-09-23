"""Phase 14 CP1: `ModelDesigner` widget tests.

qtbot + tmp_path only -- no screenshots, sleeps, or unregistered markers.
Every case inspects widget state programmatically (spin/combo/checkbox
values, layer-dict contents, status/shape-trace label text) and drives the
existing canonical `model_definition` API (`ModelSpec`/`LayerSpec`
dataclasses, `load_model_spec`/`save_model_spec`, `validate_model_spec`) to
build fixtures and to check semantic round trips -- never a parallel
schema.

Every test requests the `qtbot` fixture (even when it only calls
`qtbot.addWidget()` for cleanup) so a `QApplication` is guaranteed to exist
before any `QWidget` subclass -- including the plain `LayerListPanel` used
directly in several cases -- is constructed.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PySide6.QtWidgets import QCheckBox, QFileDialog, QLineEdit

import image_ai_studio.gui.model_designer as model_designer_module
from image_ai_studio.gui.model_designer import (
    ModelDesigner,
    LayerListPanel,
    _BranchEditor,
    _starter_model_dict,
)
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import (
    load_model_spec,
    model_spec_to_dict,
    save_model_spec,
)
from image_ai_studio.model_definition.specs import (
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

_ALL_LAYER_TYPE_NAMES = {
    "conv2d",
    "batch_norm2d",
    "relu",
    "max_pool2d",
    "adaptive_avg_pool2d",
    "flatten",
    "linear",
    "dropout",
    "residual_block",
    "branch",
    "identity",
}


def _combo_items(combo) -> set[str]:
    return {combo.itemText(i) for i in range(combo.count())}


def _select_row(panel: LayerListPanel, row: int) -> None:
    panel._list_widget.setCurrentRow(row)


def _make_panel(qtbot, layers: list[dict], *, allow_branch: bool) -> LayerListPanel:
    panel = LayerListPanel(layers, allow_branch=allow_branch)
    qtbot.addWidget(panel)
    return panel


def _make_designer(qtbot) -> ModelDesigner:
    designer = ModelDesigner()
    qtbot.addWidget(designer)
    return designer


# -- initial state / New ------------------------------------------------------


def test_initial_state_is_deterministic_valid_starter(qtbot) -> None:
    designer = _make_designer(qtbot)

    assert designer._name_edit.text() == "new_model"
    assert (
        designer._channels_spin.value(),
        designer._height_spin.value(),
        designer._width_spin.value(),
    ) == (3, 32, 32)
    layer_types = [layer["type"] for layer in designer._layer_panel.layers]
    assert layer_types == ["conv2d", "relu", "flatten", "linear"]

    model_spec = designer._build_model_spec()
    assert isinstance(model_spec, ModelSpec)


def test_new_resets_to_starter_after_modification(qtbot) -> None:
    designer = _make_designer(qtbot)

    designer._name_edit.setText("edited")
    designer._channels_spin.setValue(1)
    designer._layer_panel._type_combo.setCurrentText("dropout")
    designer._layer_panel._on_add_clicked()
    assert len(designer._layer_panel.layers) == 5

    designer._on_new_clicked()

    assert designer._name_edit.text() == _starter_model_dict()["name"]
    assert (
        designer._channels_spin.value(),
        designer._height_spin.value(),
        designer._width_spin.value(),
    ) == tuple(_starter_model_dict()["input_shape"])
    assert designer._layer_panel.layers == _starter_model_dict()["layers"]
    assert designer._status_label.text() == "New model created"


# -- ordered layer editor: add/remove/move ------------------------------------


def test_add_layer_appends_deterministic_default(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText("conv2d")
    panel._on_add_clicked()

    assert len(panel.layers) == 1
    assert panel.layers[0] == {
        "type": "conv2d",
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
    }
    assert panel.current_index() == 0


def test_remove_layer_deletes_selected_row(qtbot) -> None:
    panel = _make_panel(
        qtbot,
        [{"type": "relu", "inplace": False}, {"type": "flatten"}, {"type": "dropout", "p": 0.5}],
        allow_branch=True,
    )
    _select_row(panel, 1)
    panel._on_remove_clicked()

    assert [layer["type"] for layer in panel.layers] == ["relu", "dropout"]


def test_remove_with_no_selection_is_noop(qtbot) -> None:
    panel = _make_panel(qtbot, [{"type": "flatten"}], allow_branch=True)
    panel._list_widget.setCurrentRow(-1)
    panel._on_remove_clicked()
    assert len(panel.layers) == 1


def test_move_up_and_move_down_swap_order(qtbot) -> None:
    panel = _make_panel(
        qtbot,
        [{"type": "relu", "inplace": False}, {"type": "flatten"}, {"type": "dropout", "p": 0.5}],
        allow_branch=True,
    )
    _select_row(panel, 2)
    panel._on_move_up_clicked()
    assert [layer["type"] for layer in panel.layers] == ["relu", "dropout", "flatten"]
    assert panel.current_index() == 1

    panel._on_move_down_clicked()
    assert [layer["type"] for layer in panel.layers] == ["relu", "flatten", "dropout"]
    assert panel.current_index() == 2


def test_move_up_at_top_and_move_down_at_bottom_are_noops(qtbot) -> None:
    panel = _make_panel(qtbot, [{"type": "relu", "inplace": False}, {"type": "flatten"}], allow_branch=True)
    _select_row(panel, 0)
    panel._on_move_up_clicked()
    assert [layer["type"] for layer in panel.layers] == ["relu", "flatten"]

    _select_row(panel, 1)
    panel._on_move_down_clicked()
    assert [layer["type"] for layer in panel.layers] == ["relu", "flatten"]


# -- layer type coverage -------------------------------------------------------


def test_top_level_combo_offers_every_registered_layer_type(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    assert _combo_items(panel._type_combo) == _ALL_LAYER_TYPE_NAMES


def test_branch_sub_panel_excludes_branch_type_no_nesting(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=False)
    assert "branch" not in _combo_items(panel._type_combo)
    assert _combo_items(panel._type_combo) == _ALL_LAYER_TYPE_NAMES - {"branch"}


@pytest.mark.parametrize("type_name", sorted(_ALL_LAYER_TYPE_NAMES))
def test_every_layer_type_builds_a_param_widget_without_error(qtbot, type_name: str) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText(type_name)
    panel._on_add_clicked()
    _select_row(panel, 0)

    assert panel._param_widget is not None
    assert panel.layers[0]["type"] == type_name


# -- typed field widgets edit the underlying layer dict -----------------------


def test_int_field_widget_updates_layer_dict(qtbot) -> None:
    panel = _make_panel(
        qtbot,
        [{"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0}],
        allow_branch=True,
    )
    _select_row(panel, 0)
    edits = panel._param_widget.findChildren(QLineEdit)
    assert edits, "expected text controls for conv2d integer params"
    edits[0].setText("64")
    assert panel.layers[0]["out_channels"] == 64
    # list summary refreshes too
    assert "64" in panel._list_widget.item(0).text()


def test_bool_field_widget_updates_layer_dict(qtbot) -> None:
    panel = _make_panel(qtbot, [{"type": "relu", "inplace": False}], allow_branch=True)
    _select_row(panel, 0)
    checkboxes = panel._param_widget.findChildren(QCheckBox)
    assert len(checkboxes) == 1
    checkboxes[0].setChecked(True)
    assert panel.layers[0]["inplace"] is True


def test_float_field_widget_updates_layer_dict(qtbot) -> None:
    panel = _make_panel(qtbot, [{"type": "dropout", "p": 0.5}], allow_branch=True)
    _select_row(panel, 0)
    edits = panel._param_widget.findChildren(QLineEdit)
    assert len(edits) == 1
    edits[0].setText("0.25")
    assert panel.layers[0]["p"] == pytest.approx(0.25)


def test_optional_int_field_defaults_auto_and_can_be_set_explicit(qtbot) -> None:
    panel = _make_panel(
        qtbot,
        [{"type": "max_pool2d", "kernel_size": 2, "stride": None, "padding": 0}],
        allow_branch=True,
    )
    _select_row(panel, 0)
    checkboxes = panel._param_widget.findChildren(QCheckBox)
    assert len(checkboxes) == 1  # the "Auto" checkbox
    assert checkboxes[0].isChecked() is True
    assert panel.layers[0]["stride"] is None

    checkboxes[0].setChecked(False)
    assert panel.layers[0]["stride"] is not None
    assert isinstance(panel.layers[0]["stride"], int)


# -- BranchSpec structured, non-nested editing --------------------------------


def test_branch_layer_editor_has_two_default_branches_and_add_combo(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText("branch")
    panel._on_add_clicked()
    _select_row(panel, 0)

    assert isinstance(panel._param_widget, _BranchEditor)
    editor = panel._param_widget
    nested_panels = editor.findChildren(LayerListPanel)
    assert len(nested_panels) == 2
    for nested in nested_panels:
        assert "branch" not in _combo_items(nested._type_combo)

    layer = panel.layers[0]
    assert layer["merge"] == "add"
    assert len(layer["branches"]) == 2
    assert layer["branches"][0] == [{"type": "identity"}]
    assert layer["branches"][1] == [{"type": "identity"}]


def test_branch_editor_add_and_remove_branch_respects_minimum_two(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText("branch")
    panel._on_add_clicked()
    _select_row(panel, 0)
    editor = panel._param_widget
    layer = panel.layers[0]

    editor._on_add_branch()
    assert len(layer["branches"]) == 3

    editor._on_remove_branch(0)
    assert len(layer["branches"]) == 2

    # Below the BranchSpec minimum of 2 -- must be refused.
    editor._on_remove_branch(0)
    assert len(layer["branches"]) == 2


def test_branch_editor_merge_combo_updates_layer_dict(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText("branch")
    panel._on_add_clicked()
    _select_row(panel, 0)
    editor = panel._param_widget

    editor._merge_combo.setCurrentText("concat")
    assert panel.layers[0]["merge"] == "concat"


def test_branch_editor_nested_panel_edits_reach_branch_list(qtbot) -> None:
    panel = _make_panel(qtbot, [], allow_branch=True)
    panel._type_combo.setCurrentText("branch")
    panel._on_add_clicked()
    _select_row(panel, 0)
    editor = panel._param_widget
    nested_panels = editor.findChildren(LayerListPanel)

    first_branch_panel = nested_panels[0]
    first_branch_panel._type_combo.setCurrentText("relu")
    first_branch_panel._on_add_clicked()

    layer = panel.layers[0]
    assert [item["type"] for item in layer["branches"][0]] == ["identity", "relu"]


# -- Validate: existing shape-trace / bounded error, no re-implemented rules --


def test_validate_valid_model_shows_shape_trace(qtbot) -> None:
    designer = _make_designer(qtbot)

    designer._on_validate_clicked()

    assert designer._status_label.text() == "Valid"
    assert designer._shape_trace_label.text() != ""
    assert "Conv2d" in designer._shape_trace_label.text()
    assert "->" in designer._shape_trace_label.text()


def test_validate_invalid_shape_shows_bounded_actionable_error(qtbot) -> None:
    designer = _make_designer(qtbot)
    # Linear expects a 1D input but the preceding Conv2d output is 3D --
    # a genuine shape-connectivity failure from the existing shape_inference
    # rules (not re-implemented here).
    designer._layer_panel.set_layers(
        [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "linear", "out_features": 10, "bias": True},
        ]
    )

    designer._on_validate_clicked()

    assert designer._status_label.text() == "Invalid"
    text = designer._shape_trace_label.text()
    assert text.startswith("Validation failed:")
    assert len(text) < 300


def test_validate_structural_error_shows_bounded_actionable_error(qtbot) -> None:
    designer = _make_designer(qtbot)
    designer._name_edit.setText("")  # ModelSpec.name must be non-empty

    designer._on_validate_clicked()

    assert designer._status_label.text() == "Invalid"
    assert designer._shape_trace_label.text().startswith("Validation failed:")


# -- Load: deterministic population + error recovery --------------------------


def _sample_model_spec() -> ModelSpec:
    return ModelSpec(
        name="sample_net",
        input_shape=(3, 16, 16),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            BatchNorm2dSpec(eps=1e-4, momentum=0.2),
            ReLUSpec(inplace=True),
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=8, kernel_size=1, stride=1, padding=0)],
                    [IdentitySpec()],
                ],
                merge="add",
            ),
            MaxPool2dSpec(kernel_size=2, stride=None, padding=0),
            FlattenSpec(),
            DropoutSpec(p=0.3),
            LinearSpec(out_features=5, bias=False),
        ],
    )


def test_load_valid_file_populates_editor_deterministically(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    original = _sample_model_spec()
    path = tmp_path / "model.json"
    save_model_spec(original, path)

    designer._load_from_path(path)

    assert designer._name_edit.text() == "sample_net"
    assert (
        designer._channels_spin.value(),
        designer._height_spin.value(),
        designer._width_spin.value(),
    ) == (3, 16, 16)
    layer_types = [layer["type"] for layer in designer._layer_panel.layers]
    assert layer_types == [
        "conv2d",
        "batch_norm2d",
        "relu",
        "branch",
        "max_pool2d",
        "flatten",
        "dropout",
        "linear",
    ]
    branch_layer = designer._layer_panel.layers[3]
    assert [item["type"] for item in branch_layer["branches"][0]] == ["conv2d"]
    assert [item["type"] for item in branch_layer["branches"][1]] == ["identity"]
    assert designer._status_label.text() == "Loaded: model.json"

    rebuilt = designer._build_model_spec()
    assert model_spec_to_dict(rebuilt) == model_spec_to_dict(original)


def test_load_invalid_json_syntax_preserves_previous_valid_state(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    good_path = tmp_path / "good.json"
    save_model_spec(_sample_model_spec(), good_path)
    designer._load_from_path(good_path)
    previous_layers = [dict(layer) for layer in designer._layer_panel.layers]
    previous_name = designer._name_edit.text()

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    designer._load_from_path(bad_path)

    assert designer._status_label.text().startswith("Load failed:")
    assert designer._name_edit.text() == previous_name
    assert designer._layer_panel.layers == previous_layers


def test_load_unsupported_structure_preserves_previous_valid_state(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    good_path = tmp_path / "good.json"
    save_model_spec(_sample_model_spec(), good_path)
    designer._load_from_path(good_path)
    previous_layers = [dict(layer) for layer in designer._layer_panel.layers]

    missing_field_path = tmp_path / "missing.json"
    missing_field_path.write_text('{"name": "x", "input_shape": [3, 8, 8]}', encoding="utf-8")
    designer._load_from_path(missing_field_path)

    assert designer._status_label.text().startswith("Load failed:")
    assert designer._layer_panel.layers == previous_layers


def test_load_parameter_validation_failure_preserves_previous_valid_state(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    good_path = tmp_path / "good.json"
    save_model_spec(_sample_model_spec(), good_path)
    designer._load_from_path(good_path)
    previous_name = designer._name_edit.text()

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        '{"name": "bad", "input_shape": [3, 8, 8], '
        '"layers": [{"type": "conv2d", "out_channels": -1, "kernel_size": 3}]}',
        encoding="utf-8",
    )
    designer._load_from_path(invalid_path)

    assert designer._status_label.text().startswith("Load failed:")
    assert designer._name_edit.text() == previous_name


# -- Save: valid-only, atomic, reports path, no partial overwrite -------------


def test_save_valid_model_writes_canonical_json_and_reports_path(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    path = tmp_path / "out.json"

    designer._save_to_path(path)

    assert path.exists()
    saved = load_model_spec(path)
    assert model_spec_to_dict(saved) == model_spec_to_dict(designer._build_model_spec())
    assert designer._status_label.text() == f"Saved: {path}"


def test_save_structural_failure_does_not_overwrite_existing_file(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    path = tmp_path / "existing.json"
    save_model_spec(_sample_model_spec(), path)
    before_bytes = path.read_bytes()

    designer._name_edit.setText("")  # invalid: empty name
    designer._save_to_path(path)

    assert designer._status_label.text().startswith("Save failed:")
    assert path.read_bytes() == before_bytes


def test_save_shape_validation_failure_does_not_overwrite_existing_file(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    path = tmp_path / "existing.json"
    save_model_spec(_sample_model_spec(), path)
    before_bytes = path.read_bytes()

    designer._layer_panel.set_layers(
        [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "linear", "out_features": 10, "bias": True},
        ]
    )
    designer._save_to_path(path)

    assert designer._status_label.text().startswith("Save failed:")
    assert path.read_bytes() == before_bytes


def test_save_then_load_round_trip_preserves_layers(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)
    path = tmp_path / "roundtrip.json"

    designer._save_to_path(path)
    other = _make_designer(qtbot)
    other._load_from_path(path)

    assert other._layer_panel.layers == designer._layer_panel.layers
    assert other._name_edit.text() == designer._name_edit.text()


# -- semantic round trip: layer order, branch order, scalars, optionals ------


def test_load_then_save_preserves_full_semantic_model_spec(qtbot, tmp_path) -> None:
    """Load an existing canonical model (with a Branch, an explicit and an
    auto-default MaxPool2d stride, and various scalar types), save it back
    untouched, and confirm the two ModelSpec objects are fully equal --
    name, input_shape, layer order, branch order, scalar types, and
    optional values all survive the round trip via the *same* dataclasses
    (dataclass ``==`` is a deep structural comparison)."""
    original = ModelSpec(
        name="round_trip_net",
        input_shape=(3, 24, 24),
        layers=[
            Conv2dSpec(out_channels=12, kernel_size=3, stride=2, padding=1),
            BranchSpec(
                branches=[
                    [ResidualBlockSpec(out_channels=12, stride=1)],
                    [Conv2dSpec(out_channels=12, kernel_size=1, stride=1, padding=0), ReLUSpec(inplace=False)],
                ],
                merge="add",
            ),
            MaxPool2dSpec(kernel_size=2, stride=2, padding=0),  # explicit stride
            MaxPool2dSpec(kernel_size=2, stride=None, padding=0),  # auto (None) stride
            FlattenSpec(),
            LinearSpec(out_features=7, bias=True),
        ],
    )
    source_path = tmp_path / "source.json"
    save_model_spec(original, source_path)

    designer = _make_designer(qtbot)
    designer._load_from_path(source_path)

    dest_path = tmp_path / "dest.json"
    designer._save_to_path(dest_path)

    round_tripped = load_model_spec(dest_path)
    assert round_tripped == original
    assert model_spec_to_dict(round_tripped) == model_spec_to_dict(original)


# -- rework regressions: Load must shape-validate; widgets must not clamp ----


def test_load_shape_invalid_model_preserves_previous_valid_state(qtbot, tmp_path) -> None:
    """``load_model_spec`` only checks JSON/structure/per-field parameters --
    it never runs shape-connectivity inference. A structurally valid file
    whose Linear layer directly follows a Conv2d (a genuine shape-connectivity
    failure) must still be refused by Load, with the same "leaves prior valid
    state intact" contract as a JSON/structural failure."""
    designer = _make_designer(qtbot)
    good_path = tmp_path / "good.json"
    save_model_spec(_sample_model_spec(), good_path)
    designer._load_from_path(good_path)
    previous_name = designer._name_edit.text()
    previous_layers = [dict(layer) for layer in designer._layer_panel.layers]

    shape_invalid = ModelSpec(
        name="shape_invalid",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            LinearSpec(out_features=10, bias=True),
        ],
    )
    bad_path = tmp_path / "shape_invalid.json"
    save_model_spec(shape_invalid, bad_path)

    designer._load_from_path(bad_path)

    assert designer._status_label.text().startswith("Load failed:")
    assert designer._name_edit.text() == previous_name
    assert designer._layer_panel.layers == previous_layers


def test_input_shape_preserves_python_integer_above_qt_limit(qtbot, tmp_path) -> None:
    """Canonical input dimensions are Python ints, not Qt 32-bit ints."""
    original = ModelSpec(
        name="huge_shape_net",
        input_shape=(3, 2147483648, 1),
        layers=[IdentitySpec()],
    )
    path = tmp_path / "huge_shape.json"
    save_model_spec(original, path)

    designer = _make_designer(qtbot)
    designer._load_from_path(path)

    assert (
        designer._channels_spin.value(),
        designer._height_spin.value(),
        designer._width_spin.value(),
    ) == (3, 2147483648, 1)

    rebuilt = designer._build_model_spec()
    assert model_spec_to_dict(rebuilt) == model_spec_to_dict(original)

    saved_path = tmp_path / "huge_shape_saved.json"
    designer._save_to_path(saved_path)
    assert load_model_spec(saved_path).input_shape == (3, 2147483648, 1)


def test_layer_integer_preserves_python_integer_above_qt_limit(qtbot, tmp_path) -> None:
    original = ModelSpec(
        name="huge_layer_integer_net",
        input_shape=(1, 1, 1),
        layers=[FlattenSpec(), LinearSpec(out_features=2147483648, bias=True)],
    )
    path = tmp_path / "huge_layer_integer.json"
    save_model_spec(original, path)

    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    _select_row(designer._layer_panel, 1)
    edits = designer._layer_panel._param_widget.findChildren(QLineEdit)

    assert len(edits) == 1
    assert edits[0].text() == "2147483648"
    assert designer._layer_panel.layers[1]["out_features"] == 2147483648
    assert model_spec_to_dict(designer._build_model_spec()) == model_spec_to_dict(original)

    saved_path = tmp_path / "huge_layer_integer_saved.json"
    designer._save_to_path(saved_path)
    reloaded = _make_designer(qtbot)
    reloaded._load_from_path(saved_path)
    _select_row(reloaded._layer_panel, 1)
    reloaded_edits = reloaded._layer_panel._param_widget.findChildren(QLineEdit)
    assert reloaded_edits[0].text() == "2147483648"
    assert reloaded._build_model_spec().layers[1].out_features == 2147483648


@pytest.mark.parametrize("invalid_text", ["", "3.5", "0", "-1", "not-an-int"])
def test_invalid_input_shape_text_blocks_model_build(qtbot, invalid_text: str) -> None:
    designer = _make_designer(qtbot)
    designer._height_spin.setText(invalid_text)

    with pytest.raises(ModelValidationError):
        designer._build_model_spec()


def test_batch_norm_eps_and_momentum_widgets_do_not_clamp_extreme_valid_values(qtbot) -> None:
    """eps has no upper bound in specs.py, and momentum only needs to be in
    (0.0, 1.0]. A widget range narrower than those real constraints silently
    clamps (and therefore misrepresents/corrupts) an otherwise-valid loaded
    value -- eps=1e-12 and momentum=1e-12 must display and round-trip
    unchanged."""
    layer = {"type": "batch_norm2d", "eps": 1e-12, "momentum": 1e-12}
    panel = _make_panel(qtbot, [layer], allow_branch=True)
    _select_row(panel, 0)

    edits = panel._param_widget.findChildren(QLineEdit)
    assert len(edits) == 2
    assert [edit.text() for edit in edits] == ["1e-12", "1e-12"]
    assert [float(edit.text()) for edit in edits] == [1e-12, 1e-12]
    assert layer["eps"] == 1e-12
    assert layer["momentum"] == 1e-12


def test_batch_norm_extreme_values_round_trip_through_save_and_load(qtbot, tmp_path) -> None:
    original = ModelSpec(
        name="extreme_bn_net",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, stride=1, padding=1),
            BatchNorm2dSpec(eps=1e-12, momentum=1e-12),
            FlattenSpec(),
            LinearSpec(out_features=2, bias=True),
        ],
    )
    path = tmp_path / "extreme_bn.json"
    save_model_spec(original, path)

    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    # Select the BatchNorm row to force its parameter widgets to actually be
    # built (this is exactly where clamping would previously misrepresent --
    # and, if the user then nudged a field, corrupt -- the loaded value).
    _select_row(designer._layer_panel, 1)

    rebuilt = designer._build_model_spec()
    assert model_spec_to_dict(rebuilt) == model_spec_to_dict(original)

    dest_path = tmp_path / "extreme_bn_out.json"
    designer._save_to_path(dest_path)
    round_tripped = load_model_spec(dest_path)
    assert round_tripped == original


def test_scientific_float_text_is_authoritative_for_build(qtbot, tmp_path) -> None:
    original = ModelSpec(
        name="scientific_float_net",
        input_shape=(3, 8, 8),
        layers=[BatchNorm2dSpec(eps=1e-12, momentum=1e-12), IdentitySpec()],
    )
    path = tmp_path / "scientific_float.json"
    save_model_spec(original, path)
    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    _select_row(designer._layer_panel, 0)
    edits = designer._layer_panel._param_widget.findChildren(QLineEdit)

    edits[0].setText("2.5e-12")
    edits[1].setText("3E-12")
    rebuilt = designer._build_model_spec()

    assert rebuilt.layers[0].eps == 2.5e-12
    assert rebuilt.layers[0].momentum == 3e-12

    saved_path = tmp_path / "scientific_float_saved.json"
    designer._save_to_path(saved_path)
    reloaded = _make_designer(qtbot)
    reloaded._load_from_path(saved_path)
    assert reloaded._build_model_spec().layers[0].eps == 2.5e-12
    assert reloaded._build_model_spec().layers[0].momentum == 3e-12


@pytest.mark.parametrize("invalid_text", ["", "bad-integer", "0"])
def test_invalid_integer_text_survives_form_reconstruction_and_can_be_fixed(
    qtbot, tmp_path, invalid_text: str
) -> None:
    original = ModelSpec(
        name="integer_reconstruction",
        input_shape=(3, 8, 8),
        layers=[Conv2dSpec(out_channels=8, kernel_size=3, padding=1), IdentitySpec()],
    )
    path = tmp_path / "integer_reconstruction.json"
    save_model_spec(original, path)
    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    _select_row(designer._layer_panel, 0)
    designer._layer_panel._param_widget.findChildren(QLineEdit)[0].setText(invalid_text)

    _select_row(designer._layer_panel, 1)
    _select_row(designer._layer_panel, 0)
    reconstructed = designer._layer_panel._param_widget.findChildren(QLineEdit)[0]
    assert reconstructed.text() == invalid_text
    with pytest.raises(ModelValidationError):
        designer._build_model_spec()
    failed_path = tmp_path / "integer_must_not_save.json"
    designer._save_to_path(failed_path)
    assert designer._status_label.text().startswith("Save failed:")
    assert not failed_path.exists()

    reconstructed.setText("64")
    assert designer._build_model_spec().layers[0].out_channels == 64
    recovered_path = tmp_path / "integer_recovered.json"
    designer._save_to_path(recovered_path)
    assert load_model_spec(recovered_path).layers[0].out_channels == 64


@pytest.mark.parametrize("invalid_text", ["bad-float", "nan", "inf"])
def test_invalid_float_text_survives_form_reconstruction_and_can_be_fixed(
    qtbot, tmp_path, invalid_text: str
) -> None:
    original = ModelSpec(
        name="float_reconstruction",
        input_shape=(3, 8, 8),
        layers=[BatchNorm2dSpec(eps=1e-5, momentum=0.1), IdentitySpec()],
    )
    path = tmp_path / "float_reconstruction.json"
    save_model_spec(original, path)
    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    _select_row(designer._layer_panel, 0)
    designer._layer_panel._param_widget.findChildren(QLineEdit)[0].setText(invalid_text)

    _select_row(designer._layer_panel, 1)
    _select_row(designer._layer_panel, 0)
    reconstructed = designer._layer_panel._param_widget.findChildren(QLineEdit)[0]
    assert reconstructed.text() == invalid_text
    with pytest.raises(ModelValidationError):
        designer._build_model_spec()
    failed_path = tmp_path / "float_must_not_save.json"
    designer._save_to_path(failed_path)
    assert designer._status_label.text().startswith("Save failed:")
    assert not failed_path.exists()

    reconstructed.setText("1e-12")
    assert designer._build_model_spec().layers[0].eps == 1e-12
    recovered_path = tmp_path / "float_recovered.json"
    designer._save_to_path(recovered_path)
    assert load_model_spec(recovered_path).layers[0].eps == 1e-12


@pytest.mark.parametrize("invalid_text", ["nan", "inf", "-inf", "", "not-a-number"])
def test_invalid_visible_float_text_blocks_build_and_save(qtbot, tmp_path, invalid_text: str) -> None:
    original = ModelSpec(
        name="invalid_float_net",
        input_shape=(3, 8, 8),
        layers=[BatchNorm2dSpec(eps=1e-5, momentum=0.1), IdentitySpec()],
    )
    source_path = tmp_path / "invalid_float_source.json"
    save_model_spec(original, source_path)
    designer = _make_designer(qtbot)
    designer._load_from_path(source_path)
    _select_row(designer._layer_panel, 0)
    edits = designer._layer_panel._param_widget.findChildren(QLineEdit)
    edits[0].setText(invalid_text)

    with pytest.raises(ModelValidationError):
        designer._build_model_spec()

    destination = tmp_path / "must_remain.json"
    destination.write_text('{"sentinel": true}\n', encoding="utf-8")
    before = destination.read_bytes()
    designer._save_to_path(destination)
    assert designer._status_label.text().startswith("Save failed:")
    assert destination.read_bytes() == before


def _distinct_designer_state(qtbot, tmp_path) -> ModelDesigner:
    spec = ModelSpec(
        name="prior_distinct_model",
        input_shape=(3, 2147483648, 1),
        layers=[IdentitySpec(), FlattenSpec(), LinearSpec(out_features=37, bias=False)],
    )
    path = tmp_path / "prior_distinct.json"
    save_model_spec(spec, path)
    designer = _make_designer(qtbot)
    designer._load_from_path(path)
    _select_row(designer._layer_panel, 2)
    return designer


def _editor_data_snapshot(designer: ModelDesigner) -> dict:
    return {
        "name": designer._name_edit.text(),
        "shape_text": (
            designer._channels_spin.text(),
            designer._height_spin.text(),
            designer._width_spin.text(),
        ),
        "layers": copy.deepcopy(designer._layer_panel.layers),
        "selection": copy.deepcopy(designer._layer_panel.capture_selection_state()),
        "model": model_spec_to_dict(designer._build_model_spec()),
    }


def test_deserialize_failure_preserves_complete_editor_state(qtbot, tmp_path) -> None:
    designer = _distinct_designer_state(qtbot, tmp_path)
    before = _editor_data_snapshot(designer)
    path = tmp_path / "broken.json"
    path.write_text("{broken json", encoding="utf-8")

    designer._load_from_path(path)

    assert designer._status_label.text().startswith("Load failed:")
    assert _editor_data_snapshot(designer) == before


def test_validation_failure_preserves_complete_editor_state(qtbot, tmp_path) -> None:
    designer = _distinct_designer_state(qtbot, tmp_path)
    before = _editor_data_snapshot(designer)
    invalid = ModelSpec(
        name="shape_invalid",
        input_shape=(3, 8, 8),
        layers=[Conv2dSpec(out_channels=8, kernel_size=3), LinearSpec(out_features=2)],
    )
    path = tmp_path / "shape_invalid_distinct.json"
    save_model_spec(invalid, path)

    designer._load_from_path(path)

    assert designer._status_label.text().startswith("Load failed:")
    assert _editor_data_snapshot(designer) == before


def test_staging_failure_preserves_complete_editor_state(qtbot, tmp_path, monkeypatch) -> None:
    import image_ai_studio.gui.model_designer as model_designer_module

    designer = _distinct_designer_state(qtbot, tmp_path)
    before = _editor_data_snapshot(designer)
    path = tmp_path / "stageable.json"
    save_model_spec(_sample_model_spec(), path)

    def _fail_staging(_data) -> None:
        raise ModelValidationError("injected staging failure")

    monkeypatch.setattr(model_designer_module, "_verify_model_stageable", _fail_staging)
    designer._load_from_path(path)

    assert designer._status_label.text().startswith("Load failed:")
    assert _editor_data_snapshot(designer) == before


def test_population_failure_rolls_back_complete_editor_state(qtbot, tmp_path, monkeypatch) -> None:
    designer = _distinct_designer_state(qtbot, tmp_path)
    before = _editor_data_snapshot(designer)
    path = tmp_path / "population_target.json"
    save_model_spec(_sample_model_spec(), path)
    original_set_layers = designer._layer_panel.set_layers
    calls = 0

    def _fail_once(layers) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            designer._layer_panel.layers = [{"type": "identity"}]
            raise RuntimeError("injected population failure")
        original_set_layers(layers)

    monkeypatch.setattr(designer._layer_panel, "set_layers", _fail_once)
    designer._load_from_path(path)

    assert calls == 2
    assert designer._status_label.text().startswith("Load failed:")
    assert _editor_data_snapshot(designer) == before


def test_population_failure_restores_nested_branch_selections(qtbot, tmp_path, monkeypatch) -> None:
    original = ModelSpec(
        name="nested_selection_model",
        input_shape=(3, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [IdentitySpec(), ReLUSpec(inplace=True)],
                    [IdentitySpec(), ReLUSpec(inplace=False)],
                ],
                merge="add",
            ),
            IdentitySpec(),
        ],
    )
    source = tmp_path / "nested_selection.json"
    save_model_spec(original, source)
    designer = _make_designer(qtbot)
    designer._load_from_path(source)
    _select_row(designer._layer_panel, 0)
    branch_editor = designer._layer_panel._param_widget
    assert isinstance(branch_editor, _BranchEditor)
    branch_editor._branch_panels[0].select_row(1)
    branch_editor._branch_panels[1].select_row(0)
    before = _editor_data_snapshot(designer)

    target = tmp_path / "nested_population_target.json"
    save_model_spec(_sample_model_spec(), target)
    original_set_layers = designer._layer_panel.set_layers
    calls = 0

    def _fail_once(layers) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            designer._layer_panel.layers = [{"type": "identity"}]
            raise RuntimeError("injected nested population failure")
        original_set_layers(layers)

    monkeypatch.setattr(designer._layer_panel, "set_layers", _fail_once)
    designer._load_from_path(target)

    assert calls == 2
    assert designer._status_label.text().startswith("Load failed:")
    assert _editor_data_snapshot(designer) == before
    restored_editor = designer._layer_panel._param_widget
    assert isinstance(restored_editor, _BranchEditor)
    assert [panel.current_index() for panel in restored_editor._branch_panels] == [1, 0]


def test_refresh_list_preserves_existing_signal_block_state(qtbot) -> None:
    panel = _make_panel(qtbot, [{"type": "identity"}], allow_branch=True)
    assert panel._list_widget.signalsBlocked() is False
    panel._refresh_list()
    assert panel._list_widget.signalsBlocked() is False

    panel._list_widget.blockSignals(True)
    panel._refresh_list()
    assert panel._list_widget.signalsBlocked() is True
    panel._list_widget.blockSignals(False)


def test_refresh_list_restores_signal_block_state_after_exception(qtbot, monkeypatch) -> None:
    import image_ai_studio.gui.model_designer as model_designer_module

    panel = _make_panel(qtbot, [{"type": "identity"}], allow_branch=True)
    panel._list_widget.blockSignals(True)

    def _fail_summary(_layer) -> str:
        raise RuntimeError("injected summary failure")

    monkeypatch.setattr(model_designer_module, "_layer_summary", _fail_summary)
    with pytest.raises(RuntimeError, match="injected summary failure"):
        panel._refresh_list()
    assert panel._list_widget.signalsBlocked() is True
    panel._list_widget.blockSignals(False)


def test_save_write_failure_does_not_overwrite_existing_file(qtbot, tmp_path, monkeypatch) -> None:
    """A failure inside the atomic writer itself (not a validation failure)
    must be reported the same bounded way and must not touch an existing
    destination file's bytes."""
    import image_ai_studio.gui.model_designer as model_designer_module

    designer = _make_designer(qtbot)
    path = tmp_path / "existing.json"
    save_model_spec(_sample_model_spec(), path)
    before_bytes = path.read_bytes()

    def _boom(_model_spec, _path) -> None:
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(model_designer_module, "save_model_spec", _boom)

    designer._save_to_path(path)

    assert designer._status_label.text().startswith("Save failed:")
    assert path.read_bytes() == before_bytes


# -- CP2: explicit "Save for Training" handoff signal ------------------------


def _stub_save_dialog(monkeypatch, path) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )


def _capture_signal(designer: ModelDesigner) -> list[str]:
    emitted: list[str] = []
    designer.model_saved_for_training.connect(emitted.append)
    return emitted


def test_save_to_path_returns_true_on_success_false_on_failure(qtbot, tmp_path) -> None:
    designer = _make_designer(qtbot)

    assert designer._save_to_path(tmp_path / "ok.json") is True

    designer._name_edit.setText("")  # invalid: empty name
    assert designer._save_to_path(tmp_path / "bad.json") is False
    assert not (tmp_path / "bad.json").exists()


def test_use_for_training_emits_normalized_absolute_saved_path(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)
    target = tmp_path / "designed_model.json"
    _stub_save_dialog(monkeypatch, target)

    designer._on_use_for_training_clicked()

    expected = str(Path(target).resolve())
    assert emitted == [expected]
    assert Path(expected).is_absolute()
    # It is an ordinary canonical model-definition file.
    saved = load_model_spec(target)
    assert model_spec_to_dict(saved) == model_spec_to_dict(designer._build_model_spec())
    assert designer._status_label.text() == f"Saved for training: {expected}"


def test_use_for_training_dialog_cancel_is_complete_noop(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)
    status_before = designer._status_label.text()
    state_before = _editor_data_snapshot(designer)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )

    designer._on_use_for_training_clicked()

    assert emitted == []
    assert designer._status_label.text() == status_before
    assert _editor_data_snapshot(designer) == state_before
    assert list(tmp_path.iterdir()) == []


def test_use_for_training_invalid_model_does_not_emit_or_write(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)
    designer._name_edit.setText("")  # ModelSpec.name must be non-empty
    target = tmp_path / "must_not_exist.json"
    _stub_save_dialog(monkeypatch, target)

    designer._on_use_for_training_clicked()

    assert emitted == []
    assert not target.exists()
    assert designer._status_label.text().startswith("Save failed:")


def test_use_for_training_shape_invalid_model_does_not_emit(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)
    designer._layer_panel.set_layers(
        [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "linear", "out_features": 10, "bias": True},
        ]
    )
    target = tmp_path / "shape_invalid.json"
    _stub_save_dialog(monkeypatch, target)

    designer._on_use_for_training_clicked()

    assert emitted == []
    assert not target.exists()
    assert designer._status_label.text().startswith("Save failed:")


def test_use_for_training_atomic_write_failure_does_not_emit(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)
    state_before = _editor_data_snapshot(designer)
    path = tmp_path / "existing.json"
    save_model_spec(_sample_model_spec(), path)
    before_bytes = path.read_bytes()

    def _boom(_model_spec, _path) -> None:
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(model_designer_module, "save_model_spec", _boom)
    _stub_save_dialog(monkeypatch, path)

    designer._on_use_for_training_clicked()

    assert emitted == []
    assert path.read_bytes() == before_bytes
    assert designer._status_label.text().startswith("Save failed:")
    assert _editor_data_snapshot(designer) == state_before


def test_use_for_training_repeated_use_emits_each_success(qtbot, tmp_path, monkeypatch) -> None:
    designer = _make_designer(qtbot)
    emitted = _capture_signal(designer)

    first = tmp_path / "first.json"
    _stub_save_dialog(monkeypatch, first)
    designer._on_use_for_training_clicked()

    second = tmp_path / "second.json"
    _stub_save_dialog(monkeypatch, second)
    designer._on_use_for_training_clicked()

    assert emitted == [str(Path(first).resolve()), str(Path(second).resolve())]


def test_use_for_training_leaves_editor_state_intact(qtbot, tmp_path, monkeypatch) -> None:
    """The handoff action saves the current model as-is; it does not reset,
    reload, or otherwise disturb the editor."""
    designer = _make_designer(qtbot)
    _capture_signal(designer)
    designer._name_edit.setText("kept_name")
    before_layers = copy.deepcopy(designer._layer_panel.layers)
    _stub_save_dialog(monkeypatch, tmp_path / "m.json")

    designer._on_use_for_training_clicked()

    assert designer._name_edit.text() == "kept_name"
    assert designer._layer_panel.layers == before_layers
