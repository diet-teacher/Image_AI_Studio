"""torch.jit.trace export path.

Phase 0 validates the static trace path only -- torch.jit.script is out
of scope. TorchScript is deprecated upstream but kept as the
compatibility backend for this spike.
"""
from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn

from image_ai_studio.export.base import ModelExporter, build_metadata, write_metadata
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs

STRICT = True
CHECK_TRACE = True


class TorchScriptExporter(ModelExporter):
    def export(
        self,
        model: nn.Module,
        example_input: Tensor,
        output_path: Path,
        metadata_path: Path,
        *,
        model_name: str,
        state_dict_path: Path,
    ) -> Path:
        model = model.eval()
        example_input = example_input.to("cpu").contiguous().to(torch.float32)

        started_at = _dt.datetime.now(_dt.timezone.utc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status = "PASS"
        error_log = None
        parity_info = None

        try:
            with torch.inference_mode():
                reference_output = model(example_input)

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    traced = torch.jit.trace(
                        model, example_input, strict=STRICT, check_trace=CHECK_TRACE
                    )
                    deprecation_warnings = [str(w.message) for w in caught]

                traced.save(str(output_path))

                reloaded = torch.jit.load(str(output_path))
                reloaded.eval()
                post_export_output = reloaded(example_input)

            parity = compare_outputs(
                reference_output, post_export_output, rtol=CPU_FP32_RTOL, atol=CPU_FP32_ATOL
            )
            parity_info = parity.to_dict()
            if not parity.allclose:
                status = "FAIL"
                error_log = f"pre/post-export parity failed: {parity_info}"
        except Exception as exc:  # noqa: BLE001 - export failures must be captured, not swallowed
            status = "FAIL"
            error_log = f"{type(exc).__name__}: {exc}"
            deprecation_warnings = []

        finished_at = _dt.datetime.now(_dt.timezone.utc)

        metadata = build_metadata(
            export_backend="torchscript",
            export_mode="trace",
            artifact_path=output_path,
            model_name=model_name,
            state_dict_path=state_dict_path,
            example_input=example_input,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            error_log=error_log,
            extra={
                "trace_strict": STRICT,
                "trace_check_trace": CHECK_TRACE,
                "torch_jit_script_used": False,
                "deprecation_warnings": deprecation_warnings,
                "pre_post_export_parity": parity_info,
                "scope_note": (
                    "Only the static torch.jit.trace path was validated. "
                    "torch.jit.script is out of scope for Phase 0."
                ),
            },
        )
        write_metadata(metadata, metadata_path)
        return output_path
