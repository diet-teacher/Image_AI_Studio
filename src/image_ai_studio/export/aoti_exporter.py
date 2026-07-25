"""torch.export + AOTInductor package export.

Verified against the installed PyTorch 2.13.0 on this machine: the
public entry points are

    torch.export.export(model, (example_input,))
    torch._inductor.aoti_compile_and_package(exported_program, package_path=...)

`aoti_compile_and_package` lives under the **private** `torch._inductor`
module (leading underscore) even though it is the documented way to
produce a .pt2 package as of this version -- there is no non-underscore
alias. This is recorded in the export metadata rather than hidden.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import torch
from torch import Tensor, nn

from image_ai_studio.export.base import ModelExporter, build_metadata, write_metadata
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs

AOTI_COMPILE_API = "torch._inductor.aoti_compile_and_package"
AOTI_LOAD_API = "torch._inductor.aoti_load_package"
API_VISIBILITY = "private/internal (torch._inductor has a leading underscore; no public alias found in this version)"


class AOTInductorExporter(ModelExporter):
    def export(
        self,
        model: nn.Module,
        example_input: Tensor,
        output_path: Path,
        metadata_path: Path,
        *,
        model_name: str,
        state_dict_path: Path,
        device: str = "cpu",
    ) -> Path:
        model = model.eval().to(device)
        example_input = example_input.to(device).contiguous().to(torch.float32)

        started_at = _dt.datetime.now(_dt.timezone.utc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status = "PASS"
        error_log = None
        parity_info = None

        try:
            with torch.no_grad():
                reference_output = model(example_input)
                exported_program = torch.export.export(model, (example_input,))

            import torch._inductor as ti

            artifact_path_str = ti.aoti_compile_and_package(
                exported_program, package_path=str(output_path)
            )
            assert Path(artifact_path_str) == output_path or Path(artifact_path_str).exists()

            compiled = ti.aoti_load_package(str(output_path))
            with torch.no_grad():
                post_export_output = compiled(example_input)

            rtol = CPU_FP32_RTOL if device == "cpu" else None
            atol = CPU_FP32_ATOL if device == "cpu" else None
            if device != "cpu":
                from image_ai_studio.parity.compare_outputs import CUDA_FP32_ATOL, CUDA_FP32_RTOL

                rtol, atol = CUDA_FP32_RTOL, CUDA_FP32_ATOL

            parity = compare_outputs(reference_output, post_export_output, rtol=rtol, atol=atol)
            parity_info = parity.to_dict()
            if not parity.allclose:
                status = "FAIL"
                error_log = f"pre/post-export parity failed: {parity_info}"
        except Exception as exc:  # noqa: BLE001 - export failures must be captured, not swallowed
            status = "FAIL"
            error_log = f"{type(exc).__name__}: {exc}"

        finished_at = _dt.datetime.now(_dt.timezone.utc)

        metadata = build_metadata(
            export_backend="aot_inductor",
            export_mode=device,
            artifact_path=output_path,
            model_name=model_name,
            state_dict_path=state_dict_path,
            example_input=example_input,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            error_log=error_log,
            extra={
                "compile_api": AOTI_COMPILE_API,
                "load_api": AOTI_LOAD_API,
                "api_visibility": API_VISIBILITY,
                "beta_or_unstable": (
                    "torch.export is documented as stable in this version; "
                    "the AOTInductor packaging entry point is undocumented as "
                    "public API (private module) -- treat as subject to change."
                ),
                "pre_post_export_parity": parity_info,
            },
        )
        write_metadata(metadata, metadata_path)
        return output_path
