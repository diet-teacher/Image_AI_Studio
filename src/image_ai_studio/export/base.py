"""Minimal backend-swappable exporter interface (Python export side only;
the C++ runners intentionally do NOT share a common interface in Phase 0).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, nn

from image_ai_studio.tools.inspect_environment import inspect_environment


class ModelExporter(ABC):
    @abstractmethod
    def export(
        self,
        model: nn.Module,
        example_input: Tensor,
        output_path: Path,
        metadata_path: Path,
    ) -> Path:
        ...


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_metadata(
    *,
    export_backend: str,
    export_mode: str,
    artifact_path: Path,
    model_name: str,
    state_dict_path: Path,
    example_input: Tensor,
    started_at: _dt.datetime,
    finished_at: _dt.datetime,
    status: str,
    error_log: str | None = None,
    extra: dict | None = None,
) -> dict:
    env = inspect_environment()

    meta = {
        "export_backend": export_backend,
        "export_mode": export_mode,
        "artifact": str(artifact_path),
        "python_version": env.get("python_version"),
        "torch_version": env.get("torch_version"),
        "torch_cuda_version": env.get("torch_cuda_build_version"),
        "cudnn_version": env.get("torch_cudnn_version"),
        "os": env.get("os"),
        "architecture": env.get("architecture"),
        "visual_studio_version": env.get("visual_studio_version"),
        "msvc_version": None if platform.system() != "Windows" else env.get("compiler"),
        "compiler": env.get("compiler"),
        "cmake_version": env.get("cmake_version"),
        "libtorch_version": env.get("torch_version"),
        "libtorch_variant": "cuda-release" if env.get("cuda_is_available") else "cpu-release",
        "gpu_name": env.get("gpu_name"),
        "gpu_compute_capability": env.get("gpu_compute_capability"),
        "nvidia_driver_version": env.get("nvidia_driver_version"),
        "cuda_toolkit_version": env.get("cuda_toolkit_version"),
        "model_name": model_name,
        "state_dict_sha256": sha256_of_file(state_dict_path) if state_dict_path.exists() else None,
        "artifact_sha256": sha256_of_file(artifact_path) if artifact_path.exists() else None,
        "input_shape": list(example_input.shape),
        "input_dtype": "float32",
        "dynamic_shape": False,
        "export_started_at": started_at.isoformat(),
        "export_finished_at": finished_at.isoformat(),
        "export_duration_ms": (finished_at - started_at).total_seconds() * 1000.0,
        "artifact_size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
        "status": status,
        "error_log": error_log,
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(metadata: dict, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
