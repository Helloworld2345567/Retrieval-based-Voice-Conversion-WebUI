"""Compatibility exports for the former combined inference runtime."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from rvc.core.devices import DeviceConfig, _cuda_profile, _directml_device, select_device
from rvc.core.paths import PROJECT_ROOT, index_roots, resource_path
from rvc.vc.config import VoiceConversionConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Legacy VC configuration shape backed by the canonical VC runtime."""

    device: Any
    dtype: Any
    is_half: bool
    cuda_graph: bool
    n_cpu: int
    gpu_mem: int | None
    x_pad: int
    x_query: int
    x_center: int
    x_max: int
    hubert_root: Path
    rmvpe_root: Path
    weight_root: Path
    index_roots: tuple[Path, ...]

    @classmethod
    def create(cls, device="auto", device_id=None, *, resource_base=None):
        config = VoiceConversionConfig.create(device, device_id, resource_base=resource_base)
        memory = None
        if str(config.device).startswith("cuda"):
            try:
                import torch

                memory = int(torch.cuda.get_device_properties(config.device).total_memory / (1024**3) + 0.4)
            except Exception:
                memory = None
        return cls(
            device=config.device,
            dtype=config.dtype,
            is_half=config.is_half,
            cuda_graph=config.cuda_graph,
            n_cpu=os.cpu_count() or 1,
            gpu_mem=memory,
            x_pad=config.x_pad,
            x_query=config.x_query,
            x_center=config.x_center,
            x_max=config.x_max,
            hubert_root=config.hubert_root,
            rmvpe_root=config.rmvpe_root,
            weight_root=resource_path("weight_root", "assets/weights", base=resource_base),
            index_roots=index_roots(base=resource_base),
        )


__all__ = [
    "DeviceConfig",
    "InferenceConfig",
    "PROJECT_ROOT",
    "VoiceConversionConfig",
    "index_roots",
    "resource_path",
    "select_device",
]
