"""Explicit inference configuration without import-time hardware initialization."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def resource_path(variable, relative, *, base=None):
    """Resolve legacy resource overrides without changing the process environment."""
    value = Path(os.environ.get(variable, str(PROJECT_ROOT / relative))).expanduser()
    if not value.is_absolute() and base is not None:
        value = Path(base) / value
    return value.resolve()


def index_roots(*, base=None):
    return (
        resource_path("outside_index_root", "assets/indices", base=base),
        resource_path("index_root", "logs", base=base),
    )


def _cuda_profile(torch, index):
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        memory = torch.cuda.get_device_properties(index).total_memory / (1024**3) + 0.4
    except Exception:
        logger.debug("Cannot inspect CUDA device %s", index, exc_info=True)
        return None
    sm = major + minor / 10.0
    if memory < 4 or sm < 5.3 or (5.3 <= sm < 6.1):
        return None
    fp32 = sm == 6.1 or (sm == 7.5 and re.search(r"16\d{2}", name))
    return torch.device(f"cuda:{index}"), torch.float32 if fp32 else torch.float16, sm, memory


def _directml_device(torch, device_id=None):
    import torch_directml

    selected = torch_directml.default_device() if device_id is None else device_id
    device = torch_directml.device(selected)
    probe = torch.ones(1, dtype=torch.float32).to(device)
    (probe + 1).cpu()
    return device


def select_device(device="auto", device_id=None):
    """Return (device, dtype, memory GiB) using the existing RVC GPU eligibility."""
    import torch

    if device not in {"auto", "cpu", "cuda", "directml"}:
        raise ValueError(f"Unsupported device: {device}")
    if device_id is not None and device_id < 0:
        raise ValueError("Device ID must be nonnegative")
    if device == "cpu":
        return torch.device("cpu"), torch.float32, None
    if device in {"auto", "cuda"}:
        indices = [device_id] if device_id is not None else range(torch.cuda.device_count())
        profiles = [profile for index in indices if (profile := _cuda_profile(torch, index))]
        if profiles:
            selected, dtype, _, memory = max(profiles, key=lambda profile: (profile[2], profile[3]))
            return selected, dtype, memory
        if device == "cuda":
            raise RuntimeError("No eligible CUDA device matches the requested device ID")
    try:
        return _directml_device(torch, device_id), torch.float32, None
    except Exception as error:
        if device == "directml":
            raise RuntimeError(f"DirectML device is unavailable: {error}") from error
        logger.debug("DirectML unavailable; selecting CPU", exc_info=True)
    return torch.device("cpu"), torch.float32, None


@dataclass(frozen=True)
class InferenceConfig:
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
        import torch
        from .cuda_graph import configure_cuda_graph

        selected, dtype, memory = select_device(device, device_id)
        is_half = dtype == torch.float16
        segments = (3, 10, 60, 65) if is_half else (1, 6, 38, 41)
        gpu_mem = int(memory) if memory is not None else None
        if gpu_mem is not None and gpu_mem <= 4:
            segments = (1, 5, 30, 32)
        return cls(
            device=selected,
            dtype=dtype,
            is_half=is_half,
            cuda_graph=configure_cuda_graph(selected),
            n_cpu=os.cpu_count() or 1,
            gpu_mem=gpu_mem,
            x_pad=segments[0],
            x_query=segments[1],
            x_center=segments[2],
            x_max=segments[3],
            hubert_root=resource_path("hubert_root", "assets/hubert_base", base=resource_base),
            rmvpe_root=resource_path("rmvpe_root", "assets/rmvpe", base=resource_base),
            weight_root=resource_path("weight_root", "assets/weights", base=resource_base),
            index_roots=index_roots(base=resource_base),
        )
