"""Domain-neutral device selection with lazy hardware imports."""

from dataclasses import dataclass
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


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


def _select_device_profile(device="auto", device_id=None):
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


def select_device(device="auto", device_id=None):
    """Return ``(device, dtype, memory GiB)`` using RVC GPU eligibility rules."""
    return _select_device_profile(device, device_id)


@dataclass(frozen=True)
class DeviceConfig:
    """The shared device and precision values consumed by inference domains."""

    device: Any
    dtype: Any
    is_half: bool

    @classmethod
    def create(cls, device="auto", device_id=None):
        import torch

        selected, dtype, _ = _select_device_profile(device, device_id)
        return cls(device=selected, dtype=dtype, is_half=dtype == torch.float16)
