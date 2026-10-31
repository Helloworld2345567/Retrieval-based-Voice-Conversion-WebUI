"""Voice-conversion-specific runtime policy and resources."""

from dataclasses import dataclass
from pathlib import Path

from rvc.core.devices import DeviceConfig, select_device
from rvc.core.paths import resource_path


def configure_cuda_graph(device):
    """Load CUDA Graph support only while creating a VC runtime."""
    from .cuda_graph import configure_cuda_graph as configure

    return configure(device)


@dataclass(frozen=True)
class VoiceConversionConfig(DeviceConfig):
    x_pad: int
    x_query: int
    x_center: int
    x_max: int
    hubert_root: Path
    rmvpe_root: Path
    cuda_graph: bool

    @classmethod
    def create(cls, device="auto", device_id=None, *, resource_base=None):
        import torch

        selected, dtype, memory = select_device(device, device_id)
        is_half = dtype == torch.float16
        segments = (3, 10, 60, 65) if is_half else (1, 6, 38, 41)
        if memory is not None and int(memory) <= 4:
            segments = (1, 5, 30, 32)
        return cls(
            device=selected,
            dtype=dtype,
            is_half=is_half,
            x_pad=segments[0],
            x_query=segments[1],
            x_center=segments[2],
            x_max=segments[3],
            hubert_root=resource_path("hubert_root", "assets/hubert_base", base=resource_base),
            rmvpe_root=resource_path("rmvpe_root", "assets/rmvpe", base=resource_base),
            cuda_graph=configure_cuda_graph(selected),
        )
