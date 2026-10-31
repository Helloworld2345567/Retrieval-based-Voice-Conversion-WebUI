from dataclasses import fields
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rvc import runtime
from rvc.core import devices
from rvc.core.devices import DeviceConfig
from rvc.vc.config import VoiceConversionConfig


def fake_torch(profiles=()):
    return SimpleNamespace(
        float16="float16", float32="float32", device=lambda name: name,
        cuda=SimpleNamespace(
            is_available=lambda: bool(profiles),
            device_count=lambda: len(profiles),
            get_device_capability=lambda index: profiles[index][0],
            get_device_name=lambda index: profiles[index][1],
            get_device_properties=lambda index: SimpleNamespace(total_memory=profiles[index][2] * 1024**3),
        ),
    )


@pytest.mark.parametrize("profile,expected", [
    (((8, 6), "RTX 3060", 12), "float16"),
    (((6, 1), "GTX 1080", 8), "float32"),
    (((7, 5), "GTX 1660", 6), "float32"),
    (((7, 5), "RTX 2080", 8), "float16"),
    (((8, 6), "Small GPU", 2), None),
    (((5, 2), "Old GPU", 8), None),
    (((6, 0), "Existing ineligible GPU", 8), None),
])
def test_cuda_precision_preserves_existing_rules(profile, expected):
    result = runtime._cuda_profile(fake_torch([profile]), 0)
    assert (result[1] if result else None) == expected


def test_auto_selects_best_eligible_cuda(monkeypatch):
    torch = fake_torch([((6, 1), "GTX 1080", 8), ((8, 6), "RTX 3060", 12)])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(devices, "_directml_device", lambda *_: pytest.fail("Unexpected DirectML probe"))
    assert runtime.select_device()[:2] == ("cuda:1", "float16")
    assert runtime.select_device("cuda", 0)[:2] == ("cuda:0", "float32")


def test_auto_directml_then_cpu_and_explicit_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(devices, "_directml_device", lambda *args: "privateuseone:0")
    assert runtime.select_device()[:2] == ("privateuseone:0", "float32")

    def unavailable(*args):
        raise RuntimeError("No adapter")

    monkeypatch.setattr(devices, "_directml_device", unavailable)
    assert runtime.select_device()[:2] == ("cpu", "float32")
    with pytest.raises(RuntimeError, match="DirectML device is unavailable"):
        runtime.select_device("directml")
    with pytest.raises(RuntimeError, match="No eligible CUDA"):
        runtime.select_device("cuda")


def test_explicit_cpu_does_not_probe_accelerators(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(devices, "_directml_device", lambda *_: pytest.fail("Unexpected device probe"))
    assert runtime.select_device("cpu") == ("cpu", "float32", None)
    with pytest.raises(ValueError, match="nonnegative"):
        runtime.select_device("cuda", -1)


def test_device_config_creation_has_no_process_side_effects(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    before = (os.getcwd(), list(sys.argv), dict(os.environ))
    config = DeviceConfig.create("cpu")
    assert (os.getcwd(), sys.argv, dict(os.environ)) == before
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.is_half is False
    assert {field.name for field in fields(DeviceConfig)} == {"device", "dtype", "is_half"}


def test_voice_conversion_config_owns_vc_policy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr("rvc.vc.config.configure_cuda_graph", lambda device: False)
    config = VoiceConversionConfig.create("cpu")
    assert (config.x_pad, config.x_query, config.x_center, config.x_max) == (1, 6, 38, 41)
    assert config.hubert_root == runtime.PROJECT_ROOT / "assets/hubert_base"
    assert config.rmvpe_root == runtime.PROJECT_ROOT / "assets/rmvpe"
    assert {field.name for field in fields(VoiceConversionConfig)} == {
        "device", "dtype", "is_half", "x_pad", "x_query", "x_center", "x_max",
        "hubert_root", "rmvpe_root", "cuda_graph",
    }


def test_voice_conversion_config_applies_gpu_policy(monkeypatch):
    torch = fake_torch([((8, 6), "Small eligible GPU", 4)])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(devices, "_directml_device", lambda *_: pytest.fail("Unexpected device probe"))
    configured = []
    monkeypatch.setattr("rvc.vc.config.configure_cuda_graph", lambda device: configured.append(device) or True)
    config = VoiceConversionConfig.create("cuda")
    assert config.is_half is True
    assert (config.x_pad, config.x_query, config.x_center, config.x_max) == (1, 5, 30, 32)
    assert config.cuda_graph is True
    assert configured == ["cuda:0"]


def test_runtime_compatibility_exports_canonical_interfaces():
    from rvc.core import devices
    from rvc.core import paths

    assert runtime.DeviceConfig is devices.DeviceConfig
    assert runtime.select_device is devices.select_device
    assert runtime.resource_path is paths.resource_path


def test_runtime_compatibility_config_preserves_legacy_fields(monkeypatch):
    import inspect

    expected_order = (
        "device", "dtype", "is_half", "cuda_graph", "n_cpu", "gpu_mem",
        "x_pad", "x_query", "x_center", "x_max", "hubert_root", "rmvpe_root",
        "weight_root", "index_roots",
    )
    assert tuple(inspect.signature(runtime.InferenceConfig).parameters) == expected_order
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr("rvc.vc.config.configure_cuda_graph", lambda device: False)
    config = runtime.InferenceConfig.create("cpu")
    assert config.n_cpu >= 1
    assert config.gpu_mem is None
    assert config.weight_root == runtime.PROJECT_ROOT / "assets/weights"
    assert config.index_roots == (
        runtime.PROJECT_ROOT / "assets/indices",
        runtime.PROJECT_ROOT / "logs",
    )


@pytest.mark.parametrize("legacy,canonical", [
    ("rvc.audio", "rvc.vc.audio"),
    ("rvc.convert_cli", "rvc.vc.cli"),
    ("rvc.cuda_graph", "rvc.vc.cuda_graph"),
])
def test_legacy_vc_modules_forward_to_canonical_modules(legacy, canonical):
    assert importlib.import_module(legacy) is importlib.import_module(canonical)


def test_installed_layout_uses_external_mutable_paths(monkeypatch, tmp_path):
    from rvc.core import paths

    monkeypatch.setattr(paths, "is_source_tree", lambda: False)
    monkeypatch.delenv("RVC_DATA_DIR", raising=False)
    monkeypatch.delenv("RVC_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    if os.name == "nt":
        data, cache = tmp_path / "local" / "RVC", tmp_path / "local" / "RVC" / "Cache"
    else:
        data, cache = tmp_path / "data" / "rvc", tmp_path / "cache" / "rvc"
    assert paths.data_root() == data.resolve()
    assert paths.cache_root() == cache.resolve()
    assert paths.resource_path("weight_root", "assets/weights") == (data / "assets" / "weights").resolve()


def test_runtime_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import rvc.runtime; assert 'torch' not in sys.modules; assert 'torch_directml' not in sys.modules"],
        cwd=runtime.PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
