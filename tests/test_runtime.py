import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rvc import runtime


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
    monkeypatch.setattr(runtime, "_directml_device", lambda *_: pytest.fail("Unexpected DirectML probe"))
    assert runtime.select_device()[:2] == ("cuda:1", "float16")
    assert runtime.select_device("cuda", 0)[:2] == ("cuda:0", "float32")


def test_auto_directml_then_cpu_and_explicit_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(runtime, "_directml_device", lambda *args: "privateuseone:0")
    assert runtime.select_device()[:2] == ("privateuseone:0", "float32")

    def unavailable(*args):
        raise RuntimeError("No adapter")

    monkeypatch.setattr(runtime, "_directml_device", unavailable)
    assert runtime.select_device()[:2] == ("cpu", "float32")
    with pytest.raises(RuntimeError, match="DirectML device is unavailable"):
        runtime.select_device("directml")
    with pytest.raises(RuntimeError, match="No eligible CUDA"):
        runtime.select_device("cuda")


def test_explicit_cpu_does_not_probe_accelerators(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(runtime, "_directml_device", lambda *_: pytest.fail("Unexpected device probe"))
    assert runtime.select_device("cpu") == ("cpu", "float32", None)
    with pytest.raises(ValueError, match="nonnegative"):
        runtime.select_device("cuda", -1)


def test_config_creation_has_no_process_side_effects(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "rvc.cuda_graph", SimpleNamespace(configure_cuda_graph=lambda device: False))
    before = (os.getcwd(), list(sys.argv), dict(os.environ))
    config = runtime.InferenceConfig.create("cpu")
    assert (os.getcwd(), sys.argv, dict(os.environ)) == before
    assert (config.x_pad, config.x_query, config.x_center, config.x_max) == (1, 6, 38, 41)
    assert config.hubert_root == runtime.PROJECT_ROOT / "assets/hubert_base"


def test_runtime_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import rvc.runtime; assert 'torch' not in sys.modules; assert 'torch_directml' not in sys.modules"],
        cwd=runtime.PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
