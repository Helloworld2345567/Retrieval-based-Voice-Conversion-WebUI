"""Numerical regressions against the original models, using synthetic weights.

Requires the original Git object below; no pretrained weights or downloads.
CPU: python -m pytest tests/test_model_regression.py
DirectML smoke: set RVC_TEST_DIRECTML=1 and run the same command on Windows.
"""
import importlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
RVC_NAMES = (
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs768NSFsid_nono",
)
MODEL_NAMES = (*RVC_NAMES, "BSRoformer", "MelBandRoformer")
RVC_CONFIG = dict(
    spec_channels=17, segment_size=4, inter_channels=8, hidden_channels=8,
    filter_channels=16, n_heads=2, n_layers=1, kernel_size=3, p_dropout=0.0,
    resblock="1", resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]],
    upsample_rates=[2, 2], upsample_initial_channel=16, upsample_kernel_sizes=[4, 4],
    spk_embed_dim=3, gin_channels=4, sr=40000, is_half=False,
)
ROFORMER_CONFIG = dict(
    dim=8, depth=1, stereo=True, num_stems=1, time_transformer_depth=1,
    freq_transformer_depth=1, dim_head=4, heads=2, attn_dropout=0.0,
    ff_dropout=0.0, flash_attn=False, stft_n_fft=64, stft_hop_length=16,
    stft_win_length=64, mask_estimator_depth=1,
)


def _model_class(name, original=False):
    if name in RVC_NAMES:
        module = "infer.module.models" if original else "rvc.vc.models"
    else:
        prefix = "pymss_core.modules" if original else "rvc.separation.models"
        leaf = "bs_roformer" if name == "BSRoformer" else "mel_band_roformer"
        module = prefix + ".bs_roformer." + leaf
    return getattr(importlib.import_module(module), name)


def _run(model, fixture, device="cpu"):
    inputs = tuple(value.to(device) for value in fixture["inputs"])
    torch.manual_seed(fixture["seed"])
    # Match the runtime's DirectML path: this backend does not support
    # inference tensors, but supports the same graph under no_grad.
    context = torch.no_grad() if str(device).startswith("privateuseone") else torch.inference_mode()
    with context:
        output = model.infer(*inputs)[0] if fixture["name"] in RVC_NAMES else model(*inputs)
    return output.detach().cpu()


def _capture_original(source_root, output_dir):
    """Run in a fresh subprocess so old imports cannot contaminate new imports."""
    sys.path[:0] = [str(source_root), str(source_root / "tools")]
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        if name in RVC_NAMES:
            config = dict(RVC_CONFIG)
            width = 768 if "768" in name else 256
            torch.manual_seed(202)
            phone, lengths, sid = torch.randn(1, 12, width), torch.tensor([12]), torch.tensor([1])
            if name.endswith("_nono"):
                inputs = (phone, lengths, sid)
            else:
                inputs = (phone, lengths, torch.full((1, 12), 50), torch.full((1, 12), 220.0), sid)
        else:
            config = dict(ROFORMER_CONFIG)
            if name == "BSRoformer":
                config["freqs_per_bands"] = (8, 8, 8, 9)
            else:
                config.update(num_bands=4, sample_rate=16000, match_input_audio_length=True)
            torch.manual_seed(202)
            inputs = (torch.randn(1, 2, 256) * 0.1,)
        torch.manual_seed(101)
        model = _model_class(name, original=True)(**config).eval()
        fixture = dict(name=name, kwargs=config, inputs=inputs, seed=7321)
        fixture["output"] = _run(model, fixture)
        assert torch.isfinite(fixture["output"]).all(), name
        # Exported RVC inference checkpoints omit this training-only encoder.
        fixture["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("enc_q.")
        }
        torch.save(fixture, output_dir / (name + ".pt"))


@pytest.fixture(scope="module")
def original_fixtures(tmp_path_factory):
    archive_paths = [
        "infer/module", "tools/pymss_core/__init__.py", "tools/pymss_core/checkpoint.py",
        "tools/pymss_core/config.py", "tools/pymss_core/utils.py",
        "tools/pymss_core/modules/__init__.py", "tools/pymss_core/modules/_dsp.py",
        "tools/pymss_core/modules/bs_roformer",
    ]
    result = subprocess.run(
        ["git", "archive", "--format=tar", BASELINE_COMMIT, *archive_paths],
        cwd=PROJECT_ROOT, capture_output=True, check=False,
    )
    assert result.returncode == 0, (
        "Model regression requires Git object " + BASELINE_COMMIT + "; fetch repository history.\n"
        + result.stderr.decode(errors="replace")
    )
    temp_root = tmp_path_factory.mktemp("original-models")
    source_root, output_dir = temp_root / "source", temp_root / "results"
    source_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        archive.extractall(source_root, filter="data")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--capture", str(source_root), str(output_dir)],
        cwd=temp_root, capture_output=True, text=True, env=env, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output_dir


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_cpu_output_matches_original(original_fixtures, name, record_property):
    torch.set_num_threads(1)
    fixture = torch.load(original_fixtures / (name + ".pt"), map_location="cpu", weights_only=True)
    model = _model_class(name)(**fixture["kwargs"]).eval()
    model.load_state_dict(fixture["state_dict"], strict=True)
    output = _run(model, fixture)
    assert torch.isfinite(output).all()
    assert output.shape == fixture["output"].shape
    difference = float((output - fixture["output"]).abs().max())
    record_property("maximum_absolute_difference", difference)
    torch.testing.assert_close(output, fixture["output"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_directml_synthetic_inference(original_fixtures, name):
    if os.environ.get("RVC_TEST_DIRECTML") != "1":
        pytest.skip("Set RVC_TEST_DIRECTML=1 to exercise an actual DirectML adapter")
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())
    fixture = torch.load(original_fixtures / (name + ".pt"), map_location="cpu", weights_only=True)
    model = _model_class(name)(**fixture["kwargs"]).eval()
    model.load_state_dict(fixture["state_dict"], strict=True)
    model = model.to(device)
    output = _run(model, fixture, device)
    assert output.shape == fixture["output"].shape
    assert torch.isfinite(output).all()
    # RoFormer inference is deterministic; RVC synthesis intentionally samples noise.
    if name not in RVC_NAMES:
        torch.testing.assert_close(output, fixture["output"], rtol=5e-4, atol=1e-5)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--capture":
        _capture_original(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("This module is run with pytest, or --capture SOURCE OUTPUT.")
