import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rvc.separation import catalog, cli, download, service, worker
from rvc.separation.request import AudioEncodingParams, InferenceParams, SeparationRequest


def test_five_presets_and_legacy_aliases():
    assert len(catalog.MODELS) == 5
    assert {model.model_type for model in catalog.MODELS} == {"bs_roformer", "mel_band_roformer"}
    for model in catalog.MODELS:
        assert catalog.get_model(model.filename.upper()) is model
        assert catalog.get_model(Path(model.filename).stem) is model
        assert model.config_path.is_file()
    with pytest.raises(ValueError):
        catalog.get_model("htdemucs")


@pytest.mark.parametrize("module", ["rvc.separation.cli", "tools.pymss.cli"])
def test_listing_without_heavy_dependencies(module):
    code = f"import runpy,sys; sys.modules['torch']=None; sys.modules['av']=None; sys.argv=['{module}','list','--json']; runpy.run_module('{module}',run_name='__main__')"
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 5


def test_encoding_defaults_and_cuda_ids():
    args = cli.build_parser(legacy=True).parse_args(["infer", "vocals", "-i", "in.wav", "--device-id", "0", "--device-id", "1"])
    assert args.device_ids == [0, 1]
    assert (args.wav_bit_depth, args.flac_bit_depth, args.mp3_bit_rate, args.m4a_bit_rate) == ("FLOAT", "PCM_16", "320k", "512k")


def separation_request(**updates):
    values = {
        "model": "vocals",
        "inputs": ("input.wav",),
        "output": "results",
        "format": "wav",
        "params": InferenceParams(),
        "audio_params": AudioEncodingParams(),
    }
    values.update(updates)
    return SeparationRequest(**values)


def test_typed_request_round_trip_and_validation():
    request = separation_request(
        inputs=("音频.wav",),
        params=InferenceParams(chunk_size=128, overlap_size=32, model_dtype="fp16"),
    )
    restored = SeparationRequest.from_json(request.to_json())
    assert restored == request
    assert restored.params.model_dtype == "float16"
    assert Path(restored.inputs[0]).is_absolute()
    assert Path(restored.output).is_absolute()
    with pytest.raises(ValueError, match="smaller"):
        InferenceParams(chunk_size=128, overlap_size=128)
    with pytest.raises(ValueError, match="Unsupported inference"):
        InferenceParams.from_mapping({"unknown": 1})
    with pytest.raises(ValueError, match="path string"):
        SeparationRequest.from_dict({"model": "vocals", "inputs": [123], "output": "results", "format": "wav"})


@pytest.mark.parametrize("arguments", [["serve"], ["workflow"], ["ensemble"], ["infer", "vocals", "-i", "input", "--device-id", "-1"], ["infer", "vocals", "-i", "input", "--param", "broken"], ["infer", "vocals", "-i", "input", "--param", "batch_size=0"]])
def test_invalid_parameters_exit_two(arguments):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_output_names_and_collision(tmp_path):
    spec = catalog.get_model("lead-vocal")
    assert service.output_paths(spec, "voice.wav", tmp_path, "flac", False, False)[0].name == "voice_main_vocal.flac"
    assert service.output_paths(spec, "voice.wav", tmp_path, "flac", False, True)[0].name == "voice_karaoke.flac"
    request = {"model": "vocals", "inputs": ["a/song.wav", "b/song.mp3"], "output": str(tmp_path), "format": "wav", "save_as_folder": False, "legacy": False}
    with pytest.raises(ValueError, match="same output"):
        service.validate_outputs(request)
    request["inputs"] = [str(tmp_path / "input" / "song.wav")]
    service.validate_outputs(request)
    request["inputs"] = [str(tmp_path / "song.wav"), str(tmp_path / "song_vocals.wav")]
    with pytest.raises(ValueError, match="overwrite an input"):
        service.validate_outputs(request)


def test_input_directory_and_explicit_old_model_cache(tmp_path):
    (tmp_path / "音频 文件.WAV").write_bytes(b"audio")
    (tmp_path / "notes.txt").write_text("not audio")
    assert len(service.collect_inputs(tmp_path)) == 1
    spec = catalog.get_model("vocals")
    old = tmp_path / spec.remote_path / spec.filename
    old.parent.mkdir(parents=True)
    old.write_bytes(b"existing weight")
    assert catalog.model_path(spec, tmp_path) == old


def test_missing_download_is_validated_and_cleaned(tmp_path, monkeypatch):
    from io import BytesIO
    spec = catalog.get_model("vocals")
    monkeypatch.setattr(download, "cache_root", lambda: tmp_path / ".codex")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: BytesIO(b"incomplete"))
    with pytest.raises(ValueError, match="Incomplete"):
        download.download_model(spec, tmp_path / "weights")
    assert not (tmp_path / "weights" / spec.filename).exists()
    assert not (tmp_path / ".codex" / "downloads").exists()


def test_existing_weight_never_downloaded(tmp_path, monkeypatch):
    spec = catalog.get_model("vocals")
    path = tmp_path / spec.filename
    path.write_bytes(b"existing")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert download.download_model(spec, tmp_path) == path


def test_download_all_attempts_each_model(monkeypatch):
    attempted = []
    def fake(spec, *args, **kwargs):
        attempted.append(spec.name)
        if len(attempted) == 1:
            raise OSError("download failed")
        return spec.filename
    monkeypatch.setattr(download, "download_model", fake)
    assert cli.main(["download", "all"]) == 1
    assert len(attempted) == 5


def test_fixed_configs_are_inference_only():
    from rvc.separation.config import load_config
    for spec in catalog.MODELS:
        config = load_config(spec.config_path)
        assert "training" not in config and "augmentations" not in config
        assert set(config.separation.instruments) == set(spec.stems)
        assert config.separation.target_instrument == spec.stems[0]
        assert not any("loss" in key for key in config.model)


def test_batch_failure_is_counted(tmp_path, monkeypatch):
    import numpy as np
    from rvc.separation import audio_io, separator
    class FakeSeparator:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def separate(self, mix, **kwargs): return {"vocals": mix.T, "instrumental": mix.T}
    monkeypatch.setattr(separator, "MSSeparator", FakeSeparator)
    def load(path, **kwargs):
        if "bad" in path:
            raise ValueError("broken input")
        return np.zeros((2, 32), dtype=np.float32), 44100
    monkeypatch.setattr(audio_io, "load_audio", load)
    monkeypatch.setattr(audio_io, "save_audio", lambda path, *args: Path(path).write_bytes(b"audio"))
    request = {"model": "vocals", "model_path": "unused", "inputs": ["ok.wav", "bad.wav"], "output": str(tmp_path), "format": "wav", "save_as_folder": False, "legacy": False, "device_ids": [0], "params": {}, "tta": False, "audio_params": {}}
    result = service.run_batch(request, SimpleNamespace())
    assert result["successful"] == 1 and result["failed"] == 1
    assert len(result["outputs"]) == 2


def test_directml_fallback_fresh_process_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(worker, "cache_root", lambda: tmp_path / ".codex")
    monkeypatch.setattr(worker, "is_source_tree", lambda: True)
    requests = []
    class FakeProcess:
        def __init__(self, args, **kwargs):
            request = json.loads(Path(args[-1]).read_text())
            requests.append(request)
            self.code = 75 if len(requests) == 1 else 0
            Path(request["result_path"]).write_text(json.dumps({"successful": 1, "failed": 0}))
        def wait(self): return self.code
        def poll(self): return self.code
    monkeypatch.setattr(worker.subprocess, "Popen", FakeProcess)
    result = worker.run_directml(separation_request())
    assert result["successful"] == 1
    assert [r["params"]["model_dtype"] for r in requests] == ["float16", "float32"]
    assert requests[0]["allow_fp32_retry"] and not requests[1]["allow_fp32_retry"]
    assert not (tmp_path / ".codex" / "separation").exists()


def test_directml_worker_cannot_overwrite_request(tmp_path):
    request_path = tmp_path / "request.json"
    request = separation_request(result_path=str(request_path))
    request_path.write_text(request.to_json(), encoding="utf-8")
    original = request_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="worker result file"):
        worker.main([str(request_path)])
    assert request_path.read_text(encoding="utf-8") == original


def test_directml_interrupt_cleans_worker_and_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "cache_root", lambda: tmp_path / ".codex")
    monkeypatch.setattr(worker, "is_source_tree", lambda: False)
    terminated = []
    class Interrupted:
        def __init__(self, *a, **k): pass
        def wait(self): raise KeyboardInterrupt
    monkeypatch.setattr(worker.subprocess, "Popen", Interrupted)
    monkeypatch.setattr(worker, "terminate_worker", lambda process: terminated.append(process))
    with pytest.raises(KeyboardInterrupt):
        worker.run_directml(separation_request())
    assert len(terminated) == 1
    assert not (tmp_path / ".codex" / "separation").exists()


def test_oom_is_not_precision_retry():
    assert not service.fp16_retryable(RuntimeError("E_OUTOFMEMORY"))
    assert service.fp16_retryable(NotImplementedError("FP16 operator unsupported"))


def test_audio_bitrate_validation_matches_encoder():
    from rvc.separation.audio_io import _bitrate_to_int

    assert _bitrate_to_int("320k") == 320000
    assert _bitrate_to_int("1.5m") == 1500000
    assert AudioEncodingParams(mp3_bit_rate="1m").mp3_bit_rate == "1m"
    with pytest.raises(ValueError, match="positive bitrate"):
        AudioEncodingParams(mp3_bit_rate="1.5")


def test_invalid_audio_encoding_is_cli_error(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"audio")
    with pytest.raises(SystemExit) as error:
        cli.main(["infer", "vocals", "-i", str(path), "--mp3-bit-rate", "broken"])
    assert error.value.code == 2


def test_invalid_chunking_rejected_before_download(tmp_path, monkeypatch):
    from rvc.core.devices import DeviceConfig
    monkeypatch.setattr(DeviceConfig, "create", lambda **kwargs: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(download, "download_model", lambda *a, **k: pytest.fail("invalid request downloaded a model"))
    path = tmp_path / "input.wav"
    path.write_bytes(b"audio")
    with pytest.raises(SystemExit) as error:
        cli.main(["infer", "vocals", "-i", str(path), "--param", "chunk_size=128"])
    assert error.value.code == 2
