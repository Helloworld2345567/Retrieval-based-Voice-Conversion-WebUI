from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from rvc.separation import audio_io, catalog, service
from rvc.separation.models.bs_roformer import BSRoformer, MelBandRoformer


@pytest.mark.parametrize("extension", ["wav", "flac", "mp3", "m4a"])
def test_real_codec_roundtrip(tmp_path, extension):
    rate = 44100
    wave = (0.2 * np.sin(2 * np.pi * 440 * np.arange(4410) / rate)).astype(np.float32)
    audio = np.stack([wave, wave], axis=-1)
    path = tmp_path / f"中文 音频.{extension}"
    audio_io.save_audio(str(path), audio, rate, extension,
                        {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_16", "mp3_bit_rate": "320k", "m4a_bit_rate": "512k", "m4a_codec": "aac"})
    restored, restored_rate = audio_io.load_audio(str(path), sr=rate)
    assert restored_rate == rate and restored.shape[0] == 2
    assert restored.shape[1] >= len(wave)
    assert np.isfinite(restored).all() and np.max(np.abs(restored)) > 0.1


@pytest.mark.parametrize("kind", ["bs_roformer", "mel_band_roformer"])
def test_tiny_checkpoint_file_inference_and_stem_batches(tmp_path, monkeypatch, kind):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = dict(dim=8, depth=1, stereo=True, num_stems=1,
                      time_transformer_depth=1, freq_transformer_depth=1,
                      dim_head=4, heads=2, attn_dropout=0.0, ff_dropout=0.0,
                      flash_attn=False, stft_n_fft=64, stft_hop_length=16,
                      stft_win_length=64, mask_estimator_depth=1)
        if kind == "bs_roformer":
            kwargs["freqs_per_bands"] = [8, 8, 8, 9]
            model = BSRoformer(**kwargs)
        else:
            kwargs.update(num_bands=4, sample_rate=44100, zero_dc=False)
            model = MelBandRoformer(**kwargs)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({"audio": {"chunk_size": 256, "sample_rate": 44100},
                                               "model": kwargs,
                                               "separation": {"instruments": ["vocals", "instrumental"], "target_instrument": "vocals", "use_amp": False},
                                               "inference": {"batch_size": 1}}))
        spec = replace(catalog.get_model("vocals"), config_file=str(config_file), model_type=kind)
        monkeypatch.setattr(service, "get_model", lambda value: spec)
        checkpoint = tmp_path / "weights.ckpt"
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        del model
        source = tmp_path / "input.wav"
        wave = np.random.default_rng(1).normal(0, 0.1, (768, 2)).astype(np.float32)
        audio_io.save_audio(str(source), wave, 44100, "wav", {"wav_bit_depth": "FLOAT"})
        request = {"model": "vocals", "model_path": str(checkpoint), "inputs": [str(source)],
                   "output": str(tmp_path / "out"), "format": "wav", "save_as_folder": False,
                   "legacy": False, "device_ids": [0], "params": {"chunk_size": 256, "overlap_size": 64, "batch_size": 1, "stem_batch_size": 1},
                   "tta": False, "audio_params": {"wav_bit_depth": "FLOAT"}}
        result = service.run_batch(request, SimpleNamespace(device=torch.device("cpu"), is_half=False))
        assert result["successful"] == 1 and result["failed"] == 0
        outputs = [audio_io.load_audio(path)[0] for path in result["outputs"]]
        assert len(outputs) == 2 and all(output.shape == (2, 768) for output in outputs)
        np.testing.assert_allclose(outputs[0] + outputs[1], wave.T, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
