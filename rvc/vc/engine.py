"""Reusable offline voice conversion, independent of command-line/UI state."""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import torch

from rvc.audio import AudioLoader
from rvc.cuda_graph import clear_cuda_graph_cache
from .checkpoints import load_checkpoint, select_speaker
from .hubert import load_hubert_model
from .models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from .pipeline import Pipeline


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioResult:
    sample_rate: int
    audio: np.ndarray
    timings: tuple[float, float, float]


class VoiceConverter:
    def __init__(self, model_path, config, *, checkpoint=None):
        self.config = config
        self.model_path = Path(model_path)
        self.net_g = None
        self.hubert_model = None
        self.pipeline = None
        self.audio_loader = AudioLoader(config)
        self.closed = False
        if checkpoint is None:
            checkpoint, self.metadata = load_checkpoint(self.model_path)
        else:
            from .checkpoints import metadata_from_checkpoint

            self.metadata = metadata_from_checkpoint(checkpoint)
        constructors = {
            ("v1", True): SynthesizerTrnMs256NSFsid,
            ("v1", False): SynthesizerTrnMs256NSFsid_nono,
            ("v2", True): SynthesizerTrnMs768NSFsid,
            ("v2", False): SynthesizerTrnMs768NSFsid_nono,
        }
        model_config = list(checkpoint["config"])
        model_config[-3] = self.metadata.speaker_count
        model_config[-1] = self.metadata.sample_rate
        try:
            self.net_g = constructors[(self.metadata.version, self.metadata.uses_f0)](
                *model_config, is_half=config.is_half
            )
            loaded = self.net_g.load_state_dict(checkpoint["weight"], strict=False)
            if loaded.missing_keys:
                raise ValueError("Model is missing inference weights: %s" % ", ".join(loaded.missing_keys))
            unexpected = [key for key in loaded.unexpected_keys if not key.startswith("enc_q.")]
            if unexpected:
                logger.warning("Unused checkpoint weights: %s", ", ".join(unexpected))
            self.net_g.eval().to(device=config.device, dtype=config.dtype)
            self.pipeline = Pipeline(self.metadata.sample_rate, config)
        except Exception:
            self.close()
            raise

    @torch.no_grad()
    def convert(
        self,
        input_path,
        *,
        speaker_id=None,
        pitch=0,
        f0_method="rmvpe",
        index_path=None,
        index_rate=0.75,
        resample_sr=0,
        rms_mix_rate=1.0,
        protect=0.33,
    ):
        if self.closed:
            raise RuntimeError("VoiceConverter is closed")
        speaker_id = select_speaker(self.metadata, speaker_id)
        if f0_method not in ("pm", "rmvpe", "fcpe"):
            raise ValueError("Unsupported F0 method: %s" % f0_method)
        if not 0 <= index_rate <= 1 or not 0 <= rms_mix_rate <= 1 or not 0 <= protect <= 0.5:
            raise ValueError("Invalid index, RMS or protection ratio")
        if resample_sr != 0 and resample_sr < 16000:
            raise ValueError("Resample rate must be 0 or at least 16000")
        audio = self.audio_loader.load(input_path, 16000, mono=True)
        if audio.size <= 18 or not np.isfinite(audio).all():
            raise ValueError("Input audio is empty, too short or contains non-finite samples")
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max
        if self.hubert_model is None:
            self.hubert_model = load_hubert_model(
                self.config.hubert_root, self.config.device, self.config.is_half
            )
        times = [0.0, 0.0, 0.0]
        output = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            speaker_id,
            audio,
            times,
            int(pitch),
            f0_method,
            str(index_path) if index_path else "",
            index_rate,
            int(self.metadata.uses_f0),
            self.metadata.sample_rate,
            resample_sr,
            rms_mix_rate,
            self.metadata.version,
            protect,
        )
        sample_rate = resample_sr if resample_sr >= 16000 else self.metadata.sample_rate
        if output.size == 0:
            raise RuntimeError("Voice conversion produced empty audio")
        return AudioResult(sample_rate, output, tuple(times))

    def close(self):
        for owner in (self.net_g, self.hubert_model):
            clear_cuda_graph_cache(owner)
        if self.pipeline is not None:
            for name in ("model_rmvpe", "model_fcpe"):
                predictor = getattr(self.pipeline, name, None)
                if predictor is not None:
                    for owner in (predictor, getattr(predictor, "model", None), getattr(predictor, "mel_extractor", None), getattr(predictor, "infer_model", None)):
                        clear_cuda_graph_cache(owner)
        self.pipeline = self.net_g = self.hubert_model = None
        self.audio_loader.close()
        self.closed = True
        if str(self.config.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
