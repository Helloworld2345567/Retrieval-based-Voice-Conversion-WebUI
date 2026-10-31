"""Audio decoding and encoding shared by offline voice conversion."""

from io import BytesIO
import os
from pathlib import Path

import numpy as np

from .runtime import PROJECT_ROOT


def _ffmpeg_command(name):
    executable = PROJECT_ROOT / (name + ".exe" if os.name == "nt" else name)
    return str(executable) if executable.is_file() else name


def wav2(source, destination, output_format):
    """Transcode audio with PyAV, preserving the source rate and channels."""
    import av

    container_format = "mp4" if output_format == "m4a" else output_format
    codec = {"mp4": "aac", "ogg": "libvorbis"}.get(container_format, container_format)
    with av.open(source, "r") as incoming:
        if not incoming.streams.audio:
            raise ValueError("Input contains no audio stream")
        input_stream = incoming.streams.audio[0]
        with av.open(destination, "w", format=container_format) as outgoing:
            rate = input_stream.codec_context.sample_rate
            stream = outgoing.add_stream(codec, rate=rate) if rate else outgoing.add_stream(codec)
            channels = input_stream.codec_context.channels
            if channels in (1, 2):
                stream.layout = "mono" if channels == 1 else "stereo"
            for frame in incoming.decode(input_stream):
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
            for packet in stream.encode(None):
                outgoing.mux(packet)


def write_audio(path, audio, sample_rate, output_format):
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format in ("wav", "flac"):
        sf.write(str(path), audio, sample_rate, format=output_format)
    else:
        with BytesIO() as wav_file:
            sf.write(wav_file, audio, sample_rate, format="wav")
            wav_file.seek(0)
            with path.open("wb") as output_file:
                wav2(wav_file, output_file, output_format)


def _probe_audio(path):
    import ffmpeg

    streams = ffmpeg.probe(os.fspath(path), cmd=_ffmpeg_command("ffprobe")).get("streams", [])
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("Input contains no audio stream: %s" % path)
    return int(audio["sample_rate"]), max(1, int(audio.get("channels", 1)))


def _decode_ffmpeg(path, sample_rate=None, mono=True):
    import ffmpeg

    if sample_rate is None or not mono:
        source_rate, channels = _probe_audio(path)
        sample_rate = source_rate if sample_rate is None else sample_rate
    else:
        channels = 1
    channels = 1 if mono else channels
    output, _ = (
        ffmpeg.input(os.fspath(path), threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=channels, ar=sample_rate)
        .run(cmd=[_ffmpeg_command("ffmpeg"), "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    samples = np.frombuffer(output, np.float32)
    samples = samples[: samples.size - samples.size % channels]
    audio = samples if mono else samples.reshape(-1, channels).T.copy()
    return audio, sample_rate


class AudioLoader:
    """Per-runtime CUDA resampling with the original FFmpeg CPU fallback."""

    def __init__(self, config=None):
        self.config = config
        self.torchaudio = None
        self.resamplers = {}
        if (
            config is not None
            and str(config.device).startswith("cuda")
            and os.environ.get("RVC_AUDIO_FORCE_CPU") != "1"
        ):
            try:
                import torchaudio

                self.torchaudio = torchaudio
            except (ImportError, OSError):
                pass

    def _resample_cuda(self, audio, source_rate, target_rate, mono):
        import torch

        if not torch.is_tensor(audio):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            if source_rate == target_rate:
                if mono:
                    audio = audio.mean(dim=0, keepdim=True)
                audio = audio.detach().float().cpu().contiguous().numpy()
                return audio[0] if mono else audio
            audio = audio.to(device=self.config.device, dtype=self.config.dtype)
            if mono:
                audio = audio.mean(dim=0, keepdim=True)
            key = (source_rate, target_rate)
            if key not in self.resamplers:
                self.resamplers[key] = self.torchaudio.transforms.Resample(
                    source_rate, target_rate, dtype=self.config.dtype
                ).to(self.config.device)
            audio = self.resamplers[key](audio)
            audio = audio.detach().float().cpu().contiguous().numpy()
        return audio[0] if mono else audio

    def load(self, path, sample_rate=16000, mono=True):
        if self.torchaudio is not None:
            try:
                audio, source_rate = self.torchaudio.load(os.fspath(path))
                return self._resample_cuda(audio, source_rate, sample_rate, mono)
            except Exception:
                try:
                    audio, source_rate = _decode_ffmpeg(path, mono=False)
                    return self._resample_cuda(audio, source_rate, sample_rate, mono)
                except Exception:
                    pass
        try:
            return _decode_ffmpeg(path, sample_rate, mono)[0]
        except Exception as error:
            raise RuntimeError("Failed to load audio %s: %s" % (path, error)) from error

    def close(self):
        self.resamplers.clear()


def load_audio(path, sample_rate, force_mono=True, *, config=None):
    return AudioLoader(config).load(path, sample_rate, force_mono)
