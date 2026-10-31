import json
import subprocess
import av
import numpy as np

from rvc.core.executables import find_executable


def _executable(name):
    return find_executable(name)

def _frame_to_audio(frame, mono):
    audio = frame.to_ndarray()
    audio = audio[None, :] if audio.ndim == 1 else audio
    return (audio.mean(axis=0, keepdims=True) if mono and audio.shape[0] > 1 else audio).astype(np.float32, copy=False)

def _ffmpeg_audio_stream_info(path):
    command = [_executable('ffprobe'), '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels', '-of', 'json', str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout or '{}').get('streams') or []
    if not streams:
        raise ValueError(f'No audio stream found in {path!s}.')
    stream = streams[0]
    return (int(stream['sample_rate']), int(stream['channels']))

def _load_audio_ffmpeg(path, sr=None, mono=False, offset=0.0, duration=None):
    source_rate, source_channels = _ffmpeg_audio_stream_info(path)
    out_rate = int(sr or source_rate)
    channels = 1 if mono else source_channels
    command = [_executable('ffmpeg'), '-nostdin', '-v', 'error']
    if offset:
        command += ['-ss', str(float(offset))]
    command += ['-i', str(path), '-map', '0:a:0', '-vn']
    if duration is not None:
        command += ['-t', str(float(duration))]
    command += ['-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(out_rate), '-ac', str(channels), '-']
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype='<f4')
    complete_samples = audio.size // channels
    audio = audio[:complete_samples * channels]
    audio = audio.reshape(complete_samples, channels).T
    audio = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    return (audio[0] if mono or channels == 1 else audio, out_rate)

def _load_audio_librosa(path, sr=None, mono=False, offset=0.0, duration=None):
    import librosa
    audio, out_rate = librosa.load(path, sr=sr, mono=mono, offset=float(offset or 0.0), duration=None if duration is None else float(duration), dtype=np.float32)
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return (audio[0] if audio.ndim > 1 and (mono or audio.shape[0] == 1) else audio, int(out_rate))

def _load_audio_av(path, sr=None, mono=False, offset=0.0, duration=None):
    chunks = []
    out_rate = None
    with av.open(path) as container:
        stream = container.streams.audio[0]
        out_rate = int(sr or stream.rate)
        resampler = None
        stop_samples = None if duration is None else int(round((offset + duration) * out_rate))
        decoded = 0
        for frame in container.decode(stream):
            if resampler is None:
                resampler = av.AudioResampler(format='fltp', layout=frame.layout.name, rate=out_rate)
            for out in resampler.resample(frame):
                chunks.append((audio := _frame_to_audio(out, mono)))
                decoded += audio.shape[-1]
            if stop_samples is not None and decoded >= stop_samples:
                break
        if resampler is not None:
            for out in resampler.resample(None):
                chunks.append(_frame_to_audio(out, mono))
    start = int(round(offset * out_rate))
    stop = None if duration is None else start + int(round(duration * out_rate))
    channels = 1 if mono else 0
    audio = np.ascontiguousarray((np.concatenate(chunks, axis=-1) if chunks else np.empty((channels, 0), dtype=np.float32))[..., start:stop])
    return (audio[0] if mono or audio.shape[0] == 1 else audio, out_rate)

def load_audio(path, sr=None, mono=False, offset=0.0, duration=None):
    loaders = [('PyAV', _load_audio_av), ('librosa', _load_audio_librosa), ('ffmpeg CLI', _load_audio_ffmpeg)]
    errors = []
    for name, loader in loaders:
        try:
            return loader(path, sr=sr, mono=mono, offset=offset, duration=duration)
        except Exception as e:
            errors.append(f'{name}: {e}')
            continue
    raise RuntimeError(f"All audio loading methods failed: {'; '.join(errors)}")

def _bitrate_to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    multipliers = {'k': 1000, 'm': 1000000}
    return int(float(value[:-1]) * multipliers[value[-1]]) if value[-1:] in multipliers else int(value)

def _format_audio(audio):
    audio = np.asarray(audio)
    audio = np.ascontiguousarray(audio[:, None] if audio.ndim == 1 else audio)
    return np.ascontiguousarray(audio.astype(np.float32).T)

def save_audio(path, audio, sr, output_format, audio_params):
    output_format = output_format.lower()
    audio_array = np.asarray(audio)
    layout = 'stereo' if audio_array.ndim > 1 and audio_array.shape[1] == 2 else 'mono'
    if output_format == 'mp3':
        codec = 'libmp3lame'
    elif output_format == 'm4a':
        codec = audio_params.get('m4a_codec', 'aac')
    elif output_format == 'flac':
        codec = 'flac'
        if audio_params.get('flac_bit_depth', 'PCM_24') == 'PCM_24':
            import soundfile as sf
            return sf.write(path, audio_array, int(sr), format='FLAC', subtype='PCM_24')
    else:
        wav_codecs = {'PCM_16': 'pcm_s16le', 'PCM_24': 'pcm_s24le', 'FLOAT': 'pcm_f32le'}
        codec = wav_codecs.get(audio_params.get('wav_bit_depth', 'FLOAT'), wav_codecs['FLOAT'])
    with av.open(path, 'w') as container:
        stream = container.add_stream(codec, rate=int(sr))
        stream.layout = layout
        if output_format == 'mp3':
            stream.bit_rate = _bitrate_to_int(audio_params.get('mp3_bit_rate', '320k'))
        elif output_format == 'm4a':
            stream.bit_rate = _bitrate_to_int(audio_params.get('m4a_bit_rate', '512k'))
            if codec == 'aac_at':
                stream.codec_context.options = {'aac_at_quality': str(audio_params.get('m4a_aac_at_quality', 2))}
        frame = av.AudioFrame.from_ndarray(_format_audio(audio_array), format='fltp', layout=layout)
        frame.sample_rate = int(sr)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
