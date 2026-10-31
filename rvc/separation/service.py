"""File inference shared by the unified CLI and its legacy entry point."""
import logging
from pathlib import Path
from collections.abc import Mapping

from .catalog import get_model
from .request import SeparationRequest

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".aiff", ".aif"}


def collect_inputs(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")
    files = [str(item) for item in sorted(path.iterdir()) if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS]
    if not files:
        raise ValueError(f"No supported audio files found in {path}")
    return files


def _request(value) -> SeparationRequest:
    if isinstance(value, SeparationRequest):
        return value
    if isinstance(value, Mapping):
        return SeparationRequest.from_dict(value)
    raise TypeError("request must be a SeparationRequest")


def output_paths(spec, input_path, output, output_format, save_as_folder, legacy):
    root = Path(output)
    stem = Path(input_path).stem
    if save_as_folder:
        root /= stem
    suffixes = spec.stems if legacy else spec.suffixes
    return [root / f"{stem}_{suffix}.{output_format}" for suffix in suffixes]


def validate_outputs(request):
    request = _request(request)
    spec = get_model(request.model)
    seen = set()
    inputs = {str(Path(path).resolve()).casefold() for path in request.inputs}
    for path in request.inputs:
        for target in output_paths(spec, path, request.output, request.format, request.save_as_folder, request.legacy):
            key = str(target.resolve()).casefold()
            if key in seen:
                raise ValueError(f"Multiple inputs would write the same output: {target}")
            seen.add(key)
            if key in inputs:
                raise ValueError(f"Output would overwrite an input: {target}")


def fp16_retryable(error):
    message = str(error).lower()
    oom = any(word in message for word in ("out of memory", "e_outofmemory", "not enough memory", "failed to allocate", "allocation failed"))
    return not oom and isinstance(error, (RuntimeError, NotImplementedError, FloatingPointError, TypeError, ValueError))


class PrecisionRetry(RuntimeError):
    pass


def run_batch(request, runtime):
    from .audio_io import load_audio, save_audio
    from .separator import MSSeparator

    request = _request(request)
    spec = get_model(request.model)
    logger = logging.getLogger("rvc.separation")
    results = {"successful": 0, "failed": 0, "outputs": []}
    retry = request.allow_fp32_retry
    try:
        separator = MSSeparator(spec, request.model_path, runtime,
                                device_ids=request.device_ids,
                                inference_params=request.params.to_dict(), use_tta=request.tta)
    except Exception as error:
        if retry and fp16_retryable(error):
            raise PrecisionRetry(str(error)) from error
        raise
    with separator:
        for input_path in request.inputs:
            try:
                mix, sr = load_audio(input_path, sr=44100, mono=False)
                targets = output_paths(spec, input_path, request.output, request.format,
                                       request.save_as_folder, request.legacy)
                batch_size = getattr(separator, "stem_batch_size", 0)
                if batch_size <= 0 or getattr(separator, "output_normalize", False):
                    batch_size = len(spec.stems)
                for offset in range(0, len(spec.stems), batch_size):
                    group = spec.stems[offset:offset + batch_size]
                    try:
                        separated = separator.separate(mix, stems=group)
                    except Exception as error:
                        if retry and not results["successful"] and fp16_retryable(error):
                            raise PrecisionRetry(str(error)) from error
                        raise
                    for stem, target in zip(group, targets[offset:offset + batch_size]):
                        if stem not in separated:
                            raise RuntimeError(f"Model did not produce required stem: {stem}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        save_audio(str(target), separated[stem], sr, request.format, request.audio_params)
                        if not target.is_file() or not target.stat().st_size:
                            raise RuntimeError(f"Encoder produced no output: {target}")
                    del separated
                results["successful"] += 1
                results["outputs"].extend(str(path) for path in targets)
                logger.info("Separated %s", input_path)
            except PrecisionRetry:
                raise
            except Exception as error:
                results["failed"] += 1
                logger.error("Cannot separate %s: %s", input_path, error)
    return results
