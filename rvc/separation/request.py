"""Typed, validated requests for separation inference.

This module intentionally has no inference or torch imports.  It is used by
both the lightweight CLI parser and the DirectML worker process.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import re
from collections.abc import Mapping, Iterator
from typing import Any, ClassVar


_PARAM_NAMES = frozenset(
    {
        "batch_size",
        "chunk_size",
        "overlap_size",
        "standardize",
        "normalize",
        "use_amp",
        "model_dtype",
        "cuda_attention_backend",
        "stem_batch_size",
    }
)
_BOOL_PARAMS = frozenset({"standardize", "normalize", "use_amp"})
_POSITIVE_INT_PARAMS = frozenset({"batch_size", "chunk_size"})
_NONNEGATIVE_INT_PARAMS = frozenset({"overlap_size", "stem_batch_size"})
_ATTENTION_BACKENDS = frozenset(
    {
        "auto",
        "torch",
        "default",
        "sdpa",
        "flash",
        "flash_attention",
        "cudnn",
        "cudnn_attn",
        "cudnn_attention",
        "efficient",
        "mem_efficient",
        "memory_efficient",
        "math",
        "xformers",
    }
)
_MODEL_DTYPE_ALIASES = {
    "auto": "auto",
    "fp16": "float16",
    "half": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
}
_AUDIO_FORMATS = frozenset({"wav", "flac", "mp3", "m4a"})
_WAV_BIT_DEPTHS = frozenset({"FLOAT", "PCM_16", "PCM_24"})
_FLAC_BIT_DEPTHS = frozenset({"PCM_16", "PCM_24"})
_BIT_RATE_RE = re.compile(r"^(?:[1-9]\d*|[1-9]\d*(?:\.\d+)?[kKmM])$")


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Invalid {name}: expected true or false")
    return value


def _strict_string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        suffix = " non-empty" if nonempty else ""
        raise ValueError(f"Invalid {name}: expected a{suffix} string")
    return value


def _absolute_path(value: Any, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Invalid {name}: expected a non-empty path string")
    return str(Path(value).expanduser().resolve())


def normalize_model_dtype(value: Any) -> str:
    """Validate and canonicalize the DirectML precision value."""
    if not isinstance(value, str):
        raise ValueError("Invalid model_dtype: expected auto, float16 or float32")
    normalized = value.strip().lower().replace("torch.", "")
    try:
        return _MODEL_DTYPE_ALIASES[normalized]
    except KeyError as error:
        raise ValueError("model_dtype must be auto, float16, or float32") from error


def _validate_bit_rate(value: Any, name: str) -> int | str:
    if type(value) is int:
        if value <= 0:
            raise ValueError(f"Invalid {name}: expected a positive bitrate")
        return value
    if not isinstance(value, str) or not _BIT_RATE_RE.fullmatch(value.strip()):
        raise ValueError(f"Invalid {name}: expected a positive bitrate such as 320k")
    return value.strip()


@dataclass(frozen=True)
class AudioEncodingParams(Mapping[str, Any]):
    """Validated output codec settings.

    The mapping interface keeps the value object compatible with the audio
    writer while callers use typed, immutable attributes at the request seam.
    """

    wav_bit_depth: str = "FLOAT"
    flac_bit_depth: str = "PCM_16"
    mp3_bit_rate: int | str = "320k"
    m4a_bit_rate: int | str = "512k"
    m4a_codec: str = "aac"
    m4a_aac_at_quality: int = 2

    _KNOWN: ClassVar[frozenset[str]] = frozenset(
        {"wav_bit_depth", "flac_bit_depth", "mp3_bit_rate", "m4a_bit_rate", "m4a_codec", "m4a_aac_at_quality"}
    )

    def __post_init__(self) -> None:
        if self.wav_bit_depth not in _WAV_BIT_DEPTHS:
            raise ValueError(f"Invalid wav_bit_depth: expected one of {sorted(_WAV_BIT_DEPTHS)}")
        if self.flac_bit_depth not in _FLAC_BIT_DEPTHS:
            raise ValueError(f"Invalid flac_bit_depth: expected one of {sorted(_FLAC_BIT_DEPTHS)}")
        _validate_bit_rate(self.mp3_bit_rate, "mp3_bit_rate")
        _validate_bit_rate(self.m4a_bit_rate, "m4a_bit_rate")
        _strict_string(self.m4a_codec, "m4a_codec")
        if type(self.m4a_aac_at_quality) is not int or self.m4a_aac_at_quality < 0:
            raise ValueError("Invalid m4a_aac_at_quality: expected a nonnegative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AudioEncodingParams":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("audio_params must be an object")
        unknown = set(value) - cls._KNOWN
        if unknown:
            raise ValueError("Unsupported audio parameters: " + ", ".join(sorted(unknown)))
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wav_bit_depth": self.wav_bit_depth,
            "flac_bit_depth": self.flac_bit_depth,
            "mp3_bit_rate": self.mp3_bit_rate,
            "m4a_bit_rate": self.m4a_bit_rate,
            "m4a_codec": self.m4a_codec,
            "m4a_aac_at_quality": self.m4a_aac_at_quality,
        }

    def __getitem__(self, key: str) -> Any:
        if key not in self._KNOWN:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KNOWN)

    def __len__(self) -> int:
        return len(self._KNOWN)


@dataclass(frozen=True)
class InferenceParams(Mapping[str, Any]):
    """The allow-listed ``--param`` values used by a separator."""

    batch_size: int | None = None
    chunk_size: int | None = None
    overlap_size: int | None = None
    standardize: bool | None = None
    normalize: bool | None = None
    use_amp: bool | None = None
    model_dtype: str | None = None
    cuda_attention_backend: str | None = None
    stem_batch_size: int | None = None

    _KNOWN: ClassVar[frozenset[str]] = _PARAM_NAMES

    def __post_init__(self) -> None:
        for name in _POSITIVE_INT_PARAMS | _NONNEGATIVE_INT_PARAMS:
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not int or (name in _POSITIVE_INT_PARAMS and value < 1) or (
                name in _NONNEGATIVE_INT_PARAMS and value < 0
            ):
                qualifier = "positive" if name in _POSITIVE_INT_PARAMS else "nonnegative"
                raise ValueError(f"Invalid {name}: expected a {qualifier} integer")
        for name in _BOOL_PARAMS:
            value = getattr(self, name)
            if value is not None:
                _strict_bool(value, name)
        if self.model_dtype is not None:
            object.__setattr__(self, "model_dtype", normalize_model_dtype(self.model_dtype))
        if self.cuda_attention_backend is not None:
            if not isinstance(self.cuda_attention_backend, str):
                raise ValueError("Invalid cuda_attention_backend")
            normalized = self.cuda_attention_backend.strip().lower().replace("-", "_")
            if normalized not in _ATTENTION_BACKENDS:
                raise ValueError("Invalid cuda_attention_backend")
            object.__setattr__(self, "cuda_attention_backend", normalized)
        if self.chunk_size is not None and self.overlap_size is not None and self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "InferenceParams":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("params must be an object")
        unknown = set(value) - cls._KNOWN
        if unknown:
            raise ValueError("Unsupported inference parameters: " + ", ".join(sorted(unknown)))
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in ((field.name, getattr(self, field.name)) for field in fields(self)) if value is not None}

    def with_model_dtype(self, value: str) -> "InferenceParams":
        return replace(self, model_dtype=value)

    def __getitem__(self, key: str) -> Any:
        if key not in self._KNOWN or getattr(self, key) is None:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def parse_params(values: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Parse and strictly validate repeated ``--param KEY=VALUE`` options."""
    result: dict[str, Any] = {}
    for value in values or []:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError(f"Expected key=value, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key or not raw.strip():
            raise ValueError(f"Expected a non-empty key and value, got {value!r}")
        if raw.lower() in {"true", "false"}:
            parsed: Any = raw.lower() == "true"
        else:
            try:
                parsed = int(raw)
            except ValueError:
                try:
                    parsed = float(raw)
                except ValueError:
                    parsed = raw
        result[key] = parsed
    # Constructing the value object is the single validation path for CLI params.
    InferenceParams.from_mapping(result)
    return result


@dataclass(frozen=True)
class SeparationRequest:
    """A complete immutable separation invocation and worker message."""

    model: str
    inputs: tuple[str, ...]
    output: str
    format: str
    legacy: bool = False
    save_as_folder: bool = False
    device_ids: tuple[int, ...] = (0,)
    params: InferenceParams = InferenceParams()
    tta: bool = False
    debug: bool = False
    audio_params: AudioEncodingParams = AudioEncodingParams()
    model_path: str | None = None
    allow_fp32_retry: bool = False
    result_path: str | None = None

    _KNOWN: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "inputs",
            "output",
            "format",
            "legacy",
            "save_as_folder",
            "device_ids",
            "params",
            "tta",
            "debug",
            "audio_params",
            "model_path",
            "allow_fp32_retry",
            "result_path",
        }
    )

    def __post_init__(self) -> None:
        _strict_string(self.model, "model")
        if isinstance(self.inputs, (str, bytes)) or not isinstance(self.inputs, (tuple, list)):
            raise ValueError("inputs must be a sequence of paths")
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        object.__setattr__(self, "inputs", tuple(_absolute_path(path, "input path") for path in self.inputs))
        object.__setattr__(self, "output", _absolute_path(self.output, "output"))
        if not isinstance(self.format, str) or self.format.lower() not in _AUDIO_FORMATS:
            raise ValueError(f"Invalid format: expected one of {sorted(_AUDIO_FORMATS)}")
        object.__setattr__(self, "format", self.format.lower())
        for name in ("legacy", "save_as_folder", "tta", "debug", "allow_fp32_retry"):
            _strict_bool(getattr(self, name), name)
        if isinstance(self.device_ids, (str, bytes)) or not isinstance(self.device_ids, (tuple, list)):
            raise ValueError("device_ids must be a sequence of nonnegative integers")
        device_ids = tuple(self.device_ids)
        if not device_ids:
            raise ValueError("device_ids must not be empty")
        if any(type(index) is not int or index < 0 for index in device_ids) or len(set(device_ids)) != len(device_ids):
            raise ValueError("Device IDs must be distinct nonnegative integers")
        object.__setattr__(self, "device_ids", device_ids)
        object.__setattr__(self, "params", InferenceParams.from_mapping(self.params))
        object.__setattr__(self, "audio_params", AudioEncodingParams.from_mapping(self.audio_params))
        for name in ("model_path", "result_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _absolute_path(value, name))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeparationRequest":
        if not isinstance(value, Mapping):
            raise ValueError("request must be an object")
        unknown = set(value) - cls._KNOWN
        if unknown:
            raise ValueError("Unsupported request fields: " + ", ".join(sorted(unknown)))
        required = {"model", "inputs", "output", "format"}
        missing = required - set(value)
        if missing:
            raise ValueError("Missing request fields: " + ", ".join(sorted(missing)))
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "SeparationRequest":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid request JSON: {error}") from error
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "inputs": list(self.inputs),
            "output": self.output,
            "format": self.format,
            "legacy": self.legacy,
            "save_as_folder": self.save_as_folder,
            "device_ids": list(self.device_ids),
            "params": self.params.to_dict(),
            "tta": self.tta,
            "debug": self.debug,
            "audio_params": self.audio_params.to_dict(),
            "model_path": self.model_path,
            "allow_fp32_retry": self.allow_fp32_retry,
            "result_path": self.result_path,
        }

    def to_json(self, *, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=sort_keys)

    def with_updates(self, **changes: Any) -> "SeparationRequest":
        unknown = set(changes) - self._KNOWN
        if unknown:
            raise ValueError("Unsupported request fields: " + ", ".join(sorted(unknown)))
        return replace(self, **changes)

    def with_result_path(self, result_path: str | Path) -> "SeparationRequest":
        return replace(self, result_path=str(result_path))

    def with_retry(self, allow_fp32_retry: bool) -> "SeparationRequest":
        return replace(self, allow_fp32_retry=allow_fp32_retry)

    def for_precision_attempt(
        self,
        model_dtype: str,
        *,
        result_path: str | Path,
        allow_fp32_retry: bool,
    ) -> "SeparationRequest":
        return replace(
            self,
            params=self.params.with_model_dtype(model_dtype),
            result_path=str(result_path),
            allow_fp32_retry=allow_fp32_retry,
        )

    # Explicit aliases make the update intent clear at call sites and preserve
    # a small, discoverable API for integrations that call the seam directly.
    with_precision_attempt = for_precision_attempt
    with_model_dtype = lambda self, value: replace(self, params=self.params.with_model_dtype(value))

    @property
    def output_format(self) -> str:
        return self.format

    @property
    def input_paths(self) -> tuple[str, ...]:
        return self.inputs


# Short aliases for integrations that used the value-object names before the
# request itself was finalized.
AudioParams = AudioEncodingParams
RequestParams = InferenceParams
