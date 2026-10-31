"""Checkpoint metadata and feature-index selection without device setup."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Speaker:
    id: int
    name: str


@dataclass(frozen=True)
class ModelMetadata:
    version: str
    uses_f0: bool
    sample_rate: int
    speaker_count: int
    speakers: tuple[Speaker, ...]


def metadata_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Model must contain a checkpoint dictionary")
    weight = checkpoint.get("weight", {})
    embedding = weight.get("emb_g.weight") if isinstance(weight, dict) else None
    if embedding is None or len(embedding.shape) != 2 or embedding.shape[0] < 1:
        raise ValueError("Model does not contain a usable weight/emb_g.weight")
    config = checkpoint.get("config")
    if not isinstance(config, (list, tuple)) or len(config) != 18:
        raise ValueError("Model must contain the 18-value RVC config")
    version = checkpoint.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ValueError("Unsupported RVC model version: %r" % version)
    f0 = checkpoint.get("f0", 1)
    if f0 not in (0, 1):
        raise ValueError("Model f0 must be 0 or 1")
    sample_rate = config[-1]
    if isinstance(sample_rate, str):
        sample_rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sample_rate, sample_rate)
    sample_rate = int(sample_rate)
    if sample_rate < 16000:
        raise ValueError("Invalid model sample rate: %s" % sample_rate)
    speaker_count = int(embedding.shape[0])
    speakers = []
    seen = set()
    for item in checkpoint.get("speaker_info", []) or []:
        try:
            speaker_id, name = int(item["id"]), str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= speaker_id < speaker_count and name and speaker_id not in seen:
            speakers.append(Speaker(speaker_id, name))
            seen.add(speaker_id)
    return ModelMetadata(version, bool(f0), sample_rate, speaker_count, tuple(sorted(speakers, key=lambda item: item.id)))


def load_checkpoint(path):
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    return checkpoint, metadata_from_checkpoint(checkpoint)


def select_speaker(metadata, requested_id=None):
    if metadata.speakers:
        ids = {speaker.id for speaker in metadata.speakers}
        speaker_id = metadata.speakers[0].id if requested_id is None else requested_id
        if speaker_id not in ids:
            raise ValueError("Speaker ID %s is not declared; available IDs: %s" % (speaker_id, ", ".join(map(str, sorted(ids)))))
        return speaker_id
    speaker_id = 0 if requested_id is None else requested_id
    if not 0 <= speaker_id < metadata.speaker_count:
        raise ValueError("Speaker ID must be between 0 and %s" % (metadata.speaker_count - 1))
    return speaker_id


def find_index(model_name, speaker_id, roots):
    model_stem = Path(model_name).stem
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return None
    candidates = []
    for root_number, index_root in enumerate(roots):
        index_root = Path(index_root)
        if not index_root.is_dir():
            continue
        for path in index_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".index" or "trained" in path.name.lower():
                continue
            lower_index = path.stem.lower()
            experiment = experiment_name.lower()
            match = re.search(r"_spkid(\d+)$", path.stem, re.IGNORECASE)
            indexed_id = int(match.group(1)) if match else None
            if indexed_id is not None and indexed_id != speaker_id:
                continue
            standard = lower_index.startswith(experiment + "_added_") or any("_" + experiment + "_" + version in lower_index for version in ("v1", "v2"))
            if standard or model_stem.lower() in lower_index:
                score = (0 if indexed_id == speaker_id else 1, 0 if standard else 1, root_number, -path.stat().st_mtime, str(path).lower())
                candidates.append((score, path.resolve()))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
