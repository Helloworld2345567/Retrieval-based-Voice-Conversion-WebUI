"""The five separation presets formerly exposed by the RVC WebUI."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    model_type: str
    filename: str
    config_file: str
    remote_path: str
    stems: tuple[str, str]
    suffixes: tuple[str, str]
    batch_size: int
    overlap_size: int
    size_bytes: int

    @property
    def config_path(self):
        return RESOURCE_ROOT / self.config_file


MODELS = (
    ModelSpec("dereverb", "去混响", "mel_band_roformer",
              "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
              "dereverb_mel_band_roformer_anvuew.yaml", "reverb_echo_control/dereverb",
              ("noreverb", "reverb"), ("noreverb", "reverb"), 1, 176400, 913107578),
    ModelSpec("dereverb-aggressive", "去混响（激进）", "mel_band_roformer",
              "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
              "dereverb_mel_band_roformer_anvuew.yaml", "reverb_echo_control/dereverb",
              ("noreverb", "reverb"), ("noreverb", "reverb"), 1, 176400, 913107578),
    ModelSpec("vocals", "去伴奏", "bs_roformer",
              "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
              "model_bs_roformer_ep_368_sdr_12.9628.yaml", "vocal/vocal_extraction",
              ("vocals", "instrumental"), ("vocals", "instrumental"), 1, 264600, 639317465),
    ModelSpec("vocals-aggressive", "去伴奏（激进）", "bs_roformer",
              "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
              "model_bs_roformer_ep_317_sdr_12.9755.yaml", "vocal/vocal_extraction",
              ("vocals", "other"), ("vocals", "instrumental"), 4, 176400, 639331213),
    ModelSpec("lead-vocal", "提主旋律", "mel_band_roformer",
              "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
              "config_mel_band_roformer_karaoke.yaml", "karaoke",
              ("karaoke", "other"), ("main_vocal", "off_vocal"), 1, 264600, 913096801),
)


def get_model(value):
    key = str(value).strip().lower()
    for model in MODELS:
        if key in {model.name, model.label.lower(), model.filename.lower(), Path(model.filename).stem.lower()}:
            return model
    raise ValueError(f"Unknown separation model: {value}. Available presets: " + ", ".join(m.name for m in MODELS))


def model_root(directory=None):
    value = directory or os.environ.get("PYMSS_MODEL_DIR") or os.environ.get("weight_pymss_root")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "assets" / "pymss_weights"


def model_path(spec, directory=None):
    root = model_root(directory)
    flat = root / spec.filename
    old = root / spec.remote_path / spec.filename
    return old if not flat.is_file() and old.is_file() else flat
