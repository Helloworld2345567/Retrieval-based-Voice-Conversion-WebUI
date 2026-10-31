"""Contracts for retained public APIs and intentionally narrow legacy adapters."""

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_isolated(source):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [str(PYTHON), "-B", "-c", source],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_public_package_imports_are_lightweight():
    result = _run_isolated(
        "import sys\n"
        "import rvc\n"
        "import rvc.vc\n"
        "import rvc.separation\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'av' not in sys.modules\n"
        "assert {'VoiceConverter', 'VoiceConversionConfig', 'AudioResult', "
        "'Speaker', 'ModelMetadata', 'MSSeparator', 'ModelSpec', "
        "'SeparationRequest', 'InferenceParams', 'AudioEncodingParams'} <= set(rvc.__all__)\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_exports_forward_to_retained_types():
    import rvc
    from rvc.separation.catalog import ModelSpec
    from rvc.separation.request import AudioEncodingParams, InferenceParams, SeparationRequest
    from rvc.vc.checkpoints import ModelMetadata, Speaker
    from rvc.vc.config import VoiceConversionConfig

    assert rvc.VoiceConversionConfig is VoiceConversionConfig
    assert rvc.Speaker is Speaker
    assert rvc.ModelMetadata is ModelMetadata
    assert rvc.ModelSpec is ModelSpec
    assert rvc.SeparationRequest is SeparationRequest
    assert rvc.InferenceParams is InferenceParams
    assert rvc.RequestParams is InferenceParams
    assert rvc.AudioEncodingParams is AudioEncodingParams
    assert rvc.AudioParams is AudioEncodingParams


def test_runtime_exports_preserve_class_identity():
    import rvc
    import rvc.separation as separation
    import rvc.vc as vc
    from rvc.separation.separator import MSSeparator
    from rvc.vc.engine import AudioResult, VoiceConverter

    assert rvc.VoiceConverter is vc.VoiceConverter is VoiceConverter
    assert rvc.AudioResult is vc.AudioResult is AudioResult
    assert rvc.MSSeparator is separation.MSSeparator is MSSeparator


def test_legacy_inference_modules_forward_retained_implementations():
    from infer.audio import load_audio, wav2
    from infer.fcpe import FCPEInfer
    from infer.module.attentions import Encoder
    from infer.module.commons import sequence_mask
    from infer.module.models import SynthesizerTrnMs768NSFsid
    from infer.module.modules import ResidualCouplingLayer
    from infer.rmvpe import RMVPE
    from infer.vc.pipeline import Pipeline, change_rms
    from rvc.vc import attentions, audio, commons, fcpe, models, modules, pipeline, rmvpe

    assert load_audio is audio.load_audio
    assert wav2 is audio.wav2
    assert FCPEInfer is fcpe.FCPEInfer
    assert RMVPE is rmvpe.RMVPE
    assert Encoder is attentions.Encoder
    assert sequence_mask is commons.sequence_mask
    assert SynthesizerTrnMs768NSFsid is models.SynthesizerTrnMs768NSFsid
    assert ResidualCouplingLayer is modules.ResidualCouplingLayer
    assert Pipeline is pipeline.Pipeline
    assert change_rms is pipeline.change_rms


def test_legacy_inference_modules_do_not_restore_removed_apis():
    import infer.module.attentions as attentions
    import infer.module.models as models
    import infer.module.modules as modules
    import infer.vc as vc

    assert not hasattr(attentions, "Decoder")
    assert not hasattr(models, "PosteriorEncoder")
    assert not hasattr(models, "MultiPeriodDiscriminator")
    assert not hasattr(modules, "ConvFlow")
    assert not hasattr(vc, "VC")


def test_compatibility_module_imports_are_lightweight():
    result = _run_isolated(
        "import sys\n"
        "import infer.audio, infer.fcpe, infer.hubert, infer.rmvpe\n"
        "import infer.module, infer.module.attentions, infer.module.commons\n"
        "import infer.module.models, infer.module.modules\n"
        "import infer.vc, infer.vc.pipeline, infer.vc.utils\n"
        "import tools.cuda_graph, tools.pymss\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'av' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_hubert_exports_only_retained_features(monkeypatch, tmp_path):
    import infer.hubert as legacy
    import rvc.vc.hubert as retained

    assert legacy.HubertModelWithFinalProj is retained.HubertModelWithFinalProj
    assert legacy.extract_hubert_features is retained.extract_hubert_features
    assert not hasattr(legacy, "hubert_audio_requires_normalization")

    calls = []
    monkeypatch.setattr(retained, "load_hubert_model", lambda *args: calls.append(args) or "model")
    monkeypatch.setattr(legacy, "HUBERT_MODEL_PATH", tmp_path)
    assert legacy.load_hubert_model("cpu", True) == "model"
    assert calls == [(tmp_path, "cpu", True)]


def test_legacy_vc_utils_forward_index_and_hubert_helpers(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from infer.vc.utils import get_index_path_from_model, load_hubert
    import rvc.vc.hubert as retained

    index_root = tmp_path / "indices"
    index_root.mkdir()
    expected = index_root / "voice_added_v2.index"
    expected.touch()
    monkeypatch.setenv("outside_index_root", str(index_root))
    monkeypatch.setenv("index_root", str(tmp_path / "missing"))
    assert get_index_path_from_model("voice_e1_s2.pth") == str(expected.resolve())

    calls = []
    monkeypatch.setattr(retained, "load_hubert_model", lambda *args: calls.append(args) or "hubert")
    config = SimpleNamespace(hubert_root=tmp_path / "hubert", device="cpu", is_half=False)
    assert load_hubert(config) == "hubert"
    assert calls == [(config.hubert_root, "cpu", False)]


def test_cuda_graph_adapter_forwards_retained_api():
    import tools.cuda_graph as legacy
    import rvc.vc.cuda_graph as retained

    for name in (
        "ENV_NAME",
        "MAX_CACHE_ENV",
        "clear_cuda_graph_cache",
        "configure_cuda_graph",
        "cuda_graph_enabled",
        "detect_cuda_graph_support",
        "get_cuda_graph_stats",
        "run_cuda_graph",
    ):
        assert getattr(legacy, name) is getattr(retained, name)


def test_pymss_exports_only_fixed_five_implementation():
    import tools.pymss as legacy
    from rvc.separation import audio_io, catalog, download, separator

    assert legacy.MSSeparator is separator.MSSeparator
    assert legacy.ModelSpec is catalog.ModelSpec
    assert legacy.MODELS is catalog.MODELS
    assert legacy.get_model is catalog.get_model
    assert legacy.model_path is catalog.model_path
    assert legacy.download_model is download.download_model
    assert legacy.load_audio is audio_io.load_audio
    assert legacy.save_audio is audio_io.save_audio
    assert len(legacy.MODELS) == 5
    for unsupported in (
        "WorkflowRunner",
        "create_separator",
        "ensemble_audios",
        "run_workflow_file",
        "save_ensemble_audio",
    ):
        assert not hasattr(legacy, unsupported)
