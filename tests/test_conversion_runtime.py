import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
import torch

from rvc import convert_cli
from rvc import audio as audio_module
from rvc.audio import AudioLoader, write_audio
from rvc.vc.checkpoints import ModelMetadata
from rvc.vc.engine import AudioResult, VoiceConverter
from rvc.vc.models import SynthesizerTrnMs256NSFsid_nono
from rvc.vc.pipeline import Pipeline
from rvc.vc.rmvpe import RMVPE


def pipeline_config(root):
    return SimpleNamespace(device=torch.device("cpu"), is_half=False, x_pad=1, x_query=6, x_center=38, x_max=41, rmvpe_root=root, cuda_graph=False)


class ConversionRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rvc-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.sample_rate = 32000
        self.audio = (np.sin(2 * np.pi * 220 * np.arange(16000) / self.sample_rate) * 8000).astype(np.int16)

    def test_four_output_formats_decode_to_nonempty_finite_audio(self):
        for output_format in ("wav", "flac", "mp3", "m4a"):
            with self.subTest(output_format=output_format):
                output = self.root / ("output." + output_format)
                write_audio(output, self.audio, self.sample_rate, output_format)
                decoded = AudioLoader().load(output, 16000)
                self.assertGreater(output.stat().st_size, 100)
                self.assertGreater(decoded.size, 7000)
                self.assertLess(decoded.size, 10000)
                self.assertTrue(np.isfinite(decoded).all())
                self.assertGreater(np.abs(decoded).max(), 0.05)

    def test_matching_sample_rate_preserves_float32_audio(self):
        loader = AudioLoader(SimpleNamespace(device="cpu", dtype=torch.float16))
        source = torch.tensor([[0.1234567, -0.654321]], dtype=torch.float32)
        decoded = loader._resample_cuda(source, 16000, 16000, True)
        np.testing.assert_array_equal(decoded, source.numpy()[0])

    def test_bundled_ffmpeg_and_ffprobe_do_not_depend_on_working_directory(self):
        suffix = ".exe" if os.name == "nt" else ""
        for name in ("ffmpeg", "ffprobe"):
            (self.root / (name + suffix)).touch()
        with patch.object(audio_module, "PROJECT_ROOT", self.root), patch("ffmpeg.probe", return_value={"streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}]}) as probe, patch("ffmpeg.input") as input_stream:
            input_stream.return_value.output.return_value.run.return_value = (np.zeros(100, np.float32).tobytes(), b"")
            audio_module._decode_ffmpeg(self.root / "input.wav", sample_rate=None)
            self.assertEqual(probe.call_args.kwargs["cmd"], str(self.root / ("ffprobe" + suffix)))
            self.assertEqual(input_stream.return_value.output.return_value.run.call_args.kwargs["cmd"][0], str(self.root / ("ffmpeg" + suffix)))

    def test_rmvpe_onnx_uses_the_requested_directml_adapter(self):
        ort = SimpleNamespace(InferenceSession=Mock())
        extractor = Mock()
        with patch.dict("sys.modules", {"onnxruntime": ort}), patch("rvc.vc.rmvpe.MelSpectrogram", return_value=extractor):
            RMVPE(str(self.root / "rmvpe.pt"), is_half=False, device=torch.device("privateuseone:3"))
        self.assertEqual(ort.InferenceSession.call_args.kwargs["providers"], [("DmlExecutionProvider", {"device_id": 3})])

    def test_corrupt_index_fails_before_synthesis(self):
        index_path = self.root / "corrupt.index"
        index_path.write_bytes(b"not a FAISS index")
        pipeline = Pipeline(32000, pipeline_config(self.root))
        with self.assertRaises(RuntimeError):
            pipeline.pipeline(None, None, 0, self.audio.astype(np.float32), [0, 0, 0], 0, "pm", str(index_path), 0.75, 0, 32000, 0, 1.0, "v1", 0.33)

    def test_empty_and_wrong_dimension_indexes_fail(self):
        for dimension, add_vector in ((256, False), (128, True)):
            with self.subTest(dimension=dimension, add_vector=add_vector):
                index = faiss.IndexFlatL2(dimension)
                if add_vector:
                    index.add(np.zeros((1, dimension), np.float32))
                path = self.root / "index.index"
                faiss.write_index(index, str(path))
                pipeline = Pipeline(32000, pipeline_config(self.root))
                with self.assertRaisesRegex(ValueError, "dimension"):
                    pipeline.pipeline(None, None, 0, self.audio.astype(np.float32), [0, 0, 0], 0, "pm", str(path), 0.75, 0, 32000, 0, 1.0, "v1", 0.33)

    def test_exact_index_match_and_small_index_keep_features_finite(self):
        index = faiss.IndexFlatL2(256)
        vectors = np.zeros((1, 256), np.float32)
        index.add(vectors)
        captured = []

        def synthesize(phone, lengths, speaker):
            captured.append(phone)
            return (torch.zeros((1, 1, 100)),)

        pipeline = Pipeline(32000, pipeline_config(self.root))
        with patch("rvc.vc.pipeline.extract_hubert_features", return_value=torch.zeros((1, 2, 256))):
            result = pipeline.vc(None, SimpleNamespace(infer=synthesize), torch.tensor([0]), np.zeros(320, np.float32), None, None, [0, 0, 0], index, vectors, 0.75, "v1", 0.33)
        self.assertTrue(torch.isfinite(captured[0]).all())
        self.assertEqual(result.shape, (100,))

    def test_nonfinite_synthesis_fails_before_pcm_conversion(self):
        pipeline = Pipeline(32000, pipeline_config(self.root))
        output = np.full(2 * pipeline.t_pad_tgt + 100, np.nan, np.float32)
        with patch.object(pipeline, "vc", return_value=output), self.assertRaisesRegex(RuntimeError, "non-finite"):
            pipeline.pipeline(None, None, 0, self.audio.astype(np.float32), [0, 0, 0], 0, "rmvpe", "", 0, 0, 32000, 0, 1.0, "v1", 0.33)

    def test_no_f0_model_does_not_load_a_pitch_predictor(self):
        pipeline = Pipeline(32000, pipeline_config(self.root))
        output = np.zeros(2 * pipeline.t_pad_tgt + 100, np.float32)
        with patch.object(pipeline, "vc", return_value=output), patch.object(pipeline, "get_f0", side_effect=AssertionError("pitch predictor was called")):
            result = pipeline.pipeline(None, None, 0, self.audio.astype(np.float32), [0, 0, 0], 0, "rmvpe", "", 0, 0, 32000, 0, 1.0, "v1", 0.33)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, np.int16)

    def test_batch_failure_returns_nonzero_and_keeps_successful_output(self):
        model_path = self.root / "voice.pth"
        model_path.touch()
        source = self.root / "input"
        source.mkdir()
        (source / "a.wav").touch()
        (source / "b.wav").touch()
        metadata = ModelMetadata("v1", False, self.sample_rate, 1, ())
        result = AudioResult(self.sample_rate, self.audio, (0, 0, 0))
        config = SimpleNamespace(device="cpu", dtype=torch.float32)
        with patch.object(convert_cli, "load_checkpoint", return_value=({}, metadata)), patch("rvc.vc.config.VoiceConversionConfig.create", return_value=config), patch("rvc.vc.engine.VoiceConverter") as converter, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            converter.return_value.__enter__.return_value.convert.side_effect = [result, ValueError("bad input")]
            exit_code = convert_cli.main(["--model", str(model_path), "--input", str(source), "--output", str(self.root / "output"), "--index-rate", "0"])
        self.assertEqual(exit_code, 1)
        self.assertTrue((self.root / "output/a.wav").is_file())
        self.assertFalse((self.root / "output/b.wav").exists())

    def test_cli_loads_real_checkpoint_runs_pipeline_and_releases_models(self):
        # A 160x decoder converts the 100 Hz RVC features to 16 kHz PCM.
        model_config = [9, 32, 4, 4, 8, 2, 1, 3, 0.0, "1", [3], [[1, 3, 5]], [10, 4, 2, 2], 32, [20, 8, 4, 4], 2, 4, 16000]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            network = SynthesizerTrnMs256NSFsid_nono(*model_config)
        model_path = self.root / "voice.pth"
        torch.save({"weight": network.state_dict(), "config": model_config, "version": "v1", "f0": 0, "speaker_info": [{"id": 1, "name": "test"}]}, model_path)
        del network
        input_path = self.root / "input.wav"
        output_path = self.root / "output.wav"
        write_audio(input_path, self.audio, self.sample_rate, "wav")
        instances = []

        class TrackingConverter(VoiceConverter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)

        def features(model, source, version, padding_mask=None):
            dimension = 256 if version == "v1" else 768
            frames = max(1, source.shape[-1] // 320)
            return torch.zeros((1, frames, dimension), device=source.device, dtype=source.dtype)

        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with patch("rvc.vc.engine.VoiceConverter", TrackingConverter), patch("rvc.vc.engine.load_hubert_model", return_value=SimpleNamespace()), patch("rvc.vc.pipeline.extract_hubert_features", side_effect=features), patch.object(Pipeline, "get_f0", side_effect=AssertionError("noF0 model loaded pitch predictor")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = convert_cli.main(["--model", str(model_path), "--input", str(input_path), "--output", str(output_path), "--index-rate", "0", "--device", "cpu", "--f0-method", "pm"])
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(exit_code, 0)
        decoded = AudioLoader().load(output_path, 16000)
        self.assertEqual(decoded.shape, (8000,))
        self.assertTrue(np.isfinite(decoded).all())
        self.assertGreater(np.abs(decoded).max(), 0)
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].closed)
        self.assertIsNone(instances[0].net_g)
        self.assertIsNone(instances[0].hubert_model)
        self.assertIsNone(instances[0].pipeline)


if __name__ == "__main__":
    unittest.main()
