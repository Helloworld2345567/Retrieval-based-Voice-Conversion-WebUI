import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rvc import convert_cli
from rvc.vc.checkpoints import find_index, metadata_from_checkpoint, select_speaker


def checkpoint(speakers=None, count=8, **values):
    return {
        "weight": {"emb_g.weight": SimpleNamespace(shape=(count, 8))},
        "config": [0] * 17 + [40000],
        "speaker_info": speakers or [],
        **values,
    }


class ConversionMetadataTests(unittest.TestCase):
    def test_four_checkpoint_variants(self):
        for version in ("v1", "v2"):
            for f0 in (0, 1):
                with self.subTest(version=version, f0=f0):
                    metadata = metadata_from_checkpoint(checkpoint(version=version, f0=f0))
                    self.assertEqual((metadata.version, metadata.uses_f0), (version, bool(f0)))

    def test_speaker_metadata_is_normalized_and_default_is_declared(self):
        metadata = metadata_from_checkpoint(checkpoint([
            {"id": 5, "name": "Five"}, {"id": 2, "name": "Two"},
            {"id": 2, "name": "Duplicate"}, {"id": 9, "name": "Out of range"},
            {"id": -1, "name": "Negative"}, {"oops": "Missing"},
        ]))
        self.assertEqual([(item.id, item.name) for item in metadata.speakers], [(2, "Two"), (5, "Five")])
        self.assertEqual(select_speaker(metadata), 2)
        self.assertEqual(select_speaker(metadata, 5), 5)
        with self.assertRaises(ValueError):
            select_speaker(metadata, 0)

    def test_models_without_metadata_use_embedding_range(self):
        metadata = metadata_from_checkpoint(checkpoint(count=128))
        self.assertEqual(select_speaker(metadata), 0)
        self.assertEqual(select_speaker(metadata, 127), 127)
        with self.assertRaises(ValueError):
            select_speaker(metadata, 128)

    def test_invalid_model_metadata_is_rejected(self):
        for values in ({"version": "v3"}, {"f0": 2}, {"config": []}, {"weight": {}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                metadata_from_checkpoint(checkpoint(**values))


class ConversionPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rvc-cli-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write(self, path):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def test_index_matches_speaker_and_preferred_root(self):
        preferred = self.write("indices/demo_added_fixture_spkid2.index")
        self.write("logs/demo_added_fixture_spkid2.index")
        self.write("indices/demo_added_fixture_spkid5.index")
        self.write("indices/demo_trained_fixture_spkid2.index")
        found = find_index("demo_e20_s100.pth", 2, (self.root / "indices", self.root / "logs"))
        self.assertEqual(found, preferred)

    def test_index_disabled_does_not_require_file(self):
        self.assertIsNone(convert_cli.resolve_index("missing.index", "model", 0, 0, self.root))

    def test_explicit_trained_index_resolves_added_name(self):
        expected = self.write("demo_added.index")
        self.assertEqual(convert_cli.resolve_index("demo_trained.index", "model", 0, 1, self.root), expected)

    def test_missing_index_fails(self):
        with self.assertRaises(FileNotFoundError):
            convert_cli.resolve_index("missing.index", "model", 0, 1, self.root)

    def test_batch_recursion_preserves_relative_directories(self):
        source = self.write("input/sub/song.flac")
        jobs = convert_cli.collect_jobs("input", "output", "wav", True, self.root)
        self.assertEqual(jobs, [(source, self.root / "output/sub/song.wav", "wav")])
        self.assertFalse((self.root / "output").exists())

    def test_existing_output_subtree_is_not_reprocessed(self):
        self.write("input/song.wav")
        self.write("input/output/previous.wav")
        jobs = convert_cli.collect_jobs("input", "input/output", None, True, self.root)
        self.assertEqual(len(jobs), 1)

    def test_output_ancestor_keeps_input_files(self):
        source = self.write("data/sub/song.wav")
        jobs = convert_cli.collect_jobs("data/sub", "data", None, True, self.root)
        self.assertEqual(jobs, [(source, self.root / "data/song.wav", "wav")])

    def test_output_collisions_and_input_overwrite_fail(self):
        self.write("input/song.wav")
        self.write("input/song.flac")
        with self.assertRaisesRegex(ValueError, "same output"):
            convert_cli.collect_jobs("input", "output", None, False, self.root)
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            convert_cli.collect_jobs("input/song.wav", "input/song.wav", None, False, self.root)

    def test_format_override_and_dotted_directory(self):
        source = self.write("song.wav")
        directory = self.root / "output.folder"
        directory.mkdir()
        jobs = convert_cli.collect_jobs(str(source), str(directory), "mp3", False, self.root)
        self.assertEqual(jobs[0][1], directory / "song.mp3")

    def test_legacy_entrypoint_resolves_relative_environment_from_project(self):
        model = self.write("project/models/voice.pth")
        index = self.write("project/indexes/voice_added_fixture.index")
        outside = self.root / "outside"
        outside.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(outside)
            with patch.dict(os.environ, {"weight_root": "models", "outside_index_root": "indexes", "index_root": "logs"}), patch.object(convert_cli, "PROJECT_ROOT", self.root / "project"), patch.object(convert_cli, "load_checkpoint", return_value=({}, metadata_from_checkpoint(checkpoint()))) as loader, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(convert_cli.main(["--model", "voice.pth", "--list-speakers"], legacy=True), 0)
                loader.assert_called_once_with(model)
                self.assertEqual(convert_cli.resolve_index(None, "voice.pth", 0, 0.75, self.root / "project"), index)
                self.assertEqual(Path.cwd(), outside)
                self.assertEqual(os.environ["weight_root"], "models")
        finally:
            os.chdir(previous)


class ConversionCliTests(unittest.TestCase):
    def test_cli_defaults_and_fcpe(self):
        args = convert_cli.build_parser().parse_args(["--model", "voice.pth", "--list-speakers", "--f0-method", "fcpe"])
        self.assertEqual((args.f0_method, args.index_rate, args.rms_mix_rate, args.protect, args.device), ("fcpe", 0.75, 1.0, 0.33, "auto"))

    def test_entrypoint_relative_path_base_and_no_global_mutation(self):
        before = (Path.cwd(), dict(os.environ), sys_argv())
        with patch.object(convert_cli, "_run", return_value=0) as run:
            self.assertEqual(convert_cli.main(["--model", "voice.pth", "--list-speakers"]), 0)
            self.assertEqual(run.call_args.args[1], Path.cwd())
            self.assertEqual(convert_cli.main(["--model", "voice.pth", "--list-speakers"], legacy=True), 0)
            self.assertEqual(run.call_args.args[1], convert_cli.PROJECT_ROOT)
        self.assertEqual(before, (Path.cwd(), dict(os.environ), sys_argv()))

    def test_argument_failure_exits_two(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            convert_cli.main(["--model", "voice.pth", "--list-speakers", "--protect", "nan"])
        self.assertEqual(error.exception.code, 2)

    def test_runtime_failure_and_interrupt_exit_codes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with patch.object(convert_cli, "_run", side_effect=ValueError("bad model")):
                self.assertEqual(convert_cli.main(["--model", "voice.pth", "--list-speakers"]), 1)
            with patch.object(convert_cli, "_run", side_effect=KeyboardInterrupt):
                self.assertEqual(convert_cli.main(["--model", "voice.pth", "--list-speakers"]), 130)


def sys_argv():
    import sys
    return sys.argv[:]


if __name__ == "__main__":
    unittest.main()
