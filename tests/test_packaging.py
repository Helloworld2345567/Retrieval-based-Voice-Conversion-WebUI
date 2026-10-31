"""Packaging metadata and immutable resource contracts."""

from importlib.resources import files
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_installable_cli_and_source_layout():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["build-system"]["build-backend"] == "setuptools.build_meta"
    assert metadata["project"]["requires-python"] == ">=3.12,<3.13"
    assert metadata["project"]["readme"] == "README.md"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == [
        "LICENSE",
        "MIT协议暨相关引用库协议",
        "rvc/separation/LICENSE.pymss",
        "rvc/separation/LICENSE.pymss-core",
    ]
    assert metadata["project"]["dynamic"] == ["dependencies"]
    assert metadata["project"]["scripts"] == {"rvc": "rvc.cli:main"}
    assert metadata["tool"]["setuptools"]["dynamic"]["dependencies"] == {
        "file": ["requirements-inference.txt"]
    }
    dependencies = (ROOT / "requirements-inference.txt").read_text(encoding="utf-8")
    assert "torch>=2.4.1,<2.8" in dependencies
    assert "torchaudio>=2.4.1,<2.8" in dependencies
    assert "hyper-connections>=0.1.8,<0.4.3" in dependencies
    discovery = metadata["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["."]
    assert discovery["include"] == ["rvc*", "infer*", "tools*"]
    assert discovery["exclude"] == ["tests*", "assets*"]


def test_sdist_manifest_includes_user_documentation():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include assets/README.md" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "include requirments_*_py312.txt" in manifest


def test_separation_resources_are_package_readable():
    package = files("rvc.separation")
    expected = {
        "LICENSE.pymss",
        "LICENSE.pymss-core",
        "resources/config_mel_band_roformer_karaoke.yaml",
        "resources/dereverb_mel_band_roformer_anvuew.yaml",
        "resources/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "resources/model_bs_roformer_ep_368_sdr_12.9628.yaml",
    }

    assert all(package.joinpath(*relative.split("/")).is_file() for relative in expected)
