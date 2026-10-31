import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def invoke(arguments, cwd=ROOT):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *arguments], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize("module,args", [
    ("rvc", ["--help"]),
    ("rvc", ["convert", "--help"]),
    ("rvc", ["separate", "infer", "--help"]),
    ("rvc", ["separate", "list", "--json"]),
    ("tools.pymss.cli", ["infer", "--help"]),
    ("tools.pymss.cli", ["list"]),
])
def test_help_and_catalog_do_not_import_engines(module, args, tmp_path):
    script = (
        "import sys,runpy; "
        f"sys.argv={[module, *args]!r}; "
        "\ntry: runpy.run_module(" + repr(module) + ", run_name='__main__')"
        "\nexcept SystemExit as e: assert e.code in (None,0), e.code"
        "\nassert not {'torch','torch_directml','gradio','fastapi','FreeSimpleGUI'}.intersection(sys.modules)"
    )
    result = invoke(["-c", script], tmp_path)
    assert result.returncode == 0, result.stderr


def test_legacy_conversion_script_runs_outside_repository(tmp_path):
    result = invoke([str(ROOT / "infer/cli.py"), "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "fcpe" in result.stdout and "directml" in result.stdout


@pytest.mark.parametrize("args", [
    ["train"], ["serve"], ["separate", "serve"], ["separate", "ensemble"], ["separate", "workflow"],
    ["convert", "--model", "absent.pth", "--input", "x", "--output", "y", "--protect", "3"],
])
def test_removed_commands_and_invalid_arguments_fail(args):
    assert invoke(["-m", "rvc", *args]).returncode == 2


def test_missing_model_returns_failure():
    result = invoke(["-m", "rvc", "convert", "--model", "absent.pth", "--input", "x", "--output", "y"])
    assert result.returncode == 1
    assert "Model not found" in result.stderr
