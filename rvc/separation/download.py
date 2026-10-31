"""Download a selected preset's weight; model configs ship with the package."""
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

from .catalog import PROJECT_ROOT, model_path

SOURCES = {
    "modelscope": "https://www.modelscope.cn/models/baicai1145/pymss/resolve/master",
    "huggingface": "https://huggingface.co/baicai1145/pymss/resolve/main",
    "hf-mirror": "https://hf-mirror.com/baicai1145/pymss/resolve/main",
}


def download_model(spec, directory=None, *, source="modelscope", endpoint=None, force=False):
    path = model_path(spec, directory)
    if path.is_file() and not force:
        return path
    base = endpoint.rstrip("/") if endpoint else SOURCES[source]
    relative = spec.remote_path + "/" + spec.filename
    url = base + "/" + urllib.parse.quote(relative, safe="/")
    temp_root = PROJECT_ROOT / ".codex" / "downloads"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="separation-", dir=temp_root) as temp:
            staged = Path(temp) / spec.filename
            with urllib.request.urlopen(url, timeout=60) as response, staged.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if staged.stat().st_size != spec.size_bytes:
                raise ValueError(f"Incomplete model download: {spec.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # The cache may be on another volume; stream into an adjacent file
            # before atomically replacing the final weight.
            with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", dir=path.parent, delete=False) as output:
                adjacent = Path(output.name)
                try:
                    with staged.open("rb") as incoming:
                        while chunk := incoming.read(1024 * 1024):
                            output.write(chunk)
                except BaseException:
                    output.close()
                    adjacent.unlink(missing_ok=True)
                    raise
            try:
                os.replace(adjacent, path)
            finally:
                adjacent.unlink(missing_ok=True)
    finally:
        try:
            temp_root.rmdir()
        except OSError:
            pass
    return path
