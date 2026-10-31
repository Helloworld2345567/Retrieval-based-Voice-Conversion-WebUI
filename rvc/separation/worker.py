"""DirectML subprocess lifetime and precision fallback."""

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from rvc.core.paths import PROJECT_ROOT, cache_root, is_source_tree

from .request import SeparationRequest, normalize_model_dtype


def precision_attempts(value=None):
    normalized = normalize_model_dtype(value or os.environ.get("PYMSS_DML_MODEL_DTYPE", "auto"))
    return ("float16", "float32") if normalized == "auto" else (normalized,)


def terminate_worker(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_directml(request):
    if not isinstance(request, SeparationRequest):
        request = SeparationRequest.from_dict(request)
    attempts = precision_attempts(request.params.get("model_dtype"))
    temp_root = cache_root() / "separation"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="worker-", dir=temp_root) as directory:
            request_path = Path(directory) / "request.json"
            result_path = Path(directory) / "result.json"
            for index, dtype in enumerate(attempts):
                payload = request.for_precision_attempt(
                    dtype,
                    result_path=result_path,
                    allow_fp32_retry=index + 1 < len(attempts),
                )
                request_path.write_text(payload.to_json(), encoding="utf-8")
                result_path.unlink(missing_ok=True)
                kwargs = {}
                if is_source_tree():
                    env = os.environ.copy()
                    env["PYTHONPATH"] = os.pathsep.join(
                        filter(None, (str(PROJECT_ROOT), env.get("PYTHONPATH")))
                    )
                    kwargs["env"] = env
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen([sys.executable, "-m", "rvc.separation.worker", str(request_path)], **kwargs)
                try:
                    code = process.wait()
                finally:
                    terminate_worker(process)
                if code == 75 and index + 1 < len(attempts):
                    logging.getLogger("rvc.separation").warning("DirectML FP16 failed; retrying in a fresh FP32 process")
                    continue
                if code == 130:
                    raise KeyboardInterrupt
                if code not in {0, 1} or not result_path.is_file():
                    raise RuntimeError(f"DirectML worker exited with code {code}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
        raise RuntimeError("DirectML precision attempts exhausted")
    finally:
        try:
            temp_root.rmdir()
        except OSError:
            pass


def main(argv=None):
    from rvc.core.devices import DeviceConfig
    from .service import PrecisionRetry, run_batch

    path = Path((argv or sys.argv[1:])[0]).resolve()
    request = SeparationRequest.from_json(path.read_text(encoding="utf-8"))
    if request.result_path is None:
        raise ValueError("DirectML worker request requires result_path")
    result_path = Path(request.result_path).resolve()
    if result_path != path.with_name("result.json"):
        raise ValueError("DirectML worker result_path must be the worker result file")
    logging.basicConfig(level=logging.DEBUG if request.debug else logging.INFO, format="%(message)s")
    try:
        runtime = DeviceConfig.create(device="directml", device_id=request.device_ids[0])
        result = run_batch(request, runtime)
        code = 1 if result["failed"] else 0
    except PrecisionRetry as error:
        result, code = {"error": str(error)}, 75
    except KeyboardInterrupt:
        result, code = {"error": "Interrupted"}, 130
    except Exception as error:
        result, code = {"error": str(error)}, 1
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
