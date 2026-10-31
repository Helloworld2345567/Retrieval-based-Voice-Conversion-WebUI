"""DirectML subprocess lifetime and precision fallback."""
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .catalog import PROJECT_ROOT


def precision_attempts(value=None):
    value = str(value or os.environ.get("PYMSS_DML_MODEL_DTYPE", "auto")).lower()
    aliases = {"fp16": "float16", "half": "float16", "fp32": "float32", "float": "float32"}
    value = aliases.get(value, value)
    if value == "auto":
        return ("float16", "float32")
    if value not in {"float16", "float32"}:
        raise ValueError("PYMSS_DML_MODEL_DTYPE/model_dtype must be auto, float16 or float32")
    return (value,)


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
    attempts = precision_attempts(request["params"].get("model_dtype"))
    temp_root = PROJECT_ROOT / ".codex" / "separation"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="worker-", dir=temp_root) as directory:
            request_path = Path(directory) / "request.json"
            result_path = Path(directory) / "result.json"
            for index, dtype in enumerate(attempts):
                payload = {**request, "params": {**request["params"], "model_dtype": dtype},
                           "allow_fp32_retry": index + 1 < len(attempts), "result_path": str(result_path)}
                request_path.write_text(json.dumps(payload), encoding="utf-8")
                result_path.unlink(missing_ok=True)
                kwargs = {"cwd": str(PROJECT_ROOT)}
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
    from rvc.runtime import InferenceConfig
    from .service import PrecisionRetry, run_batch

    path = Path((argv or sys.argv[1:])[0])
    request = json.loads(path.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if request["debug"] else logging.INFO, format="%(message)s")
    try:
        runtime = InferenceConfig.create(device="directml", device_id=request["device_ids"][0])
        result = run_batch(request, runtime)
        code = 1 if result["failed"] else 0
    except PrecisionRetry as error:
        result, code = {"error": str(error)}, 75
    except KeyboardInterrupt:
        result, code = {"error": "Interrupted"}, 130
    except Exception as error:
        result, code = {"error": str(error)}, 1
    Path(request["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
