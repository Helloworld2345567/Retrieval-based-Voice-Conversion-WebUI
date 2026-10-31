"""CLI parsing is lightweight; device/model imports occur only for inference."""

import argparse
import json
import logging
from pathlib import Path
import sys

from .catalog import MODELS, get_model, model_path
from .request import AudioEncodingParams, InferenceParams, SeparationRequest, parse_params


def build_parser(*, legacy=False):
    parser = argparse.ArgumentParser(prog="pymss" if legacy else "rvc separate", description="Five-model vocal separation and dereverb inference.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List the five supported presets.")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--all", action="store_true", help="List all five supported presets.")
    listing.add_argument("--category", help="Filter by catalog category.")
    info = commands.add_parser("info", help="Show a preset's model and output paths.")
    info.add_argument("model")
    info.add_argument("--model-dir")
    download = commands.add_parser("download", help="Download one preset, or all five.")
    download.add_argument("model")
    download.add_argument("--force", action="store_true")
    download.add_argument("--supported-only", action="store_true")
    infer = commands.add_parser("infer", help="Separate one audio file or a directory.")
    infer.add_argument("model")
    for command in (download, infer):
        command.add_argument("--model-dir", help="Model directory; defaults to assets/pymss_weights or PYMSS_MODEL_DIR.")
        command.add_argument("--source", choices=["modelscope", "huggingface", "hf-mirror"], default="modelscope")
        command.add_argument("--endpoint", help="Custom download endpoint serving catalog relative paths.")
    infer.add_argument("-i", "--input", required=True)
    infer.add_argument("-o", "--output", default="results")
    infer.add_argument("--download", action="store_true", help="Ensure weights exist; missing weights download automatically.")
    infer.add_argument("--save-as-folder", action="store_true")
    infer.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    infer.add_argument("--device-id", dest="device_ids", type=int, action="append", help="Device index; repeat for CUDA multi-GPU.")
    infer.add_argument("--format", choices=["wav", "flac", "mp3", "m4a"], default="wav", dest="output_format")
    infer.add_argument("--wav-bit-depth", choices=["FLOAT", "PCM_16", "PCM_24"], default="FLOAT")
    infer.add_argument("--flac-bit-depth", choices=["PCM_16", "PCM_24"], default="PCM_16")
    infer.add_argument("--mp3-bit-rate", default="320k")
    infer.add_argument("--m4a-bit-rate", default="512k")
    infer.add_argument("--m4a-codec", default="aac")
    infer.add_argument("--m4a-aac-at-quality", type=int, default=2)
    infer.add_argument("--tta", action="store_true")
    infer.add_argument("--debug", action="store_true")
    infer.add_argument("--param", action="append", default=[], help="Inference override, e.g. batch_size=1.")
    return parser


def describe(spec, directory=None):
    return {"name": spec.filename, "preset": spec.name, "label": spec.label,
            "model_type": spec.model_type, "supported": True,
            "model_path": str(model_path(spec, directory)), "config_path": str(spec.config_path),
            "stems": spec.stems, "output_suffixes": spec.suffixes, "size_bytes": spec.size_bytes}


def _build_request(args, *, legacy):
    from .service import collect_inputs

    return SeparationRequest(
        model=get_model(args.model).name,
        inputs=tuple(collect_inputs(args.input)),
        output=str(Path(args.output).expanduser().resolve()),
        format=args.output_format,
        legacy=legacy,
        save_as_folder=args.save_as_folder,
        device_ids=tuple(args.device_ids or [0]),
        params=args.inference_params,
        tta=args.tta,
        debug=args.debug,
        audio_params=args.audio_params,
    )


def cmd_infer(args, *, legacy, parser):
    from .service import validate_outputs

    spec = get_model(args.model)
    request = _build_request(args, legacy=legacy)
    validate_outputs(request)

    from rvc.core.devices import DeviceConfig

    runtime = DeviceConfig.create(device=args.device, device_id=args.device_ids[0] if args.device_ids else None)
    selected_type = str(runtime.device).split(":", 1)[0]
    if len(request.device_ids) > 1 and selected_type != "cuda":
        parser.error("Multiple device IDs require CUDA")
    chunk_size = request.params.get("chunk_size", 88200 if selected_type == "privateuseone" else 352800)
    overlap_size = request.params.get("overlap_size", 22050 if selected_type == "privateuseone" else spec.overlap_size)
    if overlap_size >= chunk_size:
        parser.error("overlap_size must be smaller than chunk_size")
    if args.device_ids is None and selected_type in {"cuda", "privateuseone"}:
        request = request.with_updates(device_ids=(runtime.device.index or 0,))

    path = model_path(spec, args.model_dir)
    if not path.is_file():
        from .download import download_model

        path = download_model(spec, args.model_dir, source=args.source, endpoint=args.endpoint)
    request = request.with_updates(model_path=str(path))
    if selected_type == "privateuseone":
        from .worker import run_directml

        result = run_directml(request)
    else:
        from .service import run_batch

        result = run_batch(request, runtime)
    print(f"Processed {result['successful']} file(s), failed {result['failed']}.")
    return 1 if result["failed"] or not result["successful"] else 0


def main(argv=None, *, legacy=False):
    parser = build_parser(legacy=legacy)
    args = parser.parse_args(argv)
    if args.command == "infer":
        try:
            args.inference_params = InferenceParams.from_mapping(parse_params(args.param))
            args.audio_params = AudioEncodingParams(
                wav_bit_depth=args.wav_bit_depth,
                flac_bit_depth=args.flac_bit_depth,
                mp3_bit_rate=args.mp3_bit_rate,
                m4a_bit_rate=args.m4a_bit_rate,
                m4a_codec=args.m4a_codec,
                m4a_aac_at_quality=args.m4a_aac_at_quality,
            )
            device_ids = tuple(args.device_ids or [0])
            if any(index < 0 for index in device_ids) or len(set(device_ids)) != len(device_ids):
                raise ValueError("Device IDs must be distinct nonnegative integers")
            if len(device_ids) > 1 and args.device in {"cpu", "directml"}:
                raise ValueError("Multiple device IDs require CUDA")
        except ValueError as error:
            parser.error(str(error))
    try:
        if args.command == "list":
            models = [m for m in MODELS if not args.category or args.category.lower() in m.remote_path.lower()]
            if args.json:
                print(json.dumps([describe(m) for m in models], ensure_ascii=False, indent=2))
            else:
                for model in models:
                    print(f"{model.filename if legacy else model.name}\t{model.label}\t{model.model_type}")
            return 0
        if args.command == "info":
            print(json.dumps(describe(get_model(args.model), args.model_dir), ensure_ascii=False, indent=2))
            return 0
        if args.command == "download":
            from .download import download_model

            failed = 0
            for spec in MODELS if args.model == "all" else (get_model(args.model),):
                try:
                    print(download_model(spec, args.model_dir, source=args.source, endpoint=args.endpoint, force=args.force))
                except Exception as error:
                    failed += 1
                    print(f"Cannot download {spec.filename}: {error}", file=sys.stderr)
            return 1 if failed else 0
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
        return cmd_infer(args, legacy=legacy, parser=parser)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"rvc separate: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
