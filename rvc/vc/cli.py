"""Command-line file conversion; importing the parser does not load models."""

import argparse
import logging
from pathlib import Path
import sys

from rvc.core.paths import PROJECT_ROOT, index_roots, resource_path
from .checkpoints import find_index, load_checkpoint, select_speaker


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
OUTPUT_FORMATS = {"wav", "flac", "mp3", "m4a"}


def build_parser():
    parser = argparse.ArgumentParser(prog="rvc convert", description="Offline RVC conversion for single-speaker and multi-speaker models.")
    parser.add_argument("--model", required=True, help="Model filename or .pth path.")
    parser.add_argument("--input", help="Input audio file or directory.")
    parser.add_argument("--output", help="Output file or directory.")
    parser.add_argument("--speaker-id", type=int, help="Speaker ID; defaults to the smallest declared ID, or 0.")
    parser.add_argument("--list-speakers", action="store_true", help="List model speakers and exit.")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--f0-method", choices=["pm", "rmvpe", "fcpe"], default="rmvpe")
    parser.add_argument("--index", help="Added index path; omitted selects a matching speaker index.")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--format", dest="output_format", choices=sorted(OUTPUT_FORMATS))
    parser.add_argument("--recursive", action="store_true", help="Scan input subdirectories.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    parser.add_argument("--device-id", type=int, help="CUDA or DirectML adapter index.")
    return parser


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_model(value, base):
    path = resolve_path(value, base)
    if not path.is_file():
        path = (resource_path("weight_root", "assets/weights", base=base) / value).resolve()
    if not path.is_file():
        raise FileNotFoundError("Model not found: %s" % value)
    if path.suffix.lower() != ".pth":
        raise ValueError("Model must be a .pth file: %s" % path)
    return path


def resolve_index(value, model_name, speaker_id, index_rate, base):
    if index_rate == 0:
        return None
    if value:
        path = resolve_path(value, base)
        if "trained" in path.name:
            path = path.with_name(path.name.replace("trained", "added"))
    else:
        path = find_index(model_name, speaker_id, index_roots(base=base))
    if path is None or not path.is_file():
        raise FileNotFoundError("No usable added index for speaker ID %s. Supply --index or use --index-rate 0." % speaker_id)
    return path.resolve()


def collect_jobs(input_value, output_value, output_format, recursive, base):
    input_path = resolve_path(input_value, base)
    output_path = resolve_path(output_value, base)
    if not input_path.exists():
        raise FileNotFoundError("Input not found: %s" % input_path)
    if input_path.is_file():
        suffix = output_path.suffix.lower().lstrip(".") if not output_path.is_dir() else ""
        if suffix and suffix not in OUTPUT_FORMATS:
            raise ValueError("Unsupported output suffix: %s" % output_path.suffix)
        selected_format = output_format or suffix or "wav"
        output_file = output_path.with_suffix("." + selected_format) if suffix else output_path / (input_path.stem + "." + selected_format)
        jobs = [(input_path, output_file, selected_format)]
    elif input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("Directory input requires an output directory")
        selected_format = output_format or "wav"
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        jobs = []
        for path in sorted(iterator):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if input_path in output_path.parents and output_path in path.parents:
                continue
            relative_parent = path.relative_to(input_path).parent if recursive else Path()
            jobs.append((path, output_path / relative_parent / (path.stem + "." + selected_format), selected_format))
        if not jobs:
            raise ValueError("No supported audio files found in: %s" % input_path)
    else:
        raise ValueError("Input must be a file or directory")
    outputs = set()
    inputs = {source for source, _, _ in jobs}
    for _, destination, _ in jobs:
        if destination in inputs:
            raise ValueError("Output would overwrite an input file: %s" % destination)
        if destination in outputs:
            raise ValueError("Multiple inputs resolve to the same output: %s" % destination)
        outputs.add(destination)
    return jobs


def _validate_arguments(parser, args):
    if not args.list_speakers and (not args.input or not args.output):
        parser.error("--input and --output are required unless --list-speakers is used")
    if not 0 <= args.index_rate <= 1:
        parser.error("--index-rate must be between 0 and 1")
    if not 0 <= args.rms_mix_rate <= 1:
        parser.error("--rms-mix-rate must be between 0 and 1")
    if not 0 <= args.protect <= 0.5:
        parser.error("--protect must be between 0 and 0.5")
    if args.resample_sr != 0 and args.resample_sr < 16000:
        parser.error("--resample-sr must be 0 or at least 16000")
    if args.device_id is not None and args.device_id < 0:
        parser.error("--device-id must be nonnegative")


def _run(args, base):
    model_path = resolve_model(args.model, base)
    checkpoint, metadata = load_checkpoint(model_path)
    if args.list_speakers:
        if metadata.speakers:
            for speaker in metadata.speakers:
                print("%s\t%s" % (speaker.id, speaker.name))
        else:
            print("0-%s" % (metadata.speaker_count - 1))
        return 0
    speaker_id = select_speaker(metadata, args.speaker_id)
    index_path = resolve_index(args.index, model_path.name, speaker_id, args.index_rate, base)
    jobs = collect_jobs(args.input, args.output, args.output_format, args.recursive, base)
    for _, destination, _ in jobs:
        if destination.exists() and (not args.overwrite or not destination.is_file()):
            raise FileExistsError("Output already exists; use --overwrite for files: %s" % destination)

    from .audio import write_audio
    from .config import VoiceConversionConfig
    from .engine import VoiceConverter

    config = VoiceConversionConfig.create(args.device, args.device_id, resource_base=base)
    print("Device: %s | Precision: %s" % (config.device, config.dtype), file=sys.stderr)
    failed = 0
    with VoiceConverter(model_path, config, checkpoint=checkpoint) as converter:
        del checkpoint
        for source, destination, output_format in jobs:
            try:
                result = converter.convert(
                    source,
                    speaker_id=speaker_id,
                    pitch=args.pitch,
                    f0_method=args.f0_method,
                    index_path=index_path,
                    index_rate=args.index_rate,
                    resample_sr=args.resample_sr,
                    rms_mix_rate=args.rms_mix_rate,
                    protect=args.protect,
                )
                write_audio(destination, result.audio, result.sample_rate, output_format)
                print(destination)
                print("Features %.2fs | F0 %.2fs | Synthesis %.2fs" % result.timings, file=sys.stderr)
            except Exception as error:
                failed += 1
                print("rvc convert: %s: %s" % (source, error), file=sys.stderr)
    return 1 if failed else 0


def main(argv=None, *, legacy=False):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return _run(args, PROJECT_ROOT if legacy else Path.cwd())
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("rvc convert: error: %s" % error, file=sys.stderr)
        return 1
