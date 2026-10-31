import gc
import logging
import re
from collections import deque
from contextlib import contextmanager
import numpy as np
import torch
from .config import load_config
from .utils import demix_track
OUTPUT_NORMALIZE_PEAK = 10 ** (-0.01 / 20)
_DML_CACHE_ATTRIBUTE_NAMES = frozenset({'cache', '_group_cache', '_layer_group_cache', '_index_cache', '_packed_layer_group_cache', '_stft_window_cache', '_pymss_cos_sin_cache', '_gamma_dtype_cache'})

def _device_type(device):
    try:
        return torch.device(device).type
    except (RuntimeError, TypeError):
        return str(device).split(':', 1)[0].lower()

def _clear_cache_container(value, seen):
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, dict):
        for cached_value in tuple(value.values()):
            _clear_cache_container(cached_value, seen)
        value.clear()
    elif isinstance(value, (list, set, deque)):
        for cached_value in tuple(value):
            _clear_cache_container(cached_value, seen)
        value.clear()
    elif isinstance(value, tuple):
        for cached_value in value:
            _clear_cache_container(cached_value, seen)

def _clear_dml_model_caches(model, logger=None):
    if model is None:
        return 0
    candidates = [model]
    model_run = getattr(model, 'model_run', None)
    if model_run is not None and model_run is not model:
        candidates.append(model_run)
    cleared = 0
    seen_modules = set()
    for candidate in candidates:
        modules = getattr(candidate, 'modules', None)
        if not callable(modules):
            continue
        try:
            module_items = tuple(modules())
        except Exception as exc:
            if logger is not None:
                logger.debug(f'Could not enumerate DirectML model caches during close: {exc}')
            continue
        for module in module_items:
            module_id = id(module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            try:
                attributes = tuple(vars(module).items())
            except TypeError:
                continue
            for name, value in attributes:
                is_cache = name in _DML_CACHE_ATTRIBUTE_NAMES or name.endswith('_cache')
                if not is_cache or not isinstance(value, (dict, list, set, deque)):
                    continue
                try:
                    _clear_cache_container(value, set())
                    cleared += 1
                except Exception as exc:
                    if logger is not None:
                        logger.debug(f'Could not clear DirectML cache {name}: {exc}')
    return cleared

def _unwrap_state_dict(state_dict):
    for key in ('state', 'state_dict', 'model_state_dict'):
        if key in state_dict:
            return state_dict[key]
    return state_dict

@contextmanager
def _skip_torch_default_init():
    classes = (torch.nn.Linear, torch.nn.Bilinear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.Embedding, torch.nn.EmbeddingBag, torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM, torch.nn.MultiheadAttention)
    saved = {cls: cls.reset_parameters for cls in classes if hasattr(cls, 'reset_parameters')}
    try:
        for cls in saved:
            cls.reset_parameters = lambda self: None
        yield
    finally:
        for cls, reset_parameters in saved.items():
            cls.reset_parameters = reset_parameters

def _infer_mel_band_roformer_mlp_hidden_layers(state_dict):
    pattern = re.compile('(?:^|\\.)mask_estimators\\.0\\.to_freqs\\.0\\.0\\.(\\d+)\\.weight$')
    layer_indices = sorted({int(match.group(1)) for key in state_dict for match in [pattern.search(key)] if match})
    if not layer_indices:
        return None
    return len(layer_indices) - 1

def _coerce_low_precision_to_float32(module):
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is not None and param.dtype in {torch.float16, torch.bfloat16}:
                child._parameters[name] = torch.nn.Parameter(param.detach().float(), requires_grad=param.requires_grad)
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.dtype in {torch.float16, torch.bfloat16}:
                child._buffers[name] = buffer.float()

def _normalize_model_dtype(value):
    normalized = str(value or 'auto').strip().lower().replace('torch.', '')
    aliases = {'auto': 'auto', 'fp16': 'float16', 'half': 'float16', 'float16': 'float16', 'fp32': 'float32', 'float': 'float32', 'float32': 'float32'}
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("model_dtype must be 'auto', 'float16', or 'float32'") from error

def _prepare_mix_channels(mix, is_stereo, logger):
    if is_stereo and len(mix.shape) == 1:
        logger.warning('Track is mono, but model is stereo, adding a second channel.')
        return np.stack([mix, mix], axis=0)
    if is_stereo and mix.ndim == 2 and mix.shape[0] > 2:
        logger.warning('Track has more than 2 channels, taking mean of all channels and adding a second channel.')
        mono = np.mean(mix, axis=0)
        return np.stack([mono, mono], axis=0)
    if not is_stereo and len(mix.shape) != 1:
        logger.warning('Track has more than 1 channels, but model is mono, taking mean of all channels.')
        return np.mean(mix, axis=0)
    return mix

def _standardize_mix(mix, enabled, logger):
    if not enabled:
        return (mix, None)
    mono = mix.mean(0)
    mean = mono.mean()
    std = mono.std()
    if std == 0:
        return mix, None
    logger.debug(f'Standardize mix with mean: {mean}, std: {std}')
    return ((mix - mean) / std, (mean, std))

def _normalize_outputs(results, enabled, logger, target_peak=OUTPUT_NORMALIZE_PEAK):
    if not enabled:
        return results
    peak = max((float(np.max(np.abs(np.asarray(audio)))) for audio in results.values() if np.asarray(audio).size), default=0.0)
    if peak <= 0.0 or not np.isfinite(peak):
        logger.debug('Skipping output normalize because peak is zero or not finite.')
        return results
    gain = target_peak / peak
    logger.debug(f'Normalize output stems with peak: {peak}, target_peak: {target_peak}, gain: {gain}')
    return {stem: np.asarray(audio) * gain for stem, audio in results.items()}

def _destandardize(estimates, stats):
    return estimates if stats is None else estimates * stats[1] + stats[0]

def _tta_variants(mix, use_tta, logger):
    if not use_tta:
        return [mix.copy()]
    variants = [mix.copy(), mix[::-1].copy(), -1.0 * mix.copy()]
    logger.debug(f'User needs to apply TTA, total tracks: {len(variants)}')
    return variants

def _merge_tta_results(results):
    waveforms = results[0]
    for index, result in enumerate(results[1:], start=1):
        for stem, audio in result.items():
            waveforms[stem] += audio[::-1].copy() if index == 1 else -1.0 * audio
    for stem in waveforms:
        waveforms[stem] /= len(results)
    return waveforms

def _build_results(waveforms, instruments, mix_orig, config, standardize_stats, logger):
    target_instrument = config.separation.target_instrument
    if target_instrument is None:
        return {instr: _destandardize(waveforms[instr].T, standardize_stats) for instr in instruments}
    results = {}
    target_audio = _destandardize(waveforms[target_instrument].T, standardize_stats)
    if target_instrument in instruments:
        results[target_instrument] = target_audio
    other_instruments = [instr for instr in config.separation.instruments if instr != target_instrument]
    logger.debug(f'target_instrument is not null, extracting instrumental from {target_instrument}, other_instruments: {other_instruments}')
    if other_instruments:
        secondary = other_instruments[0]
        if secondary in instruments:
            results[secondary] = mix_orig.T - target_audio
    return results

def _resolve_instruments(config, stems=None):
    instruments = config.separation.instruments.copy()
    if stems is None:
        source_indices = None if config.separation.target_instrument is None else (0,)
        return (instruments, source_indices)
    stem_list = [stems] if isinstance(stems, str) else list(stems)
    lower_to_index = {instr.lower(): index for index, instr in enumerate(instruments)}
    selected, indices = ([], [])
    for stem in stem_list:
        key = stem.lower()
        if key not in lower_to_index:
            raise ValueError(f'Invalid instrument key: {stem}. Valid instrument keys: {instruments}')
        index = lower_to_index[key]
        if index in indices:
            continue
        selected.append(instruments[index])
        indices.append(index)
    if not selected:
        raise ValueError('stems must not be empty')
    source_indices = tuple(indices) if config.separation.target_instrument is None else (0,)
    return (selected, source_indices)


class MSSeparator:
    """Load one preset once and separate any number of waveforms."""

    def __init__(self, spec, model_path, runtime, *, device_ids=None,
                 inference_params=None, use_tta=False, progress_callback=None):
        from .models.bs_roformer import BSRoformer, MelBandRoformer

        self.spec = spec
        self.model_type = spec.model_type
        self.device = runtime.device
        self.logger = logging.getLogger("rvc.separation")
        self.use_tta = use_tta
        self.progress_callback = progress_callback
        self.model = None
        self.config = load_config(spec.config_path)
        self._cudnn_benchmark_initial = torch.backends.cudnn.benchmark
        params = dict(inference_params or {})
        self.output_normalize = params.pop("normalize", False)
        allowed = {"batch_size", "chunk_size", "overlap_size", "standardize", "use_amp",
                   "model_dtype", "cuda_attention_backend", "stem_batch_size"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError("Unsupported inference parameter(s): " + ", ".join(sorted(unknown)))
        self.stem_batch_size = params.pop("stem_batch_size", 0)
        is_dml = _device_type(self.device) == "privateuseone"
        self.config.audio.chunk_size = params.pop("chunk_size", 88200 if is_dml else 352800)
        self.config.inference.batch_size = params.pop("batch_size", 1 if is_dml else spec.batch_size)
        self.config.inference.overlap_size = params.pop("overlap_size", 22050 if is_dml else spec.overlap_size)
        self.config.inference.normalize = params.pop("standardize", False)
        self.config.separation.use_amp = bool(params.pop("use_amp", runtime.is_half and _device_type(self.device) == "cuda"))
        if self.config.audio.chunk_size <= 0 or self.config.inference.batch_size <= 0:
            raise ValueError("chunk_size and batch_size must be positive")
        if not 0 <= self.config.inference.overlap_size < self.config.audio.chunk_size:
            raise ValueError("overlap_size must be nonnegative and smaller than chunk_size")
        dtype = _normalize_model_dtype(params.pop("model_dtype", "auto"))
        try:
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
            except (TypeError, ValueError, RuntimeError):
                state = torch.load(model_path, map_location="cpu", weights_only=True)
            state = _unwrap_state_dict(state)
            kwargs = dict(self.config.model)
            if spec.model_type == "mel_band_roformer":
                kwargs["zero_dc"] = False
                kwargs["mlp_hidden_layers"] = _infer_mel_band_roformer_mlp_hidden_layers(state)
            with _skip_torch_default_init():
                self.model = (BSRoformer if spec.model_type == "bs_roformer" else MelBandRoformer)(**kwargs)
            self.model.load_state_dict(state, strict=True, assign=True)
            del state
            if _device_type(self.device) == "cpu" or (is_dml and dtype != "float16"):
                _coerce_low_precision_to_float32(self.model)
            elif is_dml:
                self.model.half()
            backend = params.get("cuda_attention_backend", "default")
            for module in self.model.modules():
                if hasattr(module, "set_cuda_attention_backend"):
                    module.set_cuda_attention_backend(backend)
            self.model = self.model.to(self.device).eval()
            if _device_type(self.device) == "cuda" and device_ids and len(device_ids) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            torch.backends.cudnn.benchmark = True
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def separate(self, mix, pbar=False, stems=None):
        if not np.asarray(mix).size or not np.isfinite(mix).all():
            raise ValueError("Input audio is empty or contains NaN/Inf")
        mix = _prepare_mix_channels(mix, self.config.model.get("stereo", True), self.logger)
        instruments, source_indices = _resolve_instruments(self.config, stems)
        original = mix.copy()
        mix, stats = _standardize_mix(mix, self.config.inference.normalize, self.logger)
        variants = [demix_track(self.config, self.model, torch.tensor(track, dtype=torch.float32),
                               self.device, pbar=pbar, source_indices=source_indices,
                               progress_callback=self.progress_callback)
                    for track in _tta_variants(mix, self.use_tta, self.logger)]
        results = _build_results(_merge_tta_results(variants), instruments, original,
                                 self.config, stats, self.logger)
        results = _normalize_outputs(results, self.output_normalize, self.logger)
        if any(not np.isfinite(value).all() for value in results.values()):
            raise FloatingPointError("Model output contains NaN/Inf")
        return results

    def close(self):
        model, self.model = self.model, None
        try:
            if _device_type(self.device) == "privateuseone":
                _clear_dml_model_caches(model, self.logger)
            if model is not None:
                model.to("cpu")
        finally:
            del model
            torch.backends.cudnn.benchmark = self._cudnn_benchmark_initial
            gc.collect()
            if _device_type(self.device) == "cuda":
                torch.cuda.empty_cache()
