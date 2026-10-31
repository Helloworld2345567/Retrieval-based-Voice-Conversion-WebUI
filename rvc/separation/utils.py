from contextlib import contextmanager, nullcontext
import numpy as np
import torch
import torch.nn as nn
from .progress import _ProgressContext

def _model_target(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def _getWindowingArray(window_size, fade_size):
    if fade_size <= 0:
        return torch.ones(window_size)
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def _build_chunk_plan(total_length, chunk_size, step, fade_size):
    starts = list(range(0, total_length, step))
    normal_window = _getWindowingArray(chunk_size, fade_size)

    def window_for(start):
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            return normal_window
        window = normal_window.clone()
        if start == 0:
            window[:fade_size] = 1
        if start + length >= total_length:
            window[max(0, length - fade_size):length] = 1
        return window
    return (starts, [window_for(start) for start in starts])

def _get_inference_step(config, chunk_size):
    overlap_size = int(config.inference.get('overlap_size', chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError('inference.overlap_size must be >= 0 and < audio.chunk_size')
    return chunk_size - overlap_size

def _complete_chunk_count(total_length, chunk_size, step):
    return 0 if total_length < chunk_size else (total_length - chunk_size) // step + 1

def _fold_windows(counter, windows, step, start_offset=0):
    n_chunks = windows.shape[0]
    if n_chunks == 0:
        return
    chunk_size = windows.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    folded_counter = nn.functional.fold(windows.transpose(0, 1).unsqueeze(0), output_size=(1, output_length), kernel_size=(1, chunk_size), stride=(1, step))
    counter[..., start_offset:start_offset + output_length] += folded_counter.view(1, 1, output_length)

def _fold_chunk_batch(result, chunks, windows, step, start_offset=0):
    n_chunks = chunks.shape[0]
    if n_chunks == 0:
        return
    chunk_size = chunks.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    n_sources, n_channels = chunks.shape[1:3]
    folded = nn.functional.fold((chunks * windows[:, None, None, :]).permute(1, 2, 3, 0).reshape(1, n_sources * n_channels * chunk_size, n_chunks), output_size=(1, output_length), kernel_size=(1, chunk_size), stride=(1, step))
    result[..., start_offset:start_offset + output_length] += folded.view(n_sources, n_channels, output_length)

def _ensure_source_dim(x, chunk_batch):
    return x.unsqueeze(1) if x.ndim == chunk_batch.ndim else x

def _fit_tensor_length(x, length):
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x

def _autocast(device, enabled):
    device_type = torch.device(device).type
    if enabled and device_type == 'cuda':
        return torch.amp.autocast(device_type, dtype=torch.float16)
    return nullcontext()

def _inference_context(device):
    if torch.device(device).type == 'privateuseone':
        return torch.no_grad()
    return torch.inference_mode()

def _source_names(config):
    return config.separation.instruments if config.separation.target_instrument is None else [config.separation.target_instrument]

def _normalize_source_indices(config, source_indices):
    if source_indices is None:
        return None
    source_count = len(_source_names(config))
    indices = tuple((int(index) for index in source_indices))
    if not indices:
        raise ValueError('source_indices must not be empty')
    if len(set(indices)) != len(indices):
        raise ValueError('source_indices must not contain duplicates')
    if min(indices) < 0 or max(indices) >= source_count:
        raise ValueError(f'source_indices must be in range [0, {source_count})')
    return indices

def _source_count(config, source_indices=None):
    return len(_source_names(config)) if source_indices is None else len(source_indices)

def _sources_to_dict(config, estimated_sources, source_indices=None):
    names = _source_names(config)
    if source_indices is not None:
        names = [names[index] for index in source_indices]
    return {k: v for k, v in zip(names, estimated_sources)}

def _prepare_mix_for_chunks(mix, border):
    length_init = mix.shape[-1]
    mix = mix.unsqueeze(0) if mix.ndim == 1 else mix
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
    return (mix, length_init)

def _init_overlap_buffers(config, mix, device, use_fast_path, source_indices=None):
    req_shape = (_source_count(config, source_indices),) + tuple(mix.shape)
    result_device = device if use_fast_path else 'cpu'
    counter_shape = (1, 1, mix.shape[1])
    result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
    counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
    return (result, counter)

def _model_mix(mix, device):
    return mix.to(device) if torch.device(device).type != 'cpu' else mix

@contextmanager
def _model_source_context(model, source_indices):
    target = _model_target(model)
    sentinel = object()
    previous = getattr(target, '_pymss_source_indices', sentinel)
    if source_indices is not None:
        target._pymss_source_indices = source_indices
    try:
        yield
    finally:
        if previous is sentinel:
            if hasattr(target, '_pymss_source_indices'):
                delattr(target, '_pymss_source_indices')
        else:
            target._pymss_source_indices = previous

def _select_sources(chunks, source_indices, already_selected=False):
    if source_indices is None or already_selected:
        return chunks
    index = torch.as_tensor(source_indices, device=chunks.device)
    return chunks.index_select(1, index)

def _run_model_chunk(model, arr, chunk_size, source_indices=None):
    target = _model_target(model)
    chunks = _fit_tensor_length(_ensure_source_dim(model(arr), arr).float(), chunk_size)
    already_selected = source_indices is not None and hasattr(target, '_active_source_indices') and (chunks.shape[1] == len(source_indices))
    return _select_sources(chunks, source_indices, already_selected=already_selected)

def _extract_chunk(mix, start, chunk_size):
    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start:start + chunk_size]
    if length == chunk_size:
        return (part, length)
    if length > chunk_size // 2 + 1:
        part = nn.functional.pad(part, (0, chunk_size - length), mode='reflect')
    else:
        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode='constant', value=0)
    return (part, length)

def _add_weighted_chunk(result, counter, chunk, window, start, length):
    device = result.device
    window = window.to(device=device, dtype=torch.float32)[:length]
    result[..., start:start + length] += chunk[..., :length].to(device=device, dtype=torch.float32) * window
    counter[..., start:start + length] += window

def _run_complete_chunks(model, mix, windows, result, counter, chunk_size, step, batch_size, progress, source_indices=None):
    n_chunks = _complete_chunk_count(mix.shape[1], chunk_size, step)
    if n_chunks == 0:
        return 0
    n_complete = n_chunks
    if len(windows) > n_chunks:
        n_complete -= n_complete % batch_size
    if n_complete == 0:
        return 0
    inputs = mix.unfold(-1, chunk_size, step).permute(1, 0, 2)[:n_complete]
    fold_windows = torch.stack(windows[:n_complete], dim=0).to(device=result.device, dtype=torch.float32)
    _fold_windows(counter, fold_windows, step)
    for batch_start in range(0, n_complete, batch_size):
        batch_end = min(batch_start + batch_size, n_complete)
        chunks = _run_model_chunk(model, inputs[batch_start:batch_end].contiguous(), chunk_size, source_indices)
        _fold_chunk_batch(result, chunks, fold_windows[batch_start:batch_end], step, start_offset=batch_start * step)
        progress.update(step * (batch_end - batch_start))
    return n_complete

def _run_tail_chunks(model, mix, starts, windows, result, counter, chunk_size, step, batch_size, first_chunk, progress, source_indices=None):
    for batch_start in range(first_chunk, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch = [(_extract_chunk(mix, starts[idx], chunk_size), idx) for idx in batch_indices]
        batch_data = [chunk for (chunk, _), _ in batch]
        chunks = _run_model_chunk(model, torch.stack(batch_data, dim=0), chunk_size, source_indices)
        for j, ((_, length), idx) in enumerate(batch):
            start = starts[idx]
            _add_weighted_chunk(result, counter, chunks[j], windows[idx], start, length)
        progress.update(step * len(batch_data))

def _finalize_overlap(result, counter, length_init, border):
    if length_init > 2 * border and border > 0:
        start, end = (border, border + length_init)
    else:
        start, end = (0, result.shape[-1])
    result = result[..., start:end]
    counter = counter[..., start:end]
    output_shape = result.shape[:-1] + (end - start,)
    if torch.device(result.device).type != 'cuda':
        estimated_sources = (result / counter).cpu().numpy()
        if not np.isfinite(estimated_sources).all():
            raise FloatingPointError('Model output contains NaN/Inf')
        return estimated_sources
    counter_min, counter_max = torch.aminmax(counter)
    divide_counter = bool((counter_min - 1).abs().item() > 1e-06 or (counter_max - 1).abs().item() > 1e-06)
    samples_per_chunk = max(1, 512 * 1024 * 1024 // (max(1, result.shape[0] * result.shape[1]) * 4))
    estimated_sources_t = torch.empty(output_shape, dtype=torch.float32, device='cpu')
    for offset in range(0, result.shape[-1], samples_per_chunk):
        chunk_end = min(offset + samples_per_chunk, result.shape[-1])
        source = result[..., offset:chunk_end]
        if divide_counter:
            source = source / counter[..., offset:chunk_end]
        estimated_sources_t[..., offset:chunk_end].copy_(source)
    estimated_sources = estimated_sources_t.numpy()
    if divide_counter:
        if not np.isfinite(estimated_sources).all():
            raise FloatingPointError('Model output contains NaN/Inf')
    return estimated_sources

def demix_track(config, model, mix, device, pbar=False, source_indices=None, progress_callback=None):
    C = config.audio.chunk_size
    sample_rate = int(config.audio.get('sample_rate', 44100))
    source_indices = _normalize_source_indices(config, source_indices)
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size
    mix, length_init = _prepare_mix_for_chunks(mix, border)
    chunk_starts, chunk_windows = _build_chunk_plan(mix.shape[1], C, step, fade_size)
    device_type = torch.device(device).type
    use_complete_fast_path = device_type in ('cuda', 'cpu')
    mix_device = _model_mix(mix, device)
    with _autocast(device, config.separation.get('use_amp', True)):
        with _inference_context(device):
            result, counter = _init_overlap_buffers(config, mix, device, use_complete_fast_path, source_indices)
            progress = _ProgressContext(pbar, mix.shape[1], progress_callback, sample_rate=sample_rate)
            with _model_source_context(model, source_indices):
                complete_chunks = 0
                if use_complete_fast_path:
                    complete_chunks = _run_complete_chunks(model, mix_device, chunk_windows, result, counter, C, step, batch_size, progress, source_indices)
                _run_tail_chunks(model, mix_device, chunk_starts, chunk_windows, result, counter, C, step, batch_size, complete_chunks, progress, source_indices)
                progress.emit(mix.shape[1])
            progress.close()
            estimated_sources = _finalize_overlap(result, counter, length_init, border)
    return _sources_to_dict(config, estimated_sources, source_indices)
