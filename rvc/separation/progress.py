import math
from time import time
from tqdm.auto import tqdm

def _format_progress_time(value):
    seconds = max(0, int(round(value or 0)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:d}:{seconds:02d}'

def _format_progress_rtf(bar):
    done = float(bar.n or 0)
    if done <= 0:
        return '--'
    elapsed = max(0.0, time() - getattr(bar, 'start_t', time()))
    return f'{elapsed / done:.2f}'

def _update_time_progress_bar(bar):
    if bar is None:
        return
    rtf = _format_progress_rtf(bar)
    bar._pymss_audio_progress = f'{_format_progress_time(bar.n)}/{_format_progress_time(bar.total)}, RTF={rtf}'
    bar.refresh()

class _TimeProgressTqdm(tqdm):

    @property
    def format_dict(self):
        data = super().format_dict
        data['audio_progress'] = getattr(self, '_pymss_audio_progress', '')
        return data

class _ProgressContext:

    def __init__(self, pbar=False, total=1, callback=None, done=0, message='Processing audio', sample_rate=None):
        self.enabled = bool(pbar or callback)
        self.bar = None
        self.callback = callback
        self.done = done
        self.total = total
        self.message = message
        self.sample_rate = int(sample_rate or 0)
        if not self.enabled:
            return
        self.total = int(self.total or 1)
        self.done = min(max(0, int(self.done or 0)), self.total)
        if pbar:
            bar_kwargs = {'total': self._display_total(), 'desc': message, 'leave': False}
            if self.sample_rate > 0:
                bar_kwargs.update({'unit': '', 'bar_format': '{l_bar}{bar}| {audio_progress}'})
                self.bar = _TimeProgressTqdm(**bar_kwargs)
            else:
                self.bar = tqdm(**bar_kwargs)
            if self.sample_rate > 0:
                _update_time_progress_bar(self.bar)
        if self.bar is not None and self.done:
            self.bar.update(self._display_value(self.done))
            if self.sample_rate > 0:
                _update_time_progress_bar(self.bar)
        self.emit()

    def _display_value(self, value):
        if self.sample_rate <= 0:
            return int(value)
        if int(value) >= self.total:
            return self._display_total()
        return min(self._display_total(), int(int(value) // self.sample_rate))

    def _display_total(self):
        if self.sample_rate <= 0:
            return self.total
        return max(1, int(math.ceil(self.total / self.sample_rate)))

    def emit(self, done=None):
        if not self.enabled:
            return
        if done is not None:
            next_done = min(max(0, int(done)), self.total)
            if self.bar is not None:
                self.bar.update(self._display_value(next_done) - self._display_value(self.done))
                if self.sample_rate > 0:
                    _update_time_progress_bar(self.bar)
            self.done = next_done
        if self.callback is None:
            return
        self.callback(self._display_value(self.done), self._display_total(), self.message)

    def update(self, amount):
        if not self.enabled:
            return
        amount = int(amount)
        self.emit(self.done + amount)

    def close(self):
        if not self.enabled:
            return
        if self.bar:
            self.bar.close()
