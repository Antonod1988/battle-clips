"""The clip's sound from its cue sheet: effects, voices and the score of the recipe's mood through the audio module,
mastered to -14 LUFS with the true peak of the encoded file held under -1 dBFS, then muxed into the videos with the
picture stream copied.

Dense dry steel (many short clicks) has a high crest factor: brought up to -14 LUFS its transients are far above the
body of the mix, and ffmpeg's limiter (1.5 ms attack) lets them through, so the AAC file peaked over 0 dBFS. So the
mix is first brought to its final loudness here and its peaks caught by a look-ahead limiter that watches a 4x
oversampled copy (true peaks), then mastered (the master's own gain is then about 0 dB).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d
from scipy.signal import lfilter, resample_poly

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'audio'))
sys.path.insert(0, ROOT)
import dsp  # noqa: E402
import mix  # noqa: E402
import sfx_city  # noqa: E402,F401  (registers the city / titan / shield kinds)
import scores  # noqa: E402
import config  # noqa: E402


def lufs_of(wav):
    p = subprocess.run([config.FFMPEG, '-hide_banner', '-nostdin', '-i', wav, '-af',
                        'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True)
    m = json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}', p.stderr)[-1])
    return float(m['input_i'])


def true_peak_limit(x, ceiling_db=-2.5, look=0.003, release=0.08):
    """A look-ahead limiter on the true peak: the peak is read from a 4x oversampled copy, the gain falls ahead of
    every peak (over `look` seconds) and recovers over `release`. x: (n, 2) float."""
    up = resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).max(axis=1)
    pk = pk[:len(pk) // 4 * 4].reshape(-1, 4).max(axis=1)
    if len(pk) < len(x):
        pk = np.concatenate([pk, np.full(len(x) - len(pk), pk[-1] if len(pk) else 0.0)])
    pk = pk[:len(x)]
    ceil = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    w = max(1, int(look * dsp.SR))
    need = -maximum_filter1d(-need, size=2 * w + 1, mode='nearest')     # the lowest gain within +-look
    a = np.exp(-1.0 / (release * dsp.SR))
    # recover slowly, fall at once: a one-pole smoother on the reduction, then never above the needed gain
    red = 1.0 - need
    sm = lfilter([1 - a], [1, -a], red)
    g = 1.0 - np.maximum(red, sm)
    return x * g[:, None]


def make_track(cues_path, out_dir, name, mood='ninja', log=print, lufs=-14.0):
    scores.use(mood)
    sheet = json.load(open(cues_path, encoding='utf-8'))
    track = mix.render(sheet, seed=1, log=log)
    m4a = os.path.join(out_dir, f'{name}-track.m4a')
    loud = (None, None)
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, 'raw.wav')
        wavfile.write(raw, dsp.SR, track.astype(np.float32))
        gain = lufs - lufs_of(raw)
        ceil = -2.5
        for _ in range(4):
            loud_t = true_peak_limit(track * 10 ** (gain / 20), ceil)
            wav = os.path.join(td, 'mix.wav')
            wavfile.write(wav, dsp.SR, loud_t.astype(np.float32))
            mix.master(wav, m4a, config.FFMPEG, lufs=lufs, tp=-1.5)
            loud = mix.measure(m4a, config.FFMPEG)
            if loud[1] is not None and loud[1] <= -1.0:
                break
            ceil -= 1.0
    log(f'AUDIO {len(track) / dsp.SR:.2f}s, {loud[0]} LUFS, true peak {loud[1]} dBFS (look-ahead ceiling {ceil} dBTP)')
    return m4a, loud


def mux(video, m4a, out):
    mix.mux(video, m4a, out, config.FFMPEG)


if __name__ == '__main__':
    # remake only the sound of a rendered clip:  python engine/sound.py <out_dir> <name> <mood>
    out_dir, name, mood = sys.argv[1:4]
    m4a, loud = make_track(os.path.join(out_dir, f'{name}-cues.json'), out_dir, name, mood)
    for vid, src in ((f'{name}-1080p.mp4', f'{name}-silent.mp4'), (f'{name}-under-30mb.mp4', f'{name}-under-30mb.mp4')):
        s = os.path.join(out_dir, src)
        if not os.path.exists(s):
            continue
        tmp = os.path.join(out_dir, 'remux-' + vid)
        mux(s, m4a, tmp)
        os.replace(tmp, os.path.join(out_dir, vid))
        print('muxed', vid, os.path.getsize(os.path.join(out_dir, vid)))
