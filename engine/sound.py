"""The clip's sound from its cue sheet: effects, voices and the score of the recipe's mood through the audio module,
mastered to -14 LUFS with the true peak of the encoded file held under -1 dBFS (the limiter target steps down until
it is: dry steel transients come out hot after AAC), then muxed into the videos with the picture stream copied.
"""
import json
import os
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'audio'))
import dsp  # noqa: E402
import mix  # noqa: E402
import sfx_city  # noqa: E402,F401  (registers the city / titan / shield kinds)
import scores  # noqa: E402
import config  # noqa: E402


def make_track(cues_path, out_dir, name, mood='ninja', log=print):
    scores.use(mood)
    sheet = json.load(open(cues_path, encoding='utf-8'))
    track = mix.render(sheet, seed=1, log=log)
    m4a = os.path.join(out_dir, f'{name}-track.m4a')
    loud = (None, None)
    tp = -2.0
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, 'mix.wav')
        wavfile.write(wav, dsp.SR, track.astype(np.float32))
        for tp in (-2.0, -3.0, -4.0, -5.0, -6.0):
            mix.master(wav, m4a, config.FFMPEG, lufs=-14.0, tp=tp)
            loud = mix.measure(m4a, config.FFMPEG)
            if loud[1] is not None and loud[1] <= -1.0:
                break
    log(f'AUDIO {len(track) / dsp.SR:.2f}s, {loud[0]} LUFS, true peak {loud[1]} dBFS (limiter target {tp} dBTP)')
    return m4a, loud


def mux(video, m4a, out):
    mix.mux(video, m4a, out, config.FFMPEG)
