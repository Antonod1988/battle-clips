"""The mixer: a renderer's cue sheet in, a mastered track out (and muxed into the video).

Cue sheet (JSON, times in video seconds):
    {"duration": 43.77, "cues": [{"t": 3.93, "kind": "hit", "pan": -0.2, "weight": 2, ...}, ...]}
Kinds: every name in sfx.REGISTRY (an effect at t, panned by "pan", or moving from "pan" to "pan_end"; "dur" for
sounds that last), "say" (an announcer line: "text", "voice" = a voices.STYLES name) and "music" ("section" for
music.score). Unknown kinds are listed and skipped.

Mix: effects (with a room reverb send) + voices + music ducked under the voices and the big hits, then glue
compression, a limiter, ffmpeg's two-pass loudnorm to the target LUFS, AAC.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

import dsp
import sfx
import voice
import music

BIG = {'stone_crash', 'floor_slam', 'freeze', 'ice_shatter', 'pierce', 'fire_breath', 'brazier_crash', 'skull',
       'bone_pile', 'slam'}


def render(sheet, seed=1, music_db=-6.0, voice_db=6.0, sfx_db=0.0, log=print, stems=None):
    dur = float(sheet['duration'])
    n = dsp.n_of(dur + 0.5)
    dry = np.zeros((n, 2))
    send = np.zeros((n, 2))
    vox = np.zeros((n, 2))
    sections = []
    big = np.zeros(n)
    unknown = {}
    for i, c in enumerate(sheet['cues']):
        kind = c['kind']
        t = float(c['t'])
        if kind == 'music':
            sections.append((t, c.get('section', 'fight')))
            continue
        if kind == 'say':
            y = voice.line(c['text'].title() if c['text'].isupper() else c['text'], c.get('voice', 'announcer'))
            dsp.place(vox, y, t - 0.03, dsp.db(voice_db))
            continue
        if kind not in sfx.REGISTRY:
            unknown[kind] = unknown.get(kind, 0) + 1
            continue
        fn, gain, rv = sfx.REGISTRY[kind]
        y = fn(c, np.random.default_rng(seed * 100003 + i))
        y = y / (np.max(np.abs(y)) + 1e-9)
        p = float(c.get('pan', 0.0)) * 0.75
        st = dsp.pan_moving(y, p, float(c['pan_end']) * 0.75) if 'pan_end' in c else dsp.pan(y, p)
        g = dsp.db(gain + sfx_db)
        dsp.place(dry, st, t, g)
        dsp.place(send, st, t, g * rv)
        if kind in BIG or (kind == 'hit' and int(c.get('weight', 0)) >= 2):
            i0 = int(t * dsp.SR)
            m = min(n - i0, dsp.n_of(0.6))
            if m > 0:
                big[i0:i0 + m] = np.maximum(big[i0:i0 + m], np.exp(-np.arange(m) / (0.18 * dsp.SR)))
    if unknown:
        log(f'AUDIO unknown cue kinds skipped: {unknown}')
    room = dsp.ir(1.3, 0.38, damp=5500, rng=np.random.default_rng(3))
    wet = dsp.reverb(send, room, wet=1.0) - send
    fx_bus = dry + wet * 0.9
    bed = music.score(sections, dur + 0.5, seed=seed)
    if len(bed) < n:
        bed = np.concatenate([bed, np.zeros((n - len(bed), 2))])
    bed = bed[:n] / (np.max(np.abs(bed)) + 1e-9)
    # ducking: the music dips under the voices and the big hits
    venv = dsp.follower(vox, 0.03, 0.45)
    venv = venv / (np.max(venv) + 1e-9)
    duck = 1.0 - 0.72 * np.clip(venv * 3.0, 0, 1) - 0.4 * big
    duck = np.clip(duck, 0.18, 1.0)
    fx_duck = 1.0 - (1.0 - dsp.db(-9.0)) * np.clip(venv * 3.0, 0, 1)      # the effects step back for a line
    if stems is not None:
        stems.update(fx=fx_bus * fx_duck[:, None], vox=vox, music=bed * dsp.db(music_db) * duck[:, None])
    mixd = fx_bus * fx_duck[:, None] + vox + bed * dsp.db(music_db) * duck[:, None]
    mixd = dsp.compress(mixd, threshold_db=-14, ratio=2.5, attack=0.008, release=0.18, makeup_db=2.0)
    mixd = mixd / (np.max(np.abs(mixd)) + 1e-9) * 0.9
    mixd = dsp.limit(mixd, -1.0)
    return mixd[:dsp.n_of(dur)]


def master(wav_in, out_m4a, ffmpeg, lufs=-14.0, tp=-1.0, kbps=192):
    """Loudness to `lufs` (EBU R128) with the true peak held under `tp`: measure, apply the gain into a limiter with
    headroom for the AAC encoder, encode, measure the encoded file and correct the gain once."""
    p = subprocess.run([ffmpeg, '-hide_banner', '-nostdin', '-i', wav_in, '-af',
                        f'loudnorm=I={lufs}:TP={tp}:LRA=11:print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True)
    m = json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}', p.stderr)[-1])
    gain = lufs - float(m['input_i'])
    ceiling = 10 ** ((tp - 0.8) / 20)
    loud = None
    for _ in range(3):
        af = (f'volume={gain:.2f}dB,alimiter=limit={ceiling:.4f}:attack=1.5:release=60:level=false:asc=1,'
              'aresample=48000')
        subprocess.check_call([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', wav_in, '-af', af,
                               '-c:a', 'aac', '-b:a', f'{kbps}k', '-threads', '2', out_m4a])
        loud = measure(out_m4a, ffmpeg)
        if loud[0] is None or abs(loud[0] - lufs) <= 0.15:
            break
        gain += lufs - loud[0]
    return m, loud


def measure(path, ffmpeg):
    p = subprocess.run([ffmpeg, '-hide_banner', '-nostdin', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True)
    tail = p.stderr[p.stderr.rfind('Summary:'):]
    i = re.search(r'I:\s+(-?[\d.]+) LUFS', tail)
    tpk = re.search(r'Peak:\s+(-?[\d.]+) dBFS', tail)
    return (float(i.group(1)) if i else None, float(tpk.group(1)) if tpk else None)


def mux(video, audio, out, ffmpeg):
    subprocess.check_call([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', video, '-i', audio,
                           '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy', '-shortest',
                           '-movflags', '+faststart', out])


def build(cues_path, out_dir, videos, name, lufs=-14.0, seed=1, ffmpeg=voice.FFMPEG, log=print):
    """videos: {output file name: source video}; writes the mixed track and each video with it into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(cues_path, encoding='utf-8') as f:
        sheet = json.load(f)
    track = render(sheet, seed=seed, log=log)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, 'mix.wav')
        wavfile.write(wav, dsp.SR, track.astype(np.float32))
        m4a = os.path.join(out_dir, f'{name}.m4a')
        master(wav, m4a, ffmpeg, lufs=lufs)
    loud = measure(m4a, ffmpeg)
    log(f'AUDIO track {len(track) / dsp.SR:.2f}s integrated {loud[0]} LUFS true peak {loud[1]} dBFS -> {m4a}')
    for out_name, src in videos.items():
        out = os.path.join(out_dir, out_name)
        mux(src, m4a, out, ffmpeg)
        log(f'AUDIO muxed {out} ({os.path.getsize(out)} bytes)')
    return loud
