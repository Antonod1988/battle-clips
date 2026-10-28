"""Regression check for the weapon sounds (SOUND-2, kept from the version 4 branch): a clash before and after, by
numbers (nobody here can listen).

    python tools/sound_check.py <old_sfx_dir> <out.png>

<old_sfx_dir> holds an older sfx.py to compare with, e.g. the ringing version 4 one:
`git -C <mk4 worktree> show 3c155857:clips/audio/sfx.py > <dir>/sfx.py`. A healthy dry clash falls 20 dB in well
under 100 ms and is 20+ dB down 150 ms after its peak in the mix. For each kind of
clash (plain, big, on ice) over 17 differently seeded hits (as the mixer seeds each cue): the level 150 ms after the peak,
in dB below the peak, dry and as placed in the mix (with the kind's reverb send through the mixer's room); the time the hit
takes to fall 20 dB, and how much the hits differ. Writes a small before/after spectrogram of six clashes in a row.
"""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, '..', 'audio')   # the repo's audio module
sys.path.insert(0, AUDIO)
import dsp  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOM = dsp.ir(1.3, 0.38, damp=5500, rng=np.random.default_rng(3))


def as_mixed(mod, cue, seed):
    fn, gain, rv = mod.REGISTRY['clash']
    y = fn(cue, np.random.default_rng(seed))
    y = y / (np.max(np.abs(y)) + 1e-9)
    st = dsp.pan(y, 0.0) * dsp.db(gain)
    pad = np.zeros((len(st) + dsp.n_of(1.6), 2))
    pad[:len(st)] = st
    wet = dsp.reverb(pad * rv, ROOM, wet=1.0) - pad * rv
    return y, (pad + wet * 0.9)[:, 0]


def rms_env(x, win=0.005):
    w = dsp.n_of(win)
    e = np.sqrt(np.convolve(x * x, np.ones(w) / w, mode='same') + 1e-12)
    return 20 * np.log10(e)


def stats(x):
    e = rms_env(x)
    ip = int(np.argmax(e))
    j = ip + dsp.n_of(0.150)
    after = e[j] if j < len(e) else -200.0
    below = np.nonzero(e[ip:] < e[ip] - 20)[0]
    t20 = below[0] / dsp.SR if len(below) else len(x) / dsp.SR
    return e[ip] - after, t20


def main(old_dir, out_png):
    old = load('sfx_old', os.path.join(old_dir, 'sfx.py'))
    new = load('sfx_new', os.path.join(AUDIO, 'sfx.py'))
    cases = [('plain', {}), ('big', {'big': True}), ('on ice', {'ice': True, 'big': True})]
    print('clash: level 150 ms after the peak (dB below the peak), dry | in the mix; time to fall 20 dB')
    for label, cue in cases:
        for tag, mod in (('before', old), ('after', new)):
            d150, m150, t20s, cents = [], [], [], []
            for i in range(17):
                y, mixed = as_mixed(mod, dict(cue), 100003 + i)
                a, t20 = stats(y)
                b, _ = stats(mixed)
                d150.append(a)
                m150.append(b)
                t20s.append(t20)
                f, P = signal.welch(y, dsp.SR, nperseg=1024)
                cents.append(float(np.sum(f * P) / np.sum(P)))
            print(f'  {label:7s} {tag:6s} dry {np.median(d150):5.1f} dB (worst {min(d150):5.1f}) | mix '
                  f'{np.median(m150):5.1f} dB (worst {min(m150):5.1f}) | -20 dB after {np.median(t20s) * 1000:5.0f} ms'
                  f' ({min(t20s) * 1000:.0f}-{max(t20s) * 1000:.0f}) | centroid {min(cents):.0f}-{max(cents):.0f} Hz')
    # six clashes in a row, as the mix places them, before above and after below
    rows = []
    seq = [{}, {'big': True}, {}, {'ice': True}, {}, {'big': True, 'ice': True}]
    for mod in (old, new):
        x = np.zeros(dsp.n_of(3.6))
        for k, cue in enumerate(seq):
            _, m = as_mixed(mod, dict(cue), 100003 + k)
            i = dsp.n_of(0.1 + 0.55 * k)
            x[i:i + len(m)] += m[:len(x) - i]
        f, t, S = signal.spectrogram(x, dsp.SR, nperseg=1024, noverlap=896)
        keep = f <= 12000
        S = 10 * np.log10(S[keep] + 1e-12)
        S = np.clip((S - (S.max() - 70)) / 70, 0, 1)
        im = Image.fromarray((S[::-1] * 255).astype(np.uint8)).resize((1080, 220), Image.BILINEAR)
        rows.append(np.asarray(im))
    canvas = Image.new('RGB', (1100, 500), (12, 12, 16))
    d = ImageDraw.Draw(canvas)
    for r, (label, arr) in enumerate(zip(('before: ringing steel', 'after: dry steel (SOUND-2)'), rows)):
        rgb = np.stack([arr * 0.9, arr * 0.7, arr * 1.0], -1).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (10, 26 + r * 245))
        d.text((12, 8 + r * 245), label + '  -  six clashes (plain, big, plain, ice, plain, big on ice), 0-12 kHz, 3.6 s',
               fill=(255, 255, 255))
    canvas.save(out_png)
    print('png', out_png)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
