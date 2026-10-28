"""Small DSP toolkit for the clip audio (numpy + scipy, 48 kHz float). Everything here is synthesis we wrote ourselves."""
import numpy as np
from scipy import signal

SR = 48000


def n_of(dur):
    return max(1, int(round(dur * SR)))


def times(n):
    return np.arange(n, dtype=np.float64) / SR


def white(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    """1/f noise by shaping white noise in the frequency domain."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[1:] /= np.sqrt(f[1:] / 20.0)
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-9)


def brown(n, rng):
    y = np.cumsum(rng.standard_normal(n))
    y = hp(y, 25)
    return y / (np.std(y) + 1e-9)


def _sos(kind, f, order=2):
    nyq = SR / 2
    if kind == 'bp':
        lo, hi = max(10.0, f[0]), min(nyq * 0.98, f[1])
        return signal.butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
    f = min(nyq * 0.98, max(10.0, f))
    return signal.butter(order, f / nyq, btype='low' if kind == 'lp' else 'high', output='sos')


def lp(x, f, order=2):
    return signal.sosfilt(_sos('lp', f, order), x)


def hp(x, f, order=2):
    return signal.sosfilt(_sos('hp', f, order), x)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(_sos('bp', (f1, f2), order), x)


def sweep(x, fc, kind='lp', q=1.0, block=256):
    """A filter whose cutoff (or band centre) follows the array fc (one value per sample), done block-wise."""
    out = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        f = float(np.mean(fc[i:i + block]))
        if kind == 'bp':
            bw = f / max(q, 0.3)
            sos = _sos('bp', (f - bw / 2, f + bw / 2), 2)
        else:
            sos = _sos(kind, f, 2)
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0.0
        seg, zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
        out[i:i + block] = seg
    return out


def env(n, attack=0.002, decay=0.1, hold=0.0, curve=1.0):
    """Linear attack, optional hold, exponential decay (time constant `decay` seconds)."""
    t = times(n)
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    d = np.where(t < attack + hold, 1.0, np.exp(-(t - attack - hold) / max(decay, 1e-5)))
    return a * d ** curve


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2, gate=None):
    t = times(n)
    gate = (n / SR - r) if gate is None else gate
    e = np.where(t < a, t / max(a, 1e-6), np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    rel = np.clip(1 - (t - gate) / max(r, 1e-6), 0, 1)
    return np.where(t > gate, e * rel, e)


def osc(freq, n, shape='sine', phase=0.0):
    """freq: a number or an array per sample."""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    ph = phase + 2 * np.pi * np.cumsum(f) / SR
    if shape == 'sine':
        return np.sin(ph)
    if shape == 'saw':
        return 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    if shape == 'square':
        return np.sign(np.sin(ph))
    if shape == 'tri':
        return 2 * np.abs(2 * ((ph / (2 * np.pi)) % 1.0) - 1) - 1
    raise ValueError(shape)


def drop(f0, f1, n, rate=30.0):
    """A pitch that falls from f0 towards f1 exponentially (a thump's body)."""
    t = times(n)
    return f1 + (f0 - f1) * np.exp(-t * rate)


def sat(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def crackle(n, rng, rate=40.0, decay=0.002, lo=1500, hi=7000):
    """Random clicks (fire, debris, bones): a Poisson train of short filtered pops of random size."""
    x = np.zeros(n)
    k = rng.poisson(rate * n / SR)
    pos = rng.integers(0, n, k)
    amp = rng.uniform(0.2, 1.0, k) ** 2 * rng.choice([-1, 1], k)
    np.add.at(x, pos, amp)
    kern = np.exp(-np.arange(int(decay * SR * 5)) / (decay * SR))
    x = signal.fftconvolve(x, kern)[:n]
    return bp(x, lo, hi)


def partials(freqs, decays, n, amps=None, rng=None, detune=0.0):
    """A struck object: decaying sine partials (metal, glass, bone)."""
    t = times(n)
    out = np.zeros(n)
    amps = amps if amps is not None else [1.0 / (i + 1) ** 0.5 for i in range(len(freqs))]
    for f, d, a in zip(freqs, decays, amps):
        if rng is not None and detune:
            f *= 1 + rng.uniform(-detune, detune)
        out += a * np.sin(2 * np.pi * f * t + (rng.uniform(0, 6.28) if rng is not None else 0)) * np.exp(-t / d)
    return out


def karplus(f, dur, rng, damp=0.996):
    """A plucked string (the rope's twang)."""
    n = n_of(dur)
    p = max(2, int(SR / f))
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = damp * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out


def ir(dur=1.6, decay=0.5, damp=6000, rng=None, pre=0.012, stereo=True):
    """A reverb impulse response: decaying noise, darker as it fades, decorrelated left and right."""
    rng = rng or np.random.default_rng(7)
    n = n_of(dur)
    t = times(n)
    chans = []
    for c in range(2 if stereo else 1):
        x = rng.standard_normal(n) * np.exp(-t / decay)
        x = sweep(x, damp * np.exp(-t / (decay * 2.0)) + 400, 'lp', block=1024)
        x[:n_of(pre)] = 0
        chans.append(x / np.sqrt(np.sum(x ** 2)))
    return np.stack(chans, axis=1)


def reverb(x, impulse, wet=0.3):
    """x: stereo (n, 2). Convolves each side with the impulse response and mixes it in."""
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    y = np.stack([signal.fftconvolve(x[:, c], impulse[:, c])[:len(x)] for c in range(2)], axis=1)
    return x * (1 - wet) + y * wet * 2.0


def pan(mono, p=0.0):
    """Constant-power pan: p from -1 (left) to 1 (right)."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def pan_moving(mono, p0, p1):
    k = np.linspace(0, 1, len(mono))
    a = ((p0 + (p1 - p0) * k) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def place(buf, x, t0, gain=1.0):
    """Adds x (mono or stereo) into the stereo buffer at time t0 (seconds)."""
    i = int(round(t0 * SR))
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if i < 0:
        x = x[-i:]
        i = 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i] * gain


def follower(x, attack=0.005, release=0.15):
    """Amplitude envelope of a signal (for ducking and compression)."""
    a = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    ga, gr = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    # attack/release one-pole run in chunks through lfilter on the peak-held signal
    held = signal.lfilter([1 - gr], [1, -gr], a)
    return np.maximum(held, signal.lfilter([1 - ga], [1, -ga], a))


def compress(x, threshold_db=-18, ratio=3.0, attack=0.01, release=0.2, makeup_db=0.0):
    e = follower(x, attack, release) + 1e-9
    lvl = 20 * np.log10(e)
    over = np.maximum(0, lvl - threshold_db)
    g = 10 ** ((-over * (1 - 1 / ratio) + makeup_db) / 20)
    return x * (g[:, None] if x.ndim == 2 else g)


def limit(x, ceiling_db=-1.0, release=0.05):
    """A simple look-ahead-free peak limiter (the final loudness pass is ffmpeg's loudnorm)."""
    c = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    g = np.minimum(1.0, c / (peak + 1e-9))
    gr = np.exp(-1 / (release * SR))
    # gain may drop instantly, recovers smoothly
    sm = np.empty_like(g)
    cur = 1.0
    gg = g
    # vectorised enough: process in chunks with a running minimum
    for i in range(0, len(gg), 4096):
        seg = gg[i:i + 4096]
        out = np.empty_like(seg)
        for k in range(len(seg)):
            cur = seg[k] if seg[k] < cur else cur * gr + (1 - gr) * seg[k]
            out[k] = cur
        sm[i:i + 4096] = out
    return x * (sm[:, None] if x.ndim == 2 else sm)


def db(v):
    return 10 ** (v / 20)
