"""An original fighting-game track, synthesised: drums, a driving sixteenth-note bass, an arpeggio, pads, stabs, risers.

`score(sections, duration)` takes the clip's music cues [(t, section), ...] and builds a stereo bed of the clip's length.
The grid (150 bpm, D minor) is anchored on the 'fight' cue so the groove lands on the word «FIGHT!»; each bar plays the
pattern of the section active at its start; 'fatality' and 'end' also fire one-shot hits exactly on their cue.
Sections: intro, fight, frozen, fury, spear, tension, fatality, end. Unknown names fall back to 'fight'.
"""
import numpy as np

import dsp
from dsp import SR, n_of, times, white, pink, lp, hp, bp, sweep, env, osc, drop, sat

BPM = 150.0
BEAT = 60.0 / BPM
STEP = BEAT / 4
BAR = BEAT * 4


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


# chords as MIDI notes (D minor with a phrygian Eb for menace)
CH = {'Dm': (50, 53, 57), 'Bb': (46, 50, 53), 'C': (48, 52, 55), 'A': (45, 49, 52), 'Eb': (51, 55, 58), 'Gm': (43, 46, 50)}
PROG = {'fight': ['Dm', 'Dm', 'Bb', 'C'], 'fury': ['Dm', 'Bb', 'C', 'A'], 'spear': ['Eb', 'Dm', 'Eb', 'A'],
        'frozen': ['Dm', 'Eb'], 'tension': ['Dm', 'Eb'], 'intro': ['Dm'], 'fatality': ['Dm', 'Gm'], 'end': ['Dm']}


# ------------------------------------------------------------ instruments
def kick(rng, big=1.0):
    n = n_of(0.35)
    return sat(osc(drop(150, 46, n, 28), n) * env(n, 0.001, 0.11 * big) * 1.4
               + hp(white(n, rng), 3000) * env(n, 0, 0.003) * 0.3, 1.5)


def snare(rng):
    n = n_of(0.25)
    return bp(white(n, rng), 1200, 7000) * env(n, 0.001, 0.07) * 0.9 + osc(drop(240, 180, n, 30), n) * env(n, 0.001, 0.04) * 0.6


def hat(rng, open_=False):
    n = n_of(0.25 if open_ else 0.06)
    return hp(white(n, rng), 7500) * env(n, 0.0005, 0.08 if open_ else 0.012) * 0.5


def tom(rng, f=90):
    n = n_of(0.4)
    return osc(drop(f * 1.6, f, n, 18), n) * env(n, 0.002, 0.16) + bp(white(n, rng), 200, 1500) * env(n, 0, 0.02) * 0.4


def bass(note, dur, rng):
    n = n_of(dur)
    t = times(n)
    f = hz(note)
    x = osc(f, n, 'saw') + osc(f * 1.005, n, 'saw') * 0.6 + osc(f / 2, n, 'sine') * 0.8
    cut = 250 + 1400 * np.exp(-t / 0.05)
    x = dsp.sweep(x, cut, 'lp', block=128)
    return sat(x * env(n, 0.002, dur * 0.7, hold=dur * 0.3) * 0.7, 1.6)


def pluck(note, dur, rng, bright=1.0):
    n = n_of(dur)
    t = times(n)
    x = osc(hz(note), n, 'square') * 0.5 + osc(hz(note) * 2.002, n, 'saw') * 0.25
    x = dsp.sweep(x, 900 + 3500 * bright * np.exp(-t / 0.06), 'lp', block=128)
    return x * env(n, 0.002, 0.07)


def pad(notes, dur, rng, cutoff=1400, attack=0.4):
    n = n_of(dur)
    x = np.zeros(n)
    for m in notes:
        for d in (-0.006, 0.0, 0.007):
            x += osc(hz(m) * (1 + d), n, 'saw', phase=rng.uniform(0, 6.28))
    x = lp(x / (3 * len(notes)), cutoff)
    return x * dsp.adsr(n, attack, 0.3, 0.8, min(0.8, dur * 0.4))


def stab(notes, rng, dur=1.6):
    """A brassy orchestral hit."""
    n = n_of(dur)
    t = times(n)
    x = np.zeros(n)
    for m in notes:
        for d in (-0.01, 0.0, 0.012):
            x += osc(hz(m) * (1 + d), n, 'saw', phase=rng.uniform(0, 6.28))
    x = dsp.sweep(x / (3 * len(notes)), 600 + 4500 * np.exp(-t / 0.18), 'lp', block=128)
    return sat(x * env(n, 0.004, 0.45) * 1.6, 1.4)


def crash(rng, dur=2.2):
    n = n_of(dur)
    return hp(white(n, rng), 4000) * env(n, 0.001, 0.7) * 0.45 + bp(white(n, rng), 2500, 9000) * env(n, 0.001, 0.25) * 0.3


def riser(dur, rng):
    n = n_of(dur)
    t = times(n) / dur
    noise = dsp.sweep(white(n, rng), 300 + 6000 * t ** 2, 'bp', q=1.2) * t ** 2
    tone = osc(110 * 2 ** (3 * t), n, 'saw') * 0.15 * t ** 2
    return noise + lp(tone, 4000)


def drone(note, dur, rng):
    n = n_of(dur)
    t = times(n)
    x = osc(hz(note), n, 'saw') + osc(hz(note) * 1.5, n, 'saw') * 0.4 + osc(hz(note) / 2, n) * 0.8
    x = dsp.sweep(x, 220 + 180 * (1 + np.sin(2 * np.pi * 0.2 * t)), 'lp', block=512)
    return x * np.clip(t / 1.0, 0, 1) * 0.5


def heartbeat(rng):
    n = n_of(0.6)
    one = osc(drop(80, 42, n, 20), n) * env(n, 0.003, 0.09)
    two = np.zeros(n)
    k = n_of(0.22)
    two[k:] = one[:n - k] * 0.7
    return sat((one + two) * 1.5, 1.3)


# ------------------------------------------------------------ the score
def score(sections, duration, seed=11):
    """sections: [(t, name)] sorted; returns a stereo float array of `duration` seconds."""
    rng = np.random.default_rng(seed)
    n = n_of(duration + 3.0)
    L = np.zeros(n)
    R = np.zeros(n)
    sections = sorted(sections)

    def at(t):
        cur = sections[0][1] if sections else 'fight'
        for (s, name) in sections:
            if s <= t + 1e-6:
                cur = name
        return cur

    def put(x, t, g=1.0, p=0.0):
        i = int(round(t * SR))
        if i >= n or i + len(x) <= 0:
            return
        if i < 0:
            x = x[-i:]
            i = 0
        j = min(n, i + len(x))
        a = (p + 1) * np.pi / 4
        L[i:j] += x[:j - i] * g * np.cos(a)
        R[i:j] += x[:j - i] * g * np.sin(a)

    t_fight = next((s for s, nm in sections if nm == 'fight'), 0.0)
    # intro: a low drone and a riser into «FIGHT!»
    if t_fight > 0.5:
        put(drone(38, t_fight + 0.5, rng), 0.0, 0.55)
        put(pad(CH['Dm'], t_fight, rng, cutoff=900, attack=1.2), 0.0, 0.25)
        put(riser(min(1.6, t_fight), rng), t_fight - min(1.6, t_fight), 0.55)
        put(kick(rng, 1.6), t_fight, 1.0)
        put(crash(rng), t_fight, 0.8)
        put(stab(CH['Dm'] + (62,), rng), t_fight, 0.55)
    # the grid: bars from «FIGHT!» on
    bar = 0
    t = t_fight
    while t < duration:
        sec = at(t + 0.01)
        prog = PROG.get(sec, PROG['fight'])
        chord = CH[prog[bar % len(prog)]]
        root = chord[0] - 12
        if sec in ('fight', 'fury'):
            hard = sec == 'fury'
            for s in range(16):
                ts = t + s * STEP
                if s in (0, 4, 8, 12) or (s == 14 and bar % 2 == 1) or (hard and s in (3, 11)):
                    put(kick(rng), ts, 0.95)
                if s in (4, 12):
                    put(snare(rng), ts, 0.7, 0.1)
                if hard and bar % 2 == 1 and s >= 12:
                    put(snare(rng), ts, 0.35 + 0.06 * (s - 12), 0.1)
                put(hat(rng, open_=(s == 14)), ts, 0.28 if s % 4 == 2 else 0.16, 0.35)
                note = root + (12 if s in (2, 6, 10, 14) else 0) + (7 if hard and s in (7, 15) else 0)
                put(bass(note, STEP * 0.95, rng), ts, 0.42)
                arp = chord[s % 3] + 12 + (12 if hard and s % 8 >= 4 else 0)
                put(pluck(arp, STEP * 0.9, rng, 1.2 if hard else 0.9), ts + 0.003, 0.15, -0.3 + 0.6 * ((s % 4) / 3))
            put(pad(chord, BAR, rng, cutoff=1300), t, 0.16, 0.0)
            if bar % 8 == 7:
                for k, f in enumerate((140, 120, 100, 85)):
                    put(tom(rng, f), t + (12 + k) * STEP, 0.6, -0.4 + 0.25 * k)
        elif sec == 'frozen':
            for s in range(16):
                if s % 2 == 0:
                    put(hat(rng), t + s * STEP, 0.12, 0.3)
            put(bass(root, BAR * 0.95, rng), t, 0.3)
            put(pad((chord[0] + 12, chord[1] + 12, chord[2] + 24), BAR, rng, cutoff=3000, attack=0.2), t, 0.14, 0.2)
            put(heartbeat(rng), t, 0.6)
            put(heartbeat(rng), t + 2 * BEAT, 0.6)
        elif sec == 'spear':
            put(kick(rng, 1.8), t, 1.1)
            put(snare(rng), t + 2 * BEAT, 0.9)
            put(crash(rng, 1.4), t, 0.4)
            for s in range(0, 16, 2):
                put(bass(root, STEP * 1.8, rng), t + s * STEP, 0.45)
            put(pad(chord, BAR, rng, cutoff=2000, attack=0.05), t, 0.22)
        elif sec == 'tension':
            put(drone(38, BAR, rng), t, 0.5)
            put(pad((62, 63), BAR, rng, cutoff=2600, attack=0.6), t, 0.12, 0.25)
            put(heartbeat(rng), t, 0.8)
            put(heartbeat(rng), t + 2 * BEAT, 0.8)
        elif sec == 'fatality':
            put(drone(38, BAR, rng), t, 0.45)
            put(pad(chord, BAR, rng, cutoff=1000, attack=0.5), t, 0.2)
            put(tom(rng, 70), t, 0.8)
            put(tom(rng, 70), t + 2 * BEAT, 0.6)
        elif sec == 'end':
            pass
        else:
            put(drone(38, BAR, rng), t, 0.5)
        bar += 1
        t += BAR
    # one-shots exactly on their cue
    for (s, name) in sections:
        if name == 'fatality':
            put(stab((38, 50, 53, 57, 62), rng, 2.4), s, 0.85)
            put(kick(rng, 2.2), s, 1.2)
            put(crash(rng, 3.0), s, 0.7)
        if name == 'spear':
            put(riser(0.5, rng)[::-1].copy(), s - 0.5, 0.35)
        if name == 'end':
            put(stab((38, 50, 53, 57, 62, 69), rng, 3.0), s, 0.95)
            for k in range(10):
                put(tom(rng, 65 + k * 2), s + 0.4 + k * 0.06, 0.35 + 0.05 * k)
            put(kick(rng, 2.5), s + 1.05, 1.2)
            put(stab((38, 50, 53, 57, 62, 69), rng, 3.2), s + 1.05, 0.9)
            put(crash(rng, 3.5), s + 1.05, 0.8)
    out = np.stack([L, R], axis=1)
    return out[:n_of(duration)]
