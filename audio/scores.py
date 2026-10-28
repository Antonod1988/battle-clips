"""Scores by mood. 'ninja' is the shared fighting-game track (music.score: 150 bpm, D minor, taiko and arpeggios);
'heroic' is a hero-against-the-brute score (136 bpm, C minor: brass and a heroic theme over taiko, a growling low riff;
from the HULK vs CAPTAIN AMERICA clip). Both take the clip's music cues [(t, section)] and return a stereo bed.
Sections: intro, fight, fury, frozen, spear, tension, fatality, end (unknown names fall back to fight).
"""
import numpy as np

import dsp
import music
from dsp import SR, n_of, times, white, pink, lp, hp, bp, sweep, env, osc, drop, sat  # noqa: F401

BPM = 136.0
BEAT = 60.0 / BPM
STEP = BEAT / 4
BAR = BEAT * 4
hz = music.hz
CH = {'Cm': (48, 51, 55), 'Ab': (44, 48, 51), 'Eb': (51, 55, 58), 'Bb': (46, 50, 53), 'G': (43, 47, 50),
      'Fm': (41, 44, 48), 'Db': (49, 53, 56)}
PROG = {'fight': ['Cm', 'Ab', 'Eb', 'Bb'], 'fury': ['Cm', 'Db', 'Cm', 'G'], 'tension': ['Cm', 'Db'],
        'intro': ['Cm'], 'fatality': ['Cm', 'Fm'], 'end': ['Cm']}
THEME = [(0, 67, 2), (2, 72, 2), (4, 70, 1), (5, 72, 1), (6, 75, 2), (8, 74, 4), (12, 72, 2), (14, 70, 2)]


def brass(notes, dur, rng, swell=0.12):
    """A heroic brass chord: detuned saws, a swelling filter."""
    n = n_of(dur)
    t = times(n)
    x = np.zeros(n)
    for m in notes:
        for d in (-0.008, 0.0, 0.009):
            x += osc(hz(m) * (1 + d) * (1 + 0.004 * np.sin(2 * np.pi * 5 * t)), n, 'saw', phase=rng.uniform(0, 6.28))
    x = sweep(x / (3 * len(notes)), 500 + 2600 * np.clip(t / swell, 0, 1) * np.exp(-t / (dur * 1.5)), 'lp', block=128)
    return sat(x * dsp.adsr(n, 0.03, 0.2, 0.75, min(0.25, dur * 0.3)) * 1.4, 1.3)


def brute(note, dur, rng):
    """The brute's riff note: a growling, distorted low unison with a sub."""
    n = n_of(dur)
    t = times(n)
    f = hz(note)
    x = osc(f, n, 'saw') + osc(f * 1.01, n, 'saw') + osc(f * 0.5, n, 'square') * 0.6 + osc(f * 0.5, n) * 0.8
    x = sweep(x, 200 + 1200 * np.exp(-t / 0.15), 'lp', block=128)
    return sat(x * env(n, 0.005, dur * 0.8, hold=dur * 0.2) * 1.8, 2.6) * 0.6


def taiko(rng, f=60, big=1.0):
    n = n_of(0.9)
    return sat(osc(drop(f * 2.2, f, n, 14), n) * env(n, 0.002, 0.28 * big) * 1.6
               + lp(white(n, rng), 800) * env(n, 0.001, 0.02) * 0.6, 1.4)


def hvc_score(sections, duration, seed=11):
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
    if t_fight > 0.5:
        put(music.drone(36, t_fight + 0.6, rng), 0.0, 0.5)
        for k in range(int(t_fight / BEAT)):
            put(taiko(rng, 55, 0.8), k * BEAT, 0.35 + 0.4 * k * BEAT / t_fight)
        put(brass(CH['Cm'], t_fight, rng, swell=t_fight), 0.0, 0.25)
        put(music.riser(min(1.6, t_fight), rng), t_fight - min(1.6, t_fight), 0.5)
        put(taiko(rng, 50, 1.6), t_fight, 1.0)
        put(music.crash(rng), t_fight, 0.8)
        put(brass(CH['Cm'] + (60,), 1.2, rng), t_fight, 0.6)
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
                if s in (0, 6, 8, 14) or (hard and s in (3, 11)):
                    put(music.kick(rng), ts, 0.9)
                if s in (4, 12):
                    put(music.snare(rng), ts, 0.65, 0.1)
                if s % 2 == 0:
                    put(music.hat(rng, open_=(s == 14)), ts, 0.14, 0.35)
                if s in (0, 8) or (hard and s == 12):
                    put(taiko(rng, 58 if s else 52), ts, 0.55, -0.2)
                note = root + (12 if s % 4 == 2 else 0)
                put(music.bass(note, STEP * 0.9, rng), ts, 0.32)
            # the brute's riff on the first beat of every other bar; the hero's theme over it
            if bar % 2 == 0 or hard:
                put(brute(root - 12, BEAT * 1.5, rng), t, 0.5)
                put(brute(root - 12 + (1 if hard else 3), BEAT * 0.9, rng), t + BEAT * 2.5, 0.4)
            put(brass(chord, BAR, rng), t, 0.22, -0.1)
            if not hard or bar % 2 == 1:
                for (s, m, ln) in THEME:
                    put(brass((m + (0 if bar % 4 < 2 else -2),), STEP * ln * 0.95, rng, swell=0.05), t + s * STEP, 0.16, 0.25)
            if bar % 4 == 3:
                for k, f in enumerate((140, 120, 100, 85)):
                    put(music.tom(rng, f), t + (12 + k) * STEP, 0.55, -0.4 + 0.25 * k)
        elif sec == 'tension':
            put(music.drone(36, BAR, rng), t, 0.5)
            put(brute(root - 12, BEAT * 3.5, rng), t, 0.25)
            put(music.heartbeat(rng), t, 0.8)
            put(music.heartbeat(rng), t + 2 * BEAT, 0.8)
        elif sec == 'fatality':
            put(music.drone(36, BAR, rng), t, 0.45)
            put(brute(root - 12, BAR * 0.9, rng), t, 0.35)
            put(taiko(rng, 48, 1.4), t, 0.8)
            put(taiko(rng, 48, 1.2), t + 2 * BEAT, 0.6)
        elif sec == 'end':
            pass
        else:
            put(music.drone(36, BAR, rng), t, 0.5)
        bar += 1
        t += BAR
    for (s, name) in sections:
        if name == 'fury':
            put(music.riser(0.5, rng)[::-1].copy(), s - 0.5, 0.3)
            put(taiko(rng, 45, 2.0), s, 1.0)
        if name == 'fatality':
            put(brass((36, 48, 55, 60, 63), 2.6, rng, swell=0.02), s, 0.9)
            put(taiko(rng, 40, 2.4), s, 1.2)
            put(music.crash(rng, 3.0), s, 0.7)
        if name == 'end':
            put(brass((36, 48, 55, 60, 63, 67), 3.0, rng, swell=0.02), s, 0.95)
            for k in range(10):
                put(music.tom(rng, 65 + k * 2), s + 0.4 + k * 0.06, 0.35 + 0.05 * k)
            put(taiko(rng, 40, 2.6), s + 1.05, 1.2)
            put(brute(24, 2.4, rng), s + 1.05, 0.6)
            put(brass((36, 48, 55, 60, 63, 67), 3.2, rng, swell=0.02), s + 1.05, 0.9)
            put(music.crash(rng, 3.5), s + 1.05, 0.8)
    out = np.stack([L, R], axis=1)
    return out[:n_of(duration)]




MOODS = {'ninja': music.score, 'heroic': hvc_score}


def use(mood):
    """Make the mixer use this mood's score."""
    music.score = MOODS.get(mood, MOODS['ninja'])
