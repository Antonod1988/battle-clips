"""Effect kinds for city fights, titans and shields (from the HULK vs CAPTAIN AMERICA clip, made dry in HULK-2 like
SOUND-2's weapons: short rooms, metal that no longer rings for a second), registered into the shared sfx.REGISTRY
on import: roars, heavy landings, smashes and body slams, the thunderclap and its shockwave, the
shield's ringing blocks, ricochets, flight, catch and roll, glass, cars, poles and steel, live wires, walls bursting,
cave-ins, wood and plates, water. Heavy fighters' footsteps ('steps' cues with heavy=True) get their own sound. The
'hulk' voice style is added to voice.STYLES. Everything is synthesised (no samples).
"""
import numpy as np

import dsp  # noqa: F401
import sfx
import voice
import mix
from dsp import (SR, n_of, times, white, pink, brown, lp, hp, bp, sweep, env, osc, drop, sat, crackle,  # noqa: F401
                 partials)

register = sfx.register
_thump = sfx._thump


def _fit(x, n):
    out = np.zeros(n)
    m = min(n, len(x))
    out[:m] = x[:m]
    return out


def _debris(n, rng, dur, density=80.0, lo=1200, hi=6500):
    return crackle(n, rng, rate=density, decay=0.004, lo=lo, hi=hi) * env(n, 0.005, dur * 0.4)


def _tinkle(n, rng, count, lo=3000, hi=9000, spread=0.6):
    """Glass: many tiny bright struck pieces scattered over `spread` seconds."""
    out = np.zeros(n)
    for _ in range(count):
        m = n_of(rng.uniform(0.03, 0.12))
        f = rng.uniform(lo, hi)
        piece = np.sin(2 * np.pi * f * times(m)) * np.exp(-times(m) / rng.uniform(0.01, 0.04))
        i = n_of(abs(rng.normal(0, spread * 0.45)))
        if i + m < n:
            out[i:i + m] += piece * rng.uniform(0.2, 1.0)
    return out


def _metal(n, rng, base=180.0, bright=1.0, decay=0.9):
    """Struck sheet metal: inharmonic partials."""
    ratios = [1.0, 2.13, 3.47, 4.9, 6.3, 8.1, 10.4]
    return partials([base * r for r in ratios], [decay * k for k in (1.0, 0.7, 0.5, 0.36, 0.26, 0.18, 0.12)], n,
                    rng=rng, detune=0.03) * (0.6 + 0.4 * bright)


# ------------------------------------------------------------------ the Hulk
@register('roar', gain_db=-1.0, reverb=0.35)
def roar(c, rng):
    """A monster's roar: a low growling voice (a jittery saw through vowel formants) over noise, distorted."""
    dur = max(0.4, float(c.get('dur', 0.8))) + 0.3
    big = bool(c.get('big', True))
    n = n_of(dur)
    t = times(n)
    shape = np.clip(t / 0.08, 0, 1) * np.clip((dur - t) / 0.35, 0, 1)
    f0 = (78 if big else 92) * (1 + 0.08 * np.sin(2 * np.pi * 5.5 * t)) * (1 + 0.12 * np.sin(np.pi * t / dur))
    f0 = f0 * (1 + 0.03 * lp(rng.standard_normal(n), 30) * 8)
    src = osc(f0, n, 'saw') + osc(f0 * 0.5, n, 'saw') * 0.6 + pink(n, rng) * 0.35
    voc = bp(src, 520, 900, 2) * 1.2 + bp(src, 1000, 1500, 2) * 0.8 + bp(src, 2300, 3200, 2) * 0.35 + lp(src, 300) * 0.9
    rasp = bp(white(n, rng), 1200, 5000) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * f0 * t / 2))) * 0.25
    x = sat((voc + rasp) * 2.2, 2.5) * shape
    if big:
        x += _thump(n, 60, 30, 3, 0.8, rng) * 0.4 * np.clip(t / 0.1, 0, 1)
    return x


@register('hulk_land', gain_db=0.0, reverb=0.18)
def hulk_land(c, rng):
    """A giant landing: a sub boom, the road cracking, a long rain of debris."""
    n = n_of(2.4)
    x = _thump(n, 75, 22, 5, 0.35, rng) * 1.8
    x += hp(white(n, rng), 1500) * env(n, 0.0005, 0.03) * 1.2
    x += crackle(n, rng, rate=2000, decay=0.0015, lo=700, hi=5000) * env(n, 0.002, 0.08) * 2.4
    x += _debris(n, rng, 1.0, 140) * 1.8
    x += lp(brown(n, rng), 200) * env(n, 0.02, 0.35) * 0.5
    return sat(x, 1.5)


@register('smash', gain_db=-1.0, reverb=0.14)
def smash(c, rng):
    """A fist or a body smashing a floor: boom, crack, chunks; tiles add a ceramic clatter."""
    p = float(c.get('power', 1.0))
    n = n_of(1.6)
    x = _thump(n, 85, 30, 9, 0.25 + 0.12 * p, rng) * (1.2 + 0.3 * p)
    x += hp(white(n, rng), 1800) * env(n, 0.0005, 0.025) * 1.1
    x += crackle(n, rng, rate=1600, decay=0.0015, lo=800, hi=5000) * env(n, 0.002, 0.06) * 2.0
    x += _debris(n, rng, 0.8, 90 * p) * 1.5
    if c.get('floor') == 'tile':
        x += _tinkle(n, rng, 40, 1800, 5000, 0.5) * 0.6
    return sat(x, 1.4)


@register('body_slam', gain_db=-1.0, reverb=0.14)
def body_slam(c, rng):
    p = float(c.get('power', 1.5))
    n = n_of(1.8)
    x = _fit(sfx.REGISTRY['floor_slam'][0]({}, rng), n) * 0.8
    x += _fit(sfx.hit({'weight': 3, 'blood': 10}, rng), n) * 0.8
    x += _thump(n, 60, 26, 6, 0.5, rng) * 0.5 * p
    return sat(x, 1.3)


@register('clap', gain_db=1.0, reverb=0.22)
def clap(c, rng):
    """The thunderclap: a crack like a gunshot, a sub punch, a rolling boom, a ringing in the ears."""
    n = n_of(4.0)
    t = times(n)
    crack = white(n, rng) * env(n, 0.0002, 0.012) * 2.5
    x = crack + hp(white(n, rng), 3000) * env(n, 0.0, 0.05)
    x += _thump(n, 90, 24, 6, 0.4, rng) * 2.0
    x += lp(brown(n, rng), 160) * env(n, 0.05, 0.6) * 1.4
    x += _tinkle(n, rng, 120, 2500, 9000, 0.6) * 0.5
    ring = np.sin(2 * np.pi * 3900 * t) * env(n, 0.02, 0.25) * 0.04
    return sat(x * 1.2, 1.6) + ring


@register('shockwave', gain_db=1.5, reverb=0.22)
def shockwave(c, rng):
    """The last blow on the shield: a vast clang, a boom that rolls down the street, every window going."""
    n = n_of(5.0)
    x = (_metal(n, rng, base=210, bright=1.0, decay=0.08) * env(n, 0.0005, 0.12) + sfx._steel(n, rng, weight=2.5)) * 1.3
    x += _fit(clap({}, rng), n)
    x += _thump(n, 55, 18, 3, 0.5, rng) * 1.6
    x += _debris(n, rng, 1.6, 160) * 1.4
    return sat(x, 1.5)


@register('grab', gain_db=-6.0, reverb=0.15)
def grab(c, rng):
    n = n_of(0.3)
    slap = bp(white(n, rng), 600, 3500) * env(n, 0.0005, 0.03)
    return slap * 1.2 + _thump(n, 140, 80, 30, 0.05, rng)


@register('swing', gain_db=-6.0, reverb=0.15)
def swing(c, rng):
    """A man swung overhead: a long heavy whoosh."""
    n = n_of(0.6)
    t = times(n) / 0.6
    fc = 250 + 900 * np.sin(np.pi * t) ** 2
    return sweep(pink(n, rng), fc, 'bp', q=1.1) * np.sin(np.pi * t) ** 1.5 * 1.4


@register('throw', gain_db=-6.0, reverb=0.15)
def throw(c, rng):
    return swing(c, rng)


@register('rip_road', gain_db=-3.0, reverb=0.14)
def rip_road(c, rng):
    n = n_of(1.0)
    t = times(n)
    groan = sweep(osc(drop(110, 60, n, 3), n, 'saw'), 400 + 300 * np.sin(np.pi * t), 'lp', block=256) * env(n, 0.05, 0.4) * 0.5
    return sat(groan + crackle(n, rng, rate=500, decay=0.003, lo=500, hi=4000) * env(n, 0.02, 0.5) * 2.2
               + _thump(n, 90, 40, 12, 0.2, rng), 1.3)


def _heavy_steps(c, rng):
    """Footsteps; the Hulk's are deep thuds."""
    if not c.get('heavy'):
        return sfx.steps(c, rng)
    dur = float(c.get('dur', 1.0))
    rate = float(c.get('rate', 3.2))
    n = n_of(dur + 0.3)
    out = np.zeros(n)
    k = 0.0
    while k < dur:
        m = n_of(0.25)
        thud = _thump(m, 90, 38, 18, 0.08, rng) * 1.6 + lp(white(m, rng), 900) * env(m, 0.001, 0.02) * 0.4
        i = n_of(k)
        out[i:i + m] += thud[:max(0, min(m, n - i))] * rng.uniform(0.75, 1.0)
        k += 1.0 / rate * rng.uniform(0.92, 1.08)
    return out


@register('fall_whistle', gain_db=-8.0, reverb=0.25)
def fall_whistle(c, rng):
    dur = max(0.3, float(c.get('dur', 0.6)))
    n = n_of(dur)
    t = times(n) / dur
    tone = osc(1900 * (1 - 0.7 * t), n) * t * 0.25
    wind = sweep(white(n, rng), 600 + 2400 * t, 'bp', q=1.3) * t ** 1.5
    return tone + wind


# ------------------------------------------------------------------ the shield
@register('shield_block', gain_db=-2.0, reverb=0.12)
def shield_block(c, rng):
    """A blow on the shield, dry like the weapons after SOUND-2: the crack of contact, a short low clang that is gone
    within a tenth of a second, and the shove behind it (no long ring)."""
    w = int(c.get('weight', 2))
    n = n_of(0.6)
    steel = sfx._steel(n, rng, weight=1.0 + 0.3 * w, grit=1.0)
    clang = _metal(n, rng, base=rng.uniform(260, 320), bright=0.8, decay=0.05 + 0.02 * w) * env(n, 0.0005, 0.09)
    shove = _thump(n, 120, 45, 14, 0.1 + 0.04 * w, rng) * (0.8 + 0.3 * w)
    return sat(steel + clang * 0.5 + shove, 1.3)


@register('shield_ricochet', gain_db=-4.0, reverb=0.1)
def shield_ricochet(c, rng):
    """The shield bouncing off something: a dry ping (gone in a few hundredths), a short whine, a thump on a body."""
    n = n_of(0.5)
    t = times(n)
    ping = _metal(n, rng, base=rng.uniform(480, 560), bright=1.0, decay=0.03) * env(n, 0.0003, 0.05)
    whine = osc(3200 - 1400 * np.clip(t / 0.25, 0, 1), n) * env(n, 0.002, 0.1) * 0.2
    x = sfx._steel(n, rng, weight=0.8) * 0.8 + ping * 0.5 + whine
    if c.get('hit'):
        x += _thump(n, 150, 60, 20, 0.1, rng) * 1.2
    return sat(x, 1.2)


@register('shield_throw', gain_db=-6.0, reverb=0.15)
def shield_throw(c, rng):
    n = n_of(0.35)
    t = times(n) / 0.35
    return sweep(white(n, rng), 800 + 2500 * t, 'bp', q=1.5) * np.sin(np.pi * t) * 1.2


@register('shield_fly', gain_db=-11.0, reverb=0.15)
def shield_fly(c, rng):
    """The spinning shield: a whirring hum that beats as it turns."""
    dur = max(0.3, float(c.get('dur', 1.0)))
    n = n_of(dur)
    t = times(n)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 22 * t)
    x = bp(white(n, rng), 900, 3200) * am + osc(240 + 40 * np.sin(2 * np.pi * 22 * t), n) * 0.25 * am
    return x * np.clip(t / 0.05, 0, 1) * np.clip((dur - t) / 0.05, 0, 1)


@register('shield_catch', gain_db=-5.0, reverb=0.25)
def shield_catch(c, rng):
    n = n_of(1.2)
    if c.get('flat'):
        x = _metal(n, rng, base=330, bright=0.8, decay=0.6) * env(n, 0.0005, 0.6) * 0.8
        for k, dt in enumerate((0.0, 0.11, 0.19, 0.24, 0.27)):
            i = n_of(dt)
            x[i:] += _metal(n - i, rng, base=340, bright=0.6, decay=0.2) * 0.5 / (k + 1)
        return x
    return _metal(n, rng, base=360, bright=0.8, decay=0.35) * env(n, 0.0005, 0.4) * 0.6 + _thump(n, 160, 70, 25, 0.06, rng)


@register('shield_roll', gain_db=-10.0, reverb=0.3)
def shield_roll(c, rng):
    """The dropped shield rolling and wobbling down: a ringing whose beat speeds up."""
    dur = max(0.5, float(c.get('dur', 2.0)))
    n = n_of(dur)
    t = times(n) / dur
    rate = 4 + 30 * t ** 2
    ph = 2 * np.pi * np.cumsum(rate) / SR
    am = 0.5 + 0.5 * np.abs(np.sin(ph))
    x = _metal(n, rng, base=340, bright=0.6, decay=3.0) * am * 0.4 + crackle(n, rng, rate=80, lo=1500, hi=5000) * 0.3
    return x * np.clip(t * 6, 0, 1)


# ------------------------------------------------------------------ the street
@register('glass', gain_db=-6.0, reverb=0.12)
def glass(c, rng):
    p = float(c.get('power', 1.0))
    n = n_of(1.0 + 0.6 * p)
    crash = hp(white(n, rng), 2500) * env(n, 0.0005, 0.05 + 0.03 * p) * 1.5
    return crash + _tinkle(n, rng, int(50 + 70 * p), 2500, 10000, 0.4 + 0.4 * p) * 1.2


@register('car_hit', gain_db=-3.0, reverb=0.12)
def car_hit(c, rng):
    """Sheet metal crumpling: a hollow boom, a crunch, a ringing panel."""
    p = float(c.get('power', 1.0))
    n = n_of(1.4)
    x = _thump(n, 95, 40, 10, 0.25 + 0.1 * p, rng) * 1.4
    x += crackle(n, rng, rate=900, decay=0.0025, lo=400, hi=3500) * env(n, 0.002, 0.1) * 2.2
    x += _metal(n, rng, base=rng.uniform(110, 150), bright=0.5, decay=0.06) * env(n, 0.001, 0.08) * 0.6
    return sat(x * (0.8 + 0.3 * p), 1.4)


@register('car_crash', gain_db=-1.0, reverb=0.15)
def car_crash(c, rng):
    p = float(c.get('power', 1.5))
    n = n_of(2.6)
    x = _fit(car_hit({'power': p}, rng), n) * 1.2
    i = n_of(0.12)
    x[i:] += _fit(car_hit({'power': p * 0.7}, rng), n - i) * 0.7
    x += crackle(n, rng, rate=300, decay=0.004, lo=300, hi=3000) * env(n, 0.03, 0.35) * 1.4
    x += _tinkle(n, rng, 90, 2500, 9000, 0.8) * 0.8
    return sat(x, 1.4)


@register('car_alarm', gain_db=-21.0, reverb=0.2)
def car_alarm(c, rng):
    dur = max(1.0, float(c.get('dur', 6.0)))
    n = n_of(dur)
    t = times(n)
    f = np.where((t * 2.2) % 1.0 < 0.5, 760, 1120)
    x = bp(osc(f, n, 'square'), 600, 3000) * 0.6
    return x * np.clip(t / 0.05, 0, 1) * np.clip((dur - t) / 1.0, 0, 1)


@register('car_lift', gain_db=-8.0, reverb=0.3)
def car_lift(c, rng):
    n = n_of(1.0)
    t = times(n)
    creak = sweep(osc(drop(140, 90, n, 2) * (1 + 0.05 * np.sin(2 * np.pi * 7 * t)), n, 'saw'), np.full(n, 700.0), 'lp', block=256)
    return creak * env(n, 0.05, 0.4) * 0.6 + crackle(n, rng, rate=120, lo=500, hi=3000) * env(n, 0.02, 0.4)


@register('car_fly', gain_db=-9.0, reverb=0.15)
def car_fly(c, rng):
    return sfx.fly(c, rng) * 1.3


@register('pole_hit', gain_db=-4.0, reverb=0.1)
def pole_hit(c, rng):
    n = n_of(1.6)
    return _metal(n, rng, base=rng.uniform(150, 190), bright=0.9, decay=0.05) * env(n, 0.0005, 0.08) + sfx._steel(n, rng, weight=1.2) * 0.8 + \
        _thump(n, 120, 50, 15, 0.1, rng)


@register('metal_crash', gain_db=-3.0, reverb=0.12)
def metal_crash(c, rng):
    n = n_of(1.8)
    x = _metal(n, rng, base=rng.uniform(120, 160), bright=1.0, decay=0.07) * env(n, 0.0005, 0.12) + sfx._steel(n, rng, weight=1.5)
    for k in range(4):
        i = n_of(0.1 + 0.09 * k + rng.uniform(0, 0.04))
        x[i:] += _metal(n - i, rng, base=rng.uniform(140, 220), bright=0.7, decay=0.04) * 0.5 / (k + 1)
    return sat(x + _thump(n, 100, 45, 12, 0.15, rng), 1.3)


@register('sparks', gain_db=-10.0, reverb=0.2)
def sparks(c, rng):
    """Live wires: a buzzing hum cut in and out, and crackling arcs."""
    dur = max(0.3, float(c.get('dur', 0.6)))
    n = n_of(dur)
    t = times(n)
    gate = (lp(rng.standard_normal(n), 25) > 0).astype(float)
    buzz = sat(bp(osc(120, n, 'saw'), 100, 2500) * 3, 2.0) * gate * 0.5
    arcs = crackle(n, rng, rate=300, decay=0.0008, lo=2500, hi=9000) * 2.0
    return (buzz + arcs) * np.clip((dur - t) / 0.2, 0, 1)


@register('electrocute', gain_db=-2.0, reverb=0.25)
def electrocute(c, rng):
    dur = max(0.5, float(c.get('dur', 1.2)))
    n = n_of(dur)
    t = times(n)
    gate = 0.4 + 0.6 * (lp(rng.standard_normal(n), 18) > -0.1)
    buzz = sat(bp(osc(60 * (1 + 0.02 * np.sin(2 * np.pi * 3 * t)), n, 'saw'), 60, 3000) * 4, 2.5) * gate
    zap = crackle(n, rng, rate=900, decay=0.0006, lo=2000, hi=10000) * 2.5
    return (buzz + zap) * np.clip(t / 0.01, 0, 1) * np.clip((dur - t) / 0.15, 0, 1)


@register('fire', gain_db=-18.0, reverb=0.1)
def fire(c, rng):
    return sfx.coals(c, rng) * 1.2


@register('wall_break', gain_db=-1.0, reverb=0.18)
def wall_break(c, rng):
    """A wall burst through: a deep boom, cracking brick, a long rubble fall, dust."""
    p = float(c.get('power', 1.0))
    n = n_of(2.4)
    x = _thump(n, 80, 30, 8, 0.3 + 0.1 * p, rng) * 1.5
    x += hp(white(n, rng), 1600) * env(n, 0.0005, 0.03) * 1.2
    x += crackle(n, rng, rate=1400, decay=0.002, lo=600, hi=4500) * env(n, 0.003, 0.15) * 2.2
    x += _debris(n, rng, 1.2, 110 * p, 600, 4500) * 1.8
    x += lp(white(n, rng), 1800) * env(n, 0.05, 0.4) * 0.15
    return sat(x, 1.4)


@register('cave_in', gain_db=0.0, reverb=0.25)
def cave_in(c, rng):
    n = n_of(3.6)
    x = _fit(wall_break({'power': 2.0}, rng), n) * 1.2
    x += lp(brown(n, rng), 140) * env(n, 0.1, 1.6) * 1.2
    x += _debris(n, rng, 3.4, 200, 400, 4000) * 1.8
    return sat(x, 1.4)


@register('table_break', gain_db=-3.0, reverb=0.12)
def table_break(c, rng):
    n = n_of(1.2)
    snap = bp(white(n, rng), 800, 4000) * env(n, 0.0003, 0.02) * 1.8
    x = snap + _thump(n, 130, 60, 20, 0.08, rng) + crackle(n, rng, rate=400, decay=0.002, lo=900, hi=5000) * env(n, 0.01, 0.2) * 1.4
    return x + _tinkle(n, rng, 30, 1800, 6000, 0.4) * 0.6


@register('wood_crash', gain_db=-1.0, reverb=0.14)
def wood_crash(c, rng):
    n = n_of(2.0)
    x = _fit(table_break(c, rng), n) * 1.2 + _thump(n, 80, 32, 8, 0.35, rng) * 1.2
    x += crackle(n, rng, rate=600, decay=0.003, lo=700, hi=4500) * env(n, 0.02, 0.6) * 1.6
    return sat(x, 1.3)


@register('water_burst', gain_db=-3.0, reverb=0.35)
def water_burst(c, rng):
    n = n_of(1.6)
    t = times(n)
    gush = sweep(white(n, rng), 3000 * np.exp(-t * 2) + 500, 'lp', block=256) * env(n, 0.005, 0.6) * 1.5
    return gush + _metal(n, rng, base=170, bright=0.5, decay=0.4) * env(n, 0.0005, 0.4) * 0.6 + _thump(n, 90, 40, 12, 0.2, rng)


@register('water', gain_db=-17.0, reverb=0.2)
def water(c, rng):
    """Water running on: a hissing gush (the geyser) or a heavy splattering fall (the main into the tunnel)."""
    dur = min(60.0, max(1.0, float(c.get('dur', 10.0))))
    n = n_of(dur)
    t = times(n)
    breathe = 0.85 + 0.15 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 6))
    if c.get('source') == 'fall':
        x = lp(pink(n, rng), 2200) * 1.2 + crackle(n, rng, rate=400, decay=0.002, lo=600, hi=4000) * 0.8
    else:
        x = bp(white(n, rng), 1500, 7000) * 0.9 + lp(pink(n, rng), 900) * 0.6
    return x * breathe * np.clip(t / 0.3, 0, 1)


# ------------------------------------------------------------------ the Hulk's voice
voice.STYLES['hulk'] = {
    'voice': 'Microsoft David Desktop', 'rate': '-30%', 'pitch': '-20%', 'hall': (1.6, 0.5, 0.3),
    'chain': 'asplit=2[a][b];[a]asetrate=48000*0.66,aresample=48000,atempo=1.25[a1];'
             '[b]asetrate=48000*0.5,aresample=48000,atempo=1.65[b1];[a1][b1]amix=inputs=2:weights=1 0.8,'
             'highpass=f=60,lowpass=f=4200,equalizer=f=150:t=q:w=1:g=6,volume=3,asoftclip=type=tanh:param=2.2,'
             'acompressor=threshold=-20dB:ratio=6:attack=3:release=120:makeup=6'}
mix.BIG |= {'hulk_land', 'smash', 'body_slam', 'clap', 'shockwave', 'wall_break', 'cave_in', 'car_crash', 'roar',
            'shield_block'}


_light_steps = sfx.REGISTRY['steps']


def _steps(c, rng):
    if c.get('heavy'):
        return _heavy_steps(c, rng)
    return _light_steps[0](c, rng)


sfx.REGISTRY['steps'] = (_steps, _light_steps[1], _light_steps[2])
