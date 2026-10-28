"""The sound effects: every one synthesised here (noise, oscillators, filters, struck partials), nothing sampled.

Each generator takes a cue (a dict from the renderer's cue sheet: t, kind, pan, dur and its own params) and a random
generator (so every instance differs a little) and returns a mono array at dsp.SR. `REGISTRY` maps a cue kind to
(generator, gain in dB); a new clip adds its own kinds with `register`.
"""
import numpy as np

from dsp import (SR, n_of, times, white, pink, brown, lp, hp, bp, sweep, env, osc, drop, sat, crackle, partials,
                 karplus)

REGISTRY = {}


def register(kind, gain_db=0.0, reverb=0.25):
    def deco(fn):
        REGISTRY[kind] = (fn, gain_db, reverb)
        return fn
    return deco


def _thump(n, f0, f1, rate, decay, rng):
    return osc(drop(f0, f1, n, rate), n) * env(n, 0.001, decay)


# ------------------------------------------------------------------ dry steel (SOUND-2: weapons that do not ring)
def _click(n, rng, lo=3000.0, dur=0.0025):
    """The crack at the instant of contact."""
    return hp(white(n, rng), lo) * env(n, 0.0, dur)


def _shing(n, rng, length=0.06, f_hi=7000.0, f_lo=2600.0, q=3.0):
    """A short noisy scrape of steel on steel: band noise sliding down, gone within `length` seconds."""
    t = times(n)
    fc = f_lo + (f_hi - f_lo) * np.exp(-t / max(0.01, length * 0.5))
    return sweep(white(n, rng), fc, 'bp', q=q) * env(n, 0.0008, length * 0.4)


def _glint(n, rng, f0, tau=0.025, level=0.12):
    """At most a glint of ring: two partials dying within a few hundredths of a second, low."""
    return partials([f0, f0 * 2.71], [tau, tau * 0.6], n, rng=rng) * level


def _steel(n, rng, weight=1.0, grit=1.0):
    """Dry steel on steel: a crack, gritty grains, a short 'shing', a body thump, a glint; every hit a little different
    (pitch, length, grit)."""
    click = _click(n, rng, rng.uniform(2800, 4500), rng.uniform(0.0015, 0.003)) * 1.5
    crack = crackle(n, rng, rate=rng.uniform(1800, 3200) * grit, decay=0.001, lo=1800, hi=9000) * \
        env(n, 0.0004, rng.uniform(0.008, 0.02) * (0.8 + 0.4 * weight)) * 2.0 * grit
    shing = _shing(n, rng, rng.uniform(0.04, 0.075) * (0.8 + 0.3 * weight), rng.uniform(5500, 8500),
                   rng.uniform(2200, 3400), rng.uniform(2.2, 4.0))
    thump = _thump(n, rng.uniform(130, 190), rng.uniform(70, 95), 30, rng.uniform(0.02, 0.03), rng) * (0.5 + 0.3 * weight)
    glint = _glint(n, rng, rng.uniform(1900, 3200), rng.uniform(0.015, 0.03), rng.uniform(0.05, 0.1))
    return click + crack + shing * 1.3 + thump + glint


def _ice_crunch(n, rng, size=1.0):
    """Ice cracking: a crisp crack and a crunch of grains, no chimes."""
    crack = hp(white(n, rng), rng.uniform(1200, 2000)) * env(n, 0.0, rng.uniform(0.004, 0.008)) * 1.4
    crunch = crackle(n, rng, rate=rng.uniform(2500, 4000), decay=0.0008, lo=1500, hi=9000) * \
        env(n, 0.001, rng.uniform(0.02, 0.04) * size) * 2.2
    return (crack + crunch) * size


def _clank(n, rng, f0):
    """A heavy, dead clank of thick metal (bronze): a noisy knock with partials that die in a few hundredths."""
    knock = bp(white(n, rng), 300, 3000) * env(n, 0.0005, rng.uniform(0.03, 0.05))
    body = partials([f0, f0 * 2.3, f0 * 3.9, f0 * 5.2], [0.06, 0.04, 0.025, 0.018], n, rng=rng, detune=0.03) * 0.5
    return knock * 1.2 + body + _thump(n, 120, 60, 20, 0.06, rng) * 0.8


# ------------------------------------------------------------------ blows
@register('hit', gain_db=-3.0, reverb=0.18)
def hit(c, rng):
    """A blow landing: a body thump, the slap of the contact, a click; heavy blows add a crunch."""
    w = int(c.get('weight', 1))
    kick = c.get('limb') == 'kick'
    dur = 0.16 + 0.12 * w
    n = n_of(dur)
    f0 = (165 if not kick else 130) - 18 * w + rng.uniform(-10, 10)
    body = _thump(n, f0, 48 + 6 * (2 - w), 24, 0.045 + 0.04 * w, rng)
    slap = bp(white(n, rng), 700 + rng.uniform(-100, 200), 3800) * env(n, 0.0008, 0.018 + 0.01 * w)
    click = hp(white(n, rng), 3500) * env(n, 0.0, 0.003)
    x = body * (1.1 + 0.25 * w) + slap * (0.7 if not kick else 0.55) + click * 0.35
    if w >= 2:
        x += crackle(n, rng, rate=900, decay=0.0015, lo=900, hi=5000) * env(n, 0.001, 0.035) * 2.5
        x += _thump(n, 70, 38, 12, 0.18, rng) * 0.8
    if c.get('blood', 0) >= 8:
        sq = sweep(white(n, rng), 1600 * np.exp(-times(n) * 9) + 250, 'bp', q=3.0)
        x += sq * env(n, 0.004, 0.06) * 0.35
    return sat(x * 1.3, 1.6)


@register('block', gain_db=-5.0, reverb=0.15)
def block(c, rng):
    """Forearms meeting a blow: a dull knock and a hard tick."""
    n = n_of(0.16)
    knock = bp(white(n, rng), 450, 1100) * env(n, 0.0005, 0.03)
    tick = hp(white(n, rng), 4000) * env(n, 0.0, 0.004)
    body = _thump(n, 140, 90, 30, 0.04, rng)
    return sat(knock * 1.4 + tick * 0.5 + body * 0.8, 1.4)


@register('whoosh', gain_db=-11.0, reverb=0.1)
def whoosh(c, rng):
    """A limb cutting the air: band noise whose centre sweeps up and back, faster blows brighter."""
    sp = float(c.get('speed', 1.0))
    dur = 0.2 / sp ** 0.4
    n = n_of(dur)
    t = times(n) / dur
    fc = (450 + 1300 * sp) * (0.35 + np.sin(np.pi * t) ** 1.5) + 200
    x = sweep(white(n, rng), fc, 'bp', q=1.6) * np.sin(np.pi * np.clip(t, 0, 1)) ** 2
    return x * (0.8 + 0.2 * sp)


@register('fly', gain_db=-14.0, reverb=0.1)
def fly(c, rng):
    """A body flying through the air: a long, low rush of wind."""
    dur = max(0.15, float(c.get('dur', 0.4)))
    n = n_of(dur)
    t = times(n) / dur
    fc = 300 + 700 * np.sin(np.pi * t)
    return sweep(pink(n, rng), fc, 'bp', q=1.0) * np.sin(np.pi * t) ** 1.5


# ---------------------------------------------------------- feet and floor
@register('steps', gain_db=-16.0, reverb=0.12)
def steps(c, rng):
    dur = float(c.get('dur', 1.0))
    rate = float(c.get('rate', 3.6))
    n = n_of(dur + 0.1)
    out = np.zeros(n)
    k = 0.0
    while k < dur:
        m = n_of(0.06)
        tap = lp(white(m, rng), 1400) * env(m, 0.001, 0.012) + _thump(m, 110, 70, 40, 0.02, rng) * 0.8
        i = n_of(k)
        out[i:i + m] += tap[:max(0, min(m, n - i))] * (0.5 if c.get('soft') else 1.0) * rng.uniform(0.7, 1.0)
        k += 1.0 / rate * rng.uniform(0.9, 1.1)
    return out


@register('dash', gain_db=-10.0, reverb=0.1)
def dash(c, rng):
    """A burst off the mark: a scuff of grit and a quick rush."""
    n = n_of(0.3)
    t = times(n)
    scuff = bp(white(n, rng), 1200, 6000) * env(n, 0.002, 0.04)
    rush = sweep(white(n, rng), 600 + 2500 * np.exp(-t * 8), 'bp', q=1.2) * env(n, 0.01, 0.1)
    return scuff * 0.8 + rush


@register('jump', gain_db=-13.0, reverb=0.1)
def jump(c, rng):
    n = n_of(0.18)
    t = times(n)
    push = _thump(n, 120, 80, 40, 0.025, rng) + lp(white(n, rng), 1500) * env(n, 0.001, 0.015)
    air = sweep(white(n, rng), 500 + 1500 * t / 0.18, 'bp', q=1.4) * env(n, 0.02, 0.06)
    return push * 0.8 + air * 0.6


@register('land', gain_db=-9.0, reverb=0.15)
def land(c, rng):
    p = float(c.get('power', 0.5))
    n = n_of(0.3)
    x = _thump(n, 120, 55, 25, 0.05 + 0.04 * p, rng) * (0.6 + 0.6 * p)
    x += bp(white(n, rng), 800, 5000) * env(n, 0.001, 0.02) * 0.5
    x += lp(white(n, rng), 3000) * env(n, 0.01, 0.12) * 0.15 * p
    return x


@register('bodyfall', gain_db=-5.0, reverb=0.2)
def bodyfall(c, rng):
    """A body slamming flat on stone: a heavy thump, the slap of cloth, a little grit."""
    p = float(c.get('power', 1.0))
    n = n_of(0.6)
    x = _thump(n, 100, 42, 18, 0.12 + 0.05 * p, rng) * 1.3
    x += bp(white(n, rng), 300, 2500) * env(n, 0.001, 0.04) * 0.8
    x += crackle(n, rng, rate=60 * p, decay=0.003, lo=1500, hi=6000) * env(n, 0.01, 0.25) * 1.2
    return sat(x * p ** 0.5, 1.3)


# ------------------------------------------------------------ stone
def _debris(n, rng, dur, density=60.0):
    x = crackle(n, rng, rate=density, decay=0.004, lo=1200, hi=6500)
    return x * env(n, 0.005, dur * 0.4)


@register('stone_crash', gain_db=-2.0, reverb=0.3)
def stone_crash(c, rng):
    """A body smashed into a stone pillar: a deep boom, the crack of stone, chips rattling down, a dust hiss."""
    p = float(c.get('power', 1.0))
    n = n_of(1.6)
    boom = _thump(n, 85, 38, 10, 0.22 + 0.08 * p, rng) * 1.5
    crack = hp(white(n, rng), 1800) * env(n, 0.0005, 0.02) * 1.2
    crunch = crackle(n, rng, rate=1500, decay=0.0015, lo=800, hi=5000) * env(n, 0.002, 0.05) * 2.0
    debris = _debris(n, rng, 1.4, 90 * p) * 1.6
    dust = lp(white(n, rng), 2500) * env(n, 0.05, 0.5) * 0.12
    return sat((boom + crack + crunch + debris + dust) * (0.8 + 0.2 * p), 1.4)


@register('floor_slam', gain_db=0.0, reverb=0.35)
def floor_slam(c, rng):
    """Driven into the floor: a sub drop under a stone crash."""
    n = n_of(2.0)
    x = np.zeros(n)
    sc = stone_crash({'power': 2.0}, rng)
    x[:len(sc)] += sc
    x += _thump(n, 70, 28, 6, 0.45, rng) * 1.4
    return sat(x, 1.3)


# ------------------------------------------------------------ the brazier
@register('brazier_crash', gain_db=-3.0, reverb=0.15)
def brazier_crash(c, rng):
    """A bronze brazier toppling: a heavy dead clank, the bowl hitting the floor, coals scattering, a gust of flame."""
    n = n_of(1.6)
    clank = _clank(n, rng, rng.uniform(260, 340)) * 1.2
    second = np.zeros(n)
    k = n_of(0.18)
    second[k:] = _clank(n - k, rng, rng.uniform(220, 300)) * 0.7
    coals = crackle(n, rng, rate=260, decay=0.003, lo=1500, hi=7000) * env(n, 0.01, 0.5) * 1.6
    gust = fire_burst({'size': 1}, rng)
    out = clank + second + coals
    m = min(len(out), len(gust))
    out[:m] += gust[:m] * 0.7
    return out


@register('coals', gain_db=-19.0, reverb=0.1)
def coals(c, rng):
    """Spilled coals burning on: crackle over a low rumble, breathing slowly."""
    dur = float(c.get('dur', 10.0))
    n = n_of(dur)
    t = times(n)
    breathe = 0.75 + 0.25 * np.sin(2 * np.pi * 0.31 * t) * np.sin(2 * np.pi * 0.07 * t + 1)
    x = crackle(n, rng, rate=22, decay=0.003, lo=1200, hi=7000) * 1.6 + lp(brown(n, rng), 280) * 0.25
    fade = np.clip(t / 0.4, 0, 1)
    return x * breathe * fade


# --------------------------------------------------------------- ice
def _chimes(n, rng, count, lo, hi, dmin, dmax, spread=0.0):
    out = np.zeros(n)
    t = times(n)
    for _ in range(count):
        f = rng.uniform(lo, hi)
        d = rng.uniform(dmin, dmax)
        t0 = rng.uniform(0, spread)
        a = rng.uniform(0.3, 1.0)
        tt = t - t0
        out += np.where(tt > 0, a * np.sin(2 * np.pi * f * tt) * np.exp(-np.clip(tt, 0, None) / d), 0.0)
    return out / max(1.0, count ** 0.5)


@register('frost', gain_db=-15.0, reverb=0.35)
def frost(c, rng):
    """Cold gathering in his hands: an icy shimmer and a breath of hissing air, rising."""
    dur = max(0.3, float(c.get('dur', 1.0)))
    n = n_of(dur)
    t = times(n)
    rise = np.clip(t / dur, 0, 1) ** 1.3
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * 11 * t)
    shimmer = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (2637, 3520, 4186, 5274)) / 4
    hiss = hp(white(n, rng), 5000) * 0.5
    return (shimmer * trem * 0.6 + hiss) * rise * np.clip((dur - t) / 0.08, 0, 1)


@register('ice_cast', gain_db=-7.0, reverb=0.3)
def ice_cast(c, rng):
    n = n_of(0.5)
    t = times(n)
    fwip = sweep(white(n, rng), 3000 + 4000 * np.exp(-t * 10), 'bp', q=2.0) * env(n, 0.003, 0.08)
    ping = _chimes(n, rng, 5, 2500, 6000, 0.08, 0.3)
    return fwip + ping * 0.8


@register('ice_fly', gain_db=-12.0, reverb=0.2)
def ice_fly(c, rng):
    """The ice ball in flight: an airy hiss with a whistle, panned from the thrower to its end."""
    dur = max(0.15, float(c.get('dur', 0.4)))
    n = n_of(dur)
    t = times(n)
    whistle = osc(1500 + 150 * np.sin(2 * np.pi * 14 * t), n) * 0.35
    air = bp(white(n, rng), 2500, 8000) * 0.7
    shape = np.clip(t / 0.03, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)
    return (whistle + air) * shape


@register('ice_hit', gain_db=-4.0, reverb=0.12)
def ice_hit(c, rng):
    """A blow on solid ice (a fist, the hammer, a blade): a hard knock and a crunchy crack, no ringing."""
    w = int(c.get('weight', 1))
    n = n_of(0.35)
    knock = _thump(n, rng.uniform(160, 200), 110, 35, 0.03 + 0.01 * w, rng) + hp(white(n, rng), 2500) * env(n, 0.0, 0.005)
    return knock + _ice_crunch(n, rng, 1.0 + 0.3 * w)


@register('freeze', gain_db=-2.0, reverb=0.35)
def freeze(c, rng):
    """Freezing solid: a crack, crystals creaking as they spread, a ringing like glass."""
    n = n_of(1.4)
    t = times(n)
    crack = hp(white(n, rng), 1500) * env(n, 0.0, 0.03) * 1.2
    grains = np.zeros(n)
    for k in range(60):
        t0 = 0.05 + 0.6 * (k / 60.0) ** 0.7
        m = n_of(0.006)
        i = n_of(t0)
        g = hp(white(m, rng), 4000) * env(m, 0.0, 0.0015)
        grains[i:i + m] += g * rng.uniform(0.2, 0.8)
    ring = _chimes(n, rng, 10, 2200, 8000, 0.15, 0.7, spread=0.15)
    thump = _thump(n, 100, 60, 20, 0.08, rng)
    return crack + grains * 1.5 + ring * 1.4 + thump * 0.6


@register('ice_splash', gain_db=-6.0, reverb=0.3)
def ice_splash(c, rng):
    n = n_of(1.0)
    burst = hp(white(n, rng), 1200) * env(n, 0.0, 0.05)
    return burst + _chimes(n, rng, 18, 2000, 9000, 0.04, 0.3, spread=0.25) * 1.5


@register('ice_shatter', gain_db=0.0, reverb=0.4)
def ice_shatter(c, rng):
    """The ice block bursting: a blast of breaking glass, shards ringing, pieces tinkling down."""
    n = n_of(2.0)
    blast = hp(white(n, rng), 900) * env(n, 0.0, 0.12) * 1.4
    shards = _chimes(n, rng, 70, 1800, 10000, 0.04, 0.35, spread=0.3) * 2.0
    tinkle = np.zeros(n)
    for _ in range(30):
        i = n_of(rng.uniform(0.3, 1.6))
        m = n_of(0.08)
        seg = _chimes(m, rng, 2, 3000, 9000, 0.01, 0.04)
        tinkle[i:i + m] += seg[:max(0, min(m, n - i))]
    boom = _thump(n, 80, 40, 10, 0.25, rng)
    return blast + shards + tinkle * 1.2 + boom * 0.9


# --------------------------------------------------------------- fire
@register('fire_burst', gain_db=-3.0, reverb=0.3)
def fire_burst(c, rng):
    """A gout of flame: a 'whoomph' of filtered noise that blooms and dies, with crackle."""
    s = float(c.get('size', 1))
    dur = 0.8 + 0.4 * s
    n = n_of(dur)
    t = times(n)
    fc = 300 + 2600 * np.exp(-((t - 0.08) / 0.12) ** 2) + 500 * np.exp(-t * 3)
    roar = sweep(pink(n, rng), fc, 'lp') * env(n, 0.035, 0.25 + 0.15 * s) * 1.6
    low = _thump(n, 70, 40, 8, 0.2, rng) * 0.6
    crack = crackle(n, rng, rate=120, decay=0.003) * env(n, 0.02, 0.4) * 1.3
    return sat(roar + low + crack, 1.2)


@register('fire_breath', gain_db=-4.0, reverb=0.3)
def fire_breath(c, rng):
    """The fire breath: a roaring torrent that flutters, with crackle and a deep rumble under it."""
    dur = max(0.5, float(c.get('dur', 2.0)))
    n = n_of(dur + 0.5)
    t = times(n)
    flutter = 0.7 + 0.3 * np.sin(2 * np.pi * 9 * t + 2 * np.sin(2 * np.pi * 1.3 * t))
    shape = np.clip(t / 0.08, 0, 1) * np.clip((dur + 0.4 - t) / 0.5, 0, 1)
    fc = 900 + 500 * np.sin(2 * np.pi * 0.7 * t) + 1500 * np.exp(-t * 4)
    roar = sweep(pink(n, rng), fc, 'lp') * 1.5
    rumble = lp(brown(n, rng), 140) * 0.8
    crack = crackle(n, rng, rate=80, decay=0.003) * 1.2
    hiss = hp(white(n, rng), 4500) * 0.12
    wane = 1.0 - 0.42 * np.clip((t - 0.9) / 1.0, 0, 1)        # the torrent eases once he is ablaze
    x = (roar * flutter + rumble + crack + hiss) * shape * wane
    onset = fire_burst({'size': 2}, rng)
    x[:len(onset)] += onset * 0.9
    return sat(x, 1.2)


@register('burning', gain_db=-12.0, reverb=0.25)
def burning(c, rng):
    dur = max(0.3, float(c.get('dur', 1.0)))
    n = n_of(dur + 0.3)
    t = times(n)
    shape = np.clip(t / 0.1, 0, 1) * np.clip((dur + 0.2 - t) / 0.3, 0, 1)
    x = lp(pink(n, rng), 1200) * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t)) + crackle(n, rng, rate=50, decay=0.003) * 1.4
    return x * shape * (1.3 if c.get('big') else 0.8)


@register('skull', gain_db=-2.0, reverb=0.45)
def skull(c, rng):
    """The mask torn off: flame bursting round a skull over an ominous low boom."""
    n = n_of(2.0)
    t = times(n)
    boom = osc(drop(70, 32, n, 3), n) * env(n, 0.01, 0.7) * 1.2
    swell = lp(white(n, rng), 3000) * np.clip(t / 0.25, 0, 1) * np.exp(-np.clip(t - 0.25, 0, None) / 0.2) * 0.5
    burst = fire_burst({'size': 1}, rng)
    x = boom + swell
    x[:len(burst)] += burst * 0.8
    return x


# ------------------------------------------------------------- the spear
@register('kunai_throw', gain_db=-6.0, reverb=0.1)
def kunai_throw(c, rng):
    """The kunai leaving the hand: a fast swish and a click of steel."""
    n = n_of(0.3)
    t = times(n)
    swish_ = sweep(white(n, rng), rng.uniform(2200, 2800) + 3000 * t / 0.3, 'bp', q=2.0) * env(n, 0.002, 0.07)
    return swish_ + _click(n, rng, 3500, 0.002) * 0.7 + _glint(n, rng, rng.uniform(3000, 3800), 0.012, 0.1)


@register('rope_zip', gain_db=-6.0, reverb=0.15)
def rope_zip(c, rng):
    """Rope paying out fast: a buzzing zip that rises in pitch and stops dead."""
    dur = max(0.1, float(c.get('dur', 0.3)))
    n = n_of(dur)
    t = times(n) / dur
    buzz = osc(90 + 160 * t, n, 'saw') * 0.35
    zip_ = sweep(white(n, rng), 700 + 2600 * t, 'bp', q=3.0)
    return (buzz + zip_) * np.clip(t / 0.1, 0, 1)


@register('pierce', gain_db=0.0, reverb=0.15)
def pierce(c, rng):
    """The kunai biting into his shoulder: a dry tick, a wet tearing squelch, a thud."""
    n = n_of(0.6)
    t = times(n)
    tick = _click(n, rng, 3000, 0.003) * 1.2
    squelch = sweep(white(n, rng), 1800 * np.exp(-t * 7) + 250, 'bp', q=4.0) * env(n, 0.003, 0.12) * 1.8
    wet = osc(drop(320, 110, n, 18), n) * env(n, 0.002, 0.06) * 0.8
    thud = _thump(n, 120, 60, 25, 0.08, rng)
    return sat(tick + squelch + wet + thud, 1.4)


@register('yank', gain_db=-3.0, reverb=0.12)
def yank(c, rng):
    """The haul on the rope: a taut, dead snap of the rope and a rush."""
    n = n_of(0.55)
    tw = np.zeros(n)
    s_ = karplus(rng.uniform(70, 85), 0.4, rng, damp=0.985)
    tw[:len(s_)] = s_
    t = times(n)
    rush = sweep(pink(n, rng), 400 + 1600 * np.exp(-t * 5), 'bp', q=1.0) * env(n, 0.01, 0.2)
    snap = bp(white(n, rng), 600, 3000) * env(n, 0.0, 0.01)
    return tw * 1.1 * env(n, 0.0, 0.1) + rush * 0.8 + snap * 0.8


@register('rip', gain_db=-6.0, reverb=0.2)
def rip(c, rng):
    n = n_of(0.35)
    t = times(n)
    return sweep(white(n, rng), 600 + 2000 * t / 0.35, 'bp', q=3.0) * env(n, 0.002, 0.06) * 1.3


@register('slowmo', gain_db=-6.0, reverb=0.4)
def slowmo(c, rng):
    """Time slowing: a sub drop and a reversed swell."""
    n = n_of(1.2)
    t = times(n)
    sub = osc(130 * np.exp(-t * 2.6) + 30, n) * env(n, 0.005, 0.5) * 1.2
    swell = lp(white(n, rng), 1800) * np.exp(-((t - 0.05) / 0.08) ** 2) * 0.4
    return sub + swell


# -------------------------------------------------------------- bones
@register('bone_crunch', gain_db=-3.0, reverb=0.25)
def bone_crunch(c, rng):
    n = n_of(0.6)
    return (crackle(n, rng, rate=2500, decay=0.002, lo=700, hi=4500) * env(n, 0.003, 0.12) * 2.2
            + _thump(n, 90, 50, 20, 0.08, rng) * 0.6)


@register('bone_pile', gain_db=-2.0, reverb=0.3)
def bone_pile(c, rng):
    """A skeleton collapsing: dry bones clattering onto stone, settling."""
    n = n_of(1.8)
    out = np.zeros(n)
    k = 0
    tt = 0.0
    while tt < 1.4:
        m = n_of(0.07)
        f = rng.uniform(800, 2300)
        clack = partials([f, f * 2.3, f * 3.9], [0.03, 0.015, 0.008], m, rng=rng) + hp(white(m, rng), 3000) * env(m, 0, 0.002)
        i = n_of(tt)
        out[i:i + m] += clack[:max(0, min(m, n - i))] * rng.uniform(0.3, 1.0) * (1 - tt / 1.6)
        tt += rng.exponential(0.03 + tt * 0.05)
        k += 1
    out += _thump(n, 90, 45, 15, 0.12, rng) * 0.9 + lp(white(n, rng), 2000) * env(n, 0.05, 0.4) * 0.1
    return out


# ------------------------------------------------------------ the title
@register('slam', gain_db=-3.0, reverb=0.45)
def slam(c, rng):
    """A taiko-like drum hit under each name of the title."""
    big = bool(c.get('big'))
    n = n_of(1.4)
    body = osc(drop(95 if big else 110, 52, n, 10), n) * env(n, 0.002, 0.35 if big else 0.25) * 1.4
    skin = bp(white(n, rng), 200, 1600) * env(n, 0.001, 0.03) * 0.7
    return sat(body + skin, 1.3)


# ======================================================================== VERSUS-MK4 (version 4)
# Weapons, the specials and the breaking stage of the two-minute clip. Same rules: synthesised here, nothing sampled.
def _metal(n, rng, f0=1180.0, dec=0.35, bright=1.0):
    """An inharmonic metal ring (a blade): partials at stiff-bar ratios."""
    ratios = [1.0, 2.76, 5.40, 8.93, 13.34]
    freqs = [f0 * r for r in ratios]
    decs = [dec * k for k in (1.0, 0.7, 0.45, 0.3, 0.2)]
    amps = [1.0, 0.8 * bright, 0.6 * bright, 0.4 * bright, 0.25 * bright]
    return partials(freqs, decs, n, amps=amps, rng=rng, detune=0.01)


@register('swish', gain_db=-10.0, reverb=0.08)
def swish(c, rng):
    """A blade cutting the air: a thin, bright band sweeping up and away (air only, no ring)."""
    sp = float(c.get('speed', 1.0))
    dur = 0.22 / sp ** 0.3 * rng.uniform(0.85, 1.15)
    n = n_of(dur)
    t = times(n) / dur
    fc = rng.uniform(1500, 2100) + rng.uniform(3200, 4400) * np.sin(np.pi * np.clip(t, 0, 1)) ** 1.2
    return sweep(white(n, rng), fc, 'bp', q=rng.uniform(2.5, 3.5)) * np.sin(np.pi * np.clip(t, 0, 1)) ** 2


@register('cut', gain_db=-3.0, reverb=0.1)
def cut(c, rng):
    """A blade through cloth and flesh: a sharp slice, a wet tear, a body jolt (no steel ring)."""
    w = int(c.get('weight', 1))
    n = n_of(0.4)
    t = times(n)
    slice_ = sweep(white(n, rng), rng.uniform(5500, 7500) * np.exp(-t * 18) + 900, 'bp', q=2.5) * \
        env(n, 0.001, rng.uniform(0.035, 0.055)) * 1.4
    wet = sweep(white(n, rng), 1500 * np.exp(-t * 9) + 260, 'bp', q=4.0) * env(n, 0.004, 0.08) * 1.2
    thump = _thump(n, 150 - 15 * w, 60, 25, 0.04 + 0.03 * w, rng) * (0.9 + 0.3 * w)
    return sat(_click(n, rng, 3000, 0.0015) * 0.8 + slice_ + wet + thump, 1.4)


@register('clash', gain_db=-3.0, reverb=0.05)
def clash(c, rng):
    """Blade on blade, dry: a crack, a short gritty 'shing', a thump, at most a glint of ring; every hit a little
    different; on ice a crunchy crack instead of a ring. Big clashes add the blades sliding off each other."""
    big = bool(c.get('big'))
    n = n_of(0.42 if big else 0.3)
    x = _steel(n, rng, 1.5 if big else 1.0, rng.uniform(0.8, 1.3))
    if big:
        k = n_of(rng.uniform(0.012, 0.03))
        x[k:] += _shing(n - k, rng, rng.uniform(0.04, 0.065), rng.uniform(3800, 5600), rng.uniform(1700, 2400), 3.0) * 0.6
    if c.get('ice'):
        x += _ice_crunch(n, rng, 1.2 if big else 1.0)
    return sat(x, 1.5)


@register('sword_draw', gain_db=-5.0, reverb=0.12)
def sword_draw(c, rng):
    """Two swords out of their scabbards: two quick rising scrapes ('shing'), a click as each clears; no ring."""
    n = n_of(0.75)
    out = np.zeros(n)
    for t0 in (0.0, rng.uniform(0.06, 0.09)):
        ln = rng.uniform(0.3, 0.38)
        m = n_of(ln)
        tt = times(m)
        fc = rng.uniform(1800, 2400) + rng.uniform(4500, 6000) * tt / ln
        scrape = sweep(white(m, rng), fc, 'bp', q=rng.uniform(4.0, 6.0)) * np.clip(tt / 0.04, 0, 1) * \
            np.clip((ln - tt) / 0.03, 0, 1)
        grit = crackle(m, rng, rate=900, decay=0.0008, lo=3000, hi=9000) * np.clip(tt / ln, 0, 1) * 0.6
        i = n_of(t0)
        out[i:i + m] += (scrape + grit) * 0.9
        j = min(n - 1, i + m - n_of(0.01))
        k = n - j
        out[j:] += (_click(k, rng, 3500, 0.002) * 1.2 + _glint(k, rng, rng.uniform(2600, 3400), 0.02, 0.1))
    return out


@register('ice_form', gain_db=-6.0, reverb=0.15)
def ice_form(c, rng):
    """Ice growing into a weapon: crackling grains packing tight, a last crunch as it sets (no chord)."""
    big = bool(c.get('big'))
    n = n_of(0.9)
    t = times(n)
    grains = np.zeros(n)
    for k in range(70 if big else 45):
        t0 = 0.6 * (k / 70.0) ** 0.6
        m = n_of(0.005)
        i = n_of(t0)
        grains[i:i + m] += hp(white(m, rng), 4500) * env(m, 0.0, 0.0012) * rng.uniform(0.2, 0.9)
    hiss = hp(white(n, rng), 5000) * np.clip(t / 0.5, 0, 1) * np.exp(-np.clip(t - 0.5, 0, None) / 0.05) * 0.3
    k = n_of(0.55)
    grains[k:] += _ice_crunch(n - k, rng, 1.0)
    low = _thump(n, 90, 60, 6, 0.2, rng) * (0.5 if big else 0.2)
    return grains * 1.6 + hiss + low


@register('blade_lock', gain_db=-8.0, reverb=0.2)
def blade_lock(c, rng):
    """Blades locked and straining: metal grinding on metal."""
    dur = max(0.2, float(c.get('dur', 0.6)))
    n = n_of(dur)
    t = times(n)
    grind = bp(white(n, rng), 1500, 5000) * (0.6 + 0.4 * np.sin(2 * np.pi * 37 * t)) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))
    buzz = osc(240 + 20 * np.sin(2 * np.pi * 5 * t), n, 'saw') * 0.15
    return (grind + buzz) * np.clip(t / 0.05, 0, 1) * np.clip((dur - t) / 0.08, 0, 1)


@register('disarm', gain_db=-4.0, reverb=0.1)
def disarm(c, rng):
    """The sword torn out of a hand: a twisted dry clash, then it whirs away spinning."""
    n = n_of(0.8)
    t = times(n)
    hit = _steel(n, rng, 1.3, 1.2)
    whir = bp(white(n, rng), 1200, 4000) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(11, 16) * t)) * env(n, 0.05, 0.3)
    return sat(hit + whir * 0.5, 1.4)


@register('blade_stick', gain_db=-7.0, reverb=0.1)
def blade_stick(c, rng):
    """A blade biting into stone: a chock, grit, a quick dead quiver."""
    n = n_of(0.35)
    t = times(n)
    chock = _thump(n, 300, 160, 40, 0.02, rng) + hp(white(n, rng), 2000) * env(n, 0.0, 0.006)
    grit = crackle(n, rng, rate=2000, decay=0.001, lo=1500, hi=7000) * env(n, 0.001, 0.02) * 1.5
    quiver = osc(rng.uniform(700, 900) + 25 * np.sin(2 * np.pi * 22 * t), n) * np.exp(-t / 0.035) * 0.25
    return chock + grit + quiver


@register('catch', gain_db=-9.0, reverb=0.08)
def catch(c, rng):
    """A hilt slapped into a palm: a slap and a click."""
    n = n_of(0.25)
    slap = bp(white(n, rng), 700, 3500) * env(n, 0.0005, 0.02)
    return slap + _click(n, rng, 3200, 0.002) * 0.8 + _glint(n, rng, rng.uniform(2200, 3000), 0.015, 0.08)


@register('stone_crack', gain_db=-3.0, reverb=0.35)
def stone_crack(c, rng):
    """Stone giving way: a deep crack, grinding, a groan."""
    n = n_of(1.2)
    t = times(n)
    crack = hp(white(n, rng), 1200) * env(n, 0.0, 0.03) * 1.4
    grind = crackle(n, rng, rate=400, decay=0.004, lo=400, hi=3000) * env(n, 0.02, 0.6) * 1.6
    groan = osc(55 + 15 * np.sin(2 * np.pi * 0.9 * t), n, 'saw') * env(n, 0.1, 0.7) * 0.25
    return sat(crack + grind + groan + _thump(n, 90, 45, 10, 0.2, rng) * 0.8, 1.3)


@register('pillar_fall', gain_db=1.0, reverb=0.4)
def pillar_fall(c, rng):
    """A stone pillar crashing down: a huge low boom, the shock of stone, rubble rolling and settling."""
    n = n_of(2.6)
    boom = _thump(n, 60, 26, 5, 0.6, rng) * 1.6
    crash_ = stone_crash({'power': 2.0}, rng)
    rubble = crackle(n, rng, rate=120, decay=0.006, lo=400, hi=4000) * env(n, 0.05, 1.2) * 1.4
    x = boom + rubble
    x[:len(crash_)] += crash_
    return sat(x, 1.3)


@register('hammer_floor', gain_db=-1.0, reverb=0.15)
def hammer_floor(c, rng):
    """The ice hammer hitting the ground: a heavy blow, stone (or planks) breaking, the ice head crunching."""
    n = n_of(1.2)
    boom = _thump(n, rng.uniform(70, 90), 32, 8, 0.3, rng) * 1.5
    hit = hp(white(n, rng), 1800) * env(n, 0.0, 0.01) * 1.2
    if c.get('wood'):
        brk = crackle(n, rng, rate=900, decay=0.004, lo=500, hi=4000) * env(n, 0.002, 0.22) * 2.2
    else:
        brk = crackle(n, rng, rate=1200, decay=0.002, lo=800, hi=5000) * env(n, 0.002, 0.1) * 2.0
    return sat(boom + hit + brk + _ice_crunch(n, rng, 1.5), 1.3)


@register('hellfire', gain_db=-3.0, reverb=0.35)
def hellfire(c, rng):
    """A column of hellfire bursting out of the stone: a rising roar with a crack at its root."""
    n = n_of(1.2)
    t = times(n)
    fc = 300 + 2400 * np.clip(t / 0.15, 0, 1) * np.exp(-np.clip(t - 0.15, 0, None) * 2.5) + 300
    roar = sweep(pink(n, rng), fc, 'lp') * env(n, 0.02, 0.45) * 1.8
    crack = hp(white(n, rng), 2000) * env(n, 0.0, 0.01) * 0.8
    low = _thump(n, 65, 35, 6, 0.3, rng) * 0.9
    return sat(roar + crack + low + crackle(n, rng, rate=140, decay=0.003) * env(n, 0.02, 0.5) * 1.2, 1.25)


@register('ice_wall', gain_db=-3.0, reverb=0.35)
def ice_wall(c, rng):
    """A wall of ice heaving up out of the floor: grinding crystal over a low rumble, a ring at the top."""
    n = n_of(1.0)
    t = times(n)
    rumble = lp(brown(n, rng), 160) * env(n, 0.05, 0.35) * 1.2
    grind = crackle(n, rng, rate=900, decay=0.0015, lo=2000, hi=9000) * env(n, 0.02, 0.3) * 1.6
    ring = _chimes(n, rng, 10, 1800, 7000, 0.1, 0.5, spread=0.3)
    return rumble + grind + ring * np.clip((t - 0.15) / 0.1, 0, 1)


@register('steam', gain_db=-9.0, reverb=0.25)
def steam(c, rng):
    dur = max(0.3, float(c.get('dur', 0.8)))
    n = n_of(dur + 0.3)
    t = times(n)
    return hp(white(n, rng), 3500) * np.clip(t / 0.05, 0, 1) * np.clip((dur + 0.25 - t) / 0.4, 0, 1)


@register('fire_dash', gain_db=-3.0, reverb=0.25)
def fire_dash(c, rng):
    """Scorpion as a streak of fire: a roaring rush."""
    dur = max(0.2, float(c.get('dur', 0.4)))
    n = n_of(dur + 0.4)
    t = times(n)
    fc = 600 + 2200 * np.sin(np.pi * np.clip(t / (dur + 0.2), 0, 1))
    rush = sweep(pink(n, rng), fc, 'lp') * np.sin(np.pi * np.clip(t / (dur + 0.4), 0, 1)) ** 0.8 * 1.6
    crack = crackle(n, rng, rate=110, decay=0.003) * env(n, 0.03, dur) * 1.2
    onset = fire_burst({'size': 1}, rng)
    x = rush + crack
    m = min(len(x), len(onset))
    x[:m] += onset[:m] * 0.6
    return sat(x, 1.2)


@register('ice_burst', gain_db=-5.0, reverb=0.4)
def ice_burst(c, rng):
    n = n_of(1.2)
    pop = hp(white(n, rng), 1500) * env(n, 0.0, 0.06) * 1.2
    return pop + _chimes(n, rng, 24, 2500, 9500, 0.08, 0.5, spread=0.3) * 1.4 + _thump(n, 120, 70, 15, 0.08, rng) * 0.5


@register('icicle', gain_db=-13.0, reverb=0.25)
def icicle(c, rng):
    n = n_of(0.4)
    return hp(white(n, rng), 2500) * env(n, 0.0, 0.015) + _chimes(n, rng, 5, 3000, 9000, 0.03, 0.15) * 1.2


@register('roof_tiles', gain_db=-4.0, reverb=0.3)
def roof_tiles(c, rng):
    """Roof tiles bursting and sliding off: a crunch, ceramic clatter, pieces skittering down the slope."""
    p = float(c.get('power', 1.0))
    n = n_of(1.4)
    out = np.zeros(n)
    tt = 0.0
    while tt < 0.5 + 0.5 * p:
        m = n_of(0.05)
        f = rng.uniform(1100, 3200)
        clack = partials([f, f * 1.7, f * 2.9], [0.02, 0.012, 0.007], m, rng=rng) + hp(white(m, rng), 3500) * env(m, 0, 0.002)
        i = n_of(tt)
        out[i:i + m] += clack[:max(0, min(m, n - i))] * rng.uniform(0.3, 1.0) * (1 - tt / 1.2)
        tt += rng.exponential(0.02 + tt * 0.04)
    crunch = crackle(n, rng, rate=1500, decay=0.0015, lo=900, hi=6000) * env(n, 0.001, 0.06) * 2.0
    return out * 1.3 + crunch * p + _thump(n, 110, 60, 20, 0.08, rng) * 0.6 * p


@register('clone', gain_db=-4.0, reverb=0.4)
def clone(c, rng):
    """The ice clone left behind: a flash of frost and a hollow crystal shimmer."""
    n = n_of(1.0)
    t = times(n)
    fw = sweep(white(n, rng), 6000 * np.exp(-t * 6) + 1500, 'bp', q=2.0) * env(n, 0.002, 0.1)
    shim = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (1760, 2217, 2637, 3520)) / 4
    return fw + shim * np.exp(-t / 0.4) * 0.7 + _chimes(n, rng, 8, 2500, 7000, 0.1, 0.4)


@register('teleport', gain_db=-3.0, reverb=0.4)
def teleport(c, rng):
    """Scorpion folding into his fire: a whoomph sucked inwards."""
    burst = fire_burst({'size': 1}, rng)
    x = burst[::-1].copy() * np.linspace(0.4, 1.0, len(burst))
    n = len(x)
    return x + _thump(n, 60, 110, 5, 0.25, rng)[::-1].copy() * 0.6


@register('teleport_in', gain_db=-2.0, reverb=0.4)
def teleport_in(c, rng):
    burst = fire_burst({'size': 2}, rng)
    n = len(burst)
    t = times(n)
    rush = sweep(white(n, rng), 3000 * np.exp(-t * 6) + 400, 'bp', q=1.2) * env(n, 0.001, 0.15)
    return burst + rush * 0.6


@register('ice_slide', gain_db=-6.0, reverb=0.2)
def ice_slide(c, rng):
    """Sliding feet first over the ice he makes: a long scraping hiss with crystal grit."""
    dur = max(0.2, float(c.get('dur', 0.6)))
    n = n_of(dur + 0.2)
    t = times(n)
    scrape = bp(white(n, rng), 2500, 9000) * (0.7 + 0.3 * np.sin(2 * np.pi * 23 * t))
    grit = crackle(n, rng, rate=300, decay=0.0015, lo=3000, hi=9000) * 0.8
    shape = np.clip(t / 0.03, 0, 1) * np.clip((dur + 0.15 - t) / 0.2, 0, 1)
    return (scrape + grit) * shape


@register('ground_wave', gain_db=-3.0, reverb=0.35)
def ground_wave(c, rng):
    """The freezing wave racing over the stone: crystal cracking ever faster over a rumble."""
    dur = max(0.3, float(c.get('dur', 0.5)))
    n = n_of(dur + 0.8)
    t = times(n)
    grains = np.zeros(n)
    k = 0.0
    while k < dur:
        m = n_of(0.006)
        i = n_of(k)
        grains[i:i + m] += hp(white(m, rng), 3000) * env(m, 0.0, 0.002) * rng.uniform(0.4, 1.0)
        k += rng.exponential(0.006)
    rumble = lp(brown(n, rng), 150) * np.clip(t / 0.05, 0, 1) * np.clip((dur + 0.3 - t) / 0.3, 0, 1)
    ring = _chimes(n, rng, 12, 2000, 8000, 0.1, 0.5, spread=dur)
    return grains * 1.8 + rumble + ring


@register('bridge_snap', gain_db=0.0, reverb=0.4)
def bridge_snap(c, rng):
    """The rope bridge giving way: a sharp crack of wood, ropes whipping, planks clattering into the drop."""
    n = n_of(2.4)
    t = times(n)
    crack = hp(white(n, rng), 900) * env(n, 0.0, 0.04) * 1.6 + _thump(n, 140, 60, 20, 0.08, rng)
    splinter = crackle(n, rng, rate=800, decay=0.004, lo=400, hi=4000) * env(n, 0.005, 0.35) * 2.0
    whip = np.zeros(n)
    for t0 in (0.05, 0.12):
        s = karplus(rng.uniform(70, 95), 0.6, rng, damp=0.99)
        i = n_of(t0)
        whip[i:i + len(s)] += s[:max(0, min(len(s), n - i))]
    planks = np.zeros(n)
    tt = 0.3
    while tt < 2.0:
        m = n_of(0.06)
        f = rng.uniform(250, 700)
        knock = partials([f, f * 2.4], [0.04, 0.02], m, rng=rng)
        i = n_of(tt)
        planks[i:i + m] += knock[:max(0, min(m, n - i))] * rng.uniform(0.2, 0.7) * (1.2 - tt / 2.0)
        tt += rng.exponential(0.09)
    return sat(crack + splinter + whip * 0.8 + planks * 0.8, 1.3)


@register('gong', gain_db=-1.0, reverb=0.5)
def gong(c, rng):
    """The shrine's gong struck by a body: a low bronze boom that swells and shimmers away."""
    n = n_of(4.0)
    t = times(n)
    freqs = [92, 147, 211, 268, 344, 431, 517, 648, 811]
    decs = [3.2, 2.6, 2.2, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6]
    body = partials(freqs, decs, n, rng=rng, detune=0.015)
    swell = 0.6 + 0.4 * np.clip(t / 0.35, 0, 1)
    shimmer = partials([1260, 1730, 2310], [1.2, 0.9, 0.7], n, rng=rng) * 0.25 * (0.6 + 0.4 * np.sin(2 * np.pi * 4.5 * t))
    hit = _thump(n, 120, 55, 15, 0.12, rng) * 0.8 + bp(white(n, rng), 300, 2000) * env(n, 0.0, 0.02) * 0.6
    return body * swell + shimmer + hit
