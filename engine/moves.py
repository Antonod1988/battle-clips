"""The move library: every move is a function of its progress u (0..1) returning a pose.

Poses are written facing right (the foe is at +x). Strikes snap out fast and come back slower; the hit lands at the
move's HIT fraction (used by the choreography to time sparks and reactions).
"""
import math
from rig import pose, lerp_pose

GUARD = dict(hip=(0.0, -30.6), lean=0.10, head=-0.04, fh=(8.6, -3.6), nh=(6.6, -1.4),
             ff=(8.6, -2.0), nf=(-9.0, -2.0), ftoe=0.0, ntoe=0.0)


def P(**kw):
    d = dict(GUARD)
    d.update(kw)
    return pose(**d)


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def out(u):
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 3


def inn(u):
    u = min(1.0, max(0.0, u))
    return u ** 2.2


def keys(u, ks):
    """ks: [(u, pose, easing), ...]; easing applies on the way into that key."""
    if u <= ks[0][0]:
        return ks[0][1]
    for (u0, p0, _), (u1, p1, e) in zip(ks, ks[1:]):
        if u <= u1:
            w = (u - u0) / max(1e-6, u1 - u0)
            return lerp_pose(p0, p1, (e or ease)(w))
    return ks[-1][1]


G = P()


def guard(u, t=0.0, **_):
    b = math.sin(t * 2 * math.pi * 1.6)
    return P(hip=(0.0, -30.6 + 0.6 * b), fh=(8.6, -3.6 + 0.5 * b), nh=(6.6, -1.4 + 0.4 * b))


def walk(u, t=0.0, speed=40.0, back=False, **_):
    """A fighter's shuffle: the feet stay planted while the body moves (u loops)."""
    period = 0.56
    ph = (t / period) % 1.0
    stride = speed * period * 0.5
    def foot(ph, base):
        # stance half: slides back under the body; swing half: lifts and moves forward
        if ph < 0.5:
            k = ph / 0.5
            return (base + stride * (0.5 - k) * (-1 if back else 1), -2.0)
        k = (ph - 0.5) / 0.5
        return (base + stride * (k - 0.5) * (-1 if back else 1), -2.0 - 3.2 * math.sin(k * math.pi))
    ff = foot(ph, 8.6)
    nf = foot((ph + 0.5) % 1.0, -9.0)
    bob = math.sin(ph * 4 * math.pi)
    return P(hip=(0.0, -30.6 + 0.7 * bob), ff=ff, nf=nf, fh=(8.6, -3.6 + 0.4 * bob), nh=(6.6, -1.2 + 0.4 * bob))


def run(u, t=0.0, speed=90.0, **_):
    period = 0.36
    ph = (t / period) % 1.0
    stride = speed * period * 0.5
    def foot(ph, base):
        if ph < 0.45:
            k = ph / 0.45
            return (base + stride * (0.5 - k), -2.0)
        k = (ph - 0.45) / 0.55
        return (base + stride * (k - 0.5), -2.0 - 7.0 * math.sin(k * math.pi))
    ff = foot(ph, 5.0)
    nf = foot((ph + 0.5) % 1.0, -2.0)
    bob = abs(math.sin(ph * 2 * math.pi))
    sw = math.sin(ph * 2 * math.pi)
    return P(hip=(1.0, -31.0 + 1.2 * bob), lean=0.42, head=-0.3, ff=ff, nf=nf,
             fh=(4.0 + 6 * sw, 2.0), nh=(2.0 - 6 * sw, 3.0))


# ---------------------------------------------------------------- strikes
JAB_HIT = 0.40


def jab(u, **_):
    ext = P(hip=(2.5, -30.4), lean=0.22, fh=(19.0, -6.0), nh=(5.6, -2.4), ff=(10.0, -2.0))
    return keys(u, [(0.0, G, None), (0.18, P(fh=(6.0, -2.4)), out), (0.40, ext, out), (0.55, ext, None), (1.0, G, ease)])


CROSS_HIT = 0.42


def cross(u, **_):
    ext = P(hip=(4.0, -30.0), lean=0.30, twist=1.0, nh=(22.0, -5.4), fh=(4.0, 0.0), ff=(11.0, -2.0), nf=(-8.0, -2.0), ntoe=0.4)
    wind = P(hip=(-1.0, -30.6), lean=0.04, nh=(2.0, -1.0))
    return keys(u, [(0.0, G, None), (0.20, wind, out), (0.42, ext, out), (0.58, ext, None), (1.0, G, ease)])


HOOK_HIT = 0.45


def hook(u, **_):
    wind = P(hip=(-1.0, -30.4), lean=0.0, nh=(-2.0, -4.0), twist=0.0)
    arc = P(hip=(3.0, -30.0), lean=0.28, twist=1.0, nh=(17.0, -8.0), nbend=-1.0, fh=(4.0, 1.0), ntoe=0.3)
    return keys(u, [(0.0, G, None), (0.22, wind, out), (0.45, arc, out), (0.6, arc, None), (1.0, G, ease)])


UPPER_HIT = 0.46


def uppercut(u, **_):
    dip = P(hip=(1.0, -24.5), lean=0.42, head=0.1, nh=(6.0, 12.0), fh=(7.0, -2.0), ff=(10.0, -2.0), nf=(-9.0, -2.0))
    rise = P(hip=(4.0, -35.0), lean=-0.22, head=-0.25, twist=1.0, nh=(9.0, -27.0), fh=(4.0, 3.0), ff=(9.0, -3.0),
             nf=(-7.0, -4.0), ntoe=0.6, ftoe=0.25)
    return keys(u, [(0.0, G, None), (0.26, dip, out), (0.46, rise, out), (0.66, rise, None), (1.0, G, ease)])


KICK_HIT = 0.45


def roundhouse(u, **_):
    chamber = P(hip=(-1.0, -31.5), lean=-0.15, nf=(8.0, -24.0), ntoe=0.9, nkb=-1.0, fh=(6.0, -6.0), nh=(2.0, -2.0))
    ext = P(hip=(-3.0, -32.5), lean=-0.42, head=0.05, nf=(30.0, -49.0), ntoe=-0.15, nkb=-1.0, ff=(4.0, -2.0),
            fh=(7.0, -6.0), nh=(-6.0, 4.0), ftoe=0.0)
    return keys(u, [(0.0, G, None), (0.22, chamber, out), (0.45, ext, out), (0.62, ext, None), (0.82, chamber, ease),
                    (1.0, G, ease)])


def front_kick(u, **_):
    chamber = P(hip=(-1.0, -31.0), lean=-0.05, nf=(6.0, -22.0), nkb=-1.0, ntoe=0.8)
    ext = P(hip=(-2.0, -31.5), lean=-0.30, nf=(27.0, -30.0), nkb=-1.0, ntoe=-0.2, ff=(4.0, -2.0), fh=(7.0, -5.0), nh=(-4.0, 2.0))
    return keys(u, [(0.0, G, None), (0.24, chamber, out), (0.45, ext, out), (0.6, ext, None), (0.85, chamber, ease), (1.0, G, ease)])


KNEE_HIT = 0.42


def knee(u, **_):
    pull = P(hip=(3.0, -32.5), lean=0.18, nf=(5.0, -18.0), nkb=-1.0, ntoe=0.9, fh=(12.0, -10.0), nh=(11.0, -9.0),
             fopen=1.0, nopen=1.0, ff=(6.0, -2.0))
    return keys(u, [(0.0, G, None), (0.18, P(fh=(13.0, -8.0), nh=(12.0, -7.0), fopen=1.0, nopen=1.0), out),
                    (0.42, pull, out), (0.6, pull, None), (1.0, G, ease)])


SWEEP_HIT = 0.5


def sweep(u, **_):
    low = P(hip=(-1.0, -18.0), lean=0.55, head=-0.2, ff=(8.0, -2.0), nf=(-12.0, -2.0), fh=(10.0, 8.0), nh=(-4.0, 10.0),
            nkb=-1.0)
    swing = P(hip=(0.0, -16.0), lean=0.65, head=-0.25, ff=(6.0, -2.0), nf=(26.0, -3.0), nkb=1.0, ntoe=-0.4,
              fh=(-4.0, 12.0), nh=(-8.0, 12.0))
    return keys(u, [(0.0, G, None), (0.25, low, out), (0.5, swing, out), (0.68, swing, None), (1.0, G, ease)])


def jump_kick(u, **_):
    """Root motion carries him; the pose: a tuck, then the flying side kick, then the landing."""
    tuck = P(hip=(0.0, -32.0), lean=-0.1, ff=(8.0, -16.0), nf=(-4.0, -14.0), fkb=-1.0, nkb=-1.0, fh=(7.0, -6.0),
             nh=(4.0, -4.0))
    kick = P(hip=(0.0, -32.0), lean=-0.55, head=0.15, ff=(29.0, -22.0), ftoe=-0.1, fkb=1.0, nf=(-2.0, -12.0), nkb=-1.0,
             fh=(4.0, -8.0), nh=(-5.0, -2.0))
    land = P(hip=(1.0, -26.0), lean=0.3, ff=(9.0, -2.0), nf=(-9.0, -2.0))
    return keys(u, [(0.0, P(hip=(0.0, -27.0), lean=0.2), None), (0.18, tuck, out), (0.42, kick, out), (0.7, kick, None),
                    (0.86, tuck, ease), (1.0, land, ease)])


JUMP_HIT = 0.55


def block(u, **_):
    b = P(hip=(-1.5, -30.0), lean=-0.05, head=0.1, fh=(7.0, -12.0), nh=(6.0, -9.0), fbend=1.0)
    return keys(u, [(0.0, G, None), (0.2, b, out), (0.8, b, None), (1.0, G, ease)])


def block_hold(u, **_):
    return P(hip=(-1.5, -30.0), lean=-0.05, head=0.1, fh=(7.0, -12.0), nh=(6.0, -9.0))


# --------------------------------------------------------------- reactions
def hit_head(u, **_):
    snap = P(hip=(-3.0, -31.0), lean=-0.38, head=-0.55, fh=(3.0, 4.0), nh=(-2.0, 5.0), ff=(6.0, -2.0), nf=(-10.0, -2.0))
    return keys(u, [(0.0, G, None), (0.15, snap, out), (0.45, snap, None), (1.0, G, ease)])


def hit_body(u, **_):
    fold = P(hip=(-2.0, -28.5), lean=0.75, head=0.35, fh=(4.0, 9.0), nh=(5.0, 8.0), fopen=1.0, nopen=1.0,
             ff=(7.0, -2.0), nf=(-9.0, -2.0))
    return keys(u, [(0.0, G, None), (0.15, fold, out), (0.55, fold, None), (1.0, G, ease)])


def stagger(u, t=0.0, **_):
    """Dazed, swaying (FINISH HIM)."""
    s = math.sin(t * 2.6)
    s2 = math.sin(t * 1.3 + 1.0)
    return P(hip=(1.5 * s, -29.5 + 0.8 * abs(s2)), lean=0.35 + 0.12 * s, head=0.35 + 0.2 * s2,
             fh=(3.0, 12.0 + s), nh=(1.0, 13.0 - s), ff=(7.0, -2.0), nf=(-7.0, -2.0), fopen=1.0, nopen=1.0)


def launched(u, **_):
    """In the air, flying backwards, limbs trailing."""
    a = P(hip=(0.0, -30.0), lean=-0.9, head=-0.5, fh=(-2.0, -12.0), nh=(-6.0, -10.0), ff=(16.0, -26.0),
          nf=(12.0, -20.0), fkb=-1.0, nkb=-1.0, fopen=1.0, nopen=1.0)
    b = P(hip=(0.0, -30.0), lean=-1.3, head=-0.3, fh=(-8.0, -6.0), nh=(-10.0, -2.0), ff=(22.0, -36.0),
          nf=(20.0, -30.0), fkb=-1.0, nkb=-1.0, fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, a, None), (1.0, b, ease)])


LIE = P(hip=(0.0, -6.0), lean=-1.52, head=0.1, fh=(-4.0, -4.0), nh=(-2.0, 3.0), ff=(16.0, -2.0), nf=(14.0, -3.0),
        fkb=-1.0, nkb=-1.0, fopen=1.0, nopen=1.0, ftoe=-0.9, ntoe=-0.9)
LIE_FRONT = P(hip=(0.0, -6.0), lean=1.5, head=-0.4, fh=(4.0, 6.0), nh=(2.0, 5.0), ff=(-15.0, -2.0), nf=(-14.0, -3.0),
              fkb=1.0, nkb=1.0, fopen=1.0, nopen=1.0, ftoe=0.9, ntoe=0.9)


def fall(u, **_):
    """From the flight to lying on his back (root carries the arc)."""
    return keys(u, [(0.0, launched(1.0), None), (0.7, P(hip=(0.0, -10.0), lean=-1.45, head=-0.2, fh=(-6.0, -6.0),
                                                         nh=(-4.0, -2.0), ff=(18.0, -14.0), nf=(15.0, -10.0),
                                                         fopen=1.0, nopen=1.0), ease), (1.0, LIE, ease)])


def lie(u, **_):
    return LIE


def get_up(u, **_):
    sit = P(hip=(-2.0, -10.0), lean=-0.6, head=0.2, fh=(-8.0, 10.0), nh=(-9.0, 9.0), ff=(10.0, -2.0), nf=(4.0, -2.0),
            fopen=1.0, nopen=1.0)
    crouch = P(hip=(0.0, -21.0), lean=0.55, head=0.0, fh=(8.0, 6.0), nh=(4.0, 8.0), ff=(8.0, -2.0), nf=(-6.0, -2.0))
    return keys(u, [(0.0, LIE, None), (0.35, sit, ease), (0.7, crouch, ease), (1.0, G, ease)])


def get_up_front(u, **_):
    push = P(hip=(-1.0, -14.0), lean=1.1, head=-0.5, fh=(8.0, 14.0), nh=(6.0, 14.0), ff=(-8.0, -2.0), nf=(-12.0, -2.0),
             fopen=1.0, nopen=1.0)
    crouch = P(hip=(0.0, -21.0), lean=0.55, head=0.0, fh=(8.0, 6.0), nh=(4.0, 8.0), ff=(8.0, -2.0), nf=(-6.0, -2.0))
    return keys(u, [(0.0, LIE_FRONT, None), (0.4, push, ease), (0.72, crouch, ease), (1.0, G, ease)])


def slam_wall(u, **_):
    """Back against a pillar: splayed, then sliding down to sit."""
    flat = P(hip=(-1.0, -32.0), lean=-0.25, head=-0.45, fh=(-6.0, -10.0), nh=(-8.0, -8.0), ff=(6.0, -6.0), nf=(3.0, -4.0),
             fopen=1.0, nopen=1.0)
    sit = P(hip=(-2.0, -12.0), lean=-0.2, head=0.35, fh=(-2.0, 12.0), nh=(-4.0, 12.0), ff=(12.0, -2.0), nf=(9.0, -2.0),
            fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, flat, None), (0.35, flat, None), (1.0, sit, inn)])


def sit_up(u, **_):
    sit = P(hip=(-2.0, -12.0), lean=-0.2, head=0.35, fh=(-2.0, 12.0), nh=(-4.0, 12.0), ff=(12.0, -2.0), nf=(9.0, -2.0),
            fopen=1.0, nopen=1.0)
    crouch = P(hip=(0.0, -22.0), lean=0.5, fh=(8.0, 5.0), nh=(5.0, 6.0), ff=(8.0, -2.0), nf=(-6.0, -2.0))
    return keys(u, [(0.0, sit, None), (0.5, crouch, ease), (1.0, G, ease)])


# ------------------------------------------------------------- specials
def ice_cast(u, **_):
    """Sub-Zero: hands drawn back together at the hip, then both palms thrust out (release at 0.55)."""
    gather = P(hip=(-2.0, -29.5), lean=-0.05, fh=(-3.0, 8.0), nh=(-5.0, 9.0), fopen=1.0, nopen=1.0, ff=(9.0, -2.0))
    push = P(hip=(3.0, -30.5), lean=0.28, fh=(17.0, -5.0), nh=(18.0, -3.0), fopen=1.0, nopen=1.0, ff=(11.0, -2.0),
             nf=(-9.0, -2.0), twist=0.8)
    return keys(u, [(0.0, G, None), (0.3, gather, out), (0.48, gather, None), (0.58, push, out), (0.8, push, None),
                    (1.0, G, ease)])


ICE_RELEASE = 0.56


def spear_throw(u, **_):
    """Scorpion: the rope hand winds back over the shoulder, then whips forward and stays out (release at 0.42)."""
    wind = P(hip=(-2.5, -31.0), lean=-0.2, head=-0.05, nh=(-9.0, -12.0), nbend=-1.0, fh=(10.0, -4.0), twist=0.0)
    throw = P(hip=(4.0, -30.5), lean=0.35, nh=(21.0, -8.0), nopen=1.0, fh=(2.0, 0.0), twist=1.0, ff=(11.0, -2.0),
              nf=(-10.0, -2.0), ntoe=0.4)
    return keys(u, [(0.0, G, None), (0.3, wind, out), (0.42, throw, out), (1.0, throw, None)])


SPEAR_RELEASE = 0.40


def spear_hold(u, **_):
    return P(hip=(4.0, -30.5), lean=0.35, nh=(21.0, -8.0), nopen=0.0, fh=(2.0, 0.0), twist=1.0, ff=(11.0, -2.0),
             nf=(-10.0, -2.0))


def yank(u, **_):
    """Both hands haul the rope in, the body thrown back."""
    a = spear_hold(0)
    b = P(hip=(-5.0, -29.0), lean=-0.45, head=0.0, nh=(-4.0, 1.0), fh=(2.0, -2.0), ff=(12.0, -2.0), nf=(-12.0, -2.0),
          twist=0.3)
    return keys(u, [(0.0, a, None), (0.35, b, out), (1.0, b, None)])


def pulled(u, **_):
    """Yanked off his feet by the shoulder, stumbling forward."""
    a = P(hip=(3.0, -30.0), lean=0.55, head=0.2, fh=(14.0, -6.0), nh=(-6.0, 4.0), fopen=1.0, nopen=1.0, ff=(16.0, -9.0),
          nf=(-10.0, -6.0), fkb=-1.0)
    return a


def breakout(u, **_):
    """Scorpion bursts out of the ice: arms thrown wide, head back, a roar."""
    tense = P(hip=(0.0, -28.0), lean=0.35, head=0.3, fh=(5.0, 4.0), nh=(3.0, 5.0))
    burst = P(hip=(0.0, -33.0), lean=-0.2, head=-0.4, fh=(13.0, -15.0), nh=(-12.0, -14.0), fopen=1.0, nopen=1.0,
              ff=(11.0, -2.0), nf=(-11.0, -2.0), twist=1.0)
    return keys(u, [(0.0, tense, None), (0.45, tense, None), (0.62, burst, out), (0.85, burst, None), (1.0, G, ease)])


def unmask(u, **_):
    """The near hand rises to the face and tears the mask away (at 0.55)."""
    reach = P(hip=(0.0, -32.0), lean=0.0, head=0.0, nh=(7.0, -10.0), nopen=1.0, fh=(5.0, 4.0))
    pull = P(hip=(-1.0, -32.5), lean=-0.15, head=-0.25, nh=(-10.0, -12.0), nopen=0.0, fh=(6.0, 3.0))
    return keys(u, [(0.0, G, None), (0.4, reach, ease), (0.55, reach, None), (0.7, pull, out), (1.0, pull, None)])


UNMASK_AT = 0.6


def breathe(u, **_):
    """Fire breath: leaning in, arms back, head thrust forward."""
    a = P(hip=(-1.0, -32.0), lean=-0.25, head=-0.35, nh=(-9.0, -10.0), fh=(-4.0, -6.0), fopen=1.0, nopen=1.0,
          ff=(10.0, -2.0), nf=(-12.0, -2.0))
    b = P(hip=(3.0, -30.5), lean=0.32, head=0.05, nh=(-11.0, 2.0), fh=(-8.0, 3.0), fopen=1.0, nopen=1.0,
          ff=(12.0, -2.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, a, None), (0.25, b, out), (1.0, b, None)])


def burning(u, t=0.0, **_):
    """Writhing in the flames."""
    s = math.sin(t * 11.0)
    c = math.sin(t * 7.0 + 1.0)
    return P(hip=(-1.0 + s, -32.0), lean=-0.2 + 0.15 * c, head=-0.4 + 0.25 * s, fh=(4.0 + 3 * s, -16.0 + 3 * c),
             nh=(-3.0 - 3 * c, -14.0 + 3 * s), fopen=1.0, nopen=1.0, ff=(7.0, -2.0), nf=(-7.0, -2.0))


def collapse(u, **_):
    """The burnt body's knees give, it folds to the floor."""
    kneel = P(hip=(-1.0, -17.0), lean=0.25, head=0.5, fh=(2.0, 12.0), nh=(-1.0, 12.0), ff=(9.0, -2.0), nf=(-8.0, -2.0),
              fkb=-1.0, nkb=-1.0, fopen=1.0, nopen=1.0)
    down = P(hip=(0.0, -6.0), lean=1.2, head=0.4, fh=(8.0, 6.0), nh=(6.0, 7.0), ff=(-10.0, -2.0), nf=(-9.0, -2.0),
             fkb=1.0, nkb=1.0, fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, burning(0, 0.0), None), (0.45, kneel, inn), (1.0, down, inn)])


def victory(u, t=0.0, **_):
    up = P(hip=(0.0, -32.5), lean=-0.1, head=-0.2, nh=(14.0, -13.0), nbend=1.0, fh=(-2.0, 9.0), fopen=0.0,
           ff=(10.0, -2.0), nf=(-10.0, -2.0), twist=0.8)
    b = math.sin(t * 3.0) * 0.3
    up['hip'] = (0.0, -32.5 + b)
    return keys(u, [(0.0, G, None), (0.4, up, out), (1.0, up, None)])


def taunt(u, t=0.0, **_):
    """Before the fight: a slow ready, fists clenched."""
    a = P(hip=(0.0, -32.0), lean=0.0, fh=(5.0, 8.0), nh=(3.0, 9.0))
    return keys(u, [(0.0, a, None), (0.6, a, None), (1.0, G, ease)])


def hold(u, pose=None, **_):
    """A pose held still (frozen in the ice)."""
    return pose


def speared(u, **_):
    """The kunai bites into his shoulder: he is wrenched round, the arm goes limp."""
    hit = P(hip=(-3.0, -31.0), lean=-0.3, head=-0.45, fh=(3.0, 10.0), nh=(-7.0, -7.0), fopen=1.0, nopen=1.0, ff=(7.0, -2.0),
            nf=(-11.0, -2.0), twist=0.0)
    hang = P(hip=(-1.0, -30.0), lean=0.15, head=0.25, fh=(4.0, 11.0), nh=(-2.0, 2.0), fopen=1.0, nopen=1.0, ff=(7.0, -2.0),
             nf=(-10.0, -2.0))
    return keys(u, [(0.0, G, None), (0.12, hit, out), (0.5, hit, None), (1.0, hang, ease)])


def dazed(u, t=0.0, **_):
    """Arriving at the end of the rope: off balance, arms loose."""
    s = math.sin(t * 9.0)
    return P(hip=(1.0, -29.0), lean=0.45, head=0.3 + 0.1 * s, fh=(7.0, 6.0), nh=(2.0, 9.0), fopen=1.0, nopen=1.0,
             ff=(6.0, -2.0), nf=(-8.0, -2.0))


def limp(u, t=0.0, speed=25.0, **_):
    """Dazed steps forward: the walk's legs under a beaten body."""
    p = walk(u, t=t * 0.8, speed=speed)
    s = math.sin(t * 2.6)
    s2 = math.sin(t * 1.3 + 1.0)
    p.update(lean=0.38 + 0.1 * s, head=0.35 + 0.15 * s2, fh=(3.0, 12.0 + s), nh=(1.0, 13.0 - s), fopen=1.0, nopen=1.0,
             hip=(p['hip'][0], p['hip'][1] + 1.5))
    return p


# ------------------------------------------------------------------ v3: cartoon action
def dash(u, t=0.0, **_):
    """A ninja dash: low, leaning hard, arms swept back, legs a blur."""
    period = 0.2
    ph = (t / period) % 1.0
    a = math.sin(ph * 2 * math.pi)
    return P(hip=(2.0, -27.5), lean=0.62, head=-0.35, fh=(-9.0, 8.0), nh=(-11.0, 6.0), fopen=1.0, nopen=1.0,
             ff=(8.0 + 6 * a, -2.0 - 4 * max(0.0, a)), nf=(-6.0 - 6 * a, -2.0 - 4 * max(0.0, -a)))


def hop_back(u, **_):
    crouch = P(hip=(0.0, -26.0), lean=0.3, ff=(8.0, -2.0), nf=(-8.0, -2.0))
    air = P(hip=(-1.0, -32.0), lean=-0.15, ff=(10.0, -12.0), nf=(-4.0, -9.0), fh=(9.0, -6.0), nh=(5.0, -4.0))
    land = P(hip=(-1.0, -26.5), lean=0.25, ff=(9.0, -2.0), nf=(-10.0, -2.0))
    return keys(u, [(0.0, G, None), (0.15, crouch, out), (0.35, air, out), (0.8, air, None), (0.9, land, ease),
                    (1.0, G, ease)])


def jump(u, **_):
    """Anticipation crouch, a stretched take-off, a tuck in the air, legs reaching for the landing."""
    crouch = P(hip=(0.0, -23.0), lean=0.35, head=0.1, fh=(4.0, 6.0), nh=(2.0, 7.0), ff=(9.0, -2.0), nf=(-9.0, -2.0))
    up = P(hip=(0.0, -34.0), lean=-0.1, head=-0.2, fh=(8.0, -16.0), nh=(5.0, -15.0), ff=(4.0, -3.0), nf=(-4.0, -4.0),
           ftoe=0.8, ntoe=0.8)
    tuck = P(hip=(0.0, -32.0), lean=0.1, ff=(7.0, -18.0), nf=(-3.0, -15.0), fh=(9.0, -6.0), nh=(6.0, -4.0))
    reach = P(hip=(0.0, -32.0), lean=0.0, ff=(8.0, -6.0), nf=(-7.0, -5.0))
    return keys(u, [(0.0, G, None), (0.12, crouch, out), (0.2, up, out), (0.45, tuck, ease), (0.85, reach, ease),
                    (1.0, P(hip=(0.0, -25.0), lean=0.3, ff=(9.0, -2.0), nf=(-9.0, -2.0)), ease)])


AIRKICK_HIT = 0.5


def air_kick(u, **_):
    tuck = P(hip=(0.0, -32.0), lean=0.0, nf=(4.0, -20.0), ff=(-2.0, -14.0), fh=(8.0, -8.0), nh=(5.0, -6.0), nkb=-1.0)
    ext = P(hip=(0.0, -32.0), lean=-0.45, head=0.1, nf=(27.0, -24.0), ntoe=-0.1, ff=(-3.0, -14.0), fh=(4.0, -10.0),
            nh=(-6.0, -4.0), nkb=-1.0)
    return keys(u, [(0.0, tuck, None), (0.3, tuck, None), (0.5, ext, out), (0.7, ext, None), (1.0, tuck, ease)])


AIRPUNCH_HIT = 0.45


def air_punch(u, **_):
    tuck = P(hip=(0.0, -32.0), lean=0.1, ff=(6.0, -16.0), nf=(-4.0, -13.0), nh=(0.0, -4.0), fh=(8.0, -6.0))
    ext = P(hip=(2.0, -32.0), lean=0.4, ff=(4.0, -14.0), nf=(-6.0, -12.0), nh=(20.0, 2.0), fh=(3.0, 0.0), twist=1.0)
    return keys(u, [(0.0, tuck, None), (0.25, tuck, None), (0.45, ext, out), (0.65, ext, None), (1.0, tuck, ease)])


AXE_HIT = 0.55


def axe_kick(u, **_):
    """In the air: the leg swings up over the head and chops straight down."""
    up = P(hip=(0.0, -32.0), lean=-0.5, nf=(10.0, -66.0), ntoe=-0.6, nkb=-1.0, ff=(-4.0, -14.0), fh=(6.0, -12.0),
           nh=(-4.0, -6.0))
    down = P(hip=(0.0, -32.0), lean=0.3, head=0.2, nf=(22.0, -18.0), ntoe=0.4, nkb=-1.0, ff=(-4.0, -14.0),
             fh=(8.0, -2.0), nh=(-6.0, -2.0))
    return keys(u, [(0.0, G, None), (0.35, up, out), (0.55, down, inn), (0.75, down, None), (1.0, down, None)])


def duck(u, **_):
    low = P(hip=(-1.0, -19.0), lean=0.55, head=0.15, fh=(8.0, -2.0), nh=(6.0, 0.0), ff=(10.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, G, None), (0.18, low, out), (1.0, low, None)])


def sidestep(u, t=0.0, **_):
    """Footwork while moving in depth: quick little steps, guard up."""
    ph = (t / 0.22) % 1.0
    a = math.sin(ph * 2 * math.pi)
    return P(hip=(0.0, -30.0 + 0.8 * abs(a)), fh=(8.6, -3.6), nh=(6.6, -1.4), ff=(8.6, -2.0 - 2.5 * max(0.0, a)),
             nf=(-9.0, -2.0 - 2.5 * max(0.0, -a)))


def kip_up(u, **_):
    """From the floor in one spring: legs tucked, a pop into the air, landing in guard."""
    tuck = P(hip=(0.0, -9.0), lean=-1.2, head=0.2, ff=(4.0, -16.0), nf=(2.0, -14.0), fh=(-6.0, -6.0), nh=(-5.0, -4.0),
             fopen=1.0, nopen=1.0)
    pop = P(hip=(0.0, -33.0), lean=-0.1, head=-0.2, ff=(6.0, -6.0), nf=(-5.0, -6.0), fh=(9.0, -14.0), nh=(-8.0, -12.0),
            fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, LIE, None), (0.3, tuck, ease), (0.55, pop, out), (0.85, P(hip=(0.0, -26.0), lean=0.3), ease),
                    (1.0, G, ease)])


ELBOW_HIT = 0.45


def elbow(u, **_):
    wind = P(hip=(-1.0, -30.5), lean=0.0, nh=(-2.0, -2.0), twist=0.0)
    ext = P(hip=(4.0, -30.0), lean=0.32, twist=1.0, nh=(6.0, -1.0), nbend=-1.0, fh=(4.0, 2.0), ff=(11.0, -2.0))
    return keys(u, [(0.0, G, None), (0.22, wind, out), (0.45, ext, out), (0.6, ext, None), (1.0, G, ease)])


def beckon(u, t=0.0, **_):
    """A taunt: bouncing, the lead hand curling 'come on'."""
    b = math.sin(t * 2 * math.pi * 2.0)
    c = math.sin(t * 2 * math.pi * 3.0)
    return P(hip=(0.0, -31.0 + 0.8 * b), lean=0.0, fh=(10.0 + 2 * c, -8.0), fopen=1.0 if c > 0 else 0.0,
             nh=(5.0, 2.0), ff=(8.0, -2.0), nf=(-9.0, -2.0))


def tumble(u, **_):
    """Flying through the air after a big hit, limbs flailing."""
    return keys(u, [(0.0, launched(0.0), None), (0.5, launched(1.0), ease),
                    (1.0, P(hip=(0.0, -30.0), lean=-1.5, head=0.1, fh=(-10.0, -2.0), nh=(-8.0, 4.0), ff=(20.0, -30.0),
                              nf=(22.0, -34.0), fopen=1.0, nopen=1.0), ease)])


def bounce(u, **_):
    """Hitting the floor flat and bouncing once."""
    return keys(u, [(0.0, LIE, None), (0.4, P(hip=(0.0, -10.0), lean=-1.2, head=-0.3, fh=(-8.0, -10.0), nh=(-6.0, -8.0),
                                             ff=(14.0, -14.0), nf=(12.0, -12.0), fopen=1.0, nopen=1.0), out),
                    (1.0, LIE, inn)])


# ------------------------------------------------------------------ v4: weapons
# Weapon angles: 'fwa'/'nwa' turn the weapon off its usual grip (radians; + turns the point down and forward when
# facing right). Scorpion fights with a sword in each hand, Sub-Zero with the ice sword in his near hand or the hammer.
def sword_guard(u, t=0.0, **_):
    """Twin swords ready: the far blade high and forward, the near one low and swept back; the weight rolls (no hop)."""
    b = math.sin(t * 2 * math.pi * 1.2)
    s = math.sin(t * 2 * math.pi * 0.7)
    return P(hip=(1.0 * s, -29.8 + 0.35 * b), lean=0.16 + 0.03 * s, fh=(10.0 + 0.8 * s, -9.0 + 0.5 * b), fwa=-0.35,
             nh=(4.0, 5.0 + 0.4 * b), nwa=-1.9, ff=(9.0, -2.0), nf=(-10.0, -2.0))


def blade_guard(u, t=0.0, **_):
    """One blade held up in front, the free hand back; the weight rolls."""
    b = math.sin(t * 2 * math.pi * 1.1)
    s = math.sin(t * 2 * math.pi * 0.6)
    return P(hip=(1.0 * s, -30.2 + 0.35 * b), lean=0.12 + 0.03 * s, nh=(10.0, -7.0 + 0.5 * b), nwa=-0.25, fh=(2.0, 2.0),
             fopen=1.0, ff=(9.0, -2.0), nf=(-9.0, -2.0))


SLASH_HIT = 0.45


def slash_down(u, hand='n', **_):
    """An overhead diagonal cut: the blade raised behind the head, then down across the foe."""
    h, wa = hand + 'h', hand + 'wa'
    wind = P(hip=(-2.0, -31.5), lean=-0.22, head=-0.1, **{h: (-3.0, -17.0), wa: -1.4})
    cut = P(hip=(4.0, -28.5), lean=0.45, head=0.1, ff=(12.0, -2.0), nf=(-10.0, -2.0), twist=0.8,
            **{h: (17.0, 3.0), wa: 1.35})
    return keys(u, [(0.0, P(**{h: (6.0, -4.0)}), None), (0.25, wind, out), (0.45, cut, out), (0.62, cut, None),
                    (1.0, P(**{h: (8.0, -5.0)}), ease)])


def slash_up(u, hand='n', **_):
    """A rising cut from low behind to high in front."""
    h, wa = hand + 'h', hand + 'wa'
    low = P(hip=(-1.0, -27.0), lean=0.35, **{h: (-6.0, 10.0), wa: 1.6})
    high = P(hip=(3.0, -33.0), lean=-0.25, head=-0.2, twist=0.9, ff=(10.0, -2.0), nf=(-9.0, -3.0), ntoe=0.4,
             **{h: (14.0, -16.0), wa: -0.9})
    return keys(u, [(0.0, P(**{h: (6.0, -4.0)}), None), (0.25, low, out), (0.45, high, out), (0.62, high, None),
                    (1.0, P(**{h: (8.0, -5.0)}), ease)])


def slash_across(u, hand='n', **_):
    """A flat cut across at chest height."""
    h, wa = hand + 'h', hand + 'wa'
    back = P(hip=(-1.5, -30.5), lean=0.0, twist=0.0, **{h: (-6.0, -6.0), wa: -1.6})
    cut = P(hip=(4.0, -30.0), lean=0.3, twist=1.0, ff=(11.0, -2.0), **{h: (18.0, -6.0), wa: 0.2})
    return keys(u, [(0.0, P(**{h: (6.0, -4.0)}), None), (0.25, back, out), (0.45, cut, out), (0.62, cut, None),
                    (1.0, P(**{h: (8.0, -5.0)}), ease)])


THRUST_HIT = 0.42


def thrust(u, hand='n', **_):
    h, wa = hand + 'h', hand + 'wa'
    pull = P(hip=(-2.0, -30.0), lean=0.0, **{h: (-2.0, -4.0), wa: 0.45})
    lunge = P(hip=(6.0, -27.5), lean=0.5, ff=(14.0, -2.0), nf=(-12.0, -2.0), twist=1.0, **{h: (21.0, -5.0), wa: 0.45})
    return keys(u, [(0.0, P(**{h: (6.0, -4.0)}), None), (0.22, pull, out), (0.42, lunge, out), (0.62, lunge, None),
                    (1.0, P(**{h: (8.0, -5.0)}), ease)])


def x_slash(u, **_):
    """Both swords at once: crossed high, then torn apart in an X through the foe."""
    cross_ = P(hip=(-1.0, -31.0), lean=-0.1, fh=(4.0, -14.0), fwa=-1.2, nh=(-2.0, -14.0), nwa=-1.4)
    cut = P(hip=(5.0, -28.0), lean=0.5, twist=0.9, fh=(16.0, 4.0), fwa=1.0, nh=(18.0, -2.0), nwa=0.6,
            ff=(12.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, sword_guard(0), None), (0.25, cross_, out), (0.45, cut, out), (0.65, cut, None),
                    (1.0, sword_guard(0), ease)])


def spin_slash(u, **_):
    """A jumping whirl: the body spins once in the air, both blades out like a blade wheel (root carries the jump)."""
    p = P(hip=(0.0, -31.0), lean=0.1, fh=(14.0, -2.0), fwa=0.3, nh=(12.0, 4.0), nwa=0.9, ff=(6.0, -12.0),
          nf=(-4.0, -10.0), fkb=-1.0, nkb=-1.0)
    p = dict(p)
    p['spin'] = 2 * math.pi * ease(min(1.0, max(0.0, (u - 0.1) / 0.75)))
    return p


def parry(u, hand='n', high=True, **_):
    """The blade brought across to catch a cut, body braced."""
    h, wa = hand + 'h', hand + 'wa'
    if high:
        catch_ = P(hip=(-2.0, -29.5), lean=-0.05, head=0.1, **{h: (9.0, -15.0), wa: -1.1})
    else:
        catch_ = P(hip=(-2.0, -28.5), lean=0.1, **{h: (12.0, -4.0), wa: -1.7})
    return keys(u, [(0.0, P(**{h: (6.0, -4.0)}), None), (0.25, catch_, out), (0.75, catch_, None),
                    (1.0, P(**{h: (8.0, -5.0)}), ease)])


def x_block(u, **_):
    """Both swords crossed in front of the face to take a heavy blow."""
    b = P(hip=(-2.5, -28.5), lean=-0.05, head=0.15, fh=(9.0, -12.0), fwa=-1.4, nh=(8.0, -10.0), nwa=-0.2)
    return keys(u, [(0.0, sword_guard(0), None), (0.2, b, out), (0.8, b, None), (1.0, sword_guard(0), ease)])


def lock(u, t=0.0, hand='n', **_):
    """Blades locked: leaning in, straining, a little tremble."""
    h, wa = hand + 'h', hand + 'wa'
    s = math.sin(t * 40.0) * 0.4
    return P(hip=(3.0 + s * 0.5, -28.8), lean=0.4, head=-0.1, ff=(12.0, -2.0), nf=(-12.0, -2.0),
             **{h: (13.0 + s, -10.0), wa: -1.1})


def draw_swords(u, **_):
    """Both hands reach over the shoulders for the hilts, then the swords sweep out (in hand from 0.38)."""
    reach = P(hip=(0.0, -31.0), lean=-0.05, head=0.1, fh=(-4.0, -15.0), nh=(-6.0, -14.0), fwa=-2.8, nwa=-2.8)
    out_ = P(hip=(-1.0, -27.5), lean=0.25, head=-0.1, fh=(16.0, -4.0), fwa=0.4, nh=(-8.0, 4.0), nwa=-2.0,
             ff=(11.0, -2.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, G, None), (0.3, reach, out), (0.42, reach, None), (0.62, out_, out), (0.85, out_, None),
                    (1.0, sword_guard(0), ease)])


DRAW_AT = 0.4


def form_weapon(u, **_):
    """Sub-Zero shapes ice in his hands: palms together at the chest, then the near hand draws out the weapon."""
    gather = P(hip=(-1.0, -30.5), lean=0.0, fh=(7.0, -4.0), nh=(5.0, -3.0), fopen=1.0, nopen=1.0, nwa=-1.6)
    pull = P(hip=(0.0, -31.0), lean=-0.1, head=-0.1, fh=(-2.0, 4.0), fopen=1.0, nh=(14.0, -12.0), nwa=-0.8)
    return keys(u, [(0.0, G, None), (0.3, gather, out), (0.55, gather, None), (0.85, pull, out), (1.0, pull, None)])


def disarmed(u, hand='n', **_):
    """The weapon torn out of his hand: the arm flung back, hand open."""
    h = hand + 'h'
    a = P(hip=(-3.0, -31.0), lean=-0.3, head=-0.3, fopen=1.0, nopen=1.0, **{h: (-8.0, -12.0)})
    return keys(u, [(0.0, G, None), (0.2, a, out), (0.6, a, None), (1.0, G, ease)])


def kick_up(u, **_):
    """A scoop kick: the toes under the sword on the floor flick it up into the air (at 0.45)."""
    set_ = P(hip=(-2.0, -29.0), lean=-0.1, nf=(10.0, -2.0), ntoe=-0.6, fh=(6.0, -4.0), nh=(-4.0, 2.0))
    flick = P(hip=(-3.0, -31.5), lean=-0.35, nf=(16.0, -26.0), ntoe=-0.3, nkb=-1.0, ff=(4.0, -2.0), fh=(7.0, -6.0),
              nh=(-6.0, 3.0))
    return keys(u, [(0.0, G, None), (0.25, set_, out), (0.45, flick, out), (0.6, flick, None), (1.0, G, ease)])


KICKUP_AT = 0.42


def catch(u, hand='n', **_):
    h = hand + 'h'
    a = P(hip=(0.0, -32.0), lean=-0.15, head=-0.25, nopen=1.0, **{h: (5.0, -21.0)})
    b = P(hip=(0.0, -31.0), lean=0.0, **{h: (8.0, -12.0), hand + 'wa': -0.6})
    return keys(u, [(0.0, G, None), (0.4, a, out), (0.55, a, None), (0.8, b, out), (1.0, sword_guard(0), ease)])


HAMMER_HIT = 0.5


def hammer_guard(u, t=0.0, **_):
    """The hammer shouldered, both hands on the haft; the weight rolls."""
    b = math.sin(t * 2 * math.pi * 1.0)
    s = math.sin(t * 2 * math.pi * 0.55)
    return P(hip=(1.0 * s, -30.0 + 0.4 * b), lean=0.1 + 0.03 * s, nh=(6.0, 0.0 + 0.5 * b), nwa=-0.5,
             fh=(6.0, 2.0 + 0.5 * b), ff=(9.0, -2.0), nf=(-10.0, -2.0))


def hammer_smash(u, **_):
    """Overhead: the hammer swung up behind the head and brought straight down onto the floor in front."""
    wind = P(hip=(-3.0, -32.0), lean=-0.4, head=-0.2, nh=(-4.0, -16.0), nwa=-1.0, fh=(-2.0, -14.0))
    down = P(hip=(5.0, -24.0), lean=0.75, head=0.25, nh=(15.0, 12.0), nwa=1.3, fh=(13.0, 12.0), ff=(13.0, -2.0),
             nf=(-12.0, -2.0))
    return keys(u, [(0.0, hammer_guard(0), None), (0.3, wind, out), (0.5, down, inn), (0.72, down, None),
                    (1.0, hammer_guard(0), ease)])


def hammer_swing(u, **_):
    """A flat swing from behind into the foe's side."""
    back = P(hip=(-3.0, -30.0), lean=-0.1, twist=0.0, nh=(-8.0, -4.0), nwa=-2.2, fh=(-6.0, -2.0))
    hit = P(hip=(5.0, -29.0), lean=0.35, twist=1.0, nh=(17.0, -5.0), nwa=0.6, fh=(15.0, -4.0), ff=(12.0, -2.0),
            nf=(-11.0, -2.0))
    return keys(u, [(0.0, hammer_guard(0), None), (0.28, back, out), (0.5, hit, out), (0.68, hit, None),
                    (1.0, hammer_guard(0), ease)])


def hammer_upper(u, **_):
    low = P(hip=(-1.0, -25.0), lean=0.5, nh=(-2.0, 12.0), nwa=-2.6, fh=(0.0, 12.0))
    high = P(hip=(4.0, -34.0), lean=-0.3, head=-0.3, twist=0.9, nh=(10.0, -20.0), nwa=-1.0, fh=(9.0, -18.0),
             ff=(10.0, -3.0), nf=(-8.0, -4.0), ntoe=0.5)
    return keys(u, [(0.0, hammer_guard(0), None), (0.28, low, out), (0.5, high, out), (0.7, high, None),
                    (1.0, hammer_guard(0), ease)])


# ------------------------------------------------------------------ v4: magic and specials
def teleport_out(u, **_):
    """Scorpion folds into his own fire: a crouch, arms crossed, sinking (hidden from 0.7)."""
    a = P(hip=(0.0, -22.0), lean=0.4, head=0.3, fh=(4.0, -6.0), nh=(5.0, -7.0))
    b = P(hip=(0.0, -14.0), lean=0.6, head=0.4, fh=(4.0, -2.0), nh=(5.0, -3.0), ff=(6.0, -2.0), nf=(-6.0, -2.0))
    return keys(u, [(0.0, G, None), (0.4, a, out), (1.0, b, inn)])


def hellfire(u, **_):
    """A burning fist driven into the stone (the columns rise from 0.45)."""
    rise = P(hip=(-1.0, -33.0), lean=-0.2, head=-0.2, nh=(2.0, -20.0), fh=(-4.0, -6.0))
    slam = P(hip=(2.0, -15.0), lean=0.85, head=0.2, nh=(16.0, 24.0), fh=(-6.0, 6.0), ff=(12.0, -2.0), nf=(-10.0, -2.0),
             nkb=-1.0)
    return keys(u, [(0.0, G, None), (0.3, rise, out), (0.45, slam, inn), (0.8, slam, None), (1.0, G, ease)])


HELLFIRE_AT = 0.45


def fire_dash(u, **_):
    """Flying flat at the foe in a streak of fire, shoulder first."""
    return P(hip=(4.0, -26.0), lean=1.15, head=-0.6, fh=(14.0, 0.0), nh=(-12.0, 2.0), fopen=0.0, ff=(-8.0, -8.0),
             nf=(-14.0, -4.0), fkb=-1.0, nkb=-1.0)


def ice_slide(u, **_):
    """Sub-Zero's slide: low, leaning back on one hand, the lead foot shooting forward along the ice."""
    s = P(hip=(-2.0, -11.0), lean=-0.85, head=0.25, nf=(22.0, -3.0), ntoe=-0.3, nkb=1.0, ff=(6.0, -2.0), fkb=-1.0,
          fh=(-10.0, 18.0), fopen=1.0, nh=(6.0, -4.0))
    return keys(u, [(0.0, P(hip=(0.0, -22.0), lean=0.3), None), (0.15, s, out), (0.85, s, None),
                    (1.0, P(hip=(0.0, -22.0), lean=0.3), ease)])


SLIDE_HIT = 0.5


def shower_cast(u, **_):
    """Both arms thrown up: the ice burst rises over the foe and the shards rain down (release at 0.5)."""
    low = P(hip=(-1.0, -26.0), lean=0.3, fh=(4.0, 6.0), nh=(2.0, 7.0), fopen=1.0, nopen=1.0)
    up = P(hip=(0.0, -34.0), lean=-0.3, head=-0.45, fh=(6.0, -22.0), nh=(3.0, -22.0), fopen=1.0, nopen=1.0,
           ff=(8.0, -3.0), nf=(-8.0, -3.0))
    return keys(u, [(0.0, G, None), (0.35, low, out), (0.5, up, out), (0.85, up, None), (1.0, G, ease)])


SHOWER_AT = 0.5


def wave_cast(u, **_):
    """Both palms slammed flat on the stone: the freezing wave runs out from them (at 0.45)."""
    up_ = P(hip=(0.0, -33.0), lean=-0.15, fh=(6.0, -18.0), nh=(4.0, -18.0), fopen=1.0, nopen=1.0)
    down = P(hip=(2.0, -14.0), lean=0.95, head=0.0, fh=(16.0, 20.0), nh=(14.0, 21.0), fopen=1.0, nopen=1.0,
             ff=(10.0, -2.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, G, None), (0.3, up_, out), (0.45, down, inn), (0.8, down, None), (1.0, G, ease)])


WAVE_AT = 0.45


def wall_cast(u, **_):
    """Palms raised from the floor: the ice wall grows up before him (from 0.4)."""
    low = P(hip=(0.0, -22.0), lean=0.6, fh=(10.0, 10.0), nh=(8.0, 11.0), fopen=1.0, nopen=1.0)
    high = P(hip=(1.0, -32.0), lean=-0.1, fh=(14.0, -14.0), nh=(12.0, -12.0), fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, G, None), (0.3, low, out), (0.6, high, out), (1.0, high, None)])


def balance(u, t=0.0, **_):
    """Up on the balustrade: arms wide, a little wobble."""
    s = math.sin(t * 5.0)
    return P(hip=(0.0, -30.5), lean=0.05 + 0.06 * s, fh=(14.0, 0.0 + 2 * s), nh=(-14.0, -1.0 - 2 * s), fopen=1.0,
             nopen=1.0, ff=(4.0, -2.0), nf=(-4.0, -2.0))


def hang(u, t=0.0, **_):
    """Hanging by the hands from an edge, legs swinging."""
    s = math.sin(t * 3.0)
    return P(hip=(0.0, -30.0), lean=-0.05 + 0.05 * s, head=-0.2, fh=(1.0, -19.0), nh=(-1.0, -19.0), fopen=1.0,
             nopen=1.0, ff=(3.0 + 3 * s, 2.0), nf=(-2.0 + 3 * s, 4.0))


def climb(u, **_):
    """From hanging, a pull and a vault up over the edge (the root carries the rise)."""
    pull = P(hip=(0.0, -30.0), lean=0.3, head=-0.1, fh=(6.0, -8.0), nh=(4.0, -8.0), fopen=1.0, nopen=1.0,
             ff=(6.0, -12.0), nf=(0.0, -8.0))
    tuck = P(hip=(0.0, -26.0), lean=0.5, fh=(8.0, 6.0), nh=(6.0, 7.0), ff=(8.0, -4.0), nf=(-4.0, -3.0))
    return keys(u, [(0.0, hang(0), None), (0.45, pull, out), (0.8, tuck, ease), (1.0, G, ease)])


def flip(u, **_):
    """A somersault in the air (the root carries the jump), forward."""
    tuck = P(hip=(0.0, -30.0), lean=0.2, ff=(7.0, -20.0), nf=(3.0, -18.0), fh=(10.0, -2.0), nh=(8.0, 0.0),
             fkb=-1.0, nkb=-1.0)
    p = keys(u, [(0.0, P(hip=(0.0, -26.0), lean=0.3), None), (0.18, tuck, out), (0.82, tuck, None),
                 (1.0, P(hip=(0.0, -26.0), lean=0.3, ff=(9.0, -2.0), nf=(-9.0, -2.0)), ease)])
    p = dict(p)
    p['spin'] = 2 * math.pi * ease(min(1.0, max(0.0, (u - 0.12) / 0.74)))
    return p


def back_flip(u, **_):
    p = dict(flip(u))
    p['spin'] = -p['spin']
    return p


def roll(u, **_):
    """A forward roll over the floor (root low)."""
    tuck = P(hip=(0.0, -16.0), lean=0.9, head=0.5, ff=(6.0, -8.0), nf=(0.0, -6.0), fh=(8.0, 4.0), nh=(6.0, 5.0),
             fkb=-1.0, nkb=-1.0)
    p = keys(u, [(0.0, P(hip=(0.0, -22.0), lean=0.5), None), (0.15, tuck, out), (0.85, tuck, None), (1.0, G, ease)])
    p = dict(p)
    p['spin'] = 2 * math.pi * ease(min(1.0, max(0.0, (u - 0.1) / 0.75)))
    return p


def teeter(u, t=0.0, **_):
    """On the very edge, arms windmilling, leaning out over the drop and back."""
    a = t * 9.0
    s = math.sin(t * 4.0)
    return P(hip=(0.0, -31.0), lean=0.3 * s, head=0.2 * s, fh=(math.cos(a) * 12, math.sin(a) * 12 - 6),
             nh=(math.cos(a + 2.4) * 12, math.sin(a + 2.4) * 12 - 6), fopen=1.0, nopen=1.0, ff=(3.0, -2.0),
             nf=(-3.0, -2.0))


def slip(u, **_):
    """Feet shoot out on the ice, he goes over backwards."""
    a = P(hip=(2.0, -30.0), lean=-0.5, head=-0.4, ff=(16.0, -10.0), nf=(10.0, -6.0), fh=(-6.0, -14.0), nh=(8.0, -16.0),
          fopen=1.0, nopen=1.0)
    return keys(u, [(0.0, G, None), (0.4, a, out), (1.0, LIE, inn)])


def superhero(u, **_):
    """A landing from high up: deep crouch, one fist on the stone, then up."""
    low = P(hip=(0.0, -15.0), lean=0.7, head=-0.3, nh=(12.0, 22.0), fh=(-6.0, 4.0), ff=(12.0, -2.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, low, None), (0.6, low, None), (1.0, G, ease)])


def dive(u, **_):
    """Head first through the air, arms reaching."""
    return P(hip=(0.0, -30.0), lean=1.45, head=-0.9, fh=(18.0, -4.0), nh=(16.0, -2.0), fopen=1.0, nopen=1.0,
             ff=(-6.0, -6.0), nf=(-10.0, -2.0))


def throw_up(u, **_):
    """Both arms flung skywards (the ice ball leaves for the sky)."""
    return shower_cast(u)


def stomp_land(u, **_):
    """Dropping feet first onto the foe (or the bridge): legs driven down, then a crouch."""
    a = P(hip=(0.0, -32.0), lean=-0.1, ff=(4.0, -2.0), nf=(-3.0, -2.0), fh=(10.0, -12.0), nh=(-10.0, -12.0))
    b = P(hip=(0.0, -20.0), lean=0.45, ff=(9.0, -2.0), nf=(-9.0, -2.0), fh=(8.0, 4.0), nh=(-4.0, 6.0))
    return keys(u, [(0.0, a, None), (0.6, a, None), (0.8, b, out), (1.0, G, ease)])


def hurl(u, hand='n', **_):
    """Throwing something held in the hand (a sword) overarm."""
    h, wa = hand + 'h', hand + 'wa'
    wind = P(hip=(-2.0, -31.0), lean=-0.25, **{h: (-8.0, -14.0), wa: -1.5})
    rel = P(hip=(4.0, -30.0), lean=0.4, twist=1.0, **{h: (19.0, -6.0), wa: 0.4})
    return keys(u, [(0.0, G, None), (0.35, wind, out), (0.5, rel, out), (1.0, rel, None)])


HURL_AT = 0.5


# ================================================================== HULK vs CAPTAIN AMERICA
# The owner disliked the v3 guard's bounce on the spot: in guard nobody hops in place any more. The stance only rolls
# the weight slowly and breathes; every pause in the fight is a stalk or a circle that really travels.
def guard(u, t=0.0, **_):
    """The calm guard: the weight rolls and the chest breathes; no foot ever lifts while the body stays put."""
    s = math.sin(t * 2 * math.pi * 0.7)
    b = math.sin(t * 2 * math.pi * 1.3)
    return P(hip=(1.2 * s, -30.2 + 0.35 * b), lean=0.12 + 0.04 * s, head=-0.04 - 0.03 * s,
             fh=(8.6 + 1.2 * s, -3.6 + 0.5 * b), nh=(6.6 - 0.9 * s, -1.4 + 0.4 * b))


def _steps(t, speed, period, base_f, base_n, back, lift_max=3.4):
    """Feet planted while the body travels: the stride follows the real speed, so a slow drift takes small steps and
    a standstill takes none (no stepping on the spot)."""
    ph = (t / period) % 1.0
    stride = min(16.0, speed * period * 0.5)
    lift = min(lift_max, 0.4 + stride * 0.32) if stride > 1.2 else stride * 0.5
    sgn = -1.0 if back else 1.0

    def foot(ph, base):
        if ph < 0.5:
            k = ph / 0.5
            return (base + sgn * stride * (0.5 - k), -2.0)
        k = (ph - 0.5) / 0.5
        return (base + sgn * stride * (k - 0.5), -2.0 - lift * math.sin(k * math.pi))
    return foot(ph, base_f), foot((ph + 0.5) % 1.0, base_n), ph, stride


CG = P(hip=(0.0, -29.8), lean=0.16, head=-0.05, fh=(10.0, -5.0), nh=(6.0, -2.6), ff=(9.0, -2.0), nf=(-9.5, -2.0))


def cap_guard(u, t=0.0, **_):
    """Captain America's guard: the shield forward on the lead arm, the other fist by the chin, weight rolling."""
    s = math.sin(t * 2 * math.pi * 0.8)
    b = math.sin(t * 2 * math.pi * 1.4)
    return P(hip=(1.0 * s, -29.8 + 0.3 * b), lean=0.16 + 0.03 * s, head=-0.05, fh=(10.0 + 0.8 * s, -5.0 + 0.4 * b),
             nh=(6.0 - 0.6 * s, -2.6 + 0.4 * b), ff=(9.0, -2.0), nf=(-9.5, -2.0))


def cap_stalk(u, t=0.0, speed=30.0, back=False, **_):
    ff, nf, ph, stride = _steps(t, speed, 0.44, 9.0, -9.5, back)
    bob = math.sin(ph * 4 * math.pi) * min(1.0, stride / 6.0)
    return P(hip=(0.0, -29.4 + 0.6 * bob), lean=0.2, head=-0.08, ff=ff, nf=nf, fh=(10.0, -5.0 + 0.4 * bob),
             nh=(6.0, -2.6 + 0.4 * bob))


def stalk(u, t=0.0, speed=30.0, back=False, **_):
    ff, nf, ph, stride = _steps(t, speed, 0.46, 8.6, -9.0, back)
    bob = math.sin(ph * 4 * math.pi) * min(1.0, stride / 6.0)
    return P(hip=(0.0, -29.6 + 0.5 * bob), lean=0.16, ff=ff, nf=nf, fh=(9.0, -4.0 + 0.4 * bob), nh=(6.6, -1.6 + 0.4 * bob))


def cap_run(u, t=0.0, speed=110.0, **_):
    p = run(u, t=t, speed=speed)
    p.update(fh=(9.0, -1.0), fopen=0.0)
    return p


HGp = P(hip=(0.0, -27.5), lean=0.32, head=-0.25, fh=(9.0, 4.0), nh=(5.0, 6.0), ff=(9.0, -2.0), nf=(-10.0, -2.0))


def hulk_guard(u, t=0.0, **_):
    """Hunched, the arms hanging forward, the chest heaving: no hop, no stomp."""
    s = math.sin(t * 2 * math.pi * 0.55)
    b = math.sin(t * 2 * math.pi * 0.9)
    return P(hip=(1.0 * s, -27.5 - 0.6 * b), lean=0.32 + 0.03 * b, head=-0.25 - 0.06 * b, fh=(9.0 + 1.0 * s, 4.0 - 1.2 * b),
             nh=(5.0 - 0.8 * s, 6.0 - 1.0 * b), ff=(9.0, -2.0), nf=(-10.0, -2.0))


def hulk_stalk(u, t=0.0, speed=30.0, back=False, **_):
    ff, nf, ph, stride = _steps(t, speed, 0.62, 9.0, -10.0, back, lift_max=4.0)
    k = min(1.0, stride / 6.0)
    bob = math.sin(ph * 4 * math.pi) * k
    sw = math.sin(ph * 2 * math.pi) * k
    return P(hip=(0.0, -27.0 + 0.9 * bob), lean=0.34 + 0.03 * bob, head=-0.26, ff=ff, nf=nf,
             fh=(9.0 + 2.0 * sw, 4.0 + bob), nh=(5.0 - 2.0 * sw, 6.0 + bob))


def hulk_run(u, t=0.0, speed=120.0, **_):
    period = 0.42
    ph = (t / period) % 1.0
    stride = min(22.0, speed * period * 0.5)

    def foot(ph, base):
        if ph < 0.45:
            k = ph / 0.45
            return (base + stride * (0.5 - k), -2.0)
        k = (ph - 0.45) / 0.55
        return (base + stride * (k - 0.5), -2.0 - 8.0 * math.sin(k * math.pi))
    ff = foot(ph, 6.0)
    nf = foot((ph + 0.5) % 1.0, -3.0)
    bob = abs(math.sin(ph * 2 * math.pi))
    sw = math.sin(ph * 2 * math.pi)
    return P(hip=(2.0, -27.5 + 1.6 * bob), lean=0.62, head=-0.5, ff=ff, nf=nf, fh=(6 + 8 * sw, 4.0), nh=(2 - 8 * sw, 5.0))


def charge(u, t=0.0, speed=150.0, **_):
    p = hulk_run(u, t=t, speed=speed)
    p.update(lean=0.78, head=-0.2, fh=(10.0, 8.0), nh=(-2.0, 10.0), twist=0.0)
    return p


def roar(u, t=0.0, **_):
    s = math.sin(t * 31.0) * 0.6
    wind = P(hip=(0.0, -25.0), lean=0.55, head=0.2, fh=(4.0, 8.0), nh=(2.0, 9.0), ff=(9.0, -2.0), nf=(-11.0, -2.0))
    big = P(hip=(-1.0 + s * 0.3, -30.0), lean=-0.15, head=-0.45, fh=(13.0, -12.0 + s), nh=(-13.0, -11.0 - s),
            ff=(11.0, -2.0), nf=(-12.0, -2.0), twist=1.0)
    return keys(u, [(0.0, HGp, None), (0.25, wind, out), (0.42, big, out), (0.85, big, None), (1.0, HGp, ease)])


SWAT_HIT = 0.46


def swat(u, **_):
    wind = P(hip=(-2.0, -27.0), lean=0.1, head=-0.2, nh=(-15.0, -8.0), fh=(8.0, 3.0), twist=0.0)
    hit = P(hip=(4.0, -27.0), lean=0.45, head=-0.25, twist=1.0, nh=(22.0, -6.0), nbend=-1.0, fh=(2.0, 6.0), ff=(12.0, -2.0),
            nf=(-9.0, -2.0))
    thru = P(hip=(5.0, -27.0), lean=0.55, twist=1.0, nh=(14.0, 8.0), fh=(0.0, 6.0), ff=(12.0, -2.0), nf=(-9.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.28, wind, out), (0.46, hit, out), (0.62, thru, out), (1.0, HGp, ease)])


HAY_HIT = 0.56


def haymaker(u, **_):
    wind = P(hip=(-4.0, -28.0), lean=-0.2, head=-0.1, nh=(-17.0, -10.0), fh=(10.0, -2.0), twist=0.0, ff=(10.0, -2.0),
             nf=(-12.0, -2.0))
    hit = P(hip=(6.0, -27.0), lean=0.55, head=-0.3, twist=1.0, nh=(26.0, -6.0), fh=(-2.0, 6.0), ff=(14.0, -2.0),
            nf=(-10.0, -2.0), ntoe=0.4)
    return keys(u, [(0.0, HGp, None), (0.38, wind, ease), (0.56, hit, inn), (0.72, hit, None), (1.0, HGp, ease)])


JABH_HIT = 0.42


def hulk_jab(u, **_):
    ext = P(hip=(3.0, -27.4), lean=0.42, head=-0.3, fh=(22.0, -5.0), nh=(5.0, 5.0), ff=(12.0, -2.0), nf=(-10.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.2, P(hip=(-1.0, -27.5), lean=0.3, head=-0.25, fh=(6.0, 0.0), nh=(5.0, 6.0)), out),
                    (0.42, ext, out), (0.58, ext, None), (1.0, HGp, ease)])


DFIST_HIT = 0.55


def double_fist(u, **_):
    up_ = P(hip=(-1.0, -31.0), lean=-0.25, head=-0.3, fh=(0.0, -24.0), nh=(-3.0, -24.0), ff=(10.0, -2.0), nf=(-11.0, -2.0))
    down = P(hip=(3.0, -22.0), lean=0.85, head=0.1, fh=(14.0, 18.0), nh=(15.0, 19.0), ff=(12.0, -2.0), nf=(-12.0, -2.0),
             twist=0.6)
    return keys(u, [(0.0, HGp, None), (0.38, up_, out), (0.55, down, inn), (0.8, down, None), (1.0, HGp, ease)])


DFISTHI_HIT = 0.55


def double_fist_high(u, **_):
    """The double fist brought down on a standing foe's head."""
    up_ = P(hip=(-1.0, -31.0), lean=-0.25, head=-0.3, fh=(0.0, -24.0), nh=(-3.0, -24.0), ff=(10.0, -2.0), nf=(-11.0, -2.0))
    down = P(hip=(3.0, -27.0), lean=0.5, head=0.0, fh=(18.0, -2.0), nh=(18.5, -1.0), ff=(12.0, -2.0), nf=(-12.0, -2.0),
             twist=0.6)
    thru = P(hip=(3.0, -25.0), lean=0.7, head=0.1, fh=(14.0, 12.0), nh=(14.0, 13.0), ff=(12.0, -2.0), nf=(-12.0, -2.0),
             twist=0.6)
    return keys(u, [(0.0, HGp, None), (0.38, up_, out), (0.55, down, inn), (0.68, thru, out), (1.0, HGp, ease)])


CLAP_AT = 0.5


def clap(u, **_):
    wide = P(hip=(-1.0, -29.0), lean=-0.1, head=-0.3, fh=(14.0, -9.0), nh=(-14.0, -9.0), fopen=1.0, nopen=1.0, twist=1.0,
             ff=(10.0, -2.0), nf=(-12.0, -2.0))
    shut = P(hip=(3.0, -28.0), lean=0.3, head=-0.25, fh=(13.0, -7.0), nh=(17.0, -7.0), fopen=1.0, nopen=1.0, twist=1.0,
             ff=(12.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.38, wide, out), (0.5, shut, inn), (0.78, shut, None), (1.0, HGp, ease)])


LAND = P(hip=(0.0, -16.0), lean=0.75, head=-0.55, fh=(14.0, 18.0), nh=(-12.0, -6.0), ff=(12.0, -2.0), nf=(-10.0, -2.0))


def leap(u, **_):
    """Anticipation squat, a stretched launch, flight with the fists up, the superhero landing."""
    crouch = P(hip=(0.0, -19.0), lean=0.6, head=-0.4, fh=(6.0, 10.0), nh=(2.0, 12.0), ff=(10.0, -2.0), nf=(-11.0, -2.0))
    launch = P(hip=(0.0, -33.0), lean=0.1, head=-0.4, fh=(8.0, -20.0), nh=(4.0, -21.0), ff=(2.0, -3.0), nf=(-6.0, -6.0),
               ftoe=0.9, ntoe=0.9)
    fly = P(hip=(0.0, -31.0), lean=0.2, head=-0.3, fh=(10.0, -14.0), nh=(6.0, -16.0), ff=(8.0, -14.0), nf=(-8.0, -10.0),
            fkb=-1.0)
    reach = P(hip=(0.0, -31.0), lean=0.3, fh=(10.0, 4.0), nh=(-6.0, -10.0), ff=(9.0, -6.0), nf=(-10.0, -5.0))
    return keys(u, [(0.0, HGp, None), (0.14, crouch, out), (0.22, launch, out), (0.4, fly, ease), (0.86, reach, ease),
                    (1.0, LAND, ease)])


def fall_in(u, **_):
    """Dropping from the sky: the flight pose and the landing (the root carries the fall)."""
    fly = P(hip=(0.0, -31.0), lean=0.2, head=-0.3, fh=(10.0, -14.0), nh=(6.0, -16.0), ff=(8.0, -14.0), nf=(-8.0, -10.0),
            fkb=-1.0)
    reach = P(hip=(0.0, -31.0), lean=0.3, fh=(10.0, 4.0), nh=(-6.0, -10.0), ff=(9.0, -6.0), nf=(-10.0, -5.0))
    return keys(u, [(0.0, fly, None), (0.8, reach, ease), (1.0, LAND, ease)])


def land_rise(u, **_):
    return keys(u, [(0.0, LAND, None), (0.35, LAND, None), (1.0, HGp, ease)])


def smash_land(u, **_):
    """Coming down with both fists first (the meteor smash)."""
    over = P(hip=(0.0, -31.0), lean=-0.2, head=-0.3, fh=(2.0, -26.0), nh=(-1.0, -26.0), ff=(6.0, -10.0), nf=(-8.0, -8.0))
    down = P(hip=(2.0, -18.0), lean=0.9, head=0.1, fh=(12.0, 20.0), nh=(13.0, 21.0), ff=(12.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, over, None), (0.75, over, None), (0.92, down, inn), (1.0, down, None)])


def lift(u, **_):
    low = P(hip=(1.0, -20.0), lean=0.7, head=-0.3, fh=(12.0, 16.0), nh=(14.0, 18.0), fopen=1.0, nopen=1.0, ff=(11.0, -2.0),
            nf=(-11.0, -2.0))
    up_ = P(hip=(-1.0, -30.0), lean=-0.15, head=-0.35, fh=(3.0, -26.0), nh=(5.0, -27.0), fopen=1.0, nopen=1.0,
            ff=(11.0, -2.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.35, low, out), (0.55, low, None), (1.0, up_, ease)])


HOLD_UP = P(hip=(-1.0, -30.0), lean=-0.15, head=-0.35, fh=(3.0, -26.0), nh=(5.0, -27.0), fopen=1.0, nopen=1.0,
            ff=(11.0, -2.0), nf=(-12.0, -2.0))


def hold_up(u, t=0.0, **_):
    s = math.sin(t * 5.0)
    p = dict(HOLD_UP)
    p['hip'] = (-1.0, -30.0 + 0.6 * s)
    return p


def hold_up_walk(u, t=0.0, speed=30.0, **_):
    ff, nf, ph, stride = _steps(t, speed, 0.62, 11.0, -12.0, False, lift_max=4.0)
    p = dict(HOLD_UP)
    p['ff'], p['nf'] = ff, nf
    p['hip'] = (-1.0, -30.0 + 0.8 * math.sin(ph * 4 * math.pi))
    return p


HEAVE_REL = 0.42


def heave(u, **_):
    back = P(hip=(-3.0, -30.5), lean=-0.35, head=-0.4, fh=(-4.0, -24.0), nh=(-2.0, -25.0), fopen=1.0, nopen=1.0,
             ff=(11.0, -2.0), nf=(-12.0, -2.0))
    rel = P(hip=(5.0, -27.0), lean=0.6, head=-0.2, fh=(20.0, -10.0), nh=(23.0, -8.0), fopen=1.0, nopen=1.0, ff=(13.0, -2.0),
            nf=(-10.0, -2.0))
    return keys(u, [(0.0, HOLD_UP, None), (0.25, back, out), (0.42, rel, inn), (0.65, rel, None), (1.0, HGp, ease)])


def grab(u, **_):
    reach = P(hip=(3.0, -27.5), lean=0.5, head=-0.25, fh=(20.0, -4.0), nh=(20.0, -2.0), fopen=1.0, nopen=1.0,
              ff=(12.0, -2.0), nf=(-10.0, -2.0), twist=0.8)
    shut = P(hip=(1.0, -28.0), lean=0.35, head=-0.25, fh=(16.0, -6.0), nh=(15.0, -4.0), fopen=0.0, nopen=0.0,
             ff=(11.0, -2.0), nf=(-10.0, -2.0), twist=0.8)
    return keys(u, [(0.0, HGp, None), (0.5, reach, out), (0.75, shut, out), (1.0, shut, None)])


def hold_front(u, t=0.0, **_):
    """Holding someone up in front by the throat (the far hand up and out)."""
    s = math.sin(t * 4.0)
    return P(hip=(-1.0, -29.0), lean=0.05, head=-0.35, fh=(15.0, -16.0 + 0.6 * s), nh=(4.0, 4.0), ff=(11.0, -2.0),
             nf=(-12.0, -2.0))


SLAM_AT = 0.6


def swing_slam(u, **_):
    """Holding a man by the ankle: swung from behind over the top into the ground in front."""
    back = P(hip=(-2.0, -29.0), lean=-0.25, head=-0.3, fh=(-14.0, -6.0), nh=(-8.0, 2.0), twist=0.0, ff=(10.0, -2.0),
             nf=(-12.0, -2.0))
    top = P(hip=(0.0, -31.0), lean=-0.1, head=-0.4, fh=(0.0, -28.0), nh=(-4.0, -6.0), twist=0.5, ff=(10.0, -2.0),
            nf=(-12.0, -2.0))
    down = P(hip=(4.0, -23.0), lean=0.8, head=0.0, fh=(20.0, 16.0), nh=(2.0, 6.0), twist=1.0, ff=(13.0, -2.0),
             nf=(-11.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.3, back, ease), (0.45, top, inn), (0.6, down, inn), (0.8, down, None),
                    (1.0, back, ease)])


HEADBUTT_HIT = 0.45


def headbutt(u, **_):
    back = P(hip=(-3.0, -29.0), lean=-0.25, head=-0.4, fh=(15.0, -12.0), nh=(10.0, -8.0))
    hit = P(hip=(5.0, -28.0), lean=0.7, head=0.25, fh=(15.0, -10.0), nh=(12.0, -6.0), ff=(12.0, -2.0), nf=(-10.0, -2.0))
    return keys(u, [(0.0, hold_front(0), None), (0.3, back, out), (0.45, hit, inn), (0.6, hit, None),
                    (1.0, hold_front(0), ease)])


STOMP_HIT = 0.5


def stomp(u, **_):
    lift_ = P(hip=(-1.0, -31.0), lean=0.1, head=-0.2, nf=(10.0, -26.0), nkb=-1.0, fh=(10.0, 0.0), nh=(-8.0, -4.0))
    down = P(hip=(2.0, -27.0), lean=0.45, head=-0.1, nf=(16.0, -2.0), fh=(12.0, 4.0), nh=(-6.0, 0.0), ff=(-4.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.32, lift_, out), (0.5, down, inn), (0.7, down, None), (1.0, HGp, ease)])


# ---------------------------------------------------------------- Captain America
BASH_HIT = 0.42


def bash(u, **_):
    """The shield driven forward on the lead arm."""
    wind = P(hip=(-1.5, -30.4), lean=0.0, fh=(5.0, -4.0), nh=(6.0, -2.6))
    ext = P(hip=(4.0, -30.0), lean=0.36, fh=(21.0, -6.0), nh=(4.0, -1.0), ff=(11.0, -2.0), nf=(-8.0, -2.0))
    return keys(u, [(0.0, CG, None), (0.2, wind, out), (0.42, ext, out), (0.58, ext, None), (1.0, CG, ease)])


UPBASH_HIT = 0.46


def up_bash(u, **_):
    """The shield's rim driven up under the chin."""
    dip = P(hip=(1.0, -24.5), lean=0.42, head=0.1, fh=(6.0, 8.0), nh=(5.0, -1.0), ff=(10.0, -2.0), nf=(-9.0, -2.0))
    rise = P(hip=(4.0, -35.0), lean=-0.22, head=-0.25, fh=(12.0, -24.0), nh=(4.0, 3.0), ff=(9.0, -3.0), nf=(-7.0, -4.0),
             ntoe=0.6, ftoe=0.25)
    return keys(u, [(0.0, CG, None), (0.26, dip, out), (0.46, rise, out), (0.66, rise, None), (1.0, CG, ease)])


def shield_block(u, **_):
    b = P(hip=(-1.5, -28.5), lean=0.22, head=0.15, fh=(9.0, -8.0), nh=(3.0, -2.0), ff=(9.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, CG, None), (0.15, b, out), (0.85, b, None), (1.0, CG, ease)])


def shield_brace(u, **_):
    """Braced behind the shield, skidding back from a huge blow."""
    return P(hip=(-2.0, -26.0), lean=0.38, head=0.2, fh=(10.0, -7.0), nh=(5.0, -3.0), ff=(12.0, -2.0), nf=(-13.0, -2.0))


THROW_REL = 0.46


def shield_throw(u, **_):
    wind = P(hip=(-2.0, -30.0), lean=-0.1, twist=0.0, fh=(-4.0, -2.0), nh=(5.0, 1.0), ff=(9.0, -2.0), nf=(-10.0, -2.0))
    rel = P(hip=(4.0, -30.4), lean=0.32, twist=0.4, fh=(20.0, -6.0), nh=(-2.0, 3.0), ff=(11.0, -2.0), nf=(-9.0, -2.0))
    return keys(u, [(0.0, CG, None), (0.3, wind, out), (0.46, rel, out), (0.7, rel, None), (1.0, CG, ease)])


def shield_catch(u, **_):
    reach = P(hip=(1.0, -30.4), lean=0.15, fh=(16.0, -9.0), nh=(5.0, -1.0))
    give = P(hip=(-2.0, -29.6), lean=0.0, fh=(9.0, -6.0), nh=(5.0, -2.0))
    return keys(u, [(0.0, CG, None), (0.3, reach, out), (0.5, reach, None), (0.7, give, out), (1.0, CG, ease)])


def shield_flip(u, back=False, **_):
    """A somersault: crouch, a tucked full turn in the air, land in guard (the root carries the arc)."""
    crouch = P(hip=(0.0, -24.0), lean=0.35, fh=(4.0, 4.0), nh=(2.0, 6.0))
    tuck = P(hip=(0.0, -31.0), lean=0.2, head=0.3, ff=(6.0, -18.0), nf=(2.0, -16.0), fh=(8.0, 2.0), nh=(7.0, 4.0), fkb=-1.0,
             nkb=-1.0)
    land = P(hip=(0.0, -25.0), lean=0.3, ff=(9.0, -2.0), nf=(-9.0, -2.0), fh=(10.0, -6.0), nh=(5.0, -2.0))
    p = dict(keys(u, [(0.0, CG, None), (0.14, crouch, out), (0.24, tuck, out), (0.78, tuck, None), (0.9, land, ease),
                      (1.0, CG, ease)]))
    k = min(1.0, max(0.0, (u - 0.18) / 0.62))
    p['spin'] = (-1 if back else 1) * 2 * math.pi * ease(k)
    return p


FLIPKICK_HIT = 0.6


def flip_kick(u, **_):
    """A backflip whose rising legs kick the chin."""
    crouch = P(hip=(0.0, -24.0), lean=0.3, fh=(6.0, -2.0), nh=(4.0, 0.0))
    kick = P(hip=(0.0, -31.0), lean=-0.3, head=-0.2, ff=(14.0, -40.0), nf=(4.0, -26.0), fkb=1.0, nkb=-1.0, ftoe=-0.4,
             fh=(-6.0, 4.0), nh=(-8.0, 6.0))
    tuck = P(hip=(0.0, -31.0), lean=0.2, head=0.3, ff=(6.0, -18.0), nf=(2.0, -16.0), fh=(8.0, 2.0), nh=(7.0, 4.0),
             fkb=-1.0, nkb=-1.0)
    p = dict(keys(u, [(0.0, CG, None), (0.25, crouch, out), (0.6, kick, out), (0.8, tuck, ease), (1.0, CG, ease)]))
    k = min(1.0, max(0.0, (u - 0.4) / 0.55))
    p['spin'] = -2 * math.pi * ease(k)
    return p


BASESLIDE_HIT = 0.45


def base_slide(u, **_):
    """A baseball slide on the hip, the lead leg kicking out low."""
    low = P(hip=(0.0, -9.0), lean=-1.05, head=0.65, ff=(23.0, -3.0), ftoe=-0.5, nf=(5.0, -2.0), nkb=-1.0, fkb=1.0,
            fh=(4.0, -12.0), nh=(-9.0, 5.0), nopen=1.0)
    return keys(u, [(0.0, CG, None), (0.18, low, out), (0.82, low, None), (1.0, CG, ease)])


def shield_roll(u, **_):
    """A dodge roll over the shoulder."""
    tuck = P(hip=(0.0, -16.0), lean=0.9, head=0.6, ff=(4.0, -10.0), nf=(0.0, -8.0), fh=(8.0, 4.0), nh=(6.0, 6.0), fkb=-1.0,
             nkb=-1.0)
    p = dict(keys(u, [(0.0, CG, None), (0.15, tuck, out), (0.85, tuck, None), (1.0, CG, ease)]))
    k = min(1.0, max(0.0, (u - 0.12) / 0.76))
    p['spin'] = 2 * math.pi * ease(k)
    return p


def crouch_look(u, t=0.0, **_):
    """Looking up, braced (as the Hulk leaps)."""
    return P(hip=(-1.0, -25.0), lean=0.1, head=-0.5, fh=(8.0, -10.0), nh=(5.0, -4.0), ff=(10.0, -2.0), nf=(-11.0, -2.0))


def held_ankle(u, t=0.0, **_):
    """Upside down by one ankle in a giant's fist, flailing (the scene turns him along the swinging arm)."""
    s = math.sin(t * 8.0)
    return P(hip=(0.0, -30.0), lean=0.0, head=0.2, ff=(1.0, -2.0), nf=(-7.0, -12.0), nkb=-1.0, fh=(-3.0 + s, -17.0),
             nh=(3.0 - s, -16.0), fopen=1.0, nopen=1.0)


def held_throat(u, t=0.0, **_):
    """Hanging by the throat, kicking, clawing at the hand."""
    s = math.sin(t * 9.0)
    return P(hip=(0.0, -30.0), lean=-0.1, head=-0.3, ff=(3.0 + 2 * s, 2.0), nf=(-3.0 - 2 * s, 3.0), ftoe=0.9, ntoe=0.9,
             fh=(6.0, -9.0), nh=(7.0, -8.0), fopen=1.0, nopen=1.0)


def thrown(u, **_):
    """Thrown: spinning end over end (the root carries the flight)."""
    p = dict(launched(min(1.0, u * 2)))
    p['spin'] = -2.6 * math.pi * u
    return p


def zapped(u, t=0.0, **_):
    """On the live rail, convulsing."""
    a = math.sin(t * 47.0)
    b = math.sin(t * 61.0 + 1.0)
    return P(hip=(0.0, -12.0 + a), lean=-0.9 + 0.25 * b, head=-0.5 + 0.3 * a, fh=(-6.0 + 5 * a, -14.0 + 4 * b),
             nh=(-10.0 - 5 * b, -12.0 + 4 * a), ff=(14.0 + 3 * b, -6.0), nf=(10.0 - 3 * a, -4.0), fopen=1.0, nopen=1.0)


def kneel_up(u, t=0.0, **_):
    """Getting up on one knee, the shield arm propping him."""
    kn = P(hip=(-1.0, -17.0), lean=0.45, head=0.3, ff=(9.0, -2.0), nf=(-6.0, -2.0), nkb=-1.0, fh=(8.0, 10.0), nh=(2.0, 8.0))
    return keys(u, [(0.0, LIE_FRONT, None), (0.5, kn, ease), (1.0, kn, None)])


def fall_back(u, **_):
    """Rising, then the legs give and he drops back."""
    kn = P(hip=(-1.0, -17.0), lean=0.45, head=0.3, ff=(9.0, -2.0), nf=(-6.0, -2.0), nkb=-1.0, fh=(8.0, 10.0), nh=(2.0, 8.0))
    return keys(u, [(0.0, kn, None), (0.35, P(hip=(0.0, -22.0), lean=0.4, head=0.4, ff=(8.0, -2.0), nf=(-6.0, -2.0),
                                              fh=(6.0, 8.0), nh=(3.0, 9.0)), ease), (1.0, LIE_FRONT, inn)])


# ---------------------------------------------------------------- the brute takes a blow (from his own stance)
def hulk_hit(u, **_):
    snap = P(hip=(-3.0, -28.0), lean=0.05, head=-0.65, fh=(6.0, 6.0), nh=(1.0, 8.0), ff=(9.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.15, snap, out), (0.4, snap, None), (1.0, HGp, ease)])


def hulk_hit_body(u, **_):
    fold = P(hip=(-2.0, -25.5), lean=0.62, head=0.05, fh=(6.0, 9.0), nh=(3.0, 10.0), ff=(9.0, -2.0), nf=(-11.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.15, fold, out), (0.45, fold, None), (1.0, HGp, ease)])


def hulk_reel(u, t=0.0, **_):
    """Rocked back by a big blow: the head thrown back, arms flung, a heavy step to catch himself."""
    back = P(hip=(-4.0, -29.5), lean=-0.35, head=-0.7, fh=(10.0, -12.0), nh=(-10.0, -8.0), ff=(12.0, -6.0), nf=(-12.0, -2.0))
    return keys(u, [(0.0, HGp, None), (0.18, back, out), (0.5, back, None), (1.0, HGp, ease)])



def lie_front(u, **_):
    return LIE_FRONT


def hold_front_walk(u, t=0.0, speed=40.0, **_):
    """Stomping along with a man held up by the throat."""
    ff, nf, ph, stride = _steps(t, speed, 0.6, 11.0, -12.0, False, lift_max=4.0)
    p = hold_front(u, t=t)
    p['ff'], p['nf'] = ff, nf
    p['hip'] = (-1.0, -29.0 + 0.8 * math.sin(ph * 4 * math.pi))
    return p




def follow(base, over):
    """Adds a follow-through to a strike: past the extension a little, then home."""
    def fn(u, **kw):
        if u < 0.62:
            return base(u, **kw)
        ext = base(0.56, **kw)
        k = (u - 0.62) / 0.38
        p = dict(ext)
        p['lean'] = ext['lean'] + over * math.sin(min(1.0, k * 2.0) * math.pi * 0.5) * (1 - k)
        return lerp_pose(p, base(1.0, **kw), ease(k))
    fn.__name__ = base.__name__
    return fn


jab = follow(jab, 0.08)
cross = follow(cross, 0.12)
hook = follow(hook, 0.14)
uppercut = follow(uppercut, -0.1)


MOVES = {k: v for k, v in globals().items() if callable(v) and k[0].islower() and k not in (
    'pose', 'lerp_pose', 'ease', 'out', 'inn', 'keys', 'P', 'follow')}

# feet stay planted while the body travels in these (plant() in the scene)
PLANTED = {jab, cross, hook, uppercut, knee, front_kick, roundhouse, ice_cast, spear_throw, yank, hit_head, hit_body,
           block, block_hold, speared, breakout, elbow, slash_down, slash_up, slash_across, thrust, x_slash, parry,
           x_block, lock, draw_swords, form_weapon, hammer_smash, hammer_swing, hammer_upper, kick_up, catch, wall_cast,
           wave_cast, shower_cast, hellfire, hurl, disarmed, bash, up_bash, swat, haymaker, hulk_jab, double_fist,
           double_fist_high, clap, headbutt, shield_block, shield_throw, shield_catch, roar, heave, grab, lift, stomp,
           swing_slam, hold_up, hold_front, hulk_hit, hulk_hit_body, hulk_reel}
# fast blows leave a smear behind the fist or foot
SMEARS = {jab, cross, hook, uppercut, roundhouse, front_kick, knee, sweep, jump_kick, air_kick, air_punch, axe_kick,
          elbow, kick_up, bash, up_bash, swat, haymaker, hulk_jab, double_fist, double_fist_high, headbutt, stomp,
          flip_kick, base_slide, shield_throw, clap, swing_slam, heave}
# moves that travel on foot (their steps follow the real speed; they cue footsteps)
TRAVEL = {walk, run, limp, stalk, cap_stalk, hulk_stalk, cap_run, hulk_run, charge, hold_up_walk, hold_front_walk,
          sidestep}
# a fighter in one of these and not travelling is standing still
STATIC = {guard, cap_guard, hulk_guard, sword_guard, blade_guard, hammer_guard}
# footstep rates per travel move
STEP_RATE = {walk: 3.6, limp: 2.6, cap_stalk: 4.6, hulk_stalk: 3.2, stalk: 4.6, run: 5.5, cap_run: 5.5, hulk_run: 4.8,
             charge: 5.0, hold_up_walk: 3.2, hold_front_walk: 3.2, sidestep: 7.0}

# a stance maps the generic names the choreography uses to a fighter's own moves (fighter data: 'stance')
STANCES = {
    'ninja': {'guard': 'guard', 'stalk': 'stalk', 'run': 'run', 'walk': 'stalk'},
    'shield': {'guard': 'cap_guard', 'stalk': 'cap_stalk', 'run': 'cap_run', 'walk': 'cap_stalk', 'flip': 'shield_flip',
               'roll': 'shield_roll', 'block': 'shield_block', 'block_hold': 'shield_block'},
    'brute': {'guard': 'hulk_guard', 'stalk': 'hulk_stalk', 'run': 'hulk_run', 'walk': 'hulk_stalk',
              'hit_head': 'hulk_hit', 'hit_body': 'hulk_hit_body'},
}
