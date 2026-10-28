"""The beat library: short phrases of a fight (an exchange, a clash of blades, a power, a trip to another area, a
finisher), each a function beat(c, A, D, t, **params) -> the scene time it ends. A is the fighter who leads the beat,
D the other. Beats read positions from the fighters' timelines, keep the action inside the current area and scale
everything (knockback, leaps, magic, destruction) by power tier. The choreographer (choreo.py) picks them; a recipe can
pin them by name.

To add a new kind of move for a new fighter: write a beat here (built from acts.py), register it in BEATS with the acts
it suits, and name it in the fighter's 'powers' list. Every fighter can then use it.
"""
import math
import numpy as np

import moves
import rig
import fx
import magic as mg
import text as T
import acts
from acts import (strike, whiff, clash, dash, jump_to, fall_lie_up, tumble_to, wep, drip, smear_floor, element_hands,
                  fire_body, glint, crater, guard, guard_move, power, leap_k, world_point, hand_pt, DUR, floor_hit)
from scene import arc, hold_y
from proj import floor_y
from px import hexc

BEATS = {}


def beat(*acts_, weight=1.0, needs=None, pool=True):
    """Register a beat for the given acts. needs(c, A, D, t) -> bool says whether it can happen now."""
    def deco(fn):
        BEATS[fn.__name__] = dict(fn=fn, acts=acts_, weight=weight, needs=needs, pool=pool)
        return fn
    return deco


# ------------------------------------------------------------------ positions inside the current area
def area(c):
    return c.w.stage.AREAS[c.area]


def floor_at(c, x, z, t):
    """The floor height to land on at (x, z): the current area's own floor when the point is in it (so a landing never
    picks a pillar cap or a rail by accident), else the highest surface there."""
    a = area(c)
    if a['x'][0] - 40 <= x <= a['x'][1] + 40 and a['z'][0] - 40 <= z <= a['z'][1] + 40:
        return c.w.stage.area_y(c.area, x, z)
    g = c.w.stage.ground(x, z, t, 999.0)
    return g if g > -900 else 0.0


def clamp_x(c, x, m=0.0):
    a = area(c)
    return max(a['x'][0] + m, min(a['x'][1] - m, x))


def clamp_z(c, z, m=0.0):
    a = area(c)
    return max(a['z'][0] + m, min(a['z'][1] - m, z))


def sd(A, D, t):
    return 1 if D.x_at(t) > A.x_at(t) else -1


def room_behind(c, D, s, t):
    """How far D can be pushed away from A (direction s) before the area's edge."""
    a = area(c)
    x = D.x_at(t)
    return (a['x'][1] - x) if s > 0 else (x - a['x'][0])


def gap(A, D, t):
    return abs(D.x_at(t) - A.x_at(t))


def fdist(A, D):
    """The fighting distance between two fighters of these sizes."""
    return 30.0 * (A.K + D.K) / 1.5


def circle(c, A, D, t, dur=None, swap=None):
    """In guard they circle and reposition (never a stomp on the spot): both travel round a point between them, and
    the pair drifts back towards the area's centre. Ends at fighting distance."""
    rng = c.rng
    dur = dur or float(rng.uniform(0.6, 1.0))
    a = area(c)
    mx = (A.x_at(t) + D.x_at(t)) / 2
    mz = (A.z_at(t) + D.z_at(t)) / 2
    cx, cz = a['center']
    mx += (cx - mx) * 0.45
    mz += (cz - mz) * 0.45
    s = sd(A, D, t)
    if swap is None:
        swap = rng.uniform() < 0.25 and a.get('size') != 'narrow'
    if swap:
        s = -s
    half = fdist(A, D) * 0.5 + 6
    dzv = float(rng.uniform(-18, 18)) if a['z'][1] - a['z'][0] > 30 else 0.0
    za = clamp_z(c, mz - dzv, 4)
    zd = clamp_z(c, mz + dzv, 4)
    xa = clamp_x(c, mx - s * half, 10)
    xd = clamp_x(c, mx + s * half, 10)
    if abs(xd - xa) < half * 1.6:
        mid = clamp_x(c, mx, half + 12)
        xa, xd = mid - s * half, mid + s * half
    A.add(t, 'stalk', dur, to=(xa, za), face=s)
    D.add(t, 'stalk', dur * float(rng.uniform(0.85, 1.0)), to=(xd, zd), face=-s)
    guard(A, t + dur)
    guard(D, t + dur)
    return t + dur


def engage(c, A, D, t, dist=None, fast=True):
    """A closes in to fighting distance (a dash if far, a stalk if near); returns when he is there."""
    dist = dist or fdist(A, D)
    s = sd(A, D, t)
    g = gap(A, D, t)
    dzz = abs(D.z_at(t) - A.z_at(t))
    if g < dist * 0.7:
        # too close (on top of him): a quick hop back to fighting distance first
        tx = clamp_x(c, D.x_at(t) - s * dist, 4)
        A.add(t, 'hop_back', 0.3, to=(tx, D.z_at(t)), air=acts.rel(A, t, 6), face=s)
        guard(A, t + 0.3)
        return t + 0.3
    if g < dist + 6 and dzz < 10:
        return t
    tx = D.x_at(t) - s * dist
    tz = D.z_at(t)
    if g > 70 and fast:
        dur = min(0.4, 0.18 + g / 600.0)
        dash(c, A, t, (tx, tz), dur)
        return t + dur + 0.02
    dur = max(0.3, min(0.8, (g - dist) / 90.0 + 0.25))
    A.add(t, 'stalk', dur, to=(tx, tz), face=s)
    guard(A, t + dur)
    return t + dur


def ensure_room(c, A, D, t, need=40.0):
    """Before a string that pushes D back: if D's back is to the area's edge, they circle round first."""
    s = sd(A, D, t)
    if room_behind(c, D, s, t) < need:
        return circle(c, A, D, t, swap=True)
    return t


def pick(c, names, avoid=()):
    names = [n for n in names if n not in avoid] or list(names)
    return names[int(c.rng.integers(len(names)))]


def wound(part, kind='stain', r=1.6, rng=None):
    rng = rng or np.random.default_rng()
    return {'part': part, 't': float(rng.uniform(0.2, 0.9)) if part != 'head' else float(rng.uniform(-1.5, 2.6)),
            's': float(rng.uniform(-0.6, 0.6)) if part != 'head' else float(rng.uniform(0.4, 2.6)), 'r': r, 'kind': kind}


def lands(c, who):
    """Winner bias: the winner's blows land more often than they are blocked."""
    return 0.82 if who == c.winner else 0.68


def last_blows(c, who):
    return c.recent.setdefault(who, [])


def note_blow(c, who, name):
    lb = last_blows(c, who)
    lb.append(name)
    del lb[:-3]


def choose_blow(c, F, kind, avoid_last=True):
    """A blow of this kind from F's kit, not the one he just threw (no move repeated back to back)."""
    names = F.spec['strikes'].get(kind) or F.spec['strikes']['heavy']
    lb = last_blows(c, F.name)
    nm = pick(c, names, avoid=lb[-1:] if avoid_last else ())
    note_blow(c, F.name, nm)
    return nm


WHERE = {'jab': 'head', 'cross': 'head', 'hook': 'head', 'uppercut': 'chin', 'roundhouse': 'head', 'front_kick': 'gut',
         'knee': 'gut', 'elbow': 'head', 'sweep': 'legs', 'bash': 'head', 'up_bash': 'chin', 'hulk_jab': 'head',
         'swat': 'head', 'haymaker': 'head', 'double_fist_high': 'head', 'stomp': 'gut', 'headbutt': 'head',
         'base_slide': 'legs'}
REACT = {'gut': 'hit_body', 'legs': 'hit_body'}


# ------------------------------------------------------------------ the intro and the words
TITLE_COLORS = {'fire': T.FIRE, 'ice': T.ICE, 'gold': T.GOLD, 'bone': T.BONE, 'blood': T.BLOOD, 'hero': T.CAP,
                'green': T.HULK, 'red': T.RED}


def intro(c, A, D, t=0.0):
    """The title over a pan across the stage; each fighter appears his own way (from fire, dropping from a height,
    walking in); FIGHT!"""
    w = c.w
    st = w.stage
    pan = getattr(st, 'TITLE_PAN', None)
    if pan:
        (x0, y0), (x1, y1) = pan['start'], pan['end']

        def pan_fn(tt, x0=x0, y0=y0, x1=x1, y1=y1):
            k = moves.ease(min(1.0, max(0.0, (tt - 0.2) / 3.0)))
            return (x0 + (x1 - x0) * k, y0 + (y1 - y0) * k)
        w.cam_keys.append((0.0, 3.25, pan_fn, True))
    w.tint.append((0.0, 3.0, (0.55, 0.5, 0.65), 0.45))
    na, nd = A.spec['name'], D.spec['name']
    fa = 'big' if len(na) <= 10 else 'mid'
    fd = 'big' if len(nd) <= 10 else 'mid'
    w.words += [T.Word(na, 0.5, 3.3, 130, 150, fa, TITLE_COLORS.get(A.spec['title_colors'], T.GOLD), drop=10),
                T.Word('VS', 1.0, 3.3, 240, 176, 'mid', T.BONE),
                T.Word(nd, 1.4, 3.3, 350, 206, fd, TITLE_COLORS.get(D.spec['title_colors'], T.GOLD), drop=10)]
    for tt, x in ((0.5, 150), (1.0, 240), (1.4, 330)):
        w.shake(tt, 1.6, 0.15)
        w.cue('slam', t=tt, x=w.cam[0] + x, big=tt > 1.2)
    w.cue('music', t=0.0, section='intro')
    tA = entrance(c, A, 2.9, A.spec.get('entrance', 'walk_in'))
    tD = entrance(c, D, 3.2, D.spec.get('entrance', 'walk_in'))
    t_f = max(4.7, tA + 0.15, tD + 0.15)
    guard(A, tA)
    guard(D, tD)
    w.words.append(T.Word('FIGHT!', t_f, t_f + 0.7, 240, 70, 'big', T.FIRE, drop=12))
    w.shake(t_f, 2.5, 0.3)
    w.flash(t_f, 0.22, 0.1)
    w.cue('say', t=t_f, text='FIGHT!', voice='announcer')
    w.cue('music', t=t_f, section='fight')
    w.mark('title', 1.6)
    w.mark('fight', t_f)
    return t_f + 0.2


def entrance(c, f, t, how):
    """How a fighter appears under the title. Returns when he is in guard."""
    w = c.w
    x, z = f.x_init, f.z_init
    g = floor_at(c, x, z, t)
    if how == 'fire_burst':
        f.hide.append((0.0, t))
        w.at(t - 0.04, lambda world: (mg.fire_puff(world, x, z, g, c.seed(), 1.3), world.cue('teleport_in', x=x)))
        f.add(t, 'superhero', 0.9, face=f.face_init)
        element_hands(c, f.name, t + 0.1, t + 1.7, 'fire')
        return t + 0.9
    if how == 'drop_down':
        # from a perch or high above: a flip down into a crouch, his element bursting round him
        h0 = 70.0 * leap_k(f)
        f.add(0.0, 'taunt', t, face=f.face_init, air=hold_y(g + h0), to=(x - f.face_init * 6, z - 20))
        f.hide.append((0.0, 1.2))
        jump_to(c, f, t, (x, z), 0.62, 22, y1=g, move='flip', settle=False, face=f.face_init)
        f.add(t + 0.62, 'superhero', 0.8, face=f.face_init)
        el = f.spec.get('element')

        def land(world):
            if el == 'ice':
                fx.burst(world, fx.FROST, x, floor_y(z) - g - 2, z, 60, (30, 110), angle=-math.pi / 2, spread=1.5,
                         life=(0.4, 0.9), seed=c.seed())
                mg.ice_coat(world, x, z, 10.0, c.seed(), h=g, k=0.35)
                world.cue('freeze', x=x)
            else:
                fx.burst(world, fx.DUST, x, floor_y(z) - g - 2, z, 40, (30, 110), angle=-math.pi / 2, spread=1.5,
                         life=(0.4, 0.9), size=(1.5, 3), seed=c.seed())
                world.cue('hulk_land' if f.heavy else 'land', x=x, power=2.0)
                if f.heavy:
                    crater(c, x, z, t + 0.62, 1.6, g=g)
            world.effects.append(fx.Ring(x, floor_y(z) - g, t + 0.62, 4, 26, 0.3, 'bfe8ff' if el == 'ice' else 'e0d8c8',
                                         squash=0.6))
        w.at(t + 0.62, land)
        if el in ('ice', 'fire'):
            element_hands(c, f.name, t + 0.7, t + 1.4, el)
        return t + 1.42
    # walk in: from off to the side, guard up
    side = -f.face_init
    f.add(0.0, 'stalk', t + 0.8, to=(x, z), face=f.face_init)
    f.x_init = x + side * 60
    return t + 0.8


def win_card(c, winner, t):
    w = c.w
    W_ = w.f[winner]
    W_.add(t, 'victory', 6.0)
    name = W_.spec['name'] + ' WINS'
    font = 'name'
    w.words.append(T.Word(name, t + 0.3, 999.0, 240, 76, font, T.GOLD, drop=6))
    w.shake(t + 0.3, 1.5, 0.2)
    w.cue('say', t=t + 0.3, text=name, voice='announcer')
    w.cue('music', t=t + 0.3, section='end')
    w.mark('wins', t + 0.3)
    return t + 5.0


# ================================================================== hand to hand
@beat('opening', 'weapons', 'magic', 'escalation', 'travel', weight=5.0)
def exchange(c, A, D, t, n=None, end='knock'):
    """A trade at close range: A's string (light blows, some blocked), D answers with his own (some blocked), A
    finishes with a heavy blow and D reels back. Armed, it is a cut instead."""
    rng = c.rng
    t = ensure_room(c, A, D, t, 50 * power(A))
    t = engage(c, A, D, t)
    if armed(A, t):
        return blade_cut(c, A, D, t)
    n = n or int(rng.integers(3, 5))
    t = string(c, A, D, t, n, final=False)
    if not D.heavy or rng.uniform() < 0.5:
        if armed(D, t):
            t = blade_cut(c, D, A, t)
        else:
            t = string(c, D, A, t + 0.05, int(rng.integers(2, 4)), final=False)
    t = engage(c, A, D, t)
    t = string(c, A, D, t + 0.05, int(rng.integers(1, 3)), final=True)
    return t + 0.12


def string(c, A, D, t, n, final=True):
    """n blows from A's kit; light ones may be blocked, the last (if final) is heavy and knocks D back."""
    rng = c.rng
    pl = lands(c, A.name)
    for i in range(n):
        last = final and i == n - 1
        kind = 'heavy' if last or (i > 1 and rng.uniform() < 0.3) else 'light'
        nm = choose_blow(c, A, kind)
        where = WHERE.get(nm, 'head')
        blocked = (not last) and rng.uniform() > pl and not D.heavy
        if blocked:
            where = 'guard'
        knock = (9 if last else 3 + 1.5 * i) * (1.3 if kind == 'heavy' else 1.0)
        size = 9 if last else 7 + i * 0.5
        wd = None if blocked else wound('head' if where in ('head', 'chin') else 'torso', 'stain',
                                        1.2 + 0.3 * rng.uniform(), rng)
        strike(c, A.name, D.name, t, nm, where, block=blocked, knock=knock, size=size,
               blood=0 if blocked else int(4 + 2 * i + (5 if last else 0)), spark='big' if last else 'hit',
               shake=2.5 if last else 1.5, react=REACT.get(where, 'hit_head'), rdur=0.45 if last else 0.3, wound=wd)
        t = t + DUR[nm] * (0.78 if not last else 1.0) + float(rng.uniform(0.0, 0.05))
    return t


@beat('opening', 'escalation', 'magic', weight=1.3)
def counter(c, A, D, t):
    """D swings big and misses (A ducks or slips aside); A punishes with a launcher and D goes down."""
    rng = c.rng
    t = engage(c, D, A, t)
    big = choose_blow(c, D, 'heavy')
    th = whiff(c, D.name, A.name, t, big, WHERE.get(big, 'head'))
    dodge = 'duck' if WHERE.get(big, 'head') in ('head', 'chin') else 'hop_back'
    if dodge == 'duck':
        A.add(th - 0.16, 'duck', 0.42, blend=0.04)
    else:
        A.add(th - 0.2, 'hop_back', 0.36, dx=-10, air=acts.rel(A, th - 0.2, 7))
    t2 = string(c, A, D, th + 0.22, int(rng.integers(2, 4)), final=False)
    lau = choose_blow(c, A, 'launcher')
    th = strike(c, A.name, D.name, t2, lau, 'chin', react=None, blood=12, size=10, spark='big', shake=3.0,
                flash=0.1, wound=wound('head', 'stain', 1.8, rng))
    s = sd(A, D, th)
    dist = 34 * power(A) * (power(D) ** -0.6)
    to = (clamp_x(c, D.x_at(th) + s * dist, 6), clamp_z(c, D.z_at(th) + float(rng.uniform(-8, 8)), 4))
    tumble_to(c, D, th, to, 0.62, 26 * min(1.6, power(A)), y1=c.w.stage.ground(to[0], to[1], th, D.y_at(th) + 2))
    t_up = fall_lie_up(c, D, th + 0.62, -8 * s * 0, 0.3, 0.5)
    smear_floor(c, D.name, th + 0.62, th + 1.0)
    A.add(th + 0.45, 'beckon', max(0.3, t_up - th - 0.45))
    guard(A, t_up)
    c.w.mark('counter', th, key=False)
    return t_up + 0.05


@beat('opening', 'escalation', weight=1.2, needs=lambda c, A, D, t: bool(A.spec['strikes']['air']))
def juggle(c, A, D, t):
    """A launcher, an air blow on the way down, a bounce off the floor."""
    rng = c.rng
    t = ensure_room(c, A, D, t, 60)
    t = engage(c, A, D, t)
    t = string(c, A, D, t, int(rng.integers(2, 4)), final=False)
    lau = choose_blow(c, A, 'launcher')
    th = strike(c, A.name, D.name, t, lau, 'chin', react=None, blood=12, size=10, spark='big', shake=3.0, flash=0.12,
                wound=wound('head', 'stain', 1.8, rng))
    s = sd(A, D, th)
    hgt = 34 * min(1.5, power(A) / max(0.6, power(D) ** 0.5))
    tumble_to(c, D, th, (clamp_x(c, D.x_at(th) + s * 10, 4), D.z_at(th) + 2), 0.75, hgt)
    A.add(th + 0.12, 'jump', 0.24, air=arc(14 * leap_k(A), A.y_at(th), A.y_at(th) + 18), blend=0.04)
    air = choose_blow(c, A, 'air')
    g = c.w.stage.ground(A.x_at(th), A.z_at(th), th, A.y_at(th) + 2)
    t2 = strike(c, A.name, D.name, th + 0.3, air, 'chest', react=None, blood=9, size=9, settle=False, land=g,
                wound=wound('torso', 'tear', 1.6, rng))
    to = (clamp_x(c, D.x_at(t2) + s * 30 * power(A), 4), clamp_z(c, D.z_at(t2) - 6, 4))
    tumble_to(c, D, t2, to, 0.62, 14)
    guard(A, t2 + 0.3)
    t_up = fall_lie_up(c, D, t2 + 0.62, -10, 0.3, 0.5)
    smear_floor(c, D.name, t2 + 0.62, t2 + 1.0)
    A.add(t2 + 0.5, 'beckon', max(0.3, t_up - (t2 + 0.5)))
    guard(A, t_up)
    c.w.mark('juggle', t2)
    return t_up + 0.05


@beat('opening', 'escalation', weight=1.0, needs=lambda c, A, D, t: bool(A.spec['strikes']['air']) and
      bool(D.spec['strikes']['leap']) and bool(A.spec['strikes']['leap']) and area(c).get('size') != 'narrow')
def air_clash(c, A, D, t):
    """Both leap at once: the flying kicks meet in mid-air with a flash; both are flipped back."""
    w = c.w
    s = sd(A, D, t)
    mid = clamp_x(c, (A.x_at(t) + D.x_at(t)) / 2, 60)
    zm = clamp_z(c, (A.z_at(t) + D.z_at(t)) / 2, 4)
    g = floor_at(c, mid, zm, t)
    A.add(t, 'stalk', 0.45, to=(mid - 48 * s, zm))
    D.add(t, 'stalk', 0.45, to=(mid + 48 * s, zm))
    tk = t + 0.5
    hA, hD = 30 * leap_k(A), 30 * leap_k(D)
    A.add(tk, 'jump_kick', 0.62, to=(mid - 22 * s, zm), ease=lambda u: min(1.0, u / 0.55) ** 0.8,
          air=arc(hA, g, g + hA * 0.6))
    D.add(tk, 'jump_kick', 0.62, to=(mid + 22 * s, zm), ease=lambda u: min(1.0, u / 0.55) ** 0.8,
          air=arc(hD, g, g + hD * 0.6))
    tc_ = tk + 0.62 * 0.55

    def kicks_meet(world):
        a, b = world.joints[A.name]['FT'], world.joints[D.name]['FT']
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        mg.clash_fx(world, x, y, zm, c.seed(), True)
        world.effects.append(fx.Ring(x, y, world.t, 4, 34, 0.3, 'fff0c0'))
        world.flash(tc_, 0.2, 0.08)
        world.cue('hit', x=mid, weight=2, limb='kick', blood=0)
        world.jolt(A.name, 4.0)
        world.jolt(D.name, 4.0)
        world.count['hits'] += 1
    w.at(tc_, kicks_meet)
    w.slowmo(tc_ - 0.03, tc_ + 0.12, 0.3)
    for f, sg, o in ((A, -s, D), (D, s, A)):
        back = 70 * power(o) / max(0.7, power(f) ** 0.5)
        to = (clamp_x(c, mid + sg * back, 4), zm)
        y0 = f.y_at(tc_ + 0.02)
        f.add(tc_ + 0.02, 'back_flip', 0.5, to=to, air=lambda u, y0=y0: y0 * (1 - u) + g * u + 14 * math.sin(math.pi * u))
        guard(f, tc_ + 0.52)
    w.trail(A.name, tk, tc_ + 0.5)
    w.trail(D.name, tk, tc_ + 0.5)
    w.mark('kicks_meet', tc_)
    return tc_ + 0.62


@beat('opening', 'escalation', weight=1.2, needs=lambda c, A, D, t: bool(D.spec['strikes']['low']) and
      bool(A.spec['strikes']['air']))
def sweep_jump(c, A, D, t):
    """D's sweep is jumped; A's flying kick spins him round and he goes down."""
    rng = c.rng
    t = engage(c, D, A, t)
    low = choose_blow(c, D, 'low')
    if low not in ('sweep',):
        low = 'sweep'
    th_sw = whiff(c, D.name, A.name, t, low, 'legs')
    g = c.w.stage.ground(A.x_at(t), A.z_at(t), t, A.y_at(t) + 2)
    A.add(th_sw - 0.18, 'jump', 0.34, air=arc(14 * leap_k(A), g, g + 18), blend=0.03)
    air = choose_blow(c, A, 'air')
    th = strike(c, A.name, D.name, th_sw + 0.14, air, 'head', react=None, blood=11, size=10, spark='big', shake=3.0,
                land=g, wound=wound('head', 'stain', 1.6, rng))
    s = sd(A, D, th)
    to = (clamp_x(c, D.x_at(th) + 34 * s * power(A), 4), clamp_z(c, D.z_at(th) - 8, 4))
    tumble_to(c, D, th, to, 0.5, 16)
    t_up = fall_lie_up(c, D, th + 0.5, -8, 0.25, 0.5)
    A.add(th + 0.5, 'beckon', max(0.3, t_up - (th + 0.5)))
    guard(A, t_up)
    c.w.mark('jump_over', th, key=False)
    return t_up + 0.05


KNOCK_KINDS = ('brazier', 'pillar', 'gong', 'car', 'lamp', 'wall', 'column', 'table', 'counter', 'window', 'hydrant',
               'pole')


@beat('opening', 'weapons', 'escalation', 'magic', weight=1.6,
      needs=lambda c, A, D, t: any(x.kind in KNOCK_KINDS for x in c.w.stage.targets(c.area, t)))
def knock_into(c, A, D, t, kind=None):
    """A big blow throws D into something in the area (a brazier, a pillar, a car, a wall): it breaks."""
    rng = c.rng
    w = c.w
    tg = c.target_for(D, t, kind, kinds=None if kind else KNOCK_KINDS)
    if tg is None:
        return exchange(c, A, D, t)
    # stand so the blow sends him at the target: A on the far side of D from it
    s = 1 if tg.x > D.x_at(t) else -1
    xa = clamp_x(c, D.x_at(t) - s * fdist(A, D), 6)
    if abs(A.x_at(t) - xa) > 8 or sd(A, D, t) != s:
        dur = 0.5
        A.add(t, 'stalk', dur, to=(xa, D.z_at(t)), face=s)
        D.add(t, guard_move(D, t), dur, face=-s)
        t += dur
    hand, armed_ = held(A, t)
    if not armed_:
        t = string(c, A, D, t, int(rng.integers(2, 4)), final=False)
    if armed_ and armed_ != 'hammer' and A.wep_at(t, 'f')[0] and A.wep_at(t, 'n')[0]:
        nm, wpn = 'x_slash', ('n', armed_)
    elif armed_ == 'hammer':
        nm, wpn = 'hammer_swing', ('n', 'hammer')
    elif armed_:
        nm, wpn = 'slash_across', (hand, armed_)
    else:
        nm, wpn = choose_blow(c, A, 'heavy'), None
    th = strike(c, A.name, D.name, t, nm, 'chest' if wpn else WHERE.get(nm, 'head'), react=None, blood=12, size=11,
                spark='big', shake=3.5, wep=wpn, settle_move=guard_move(A, t), wound=wound('torso', 'tear', 1.8, rng),
                mp={'hand': wpn[0]} if wpn and nm == 'slash_across' else None)
    dist = abs(tg.x - D.x_at(th))
    fly = max(0.35, min(0.8, dist / 160.0))
    hit_x = tg.x - s * 8
    tumble_to(c, D, th, (hit_x, tg.z + 3), fly, 14 + dist * 0.08)
    t_hit = th + fly
    pw = 1.2 * power(A)
    if wpn and wpn[1] == 'hammer':
        pw *= 1.7
    tl = tg.smash(w, t_hit, pw, s, c.seed())
    w.mark(f'into_{tg.kind}', t_hit)
    if tg.kind in ('pillar', 'wall', 'gong', 'column', 'car'):
        D.add(t_hit, 'slam_wall', 0.6, air=lambda u, y0=D.y_at(t_hit): y0 * (1 - u) ** 1.6, blend=0.02)
        D.add(t_hit + 0.6, 'sit_up', 0.6)
        t_up = t_hit + 1.2
    else:
        t_up = fall_lie_up(c, D, t_hit, -6 * s, 0.3, 0.5)
    if tg.kind == 'brazier':
        fire_body(c, D.name, t_hit, t_hit + 1.0, rate=4)
    guard(D, t_up)
    A.add(th + 0.4, 'beckon', max(0.3, t_up - th - 0.4))
    guard(A, t_up)
    return max(t_up, (tl or 0)) + 0.05


# ================================================================== weapons
def has_weapon(f, kind=None):
    ws = f.spec.get('weapons', [])
    return bool(ws) if kind is None else any(x['kind'] == kind for x in ws)


def armed(f, t):
    return f.wep_at(t, 'n')[0] is not None or f.wep_at(t, 'f')[0] is not None


def held(f, t):
    """(hand, weapon) he fights with: the near hand's, else the far hand's; (None, None) if unarmed."""
    for side in ('n', 'f'):
        k = f.wep_at(t, side)[0]
        if k:
            return side, k
    return None, None


@beat('weapons', weight=0.0, needs=lambda c, A, D, t: has_weapon(A) and not armed(A, t), pool=False)
def draw_weapons(c, A, D, t, kind=None):
    """Weapons out: A draws (steel from his back) or forms his (out of his element); D answers with his own."""
    w = c.w
    t = circle(c, A, D, t, dur=0.7)
    tA = arm(c, A, t, first=True, kind=kind)
    tD = arm(c, D, t + 0.25, first=True) if has_weapon(D) else t
    w.cue('music', t=t + 0.4, section='fury')
    t = max(tA, tD) + 0.25
    if armed(A, t) and armed(D, t):
        # steel out, straight into it: blade on blade across the area
        return blade_exchange(c, A, D, t, n=int(c.rng.integers(4, 6)))
    return t


def arm(c, f, t, first=True, kind=None):
    """f takes up his next weapon (the first, or `kind`): draws it or forms it. Returns when he is in guard."""
    w = c.w
    ws = f.spec.get('weapons', [])
    if not ws:
        return t
    spec = ws[0] if kind is None else next((x for x in ws if x['kind'] == kind), ws[0])
    k = spec['kind']
    if spec.get('how') == 'draw':
        f.add(t, 'draw_swords', 1.0)
        td = t + 1.0 * moves.DRAW_AT
        if spec.get('hands') == 'both':
            wep(f, td, 'f', k)
        wep(f, td, 'n', k)
        f.backs.append((td, 0))
        w.cue('sword_draw', t=td, who=f.name)
        glint(c, f.name, td + 0.25, 'f')
        glint(c, f.name, td + 0.3, 'n')
        guard(f, t + 1.0)
        w.mark('swords' if k == 'sword' else 'draw', td + 0.25)
        return t + 1.0
    # formed out of his element
    big = spec.get('heavy', False)
    dur = 0.9 if big else 1.05
    f.add(t, 'form_weapon', dur)
    element_hands(c, f.name, t + 0.05, t + 0.6, spec.get('element', 'ice'))
    wep(f, t + dur * (0.5 if big else 0.55), 'n', k, grow=0.45 if big else 0.4)
    w.cue('ice_form', t=t + dur * 0.45, who=f.name, big=big)
    guard(f, t + dur)
    w.mark(k, t + dur - 0.1)
    return t + dur + 0.1


@beat('weapons', 'escalation', 'magic', weight=4.5, needs=lambda c, A, D, t: armed(A, t) and armed(D, t))
def blade_exchange(c, A, D, t, n=None):
    """Blade on blade across the area: clashes alternate, sparks every time."""
    rng = c.rng
    t = ensure_room(c, A, D, t, 40)
    s = sd(A, D, t)
    t = engage(c, A, D, t, dist=fdist(A, D) + 10)
    n = n or int(rng.integers(4, 7))
    cuts = ['slash_down', 'slash_across', 'slash_up', 'thrust']
    att, dfn = A, D
    for i in range(n):
        nm = pick(c, cuts, avoid=last_blows(c, att.name)[-1:])
        note_blow(c, att.name, nm)
        if att.wep_at(t, 'n')[0] == 'hammer':
            nm = 'hammer_swing'
        ah = 'f' if (att.wep_at(t, 'f')[0] and (i % 2 == 1 or not att.wep_at(t, 'n')[0])) else 'n'
        dh = 'f' if (dfn.wep_at(t, 'f')[0] and (nm == 'slash_across' or not dfn.wep_at(t, 'n')[0])) else 'n'
        clash(c, att.name, dfn.name, t, nm, ah=ah, dh=dh, dist=28, big=(i == 0 or i == n - 1),
              high=nm != 'slash_up', push=4 + 1.5 * (i == n - 1))
        t += float(rng.uniform(0.42, 0.5))
        att, dfn = dfn, att
    c.w.mark('clashes', t - 0.3, key=False)
    return t + 0.15


@beat('weapons', 'escalation', weight=2.0, needs=lambda c, A, D, t: armed(A, t))
def blade_cut(c, A, D, t):
    """A's blade gets through (an X cut with two swords); if D is armed he answers across A's arm."""
    rng = c.rng
    t = ensure_room(c, A, D, t, 40)
    t = engage(c, A, D, t)
    hand, k = held(A, t)
    two = A.wep_at(t, 'f')[0] is not None and A.wep_at(t, 'n')[0] is not None
    if k == 'hammer':
        nm = pick(c, ['hammer_swing', 'hammer_upper'])
    elif two:
        nm = pick(c, ['x_slash', 'slash_down', 'slash_up', 'slash_across'], avoid=last_blows(c, A.name)[-1:])
    else:
        nm = pick(c, ['slash_down', 'slash_across', 'thrust'], avoid=last_blows(c, A.name)[-1:])
    note_blow(c, A.name, nm)
    ice = k in ('ice_sword', 'hammer')
    if k != 'hammer' and nm != 'thrust':
        strike(c, A.name, D.name, t, 'thrust', 'chest', wep=(hand, k), react='hit_body', rdur=0.3, knock=4, blood=8,
               size=8, settle_move=guard_move(A, t), wound=wound('torso', 'cut', 1.4, rng), mp={'hand': hand})
        t += DUR['thrust'] * 0.9
    th = strike(c, A.name, D.name, t, nm, 'chest', wep=(hand, k), react='hit_body', rdur=0.4, knock=9, blood=14,
                size=9 + (2 if k == 'hammer' else 0), shake=2.5, settle_move=guard_move(A, t),
                wound=[wound('torso', 'cut', 2.2, rng)] + ([wound('torso', 'cut', 1.8, rng)] if nm == 'x_slash' else []),
                mp={'hand': hand} if nm not in ('x_slash', 'hammer_swing', 'hammer_upper') else None)
    c.w.mark('x_cut' if nm == 'x_slash' else 'cut', th, key=nm == 'x_slash')
    if armed(D, th):
        hd, kd = held(D, th)
        nm2 = 'hammer_swing' if kd == 'hammer' else 'slash_across'
        th2 = strike(c, D.name, A.name, th + 0.42, nm2, 'shoulder', wep=(hd, kd), react='hit_head', rdur=0.4, knock=8,
                     blood=12, size=9, shake=2.5, settle_move=guard_move(D, th + 0.42),
                     wound={'part': 'narm_u', 't': 0.5, 's': 0.0, 'r': 2.0, 'kind': 'cut'},
                     mp={'hand': hd} if nm2 == 'slash_across' else None)
        drip(c, A.name, th2, 300.0, 0.8, 'arm')
        return th2 + 0.5
    return th + 0.55


@beat('weapons', weight=1.2, needs=lambda c, A, D, t: armed(A, t) and armed(D, t) and
      A.wep_at(t, 'n')[0] != 'hammer' and D.wep_at(t, 'n')[0] != 'hammer' and area(c).get('size') != 'narrow')
def blade_lock(c, A, D, t):
    """Blades locked, grinding; they hop apart and run at each other; the blades meet as they pass and the cut
    opens on D's arm a beat later."""
    w = c.w
    t = engage(c, A, D, t, dist=fdist(A, D) - 6)
    s = sd(A, D, t)
    A.add(t, 'lock', 0.7, dx=(D.x_at(t) - A.x_at(t)) * s - 24, ease=moves.out, hand='n', face=s)
    D.add(t, 'lock', 0.7, dx=-2, ease=moves.out, hand='n', face=-s)

    def lock_sparks(world, tt, dt):
        pa = world.wtips.get(A.name, {}).get('n')
        pd = world.wtips.get(D.name, {}).get('n')
        if pa and pd:
            x = (pa[1][0] + pd[1][0]) / 2
            y = (pa[1][1] + pd[1][1]) / 2
            fx.burst(world, fx.SPARK, x, y, world.f[A.name].z, 2, (30, 90), angle=-math.pi / 2, spread=1.3,
                     life=(0.1, 0.3), seed=c.seed())
    w.effects.append(fx.Emitter(t + 0.15, t + 0.7, lock_sparks))
    w.cue('blade_lock', t=t + 0.1, dur=0.6, x=A.x_at(t) + 12 * s)
    w.mark('lock', t + 0.4)
    tb = t + 0.72
    A.add(tb, 'hop_back', 0.36, dx=-26, air=acts.rel(A, tb, 7))
    D.add(tb, 'hop_back', 0.36, dx=-26, air=acts.rel(D, tb, 7))
    guard(A, tb + 0.36)
    guard(D, tb + 0.36)
    tr = tb + 0.62
    xm = clamp_x(c, (A.x_at(tr) + D.x_at(tr)) / 2, 50)
    zm = (A.z_at(tr) + D.z_at(tr)) / 2
    A.add(tr, 'dash', 0.34, to=(xm + 48 * s, zm + 4), face=s)
    D.add(tr, 'dash', 0.34, to=(xm - 48 * s, zm - 4), face=-s)
    w.trail(A.name, tr, tr + 0.34)
    w.trail(D.name, tr, tr + 0.34)
    A.add(tr + 0.34, 'slash_across', 0.38, hand='n', face=s)
    D.add(tr + 0.34, 'slash_across', 0.38, hand='n', face=-s)
    tx_ = tr + 0.17

    def pass_clash(world):
        mg.clash_fx(world, xm, floor_y(zm) - floor_at(c, xm, zm, tx_) - 30, zm, c.seed(), True)
        world.cue('clash', x=xm, big=True, ice=True)
        world.flash(tx_, 0.2, 0.08)
    w.at(tx_, pass_clash)
    w.slowmo(tx_ - 0.02, tx_ + 0.1, 0.25)
    A.add(tr + 0.72, guard_move(A, tr), 0.3, face=-s)
    D.add(tr + 0.72, guard_move(D, tr), 0.3, face=s)
    t_open = tr + 0.85

    def opens(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        x, y = j['FS']
        fx.burst(world, fx.BLOOD, x, y, f.z, 16, (40, 140), angle=-math.pi / 2, spread=1.0, life=(0.8, 1.5),
                 size=(0.8, 2.2), seed=c.seed())
        f.st.wounds.append({'part': 'farm_u', 't': 0.6, 's': 0.0, 'r': 2.0, 'kind': 'cut', 'seed': c.seed()})
        f.flash_t = t_open
        world.jolt(D.name, 4.0)
        world.cue('cut', who=D.name, weight=1, blood=16)
        world.count['hits'] += 1
        world.count['cuts'] += 1
    w.at(t_open, opens)
    D.add(t_open, 'hit_body', 0.4, blend=0.03)
    guard(D, t_open + 0.4)
    w.mark('pass', tx_)
    return t_open + 0.5


@beat('weapons', weight=0.0, needs=lambda c, A, D, t: armed(A, t) and armed(D, t), pool=False)
def disarm(c, A, D, t):
    """D catches A's cut and twists it: one of A's blades spins away and sticks in the floor. With his other blade A
    shatters D's (ice) weapon; D forms his next one in a breath (a heavier one)."""
    w = c.w
    t = engage(c, A, D, t, dist=fdist(A, D) + 4)
    tdis = clash(c, A.name, D.name, t, 'slash_down', ah='n', dh='n', dist=27, big=True)
    A.add(tdis + 0.02, 'disarmed', 0.5, hand='n', blend=0.03)
    kind = A.wep_at(t, 'n')[0]
    wep(A, tdis + 0.02, 'n', None)
    s = sd(A, D, t)
    land_pt = (clamp_x(c, D.x_at(t) + s * 60, 10), clamp_z(c, D.z_at(t) + float(c.rng.uniform(-30, 30)), 8))
    t_land = tdis + 0.75
    gl = floor_at(c, land_pt[0], land_pt[1], t_land)
    c.stuck = (kind, land_pt, gl)

    def fling(world):
        tips = world.wtips.get(A.name, {})
        hand = tips['n'][0] if 'n' in tips else hand_pt(A, tdis)
        b = (land_pt[0], floor_y(land_pt[1]) - gl - 4)
        e = mg.LooseWeapon(kind, hand, b, tdis, t_land, 999.0, h=46, spins=3.0, za=A.z, zb=land_pt[1], seed=c.seed())
        world.effects.append(e)
        c.stuck_fx = e
        world.cue('disarm', who=A.name)
    w.at(tdis + 0.01, fling)
    guard(A, tdis + 0.52)
    w.mark('disarm', tdis + 0.2)
    t = tdis + 0.75
    if A.wep_at(t, 'f')[0] and D.wep_at(t, 'n')[0] in ('ice_sword',):
        th = clash(c, A.name, D.name, t, 'slash_down', ah='f', dh='n', dist=27, big=True)

        def shatter(world):
            tips = world.wtips.get(D.name, {})
            if 'n' in tips:
                hand, tip, _ = tips['n']
                mg.weapon_shatter(world, (hand[0] + tip[0]) / 2, (hand[1] + tip[1]) / 2, world.f[D.name].z, c.seed())
            world.cue('ice_shatter', who=D.name)
        w.at(th + 0.01, shatter)
        wep(D, th + 0.02, 'n', None)
        D.add(th + 0.02, 'disarmed', 0.45, hand='n', blend=0.03)
        w.mark('ice_shatter', th + 0.05)
        heavy = [x for x in D.spec.get('weapons', []) if x.get('heavy')]
        if heavy:
            return arm(c, D, th + 0.5, kind=heavy[0]['kind']) + 0.1
        return th + 0.6
    return t


@beat('weapons', 'escalation', weight=1.6, needs=lambda c, A, D, t: A.wep_at(t, 'n')[0] == 'hammer')
def heavy_smash(c, A, D, t):
    """The hammer comes down where D stood (he springs back): the floor craters (bigger for the stronger); then A
    drives him back towards something standing and the next swing smashes it."""
    w = c.w
    t = engage(c, A, D, t)
    x, z = D.x_at(t + 0.3), D.z_at(t + 0.3)
    g = w.stage.ground(x, z, t, D.y_at(t) + 2)
    th = strike(c, A.name, None, t, 'hammer_smash', at=(x + 4 * sd(A, D, t), floor_y(z) - g - 2, z), wep=('n', 'hammer'),
                settle_move='hammer_guard', quiet=True)
    D.add(th - 0.26, 'back_flip', 0.55, dx=-34 * leap_k(D), air=acts.rel(D, th - 0.26, 22 * leap_k(D)))
    guard(D, th + 0.3)
    crater(c, x + 4 * sd(A, D, t), z, th, 1.3 * power(A) * 1.4, ice=True, g=g)
    w.cue('hammer_floor', t=th, x=x)
    w.mark('smash', th)
    t = th + 0.5
    tg = c.target_for(D, t, None, kinds=('pillar', 'column', 'car', 'wall', 'lamp'), maxd=320)
    if tg is None:
        return t
    s = 1 if tg.x > D.x_at(t) else -1
    # D gives ground until his back is at the target; A follows (blows on the way: a swing, a parry)
    dpos = (tg.x - s * 18, tg.z + 8)
    walk = max(0.6, min(1.4, abs(tg.x - D.x_at(t)) / 140.0))
    A.add(t, 'stalk', walk, to=(tg.x - s * (18 + fdist(A, D) + 14), tg.z + 12), face=s)
    D.add(t, 'stalk', walk, to=dpos, face=-s)
    guard(D, t + walk)
    guard(A, t + walk)
    t += walk + 0.1
    if armed(D, t):
        # two swings on the way, both parried
        dh = 'n' if D.wep_at(t, 'n')[0] else 'f'
        clash(c, A.name, D.name, t, 'hammer_swing', ah='n', dh=dh, dist=30, push=3)
        clash(c, A.name, D.name, t + 0.5, 'hammer_swing', ah='n', dh=dh, dist=30, push=3, big=True)
        t += 1.0
    elif D.look.gear.get('shield') and D.shield_at(t):
        # two swings on the way, both taken on the shield
        strike(c, A.name, D.name, t, 'hammer_swing', 'guard', wep=('n', 'hammer'), block=True, knock=6, size=10,
               settle_move='hammer_guard')
        strike(c, A.name, D.name, t + 0.55, 'hammer_upper', 'guard', wep=('n', 'hammer'), block=True, knock=8, size=11,
               settle_move='hammer_guard')
        t += 1.1
    th = strike(c, A.name, None, t, 'hammer_swing', at=(tg.x - s * 9, floor_y(tg.z) - tg.h, tg.z + 2),
                wep=('n', 'hammer'), settle_move='hammer_guard', quiet=True)
    fl = (clamp_x(c, D.x_at(th) - s * 90, 6), clamp_z(c, D.z_at(th) + 30, 6))
    D.add(th - 0.3, 'flip', 0.62, to=fl, air=arc(36 * leap_k(D), D.y_at(th), floor_at(c, fl[0], fl[1], th)))
    guard(D, th + 0.35)
    tl = tg.smash(w, th, 2.4 * power(A), s, c.seed())
    w.mark('pillar_falls' if tg.kind == 'pillar' else f'smash_{tg.kind}', (tl or th) + 0.1)
    A.add(th + 0.35, 'hop_back', 0.42, dx=-20, air=arc(8, A.y_at(th), A.y_at(th)))
    guard(A, th + 0.8)
    return max(th + 0.9, (tl or 0) - 0.2)


@beat('weapons', weight=0.0, needs=lambda c, A, D, t: getattr(c, 'stuck', None) is not None, pool=False)
def recover_weapon(c, A, D, t):
    """The blade stuck in the floor: A walks to it, kicks it up and catches it."""
    w = c.w
    kind, land_pt, gl = c.stuck
    c.stuck = None
    A.add(t, 'stalk', 0.5, to=(land_pt[0] - 12 * A.facing_at(t), land_pt[1] + 2))
    tk = t + 0.55
    A.add(tk, 'kick_up', 0.5)
    t_kick = tk + 0.5 * moves.KICKUP_AT
    t_catch = t_kick + 0.55
    A.add(t_catch - 0.3, 'catch', 0.6, hand='n')
    hp = hand_pt(A, t_catch, 'n')

    def kick(world):
        if getattr(c, 'stuck_fx', None) is not None:
            c.stuck_fx.t1 = t_kick
        world.effects.append(mg.LooseWeapon(kind, (land_pt[0], floor_y(land_pt[1]) - gl - 4), hp, t_kick, t_catch,
                                            t_catch, h=40, spins=2.0, za=land_pt[1], zb=A.z_at(t_catch), seed=c.seed(),
                                            a0=math.pi / 2 + 0.35, stick_ang=-1.2))
        world.cue('whoosh', who=A.name, speed=1.6, limb='kick')
    w.at(t_kick, kick)
    wep(A, t_catch, 'n', kind)
    glint(c, A.name, t_catch + 0.02, 'n')
    w.at(t_catch, lambda world: world.cue('catch', who=A.name))
    guard(A, t_catch + 0.3)
    w.mark('catch', t_catch)
    D.add(t, 'stalk', 0.8, to=(clamp_x(c, A.x_at(t_catch) + 50 * sd(A, D, t), 6), A.z_at(t_catch)))
    guard(D, t + 0.8)
    return t_catch + 0.35


@beat('weapons', 'magic', weight=1.0, needs=lambda c, A, D, t: A.wep_at(t, 'n')[0] == 'hammer')
def heavy_launch(c, A, D, t):
    """The hammer's uppercut through D's crossed guard throws him across the area."""
    rng = c.rng
    t = ensure_room(c, A, D, t, 80)
    t = engage(c, A, D, t)
    D.add(t + 0.05, 'x_block' if D.wep_at(t, 'f')[0] else 'block', 0.6)
    th = strike(c, A.name, D.name, t, 'hammer_upper', 'guard', wep=('n', 'hammer'), react=None, blood=0, size=10,
                spark='big', shake=3.5, settle_move='hammer_guard', ice=True)
    s = sd(A, D, th)
    far = 120 * power(A) / max(0.7, power(D) ** 0.5)
    to = (clamp_x(c, D.x_at(th) + s * far, 6), clamp_z(c, D.z_at(th) + float(rng.uniform(-10, 16)), 6))
    tumble_to(c, D, th, to, 0.75, 40 * min(1.5, power(A)))
    t_up = fall_lie_up(c, D, th + 0.75, -14 * s, 0.25, 0.5)
    smear_floor(c, D.name, th + 0.75, th + 1.1)
    c.w.mark('launched', th + 0.3)
    return t_up + 0.05


# ================================================================== powers
def hits(c, who, n=1):
    c.w.count['hits'] += n


@beat('opening', 'magic', weight=1.4, needs=lambda c, A, D, t: has(A, 'teleport_strike'))
def teleport_strike(c, A, D, t):
    """A folds into his fire and drops out of it behind D: a blow to the back of the head; D flies."""
    w = c.w
    rng = c.rng
    tt = t
    s0 = sd(A, D, tt)
    xz, zz = D.x_at(tt + 1.0), D.z_at(tt + 1.0)
    bx = xz + s0 * 22
    if not (area(c)['x'][0] + 10 < bx < area(c)['x'][1] - 10):
        bx = xz - s0 * 22
    behind = (bx, zz)
    A.add(tt, 'teleport_out', 0.32, blend=0.04)
    A.hide.append((tt + 0.24, tt + 0.85))
    x0, z0, y0 = A.x_at(tt), A.z_at(tt), A.y_at(tt)
    w.at(tt + 0.22, lambda world: (mg.fire_puff(world, x0, z0, y0, c.seed()), world.cue('teleport', who=A.name)))
    g = w.stage.ground(behind[0], behind[1], tt + 1.1, D.y_at(tt) + 2)
    face = -1 if bx > xz else 1
    A.add(tt + 0.32, 'teleport_out', 0.53, to=behind, ease=lambda u: 1.0 if u > 0.5 else 0.0,
          air=hold_y(g + 26.0), face=face)
    w.at(tt + 0.81, lambda world: (mg.fire_puff(world, behind[0], behind[1], g + 26.0, c.seed(), 0.8),
                                   world.cue('teleport_in', x=behind[0])))
    D.add(tt + 0.3, guard_move(D, tt), 0.6, blend=0.1)
    th = strike(c, A.name, D.name, tt + 0.85, 'air_punch', 'head', react=None, blood=12, size=10, spark='big',
                shake=3.5, land=g, flash=0.1, wound=wound('head', 'stain', 1.6, rng))
    w.mark('teleport', tt + 0.86)
    s = sd(A, D, th)
    tg = c.target_for(D, th, None, kinds=('brazier', 'car', 'lamp', 'table'), dirn=s, maxd=90)
    if tg is not None:
        tumble_to(c, D, th, (tg.x - s * 6, tg.z + 2), 0.62, 18)
        tg.smash(w, th + 0.62, 1.2, s, c.seed())
        if tg.kind == 'brazier':
            fire_body(c, D.name, th + 0.62, th + 1.5, rate=4)
        w.mark(tg.kind, th + 0.62)
        D.add(th + 0.62, 'roll', 0.55, dx=-26, dz=-6, blend=0.03)
        t_end = th + 1.17
        guard(D, t_end)
    else:
        to = (clamp_x(c, D.x_at(th) + s * 50 * power(A), 6), D.z_at(th))
        tumble_to(c, D, th, to, 0.55, 18)
        t_end = fall_lie_up(c, D, th + 0.55, -8, 0.25, 0.5)
    guard(A, tt + 0.85 + DUR['air_punch'])
    return t_end + 0.05


def has(f, beat_name):
    return any(p['beat'] == beat_name for p in f.spec.get('powers', []))


def defence(f, kind):
    return any(p['beat'] == kind and p.get('defence') for p in f.spec.get('powers', []))


@beat('magic', 'escalation', weight=2.0, needs=lambda c, A, D, t: has(A, 'hellfire'))
def hellfire(c, A, D, t):
    """A's fist into the floor: a line of hellfire columns runs at D (a whole wall of them for the stronger). If D can
    raise an ice wall it stops them in steam, and A's fire dash goes through it."""
    w = c.w
    rng = c.rng
    t = circle(c, A, D, t, dur=0.5) if gap(A, D, t) < 70 else t
    if gap(A, D, t) < 70:
        s = sd(A, D, t)
        A.add(t, 'hop_back', 0.4, dx=-34, air=acts.rel(A, t, 10))
        t += 0.45
    A.add(t, 'hellfire', 0.9)
    th = t + 0.9 * moves.HELLFIRE_AT
    element_hands(c, A.name, t, th, 'fire')
    xs, zs = A.x_at(th), A.z_at(th)
    xz, zz = D.x_at(th), D.z_at(th)
    g = w.stage.ground(xs, zs, th, A.y_at(th) + 2)
    pw = power(A)
    hgt = 74.0 * (1.0 + 0.6 * pw)            # FLOW-1: taller columns for the stronger
    walled = defence(D, 'ice_wall')
    if walled:
        wall_x = xs + (xz - xs) * 0.68
        D.add(t + 0.2, 'wall_cast', 0.8)
        t_wall = t + 0.2 + 0.8 * 0.4
        n = 6
        for i in range(n):
            k = (i + 1) / (n + 1)
            w.effects.append(mg.HellColumn(xs + (wall_x - 10 - xs) * k, zs + (zz - zs) * k, th + 0.1 * i, g=g,
                                           height=hgt, seed=c.seed(), width=7 + hgt * 0.03))
        # side walls of fire fan out across the floor: the whole courtyard burns
        for sgn in (-1, 1):
            for i in range(4):
                k = (i + 1) / 5
                w.effects.append(mg.HellColumn(xs + (wall_x - xs) * k * 0.8, clamp_z(c, zs + sgn * 26 * (1 + k), 2),
                                               th + 0.15 + 0.1 * i, g=g, height=hgt * 0.7, seed=c.seed()))
        t_hit_wall = th + 0.1 * n
        wall = mg.IceWall(wall_x - 8, wall_x + 8, zz - 28, zz + 24, t_wall, t_hit_wall + 0.1, t_hit_wall + 1.15,
                          g=g, height=44 + 20 * power(D), seed=c.seed())
        w.effects.append(wall)
        w.cue('ice_wall', t=t_wall, x=wall_x)

        def steam(world, tt, dt):
            fx.burst(world, fx.STEAM, wall_x - 9 * sd(A, D, t), floor_y(zz) - g - 20, zz, 4, (10, 40),
                     angle=-math.pi / 2 - 0.5, spread=0.8, life=(0.6, 1.2), size=(1.2, 2.2), seed=c.seed())
        w.effects.append(fx.Emitter(t_hit_wall, t_hit_wall + 0.9, steam))
        w.cue('steam', t=t_hit_wall, dur=0.9, x=wall_x)
        w.mark('hellfire', th + 0.4)
        w.mark('ice_wall', t_hit_wall + 0.1)
        guard(D, t + 1.0)
        guard(A, t + 0.9)
        return fire_dash(c, A, D, t_hit_wall + 0.95, through=True)
    # nothing stops it: the columns reach him
    n = int(5 + 3 * pw)
    t_end = mg.hellfire_wall(w, xs, zs, xz + (xz - xs) * 0.15, zz, th, n, hgt, c.seed(), g=g, gap=0.09)
    t_reach = th + 0.09 * (n - 2)
    D.add(t_reach, 'hit_body', 0.4, blend=0.03)

    def burnt(world):
        f = world.f[D.name]
        f.flash_t = world.t
        world.jolt(D.name, 5.0)
        f.st.wounds.append(dict(wound('torso', 'stain', 2.0, rng), seed=c.seed()))
        world.cue('fire_burst', who=D.name, size=1)
        world.count['hits'] += 1
    w.at(t_reach, burnt)
    fire_body(c, D.name, t_reach, t_reach + 1.0, rate=5)
    s = sd(A, D, t_reach)
    to = (clamp_x(c, D.x_at(t_reach) + s * 40 * pw, 6), D.z_at(t_reach))
    tumble_to(c, D, t_reach + 0.1, to, 0.6, 30 * pw)
    t_up = fall_lie_up(c, D, t_reach + 0.7, -8, 0.3, 0.5)
    guard(A, t + 0.9)
    w.mark('hellfire', th + 0.4)
    return t_up + 0.05


@beat('magic', 'opening', weight=1.4, needs=lambda c, A, D, t: has(A, 'fire_dash'))
def fire_dash(c, A, D, t, through=False):
    """A flies flat at D in a streak of fire, shoulder first, and rams him."""
    w = c.w
    rng = c.rng
    s = sd(A, D, t)
    td = t
    dur = 0.32
    to = (D.x_at(td + 0.3) - s * 16, D.z_at(td + 0.3))
    A.add(td, 'fire_dash', dur, to=to, ease=lambda u: u ** 1.4, face=s)
    mg.fire_trail(w, A.name, td, td + dur)
    w.trail(A.name, td, td + dur)
    w.cue('fire_dash', t=td, dur=dur, who=A.name)
    th = td + dur

    def rammed(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        x, y = j['C']
        world.effects.append(fx.HitSpark(x, y, th, 12, 'big', c.seed()))
        fx.burst(world, fx.FLAME, x, y, f.z, 40, (40, 140), life=(0.25, 0.55), size=(1, 2.2), seed=c.seed())
        fx.burst(world, fx.BLOOD, x, y, f.z, 14, (60, 170), angle=-0.55 if s > 0 else -(math.pi - 0.55), spread=0.7,
                 life=(0.8, 1.6), size=(0.8, 2.2), seed=c.seed())
        world.flash(th, 0.25, 0.1, 'ffd0a0')
        world.shake(th, 4.5, 0.4)
        f.flash_t = th
        f.st.wounds.append(dict(wound('torso', 'stain', 2.0, rng), seed=c.seed()))
        world.cue('hit', who=D.name, weight=2, limb='punch', blood=14)
        world.cue('fire_burst', who=D.name, size=1)
        world.count['hits'] += 1
    w.at(th, rammed)
    w.slowmo(th, th + 0.012, 0.06, ramp=0.0)
    guard(A, th)
    to2 = (clamp_x(c, D.x_at(th) + s * 64 * power(A), 6), clamp_z(c, D.z_at(th) - 20, 6))
    tumble_to(c, D, th, to2, 0.6, 20)
    fire_body(c, D.name, th, th + 0.9, rate=4)
    t_up = fall_lie_up(c, D, th + 0.6, -8, 0.25, 0.5)
    w.mark('fire_dash', th)
    if through:
        return t_up + 0.05
    # he is on him as he gets up: a burning string
    t2 = engage(c, A, D, t_up - 0.1)
    return string(c, A, D, t2, int(rng.integers(2, 4)), final=True) + 0.2


@beat('magic', 'escalation', weight=1.6, needs=lambda c, A, D, t: has(A, 'fire_tornado') and A.tier >= 3)
def fire_tornado(c, A, D, t):
    """FLOW-1: A whirls his fire up round D: a tornado of hellfire tall as the hall lifts him, burns him and throws
    him out."""
    w = c.w
    rng = c.rng
    if gap(A, D, t) < 60:
        A.add(t, 'hop_back', 0.4, dx=-30, air=acts.rel(A, t, 10))
        t += 0.45
    A.add(t, 'hellfire', 0.9)
    th = t + 0.9 * moves.HELLFIRE_AT
    element_hands(c, A.name, t, th + 1.2, 'fire')
    x, z = D.x_at(th), D.z_at(th)
    g = w.stage.ground(x, z, th, D.y_at(th) + 2)
    pw = power(A)
    hgt = 90.0 * pw
    t1 = th + 1.6
    w.effects.append(mg.FireTornado(x, z, g, th, t1, hgt, 9 + 4 * pw, c.seed()))
    w.cue('hellfire', t=th, x=x)
    w.cue('fire_breath', t=th + 0.1, dur=1.4, x=x)
    w.shake(th, 3.0, 1.2)
    # he is lifted spinning in it, then thrown out
    D.add(th + 0.1, 'tumble', 1.1, air=lambda u, g=g: g + 34 * pw * math.sin(math.pi * min(1.0, u * 0.9)),
          ease=lambda u: u, dx=0)
    fire_body(c, D.name, th + 0.1, th + 2.4, rate=7, big=True)

    def scorch(world):
        f = world.f[D.name]
        f.flash_t = world.t
        world.jolt(D.name, 6.0)
        f.st.char = max(f.st.char, 0.25)
        f.st.wounds.append(dict(wound('torso', 'stain', 2.2, rng), seed=c.seed()))
        world.count['hits'] += 1
        world.cue('fire_burst', who=D.name, size=2)
        world.shake(world.t, 2.5, 0.2)
    for k in range(3):                       # the tornado strikes him three times as he spins in it
        w.at(th + 0.35 + 0.3 * k, scorch)

    def unchar(world):
        world.f[D.name].st.char = 0.0
    w.at(th + 3.0, unchar)
    s = sd(A, D, th)
    to = (clamp_x(c, x + s * 70 * pw, 6), clamp_z(c, z + float(rng.uniform(-20, 20)), 6))
    tumble_to(c, D, th + 1.2, to, 0.6, 20, y0=g + 30 * pw * math.sin(math.pi * 0.98))
    t_up = fall_lie_up(c, D, th + 1.8, -8, 0.35, 0.6, slow=True)
    A.add(th + 0.6, 'beckon', max(0.3, t_up - th - 0.6))
    guard(A, t_up)
    w.mark('fire_tornado', th + 0.6)
    return t_up + 0.05


@beat('opening', 'magic', weight=1.4, needs=lambda c, A, D, t: has(A, 'ice_ball'))
def ice_ball(c, A, D, t):
    """A throws an ice ball: D flips over it (it bursts on the floor) or takes it on his shield or in the chest."""
    w = c.w
    rng = c.rng
    if gap(A, D, t) < 60:
        A.add(t, 'hop_back', 0.36, dx=-30, air=acts.rel(A, t, 8))
        guard(A, t + 0.36)
        t += 0.4
    A.add(t, 'ice_cast', 0.6, ease=moves.out)
    element_hands(c, A.name, t + 0.05, t + 0.6 * moves.ICE_RELEASE, 'ice')
    rel_ = t + 0.6 * moves.ICE_RELEASE
    guard(A, t + 0.6)
    hx, hy = hand_pt(A, rel_, 'n')
    dodge = D.spec.get('tier', 3) <= 2 or rng.uniform() < 0.5
    if D.look.gear.get('shield'):
        dodge = False
    tz = D.z_at(rel_ + 0.3)
    if dodge:
        end = (D.x_at(rel_ + 0.3) + 4 * sd(A, D, t), floor_y(tz) - floor_at(c, D.x_at(rel_), tz, rel_) - 2)
    else:
        end = world_point(D, rel_ + 0.3, 'chest')
    t_end = rel_ + math.hypot(end[0] - hx, end[1] - hy) / 300.0
    w.effects.append(fx.IceBall((hx, hy), end, rel_, t_end, c.seed(), za=A.z_at(rel_), zb=tz))
    w.cue('ice_cast', t=rel_, who=A.name)
    w.cue('ice_fly', t=rel_, dur=t_end - rel_, who=A.name, to_x=end[0])
    if dodge:
        D.add(rel_ + 0.05, 'back_flip', 0.55, air=arc(28 * leap_k(D), D.y_at(rel_), D.y_at(rel_)),
              dx=-6, blend=0.04)
        guard(D, rel_ + 0.62)
        g = floor_at(c, end[0], tz, t_end)

        def shatter(world):
            fx.burst(world, fx.ICE, end[0], end[1], tz, 26, (40, 140), angle=-math.pi / 2, spread=1.3, life=(0.6, 1.4),
                     size=(1, 2.2), seed=c.seed())
            fx.burst(world, fx.FROST, end[0], end[1], tz, 24, (20, 80), life=(0.3, 0.8), seed=c.seed())
            world.effects.append(fx.Ring(end[0], end[1], t_end, 3, 18, 0.25, 'bfe8ff', squash=0.6))
            mg.ice_coat(world, end[0], tz, 8.0, c.seed(), h=g, k=0.45)
            world.cue('ice_splash', x=end[0])
        w.at(t_end, shatter)
        w.mark('ice_ball', rel_ + 0.3)
        return max(t_end, rel_ + 0.66) + 0.05
    shield = bool(D.look.gear.get('shield'))
    if shield:
        D.add(t_end - 0.15, 'shield_block', 0.5)
        guard(D, t_end + 0.35)

    def struck(world):
        f = world.f[D.name]
        x, y = end
        fx.burst(world, fx.ICE, x, y, f.z, 30, (40, 160), life=(0.6, 1.3), size=(1, 2.2), seed=c.seed())
        fx.burst(world, fx.FROST, x, y, f.z, 30, (20, 90), life=(0.3, 0.8), seed=c.seed())
        world.effects.append(fx.HitSpark(x, y, t_end, 9, 'block', c.seed()))
        world.shake(t_end, 2.5, 0.25)
        if shield:
            world.cue('shield_block', who=D.name, weight=2, big=True)
            world.count['blocks'] += 1
        else:
            world.cue('ice_hit', who=D.name, weight=2)
            f.flash_t = t_end
            f.st.frozen = 0.6
            world.count['hits'] += 1
    w.at(t_end, struck)
    if not shield:
        w.at(t_end + 0.8, lambda world: setattr(world.f[D.name].st, 'frozen', 0.0))
        D.add(t_end, 'hit_body', 0.8, dx=-12 * power(A), ease=moves.out, blend=0.03)
        guard(D, t_end + 0.8)
    else:
        D.add(t_end, 'shield_brace', 0.4, dx=-14 * power(A), ease=moves.out)
        guard(D, t_end + 0.4)
    w.mark('ice_ball', t_end)
    return t_end + 0.85


@beat('opening', 'magic', weight=1.3, needs=lambda c, A, D, t: has(A, 'ice_slide'))
def ice_slide(c, A, D, t):
    """A shoots along the floor feet first on a track of ice and sweeps D off his feet."""
    w = c.w
    ts = t
    s = sd(A, D, ts)
    if gap(A, D, t) < 50:
        A.add(t, 'hop_back', 0.36, dx=-28, air=acts.rel(A, t, 8))
        guard(A, t + 0.36)
        ts = t + 0.4
    A.add(ts, 'ice_slide', 0.7, to=(D.x_at(ts + 0.35) + 4 * s, D.z_at(ts) + 1), ease=moves.out)
    mg.slide_track(w, A.name, ts + 0.08, ts + 0.6)
    w.cue('ice_slide', t=ts, dur=0.6, who=A.name)
    w.trail(A.name, ts, ts + 0.5)
    th = ts + 0.32

    def swept(world):
        world.cue('hit', who=D.name, weight=1, limb='kick', blood=0)
        f = world.f[D.name]
        g = world.stage.ground(f.x, f.z, th, f.y + 2)
        fx.burst(world, fx.ICE, f.x, floor_y(f.z) - g - 4, f.z, 14, (30, 90), angle=-math.pi / 2, spread=1.2,
                 life=(0.4, 0.9), seed=c.seed())
        world.effects.append(fx.HitSpark(f.x, floor_y(f.z) - g - 5, th, 8, 'block', c.seed()))
        world.shake(th, 2.0, 0.2)
        world.count['hits'] += 1
    w.at(th, swept)
    D.add(th, 'slip', 0.42, air=lambda u, y0=D.y_at(th): y0 + 10 * math.sin(math.pi * min(1.0, u * 1.2)), blend=0.03)
    guard(A, ts + 0.7)
    t_up = fall_lie_up(c, D, th + 0.42, -4, 0.25, 0.5, h=6)
    A.add(ts + 1.0, 'beckon', max(0.3, t_up - (ts + 1.0)))
    guard(A, t_up)
    w.mark('ice_slide', th)
    return t_up + 0.05


@beat('magic', 'escalation', 'opening', weight=1.3, needs=lambda c, A, D, t: has(A, 'ground_wave'))
def ground_wave(c, A, D, t):
    """Both palms slammed on the floor: a wave of ice spikes runs at D (taller for the stronger). D jumps it and
    answers with an axe kick, or it catches him."""
    w = c.w
    rng = c.rng
    if gap(A, D, t) < 50:
        A.add(t, 'hop_back', 0.36, dx=-26, air=acts.rel(A, t, 8))
        t += 0.4
    A.add(t, 'wave_cast', 0.8, ease=moves.out)
    t_wave = t + 0.8 * moves.WAVE_AT
    s = sd(A, D, t_wave)
    xs = A.x_at(t_wave) + 10 * s
    g = w.stage.ground(xs, A.z_at(t_wave), t_wave, A.y_at(t_wave) + 2)
    pw = power(A)
    x1 = clamp_x(c, D.x_at(t_wave) + 30 * s, 0)
    mg.ground_wave(w, xs, A.z_at(t_wave), x1, D.z_at(t_wave), t_wave, 0.45, c.seed(), g=g, spike=11.0 * (0.8 + 0.5 * pw))
    guard(A, t + 0.8)
    w.mark('wave', t_wave + 0.3)
    if D.spec['strikes']['air'] and rng.uniform() < 0.7:
        D.add(t_wave + 0.18, 'flip', 0.55, air=arc(30 * leap_k(D), D.y_at(t_wave), D.y_at(t_wave)), dx=0, blend=0.04)
        ta = t_wave + 0.6
        A.add(ta - 0.1, guard_move(A, ta), 0.2)
        th = strike(c, D.name, A.name, ta, 'axe_kick', 'head', react='hit_head', rdur=0.4, knock=10, blood=10, size=10,
                    spark='big', shake=3.0, land=D.y_at(t_wave), wound=wound('head', 'stain', 2.0, rng))
        t2 = string(c, D, A, th + 0.32, int(rng.integers(2, 4)), final=True)
        return t2 + 0.3
    # caught: frozen to the shins, he takes a blow
    tc_ = t_wave + 0.35

    def caught(world):
        f = world.f[D.name]
        f.st.frozen = 0.5
        world.cue('freeze', who=D.name)
        world.count['hits'] += 1
    w.at(tc_, caught)
    D.add(tc_, 'hit_body', 0.6, blend=0.03)
    w.at(tc_ + 0.9, lambda world: setattr(world.f[D.name].st, 'frozen', 0.0))
    t2 = engage(c, A, D, tc_ + 0.1)
    th = strike(c, A.name, D.name, t2, choose_blow(c, A, 'heavy'), 'head', knock=14, blood=10, size=10, spark='big',
                shake=3.0, wound=wound('head', 'stain', 1.8, rng))
    return th + 0.6


@beat('magic', 'escalation', weight=1.2, needs=lambda c, A, D, t: has(A, 'ice_shower'))
def ice_shower(c, A, D, t):
    """A throws ice into the sky over D; it rains down round him; one shard finds him."""
    w = c.w
    rng = c.rng
    A.add(t, 'shower_cast', 0.8)
    element_hands(c, A.name, t, t + 0.4, 'ice')
    t_rel = t + 0.8 * moves.SHOWER_AT
    sx, sz = D.x_at(t_rel), D.z_at(t_rel)
    w.cue('ice_cast', t=t_rel, who=A.name)
    t_hit = t_rel + 0.95
    hx, hz = clamp_x(c, sx - 16 * sd(A, D, t), 6), clamp_z(c, sz + 10, 4)
    D.add(t_rel + 0.3, 'stalk', 0.35, to=(hx, hz))
    guard(D, t_rel + 0.65)
    pw = power(A)
    mg.ice_shower(w, sx - 6, sz + 4, t_rel + 0.15, int(22 * pw), 34 * pw, c.seed(),
                  hits=[(t_hit - 0.35, hx + 1, hz), (t_hit, hx + 2, hz)], dur=1.3)

    def shard_hit(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        x, y = j['NS']
        world.effects.append(fx.HitSpark(x, y, t_hit, 7, 'cut', c.seed()))
        fx.burst(world, fx.BLOOD, x, y, f.z, 12, (50, 140), angle=-math.pi / 2, spread=1.2, life=(0.8, 1.4),
                 size=(0.8, 2.0), seed=c.seed())
        f.st.wounds.append(dict(wound('torso', 'cut', 1.8, rng), seed=c.seed()))
        f.flash_t = t_hit
        world.jolt(D.name, 5.0)
        world.shake(t_hit, 2.5, 0.25)
        world.cue('ice_hit', who=D.name, weight=1)
        world.count['hits'] += 1
    w.at(t_hit - 0.35, shard_hit)
    w.at(t_hit, shard_hit)
    D.add(t_hit - 0.35, 'hit_head', 0.3, dx=-2, ease=moves.out, blend=0.03)
    D.add(t_hit, 'hit_body', 0.45, dx=-3, ease=moves.out, blend=0.03)
    D.add(t_hit + 0.45, 'flip', 0.55, dx=-30, air=arc(18 * leap_k(D), D.y_at(t_hit), D.y_at(t_hit)))
    guard(D, t_hit + 1.0)
    guard(A, t + 0.8)
    w.mark('shower', t_rel + 0.7)
    return t_hit + 1.05


@beat('magic', 'escalation', weight=2.0, needs=lambda c, A, D, t: has(A, 'ice_spikes'))
def ice_spikes(c, A, D, t):
    """FLOW-1: A drives his palms into the floor and a forest of giant ice spikes bursts out under D (as tall as the
    hall for the strong); D is thrown high, or springs clear if he is quick."""
    w = c.w
    rng = c.rng
    if gap(A, D, t) < 50:
        A.add(t, 'hop_back', 0.36, dx=-30, air=acts.rel(A, t, 8))
        t += 0.4
    A.add(t, 'wave_cast', 0.8, ease=moves.out)
    tw = t + 0.8 * moves.WAVE_AT
    element_hands(c, A.name, t, tw, 'ice')
    x, z = D.x_at(tw + 0.2), D.z_at(tw + 0.2)
    g = w.stage.ground(x, z, tw, D.y_at(tw) + 2)
    pw = power(A)
    hgt = 70.0 * pw * float(rng.uniform(0.95, 1.15))
    n = int(5 + 2 * pw)
    t_sp = tw + 0.25
    mg.floor_freeze(w, A.x_at(tw), A.z_at(tw), g, tw, abs(x - A.x_at(tw)) + 20, c.seed(), dur=0.25)
    mg.ice_spikes(w, x, z, g, t_sp, n, hgt, 26 + 6 * pw, c.seed())
    guard(A, t + 0.8)
    w.mark('ice_spikes', t_sp + 0.12)
    w.slowmo(t_sp + 0.02, t_sp + 0.3, 0.35)
    quick = D.tier <= 2 and rng.uniform() < 0.5
    s = sd(A, D, tw)
    if quick:
        to = (clamp_x(c, x + s * 60, 6), clamp_z(c, z + 20, 4))
        D.add(t_sp - 0.12, 'flip', 0.6, to=to, air=arc(40 * leap_k(D), D.y_at(t_sp), g))
        guard(D, t_sp + 0.5)
        return t_sp + 0.8

    def impaled(world):
        f = world.f[D.name]
        f.flash_t = world.t
        world.jolt(D.name, 6.0)
        j = world.joints[D.name]
        fx.burst(world, fx.BLOOD, j['C'][0], j['C'][1], f.z, 18, (60, 180), angle=-math.pi / 2, spread=0.8,
                 life=(0.8, 1.5), size=(1, 2.4), seed=c.seed())
        f.st.wounds.append(dict(wound('fleg_u', 'cut', 2.0, rng), seed=c.seed()))
        f.st.wounds.append(dict(wound('torso', 'tear', 1.8, rng), seed=c.seed()))
        world.cue('hit', who=D.name, weight=2, limb='kick', blood=18)
        world.cue('ice_hit', who=D.name, weight=2)
        world.count['hits'] += 2
        world.flash(world.t, 0.22, 0.1, 'd8f4ff')
    w.at(t_sp + 0.05, impaled)
    lift = hgt * 0.9
    to = (clamp_x(c, x + s * 50 * pw, 6), clamp_z(c, z + float(rng.uniform(-10, 14)), 6))
    D.add(t_sp + 0.05, 'tumble', 0.9, to=to, ease=lambda u: u,
          air=arc(lift, D.y_at(t_sp), floor_at(c, to[0], to[1], t_sp + 1.0)), blend=0.02)
    w.cue('fly', t=t_sp, dur=0.9, who=D.name)
    w.trail(D.name, t_sp, t_sp + 0.9)
    t_up = fall_lie_up(c, D, t_sp + 0.95, -6, 0.35, 0.55, slow=True)
    smear_floor(c, D.name, t_sp + 0.95, t_sp + 1.3)
    A.add(t_sp + 0.6, 'beckon', max(0.3, t_up - t_sp - 0.6))
    guard(A, t_up)
    return t_up + 0.05


@beat('magic', 'escalation', weight=1.4, needs=lambda c, A, D, t: has(A, 'floor_freeze'))
def floor_freeze(c, A, D, t):
    """FLOW-1: the whole floor freezes from A's palms outward; D's feet go out from under him and A slides in with a
    kick while he is down."""
    w = c.w
    rng = c.rng
    A.add(t, 'wave_cast', 0.8, ease=moves.out)
    tw = t + 0.8 * moves.WAVE_AT
    element_hands(c, A.name, t, tw, 'ice')
    g = w.stage.ground(A.x_at(tw), A.z_at(tw), tw, A.y_at(tw) + 2)
    a = area(c)
    radius = max(a['x'][1] - a['x'][0], 120) * 0.6 * min(1.4, power(A))
    mg.floor_freeze(w, A.x_at(tw), A.z_at(tw), g, tw, radius, c.seed(), dur=0.7)
    guard(A, t + 0.8)
    w.mark('floor_freeze', tw + 0.4)
    tr = tw + gap(A, D, tw) / radius * 0.7 + 0.05
    D.add(tr, 'slip', 0.42, air=lambda u, y0=D.y_at(tr): y0 + 12 * math.sin(math.pi * min(1.0, u * 1.2)), blend=0.03)
    w.at(tr + 0.3, lambda world: (world.cue('bodyfall', who=D.name, power=1.0), world.shake(tr + 0.3, 2.0, 0.2)))
    t_down = tr + 0.42
    D.add(t_down, 'lie', 0.7)
    # the slide in while he is down
    ts = t + 0.85
    s = sd(A, D, ts)
    A.add(ts, 'ice_slide', 0.6, to=(D.x_at(t_down) - 6 * s, D.z_at(t_down) + 1), ease=moves.out)
    mg.slide_track(w, A.name, ts + 0.05, ts + 0.5)
    w.cue('ice_slide', t=ts, dur=0.5, who=A.name)
    th = ts + 0.45

    def kicked(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        world.effects.append(fx.HitSpark(j['C'][0], j['C'][1], th, 10, 'big', c.seed()))
        fx.burst(world, fx.BLOOD, j['C'][0], j['C'][1], f.z, 10, (40, 140), angle=-math.pi / 2, spread=1.0,
                 life=(0.8, 1.4), size=(0.8, 2.0), seed=c.seed())
        f.flash_t = th
        world.jolt(D.name, 5.0)
        world.cue('hit', who=D.name, weight=2, limb='kick', blood=10)
        world.count['hits'] += 1
    w.at(max(th, t_down + 0.1), kicked)
    th = max(th, t_down + 0.1)
    to = (clamp_x(c, D.x_at(th) + s * 50 * power(A), 6), clamp_z(c, D.z_at(th) - 10, 6))
    tumble_to(c, D, th, to, 0.55, 22)
    guard(A, ts + 0.6)
    t_up = fall_lie_up(c, D, th + 0.55, -6, 0.3, 0.5)
    return t_up + 0.05


@beat('magic', weight=1.2, needs=lambda c, A, D, t: has(A, 'spear_batted') and armed(D, t))
def spear_batted(c, A, D, t):
    """A's spear on its rope flies at D, who bats it away with his weapon in a burst of sparks; it is reeled back."""
    w = c.w
    held = A.wep_at(t, 'n')[0]
    A.add(t, 'spear_throw', 0.55, ease=moves.out)
    t_rel = t + 0.55 * moves.SPEAR_RELEASE
    t_hit = t_rel + 0.28
    t_back = t_hit + 0.4
    ksp = mg.DeflectSpear(A.name, D.name, t_rel, t_hit, t_back)
    w.effects.append(ksp)
    A.add(t + 0.55, 'spear_hold', t_back - t - 0.55)
    guard(A, t_back)
    if held:
        wep(A, t, 'n', None)
        wep(A, t_back, 'n', held)
    kd = D.wep_at(t, 'n')[0]
    sw = 'hammer_swing' if kd == 'hammer' else 'slash_across'
    D.add(t_hit - 0.5 * DUR[sw], sw, DUR[sw], hand='n')
    guard(D, t_hit + 0.3)
    w.cue('kunai_throw', t=t_rel, who=A.name)
    w.cue('rope_zip', t=t_rel, dur=t_hit - t_rel, who=A.name)

    def batted(world):
        x, y = ksp.stop(world)
        mg.clash_fx(world, x, y, world.f[D.name].z, c.seed(), True)
        world.cue('clash', x=x, big=True, ice=kd in ('hammer', 'ice_sword'))
    w.at(t_hit, batted)
    w.mark('kunai_batted', t_hit)
    return t_back + 0.35


@beat('magic', weight=0.0, needs=lambda c, A, D, t: has(A, 'ice_clone'), pool=False)
def ice_clone(c, A, D, t):
    """The trap: A leaves a clone of ice where he stands and slips aside; D's blow hits the clone and he freezes solid.
    A hammers the ice. If D has a way out (fire) he bursts free and throws A; otherwise the ice shatters and he falls
    out of it, hurt."""
    w = c.w
    rng = c.rng
    t = engage(c, D, A, t, dist=fdist(A, D) + 30)
    tcl = t
    clone = mg.Clone(w, A.name, tcl + 0.1, tcl + 2.9, c.seed())
    w.effects.append(clone)
    w.cue('clone', t=tcl + 0.1, x=A.x_at(tcl))
    s = sd(D, A, tcl)
    side = (clamp_x(c, A.x_at(tcl) + s * 46, 6), clamp_z(c, A.z_at(tcl) + 30, 6))
    A.add(tcl + 0.12, 'dash', 0.26, to=side)
    w.trail(A.name, tcl + 0.12, tcl + 0.38)
    A.add(tcl + 0.38, guard_move(A, tcl), 3.0)
    w.mark('clone', tcl + 0.2)
    ts = tcl + 0.35
    j0 = rig.solve(A.pose_at(tcl + 0.1), A.look.dims)
    g = w.stage.ground(A.x_at(tcl), A.z_at(tcl), tcl, A.y_at(tcl) + 2)
    cxp = (A.x_at(tcl) + A.facing_at(tcl + 0.1) * A.K * (j0['C'][0] + 3), floor_y(A.z_at(tcl)) - g + A.K * (j0['C'][1] + 4),
           A.z_at(tcl))
    D.add(ts - 0.25, 'dash', 0.22, to=(A.x_at(tcl) - s * 40, A.z_at(tcl)))
    hand_, armed_ = held(D, ts)
    if armed_ and armed_ != 'hammer':
        t_ice = strike(c, D.name, None, ts, 'slash_down', at=cxp, wep=(hand_, armed_), quiet=True, settle=False,
                       mp={'hand': hand_})
        dur_strike = 0.4
    else:
        nm = choose_blow(c, D, 'heavy')
        t_ice = strike(c, D.name, None, ts, nm, at=cxp, quiet=True, settle=False)
        dur_strike = DUR[nm]
    D.add(t_ice + 0.01, 'hold', 2.2, pose=D.pose_at(t_ice), blend=0.0)
    breakout = D.spec.get('breakout')
    t_break = t_ice + 2.0

    def freeze(world):
        f = world.f[D.name]
        f.st.frozen = 1.0
        j = world.joints[D.name]
        pts = [v for k, v in j.items() if isinstance(v, tuple) and len(v) == 2 and k != 'up']
        for s_, (hand, tip, kind) in world.wtips.get(D.name, {}).items():
            pts += [tip]
        gg = world.stage.ground(f.x, f.z, world.t, f.y + 2)
        box = (min(p[0] for p in pts) - 3, min(p[1] for p in pts) - 3, max(p[0] for p in pts) + 3,
               floor_y(f.z) - gg + 1)
        blk = fx.IceBlock(box, t_ice, t_break - 0.7, t_break, c.seed())
        blk.z = f.z
        world.effects.append(blk)
        fx.burst(world, fx.ICE, cxp[0], cxp[1], cxp[2], 30, (40, 160), life=(0.6, 1.4), size=(1, 2.2), seed=c.seed())
        fx.burst(world, fx.FROST, cxp[0], cxp[1], cxp[2], 50, (20, 90), life=(0.3, 0.9), seed=c.seed())
        world.effects.append(fx.Ring(cxp[0], cxp[1], t_ice, 3, 24, 0.3, 'bfe8ff'))
        world.flash(t_ice, 0.3, 0.12, 'cfefff')
        world.shake(t_ice, 2.5, 0.3)
        world.cue('freeze', who=D.name)
        world.cue('music', section='frozen')
    w.at(t_ice, freeze)
    clone.t1 = t_ice + 0.5
    w.mark('frozen', t_ice + 0.1)
    # A walks round and hammers the ice, once, twice
    tw_ = t_ice + 0.5
    s2 = sd(A, D, tw_)
    A.add(tw_, 'stalk', 0.45, to=(D.x_at(t_ice) - s2 * 40, D.z_at(t_ice)))
    guard(A, tw_ + 0.45)
    ka = A.wep_at(tw_, 'n')[0]
    if ka == 'hammer':
        b1, b2, wp = 'hammer_swing', 'hammer_upper', ('n', 'hammer')
    else:
        b1, b2, wp = choose_blow(c, A, 'heavy'), choose_blow(c, A, 'heavy'), None
    strike(c, A.name, D.name, tw_ + 0.5, b1, 'chest', wep=wp, react=None, blood=0, size=9, shake=2.5, ice=True,
           settle_move=guard_move(A, tw_))
    strike(c, A.name, D.name, tw_ + 1.1, b2, 'chest', wep=wp, react=None, blood=0, size=9, shake=2.5, ice=True,
           settle_move=guard_move(A, tw_))
    if breakout == 'fire':
        w.at(t_break - 0.75, lambda world: setattr(world.f[D.name].st, 'glow_eyes', 1.0))
        D.add(t_break - 0.4, 'breakout', 0.75, blend=0.0)
        t_burst = t_break - 0.4 + 0.75 * 0.6

        def burst_free(world):
            world.cue('ice_shatter', who=D.name)
            world.cue('fire_burst', who=D.name, size=2)
            world.cue('music', section='fury')
            f = world.f[D.name]
            f.st.frozen = 0.0
            f.st.glow_eyes = 0.0
            j = world.joints[D.name]
            cx, cy = j['C'][0], j['C'][1] + 4
            fx.burst(world, fx.ICE, cx, cy, f.z, 80, (60, 220), life=(0.8, 1.8), size=(1, 2.6), seed=c.seed(), up=40,
                     zs=0.8)
            fx.burst(world, fx.FROST, cx, cy, f.z, 60, (30, 120), life=(0.3, 0.8), seed=c.seed(), zs=0.8)
            fx.burst(world, fx.FLAME, cx, cy, f.z, 90, (40, 160), life=(0.25, 0.6), size=(1, 2.4), seed=c.seed(), zs=0.8)
            world.effects.append(fx.Ring(cx, cy, t_burst, 4, 60, 0.35, 'ffc070', thick=2))
            world.flash(t_burst, 0.5, 0.14, 'ffe0b0')
            world.shake(t_burst, 5.5, 0.5)
            world.f[A.name].flash_t = t_burst
            world.jolt(A.name, 6.0)
            world.count['hits'] += 1
        w.at(t_burst, burst_free)
        w.mark('breakout', t_burst)
        fire_body(c, D.name, t_burst, t_burst + 0.9, rate=6, big=True)
        s3 = sd(D, A, t_burst)
        to = (clamp_x(c, A.x_at(t_burst) + s3 * 60, 6), A.z_at(t_burst) + 6)
        tumble_to(c, A, t_burst, to, 0.45, 18)
        t_up = fall_lie_up(c, A, t_burst + 0.45, -8, 0.25, 0.5)
        guard(D, t_burst + 0.45)
        return fury(c, D, A, t_up - 0.05)
    # no way out: the ice bursts under the blows and he falls out of it
    t_sh = tw_ + 1.1 + DUR[b2] * 0.5

    def shatter(world):
        f = world.f[D.name]
        f.st.frozen = 0.0
        j = world.joints[D.name]
        fx.burst(world, fx.ICE, j['C'][0], j['C'][1], f.z, 70, (50, 200), life=(0.8, 1.6), size=(1, 2.6), seed=c.seed())
        world.cue('ice_shatter', who=D.name)
        world.count['hits'] += 1
    w.at(t_sh, shatter)
    s4 = sd(A, D, t_sh)
    to = (clamp_x(c, D.x_at(t_sh) + s4 * 50 * power(A), 6), D.z_at(t_sh))
    tumble_to(c, D, t_sh, to, 0.55, 20)
    t_up = fall_lie_up(c, D, t_sh + 0.55, -8, 0.35, 0.5, slow=True)
    return t_up + 0.05


def fury(c, A, D, t):
    """A's fury after breaking free: four or five blows in a row and a roundhouse that throws D far."""
    rng = c.rng
    t = engage(c, A, D, t)
    hand0, k = held(A, t)
    if k in ('sword', 'ice_sword'):
        hs = [h for h in ('n', 'f') if A.wep_at(t, h)[0]]
        cuts = [('slash_down', hs[0]), ('slash_up', hs[-1]), ('slash_across', hs[0])]
        for nm, hand in cuts:
            strike(c, A.name, D.name, t, nm, 'chest', wep=(hand, A.wep_at(t, hand)[0] or k), react='hit_body', rdur=0.3,
                   knock=3, blood=10, size=8, settle_move=guard_move(A, t), mp={'hand': hand},
                   wound=wound('torso', 'cut', 1.6, rng))
            t += 0.36
    else:
        for i in range(3):
            nm = choose_blow(c, A, 'light' if i < 2 else 'heavy')
            strike(c, A.name, D.name, t, nm, WHERE.get(nm, 'head'), knock=3 + i, blood=6 + 2 * i, size=8,
                   react=REACT.get(WHERE.get(nm, 'head'), 'hit_head'), rdur=0.3)
            t += DUR[nm] * 0.85
    th = strike(c, A.name, D.name, t + 0.02, 'roundhouse', 'head', react='hit_head', rdur=0.5, knock=26, blood=14,
                size=11, spark='big', shake=3.5, react_air=arc(8), settle_move=guard_move(A, t))
    c.w.mark('fury', th)
    return th + 0.6


@beat('turning', weight=1.0, pool=False)
def comeback(c, A, D, t):
    """The turning point without a trap: D gets the upper hand (a string and a big blow puts A down hard), A rises and
    answers with fury."""
    t = exchange(c, D, A, t, n=4)
    t = counter(c, D, A, t)
    return fury(c, A, D, t)


@beat('magic', 'opening', 'escalation', weight=2.0, needs=lambda c, A, D, t: has(A, 'shield_throw') and A.shield_at(t))
def shield_throw(c, A, D, t):
    """A throws his shield: it rings off something in the area (or the floor), strikes D's head and comes back to his
    hand."""
    w = c.w
    rng = c.rng
    if gap(A, D, t) < 50:
        A.add(t, 'hop_back', 0.36, dx=-30, air=acts.rel(A, t, 8))
        guard(A, t + 0.36)
        t += 0.4
    A.add(t, 'shield_throw', 0.5)
    t_rel = t + 0.5 * moves.THROW_REL
    A.shields.append((t_rel, False))
    hx, hy = hand_pt(A, t_rel, 'f')
    g = w.stage.ground(A.x_at(t_rel), A.z_at(t_rel), t_rel, A.y_at(t_rel) + 2)
    p0 = (hx, A.z_at(t_rel), floor_y(A.z_at(t_rel)) - hy)
    tg = c.target_for(D, t_rel, None, maxd=200, kinds=None, any_dir=True)
    pts = [(t_rel, p0)]
    tt = t_rel
    if tg is not None:
        q = (tg.x, tg.z, tg.h * 0.6 + floor_at(c, tg.x, tg.z, t_rel))
        tt += math.hypot(q[0] - p0[0], q[1] - p0[1]) / 380.0 + 0.05
        pts.append((tt, q))
        t_ric = tt

        def ric(world, tg=tg, q=q, tt=tt):
            world.cue('shield_ricochet', x=q[0])
            fx.burst(world, fx.SPARK, q[0], floor_y(q[1]) - q[2], q[1], 12, (40, 120), life=(0.15, 0.4), seed=c.seed())
            world.effects.append(fx.HitSpark(q[0], floor_y(q[1]) - q[2], tt, 7, 'block', c.seed()))
            if tg.kind in ('lamp', 'car', 'window', 'table', 'brazier', 'hydrant'):
                tg.smash(world, tt, 0.8, 1 if q[0] > world.f[A.name].x else -1, c.seed())
        w.at(tt, ric)
    else:
        t_ric = None
    t_head = tt + 0.2 + (abs(D.x_at(tt) - (pts[-1][1][0])) / 420.0)

    def head_pt(world):
        j = world.joints[D.name]
        f = world.f[D.name]
        return (j['Hc'][0], f.z, floor_y(f.z) - j['Hc'][1])
    pts.append((t_head, head_pt))
    t_back = t_head + 0.2 + gap(A, D, t_head) / 420.0

    def hand_back(world):
        j = world.joints[A.name]
        f = world.f[A.name]
        return (j['FW'][0], f.z, floor_y(f.z) - j['FW'][1])
    pts.append((t_back, hand_back))
    w.effects.append(fx.Shield(pts, c.seed(), colors=A.look.gear['shield'].get('colors')))
    w.cue('shield_throw', t=t_rel, who=A.name)
    w.cue('shield_fly', t=t_rel, dur=t_back - t_rel, who=A.name)

    def bonk(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        x, y = j['Hc']
        world.effects.append(fx.HitSpark(x, y, t_head, 9, 'big', c.seed()))
        fx.burst(world, fx.BLOOD, x, y, f.z, 8, (50, 140), angle=-math.pi / 2, spread=1.0, life=(0.6, 1.2),
                 size=(0.8, 1.8), seed=c.seed())
        f.flash_t = t_head
        world.jolt(D.name, 5.0)
        f.st.wounds.append(dict(wound('head', 'bruise' if D.heavy else 'stain', 1.6, rng), seed=c.seed()))
        world.cue('shield_ricochet', who=D.name)
        world.cue('hit', who=D.name, weight=1, limb='punch', blood=8)
        world.shake(t_head, 2.5, 0.25)
        world.count['hits'] += 1
    w.at(t_head, bonk)
    D.add(t_head, 'hit_head', 0.45, dx=-6 * power(A) / max(0.7, power(D) ** 0.5), ease=moves.out, blend=0.03)
    guard(D, t_head + 0.45)
    A.add(t + 0.5, guard_move(A, t), t_back - t - 0.6)
    A.add(t_back - 0.15, 'shield_catch', 0.4)
    A.shields.append((t_back, True))
    w.at(t_back, lambda world: world.cue('shield_catch', who=A.name))
    guard(A, t_back + 0.25)
    w.mark('shield_throw', t_head)
    return t_back + 0.3


@beat('magic', 'opening', 'escalation', weight=1.2, needs=lambda c, A, D, t: has(D, 'shield_brace') and D.shield_at(t)
      and D.tier < A.tier)
def shield_brace(c, A, D, t):
    """The stronger one's huge blow lands on the raised shield: a ringing flash, D skids back over the floor."""
    w = c.w
    t = engage(c, A, D, t)
    hk, k = held(A, t)
    if k == 'hammer':
        nm, wp = 'hammer_swing', ('n', 'hammer')
    elif k:
        nm, wp = 'slash_down', (hk, k)
    else:
        nm, wp = choose_blow(c, A, 'heavy'), None
    D.add(t + DUR[nm] * 0.3, 'shield_block', 0.4)
    th = strike(c, A.name, D.name, t, nm, 'guard', wep=wp, react=None, blood=0, size=11, spark='big', shake=3.0,
                settle_move=guard_move(A, t), mp={'hand': wp[0]} if wp and nm == 'slash_down' else None, quiet=True)
    s = sd(A, D, th)
    skid = 40 * power(A)

    def ring(world):
        j = world.joints[D.name]
        x, y = j['FW']
        world.effects.append(fx.HitSpark(x, y, th, 12, 'block', c.seed()))
        world.effects.append(fx.Ring(x, y, th, 3, 28, 0.25, 'c8d8ff', thick=2))
        fx.burst(world, fx.SPARK, x, y, world.f[D.name].z, 20, (60, 180), life=(0.15, 0.45), seed=c.seed())
        world.cue('shield_block', who=D.name, weight=2, big=True)
        world.shake(th, 3.5, 0.3)
        world.count['blocks'] += 1
    w.at(th, ring)
    D.add(th, 'shield_brace', 0.5, to=(clamp_x(c, D.x_at(th) + s * skid, 6), D.z_at(th)), ease=moves.out)

    def skid_dust(world, tt, dt):
        f = world.f[D.name]
        g = world.stage.ground(f.x, f.z, tt, f.y + 2)
        fx.burst(world, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, 2, (10, 30), life=(0.3, 0.6), seed=c.seed())
    w.effects.append(fx.Emitter(th, th + 0.45, skid_dust))
    guard(D, th + 0.5)
    w.mark('shield_brace', th)
    return th + 0.6


@beat('opening', 'magic', 'escalation', weight=1.4, needs=lambda c, A, D, t: has(A, 'acrobat_dodge') and
      bool(A.spec['strikes']['leap']))
def acrobat_dodge(c, A, D, t):
    """D lunges (a charge, a big blow); A somersaults over him and lands a flip kick or a jumping kick as he turns."""
    rng = c.rng
    t = engage(c, D, A, t, dist=fdist(A, D) + 30)
    s = sd(D, A, t)
    big = choose_blow(c, D, 'heavy')
    D.add(t, 'dash', 0.25, to=(A.x_at(t) - s * 10, A.z_at(t)))
    th = whiff(c, D.name, A.name, t + 0.25, big, WHERE.get(big, 'head'))
    over = (A.x_at(th) - s * 0 + s * 44, A.z_at(th))
    over = (clamp_x(c, over[0], 6), over[1])
    g = floor_at(c, over[0], over[1], th)
    A.add(th - 0.25, 'flip', 0.6, to=over, air=arc(34 * leap_k(A), A.y_at(th), g), face=-s)
    t2 = th + 0.45
    kick = 'flip_kick' if 'flip_kick' in A.spec['strikes']['leap'] else choose_blow(c, A, 'heavy')
    th2 = strike(c, A.name, D.name, t2, kick, 'chin' if kick == 'flip_kick' else WHERE.get(kick, 'head'), react='hit_head',
                 rdur=0.45, knock=10, blood=10, size=10, spark='big', shake=2.5, wound=wound('head', 'stain', 1.6, rng))
    c.w.mark('flip_over', th)
    return th2 + 0.5


# ------------------------------------------------------------------ strength (the titans)
@beat('magic', 'escalation', weight=1.6, needs=lambda c, A, D, t: has(A, 'shockwave'))
def shockwave(c, A, D, t):
    """A thunderclap: a white blast and rings; D is blown across the area, things nearby break."""
    w = c.w
    A.add(t, 'clap', 0.8)
    tc_ = t + 0.8 * moves.CLAP_AT
    j = A
    x, y = hand_pt(A, tc_, 'f')
    z = A.z_at(tc_)

    def blast(world):
        world.effects.append(fx.Ring(x, y, tc_, 4, 90, 0.5, 'ffffff', thick=3))
        world.effects.append(fx.Ring(x, y, tc_ + 0.08, 4, 140, 0.6, 'e0f0ff', thick=2))
        world.flash(tc_, 0.6, 0.15)
        world.shake(tc_, 7.0, 0.7)
        fx.burst(world, fx.DUST, x, y, z, 60, (80, 220), life=(0.4, 1.0), size=(1.5, 3), seed=c.seed())
        world.cue('clap', x=x)
        world.cue('shockwave', x=x)
        world.count['hits'] += 1
        world.f[D.name].flash_t = world.t
        world.jolt(D.name, 6.0)
        for tg in world.stage.targets(c.area, tc_):
            if abs(tg.x - x) < 120:
                tg.smash(world, tc_ + abs(tg.x - x) / 400.0, 1.5, 1 if tg.x > x else -1, c.seed())
    w.at(tc_, blast)
    w.slowmo(tc_, tc_ + 0.5, 0.35)
    s = sd(A, D, tc_)
    to = (clamp_x(c, D.x_at(tc_) + s * 140, 6), D.z_at(tc_))
    tumble_to(c, D, tc_ + 0.05, to, 0.8, 24)
    t_up = fall_lie_up(c, D, tc_ + 0.85, -8, 0.4, 0.5, slow=True)
    guard(A, t + 0.8)
    w.mark('shockwave', tc_)
    return t_up + 0.05


@beat('magic', 'escalation', weight=1.4, needs=lambda c, A, D, t: has(A, 'leap_smash'))
def leap_smash(c, A, D, t):
    """A leaps clean out of the picture and comes down fists first where D stands: the floor caves; D is thrown."""
    w = c.w
    s = sd(A, D, t)
    A.add(t, 'leap', 0.6, air=arc(160, A.y_at(t), A.y_at(t) + 180), ease=lambda u: u)
    tl = t + 1.4
    x, z = D.x_at(tl), D.z_at(tl)
    g = w.stage.ground(x, z, tl, D.y_at(tl) + 2)
    A.add(t + 0.6, 'smash_land', 0.8, to=(x - s * 6, z), air=lambda u, g=g: g + 180 * (1 - u) ** 2, ease=lambda u: u)
    D.add(t + 0.8, 'crouch_look', 0.5)
    crater(c, x, z, tl, 2.5 * power(A) / 1.8, g=g)
    w.cue('hulk_land', t=tl, x=x, power=2.0)

    def blown(world):
        world.f[D.name].flash_t = world.t
        world.jolt(D.name, 6.0)
        world.count['hits'] += 1
        world.cue('hit', who=D.name, weight=2, limb='punch', blood=6)
    w.at(tl + 0.02, blown)
    to = (clamp_x(c, x + s * 70, 6), clamp_z(c, z + 10, 6))
    tumble_to(c, D, tl, to, 0.6, 30)
    t_up = fall_lie_up(c, D, tl + 0.6, -8, 0.4, 0.5, slow=True)
    A.add(tl, 'land_rise', 0.8)
    guard(A, tl + 0.8)
    w.mark('leap_smash', tl)
    return t_up + 0.05


@beat('magic', 'escalation', weight=0.8, needs=lambda c, A, D, t: has(A, 'roar'))
def roar(c, A, D, t):
    """A roars: dust blasts off him, the eyes flare; D is shaken back a step."""
    w = c.w
    A.add(t, 'roar', 1.2)
    w.at(t + 0.4, lambda world: (setattr(world.f[A.name].st, 'roar', 1.0), setattr(world.f[A.name].st, 'anger', 1.0),
                                 world.cue('roar', who=A.name, dur=0.9, big=True), world.shake(t + 0.4, 3.0, 0.8)))
    w.at(t + 1.2, lambda world: setattr(world.f[A.name].st, 'roar', 0.0))
    D.add(t + 0.45, 'stalk', 0.5, dx=-14)
    guard(D, t + 0.95)
    guard(A, t + 1.2)
    return t + 1.25


# ================================================================== travel between areas
def travel(c, A, D, t, route):
    """The fight moves to another area along a route of the stage. A is the one the move starts with."""
    how = route['how']
    w = c.w
    rng = c.rng
    dst = route['b']
    ad = w.stage.AREAS[dst]
    if how == 'leap':
        # A springs up (or down) to the other area, D follows
        cx, cz = ad['center']
        z_l = route.get('land_z', cz)
        xa = cx + float(rng.uniform(-40, 40))
        xd = xa + 50 * (1 if D.x_at(t) > A.x_at(t) else -1)
        xa = max(ad['x'][0] + 10, min(ad['x'][1] - 10, xa))
        xd = max(ad['x'][0] + 10, min(ad['x'][1] - 10, xd))
        if abs(xd - xa) < 40:
            xd = xa + (50 if xa < (ad['x'][0] + ad['x'][1]) / 2 else -50)
        ya = w.stage.area_y(dst, xa, z_l)
        yd = w.stage.area_y(dst, xd, z_l)
        rise = max(0.0, ya - A.y_at(t))
        h = 30 + rise * 0.4
        jump_to(c, A, t, (xa, z_l), 0.7, h * leap_k(A), y1=ya, move='flip', settle=False)
        A.add(t + 0.7, 'superhero', 0.45)
        guard(A, t + 1.15)
        t2 = t + 0.45
        jump_to(c, D, t2, (xd, z_l), 0.72, h * leap_k(D), y1=yd, move='flip', settle=False)
        D.add(t2 + 0.72, 'superhero', 0.4)
        guard(D, t2 + 1.12)
        c.area = dst
        w.mark(dst, t2 + 0.8)
        return t2 + 1.15
    if how == 'knocked_off':
        # a big blow throws D off the edge down to the other area; A follows with a dive
        t = engage(c, A, D, t)
        hk, k = held(A, t)
        nm, wp = ('hammer_swing', ('n', 'hammer')) if k == 'hammer' else (choose_blow(c, A, 'heavy'), None)
        th = strike(c, A.name, D.name, t, nm, 'chest' if wp else WHERE.get(nm, 'head'), wep=wp, react=None, blood=12,
                    size=11, spark='big', shake=4.0, settle_move=guard_move(A, t), wound=wound('torso', 'tear', 2.2, rng))
        cx, cz = ad['center']
        to = (max(ad['x'][0] + 10, min(ad['x'][1] - 10, D.x_at(th) + float(rng.uniform(-30, 30)))), cz + 20)
        y1 = w.stage.area_y(dst, to[0], to[1])
        if route.get('path') == 'eave' and hasattr(w.stage, 'eave_fall'):
            t_fl = w.stage.eave_fall(c, D, th, to)
        else:
            tumble_to(c, D, th, to, 0.85, 30, y1=y1)
            t_fl = th + 0.85
        t_up = fall_lie_up(c, D, t_fl, -8, 0.3, 0.5, h=10)
        tz = th + 0.7
        al = (max(ad['x'][0] + 10, min(ad['x'][1] - 10, to[0] + 54 * (1 if A.x_at(th) > D.x_at(th) else -1))), to[1] - 10)
        jump_to(c, A, tz, al, 0.65, 16, y1=w.stage.area_y(dst, al[0], al[1]), move='flip', settle=False)
        A.add(tz + 0.65, 'superhero', 0.35)
        guard(A, tz + 1.0)
        c.area = dst
        w.mark('roof_fall' if route.get('path') == 'eave' else dst, th + 0.5)
        return max(t_up, tz + 1.0) + 0.05
    if how == 'thrown_through':
        # a kick throws D through a breakable at the edge (the balustrade) and down to the area below
        kind = route.get('target')
        tg = c.target_for(D, t, kind, any_dir=True, area=c.area)
        if tg is None:
            return t
        s = 1 if tg.x > D.x_at(t) else -1
        # put D with his back to it
        D.add(t, 'stalk', 0.6, to=(tg.x - 4 * s, tg.z - 30))
        A.add(t, 'stalk', 0.6, to=(tg.x - 4 * s - fdist(A, D) * 0.4, tg.z - 30 - fdist(A, D) * 0.9))
        guard(D, t + 0.6)
        guard(A, t + 0.6)
        t += 0.65
        th = strike(c, A.name, D.name, t, 'front_kick', 'gut', react=None, blood=8, size=10, spark='big', shake=3.5,
                    settle_move=guard_move(A, t))
        y_t = tg.h
        tumble_to(c, D, th, (tg.x, tg.z), 0.4, 8, y1=y_t)
        tg.smash(w, th + 0.4, 1.3, 1, c.seed())
        cx, cz = ad['center']
        to = (max(ad['x'][0] + 10, min(ad['x'][1] - 10, tg.x + 14)), cz)
        y1 = w.stage.area_y(dst, to[0], to[1])
        D.add(th + 0.4, 'tumble', 0.5, to=to, ease=lambda u: u, air=lambda u, y_t=y_t, y1=y1: y_t + (y1 - y_t) * u * u,
              blend=0.02)
        w.trail(D.name, th + 0.4, th + 0.9)
        t_up = fall_lie_up(c, D, th + 0.9, -8, 0.35, 0.5, h=8, g=y1)
        drip(c, D.name, th, 300.0, 0.9, 'mouth')
        td = th + 0.85
        al = (max(ad['x'][0] + 10, min(ad['x'][1] - 10, to[0] + 70)), to[1] - 4)
        jump_to(c, A, td, al, 0.75, 22, y1=w.stage.area_y(dst, al[0], al[1]), move='dive', settle=False)
        A.add(td + 0.75, 'superhero', 0.55)
        guard(A, td + 1.3)
        c.area = dst
        w.mark('rail', th + 0.4)
        w.mark(dst, td + 0.76, key=False)
        return max(t_up, td + 1.3) + 0.05
    if how == 'run_across':
        # A runs for it along the path, leaping the gap; D chases (a dash of his element if he has one)
        path = route['path']
        tt = t
        pts = []
        for p in path:
            if p[0] == 'gap':
                pts.append(('gap', p[1]))
            else:
                pts.append(p)
        cur = (A.x_at(t), A.z_at(t))
        for p in pts:
            if p[0] == 'gap':
                to = (p[1], cur[1])
                jump_to(c, A, tt, to, 0.42, 18 * leap_k(A), face=1 if to[0] > cur[0] else -1, settle=False)
                tt += 0.42
                cur = to
            else:
                d = math.hypot(p[0] - cur[0], p[1] - cur[1])
                dur = max(0.18, d / 150.0)
                A.add(tt, 'run', dur, to=p, face=1 if p[0] > cur[0] else -1)
                tt += dur
                cur = p
        guard(A, tt)
        A.add(tt, guard_move(A, tt), 0.6, face=-1 if pts[-1][0] > pts[0][0] else 1)
        # D follows a little behind
        td = t + 0.6
        cur = (D.x_at(td), D.z_at(td))
        el = D.spec.get('element')
        for i, p in enumerate(pts):
            if p[0] == 'gap':
                to = (p[1] + (8 if pts[-1][0] > pts[0][0] else -8), cur[1])
                jump_to(c, D, td, to, 0.42, 20 * leap_k(D), move='flip', face=1 if to[0] > cur[0] else -1, settle=False)
                td += 0.42
                cur = to
            else:
                last = i == len(pts) - 1
                p2 = (p[0] - (34 if pts[-1][0] > pts[0][0] else -34), p[1]) if last else p
                d = math.hypot(p2[0] - cur[0], p2[1] - cur[1])
                if i == 0 and el == 'fire' and has(D, 'fire_dash'):
                    dur = max(0.25, d / 200.0)
                    D.add(td, 'fire_dash', dur, to=p2, ease=lambda u: u ** 1.2)
                    mg.fire_trail(w, D.name, td, td + dur)
                    w.trail(D.name, td, td + dur)
                    w.cue('fire_dash', t=td, dur=dur, who=D.name)
                else:
                    dur = max(0.18, d / 160.0)
                    D.add(td, 'run', dur, to=p2)
                td += dur
                cur = p2
        guard(D, td)
        c.area = dst
        w.mark('bridge' if route.get('via') == 'bridge' else dst, t + 1.3)
        return max(tt, td) + 0.15
    if how == 'smash_through':
        # the stronger one throws the other through a wall into the next room (the breach stays)
        return w.stage.smash_through(c, A, D, t, route)
    return t


# ================================================================== climaxes (a fighter's signature)
@beat('climax', weight=0.0, pool=False)
def spear(c, A, D, t, line='GET OVER HERE!'):
    """The spear across a gap (or the area): it bites D's shoulder, the line, the yank over the void into a huge
    uppercut, an air punch, an axe kick and a slam into the floor."""
    w = c.w
    rng = c.rng
    # distance first: D gathers his power across the area (or a chasm)
    if gap(A, D, t) < 90:
        s0 = sd(A, D, t)
        far_x = clamp_x(c, D.x_at(t) + s0 * 70, 6)
        D.add(t, 'back_flip', 0.5, to=(far_x, D.z_at(t)), air=acts.rel(D, t, 18))
        guard(D, t + 0.5)
        t += 0.55
    if D.wep_at(t, 'n')[0]:
        kd = D.wep_at(t, 'n')[0]
        wep(D, t + 0.2, 'n', None)

        def drop(world, kd=kd):
            hand = (world.joints[D.name]['NW'][0], world.joints[D.name]['NW'][1])
            world.effects.append(mg.LooseWeapon(kd, hand, (hand[0] + 6, hand[1] + 6), t + 0.2, t + 0.5, t + 0.5, h=4,
                                                spins=0.3, za=D.z_at(t), zb=D.z_at(t), seed=c.seed(), a0=-1.4,
                                                stick_ang=-1.2))
            fx.burst(world, fx.ICE, hand[0], hand[1] + 6, D.z_at(t), 20, (30, 90), life=(0.6, 1.2), seed=c.seed())
            world.cue('ice_shatter', x=hand[0], small=True)
        if body_is_ice(kd):
            w.at(t + 0.2, drop)
    D.add(t, 'ice_cast' if D.spec.get('element') == 'ice' else guard_move(D, t), 1.4, ease=moves.out)
    if D.spec.get('element') in ('ice', 'fire'):
        element_hands(c, D.name, t + 0.25, t + 0.8, D.spec['element'])
    t_sp = t + 0.1
    held = A.wep_at(t_sp, 'n')[0]
    A.add(t_sp, 'spear_throw', 0.55, ease=moves.out)
    if held:
        wep(A, t_sp, 'n', None)
        A.backs.append((t_sp, 1))
    w.cam_bias.append((t_sp, t_sp + 3.0, 0.0, 10.0))
    t_rel = t_sp + 0.55 * moves.SPEAR_RELEASE
    t_bite = t_rel + 0.3
    t_yank = t_bite + 0.6
    t_pull = t_yank + 0.1
    pull_dur = 0.62
    t_out = t_pull + pull_dur
    sp = fx.Spear(A.name, D.name, t_rel, t_bite, t_out, t_out + 0.16)
    w.effects.append(sp)
    w.cue('kunai_throw', t=t_rel, who=A.name)
    w.cue('rope_zip', t=t_rel, dur=t_bite - t_rel, who=A.name)
    w.cue('slowmo', t=t_bite - 0.04)
    w.cue('pierce', t=t_bite, who=D.name)
    w.cue('music', t=t_bite, section='spear')
    w.cue('say', t=t_bite + 0.18, text=line, voice=A.spec.get('voice', 'announcer'))
    w.cue('yank', t=t_pull, who=A.name)
    w.cue('fly', t=t_pull, dur=pull_dur, who=D.name)
    w.cue('rip', t=t_out, who=D.name)
    A.add(t_sp + 0.55, 'spear_hold', t_yank - (t_sp + 0.55))
    D.add(t_bite, 'speared', t_pull - t_bite, dx=-3, ease=moves.out, blend=0.03)
    w.slowmo(t_bite - 0.04, t_bite + 0.32, 0.3)

    def bite(world):
        z = world.f[D.name]
        x, y = sp.target_point(world)
        world.effects.append(fx.HitSpark(x, y, t_bite, 10, 'big', c.seed()))
        ang = -0.4 if z.facing < 0 else -(math.pi - 0.4)
        fx.burst(world, fx.BLOOD, x, y, z.z, 34, (60, 220), angle=ang, spread=0.5, life=(0.8, 1.8), size=(1, 2.6),
                 seed=c.seed())
        fx.burst(world, fx.GORE, x, y, z.z, 6, (50, 150), angle=ang, spread=0.4, life=(0.8, 1.5), size=(1, 2),
                 seed=c.seed())
        fx.burst(world, fx.MIST, x, y, z.z, 10, (10, 50), angle=ang, spread=0.8, life=(0.25, 0.5), size=(1.5, 2.5),
                 seed=c.seed())
        z.st.wounds.append({'part': 'farm_u', 't': 0.12, 's': 0.0, 'r': 2.4, 'kind': 'tear', 'seed': c.seed()})
        z.st.wounds.append({'part': 'torso', 't': 0.95, 's': 0.6, 'r': 2.2, 'kind': 'stain', 'seed': c.seed()})
        z.flash_t = t_bite
        world.jolt(D.name, 6.0)
        world.flash(t_bite, 0.3, 0.12, 'ff3020')
        world.shake(t_bite, 3.5, 0.4)
        world.count['hits'] += 1
    w.at(t_bite, bite)
    w.mark('speared', t_bite)
    drip(c, D.name, t_bite + 0.1, 300.0, 0.4, 'shoulder')
    w.words.append(T.Word(line, t_bite + 0.18, t_out + 0.5, 240, 50, 'big', T.FIRE, drop=10))
    w.at(t_bite + 0.18, lambda world: world.shake(t_bite + 0.18, 2.5, 0.35))
    s = sd(A, D, t_yank)
    a_ = area(c)
    room = (a_['x'][1] - A.x_at(t_yank)) if s > 0 else (A.x_at(t_yank) - a_['x'][0])
    A.add(t_yank, 'yank', 0.45, dx=-4, ease=moves.out)
    if room < 80:
        # no floor in front of him (a cliff, the edge): the haul throws D over his head to land behind him
        land_x = A.x_at(t_out) - s * 30.0
    else:
        land_x = A.x_at(t_out) + s * 26.0
    gA = w.stage.ground(A.x_at(t_out), A.z_at(t_out), t_out, A.y_at(t_out) + 2)
    D.add(t_pull, 'pulled', pull_dur, to=(land_x, A.z_at(t_out)), ease=lambda u: u ** 1.5,
          air=arc(26, D.y_at(t_pull), gA), blend=0.04)
    w.trail(D.name, t_pull, t_out)
    D.add(t_out, 'dazed', 0.6, blend=0.06)
    w.mark('yank', t_pull + 0.3)

    def rip(world):
        x, y = sp.target_point(world)
        z = world.f[D.name].z
        fx.burst(world, fx.BLOOD, x, y, z, 16, (40, 150), angle=-math.pi / 2, spread=1.0, life=(0.8, 1.5), size=(1, 2.2),
                 seed=c.seed())
    w.at(t_out, rip)
    guard(A, t_yank + 0.45)
    c.area = c.area_at(A, t_out) or c.area
    # he lands dazed at A's feet: two quick blows, then the climax's string
    t2 = string(c, A, D, t_out - 0.04, 2, final=False)
    return big_combo(c, A, D, t2, chin_fx=True)


def body_is_ice(kind):
    import body
    return bool(kind and body.WEAPONS.get(kind, {}).get('ice'))


@beat('climax', weight=0.0, pool=False)
def big_combo(c, A, D, t, chin_fx=False):
    """The climax's string: a huge uppercut in slow motion, an air blow on the way down, an axe kick that slams D into
    the floor (a crater), D gets up beaten."""
    w = c.w
    rng = c.rng
    tu = t

    def chin(world):
        cx, cy = world_point(world.f[D.name], world.t, 'chin')
        z = world.f[D.name].z
        fx.burst(world, fx.GORE, cx, cy, z, 7, (60, 170), angle=-math.pi / 2 - 0.4, spread=0.5, life=(1, 1.6),
                 size=(1, 2), seed=c.seed())
        world.effects.append(fx.Ring(cx, cy, world.t, 4, 34, 0.3, 'ffffff'))
        world.f[D.name].st.mask_torn = 1.0
    lau = choose_blow(c, A, 'launcher')
    th = strike(c, A.name, D.name, tu, lau, 'chin', react=None, blood=28, size=14, spark='big', shake=5.5, flash=0.22,
                extra=chin, wound=[{'part': 'torso', 't': 0.55, 's': 0.4, 'r': 2.6, 'kind': 'tear'},
                                   {'part': 'fleg_u', 't': 0.4, 's': 0.0, 'r': 1.8, 'kind': 'tear'}])
    w.slowmo(th + 0.07, th + 0.3, 0.3)
    w.mark('uppercut', th)
    gD = w.stage.ground(D.x_at(th), D.z_at(th), th, D.y_at(th) + 2)
    tumble_to(c, D, th, (D.x_at(th) - 14 * sd(A, D, th) * -1, D.z_at(th) - 2), 0.8, 44 * min(1.4, power(A)), y1=gD)
    gA = w.stage.ground(A.x_at(th), A.z_at(th), th, A.y_at(th) + 2)
    if A.spec['strikes']['air']:
        A.add(th + 0.14, 'jump', 0.26, air=arc(14, gA, gA + 22), blend=0.04)
        air1 = 'air_punch' if 'air_punch' in A.spec['strikes']['air'] else A.spec['strikes']['air'][0]
        t2 = strike(c, A.name, D.name, th + 0.36, air1, 'head', react=None, blood=12, size=9, settle=False)
        tumble_to(c, D, t2, (D.x_at(t2) - 8 * D.facing_at(t2), D.z_at(t2)), 0.6, 8, y1=gD)
        axe = 'axe_kick' if 'axe_kick' in A.spec['strikes']['air'] else air1
        t3 = strike(c, A.name, D.name, t2 + 0.05, axe, 'chest', react=None, blood=16, size=13, spark='big', shake=5.0,
                    settle=False, land=gA)
        t_land = t2 + 0.05 + DUR[axe]
    else:
        t3 = strike(c, A.name, D.name, th + 0.5, choose_blow(c, A, 'heavy'), 'chest', react=None, blood=16, size=13,
                    spark='big', shake=5.0)
        t_land = t3 + 0.3
    guard(A, t_land)
    A.add(t_land + 0.25, 'hop_back', 0.42, dx=-40, air=arc(10, gA, gA))
    guard(A, t_land + 0.67)
    y_at = D.y_at(t3)
    D.add(t3, 'tumble', 0.12, dx=-4, air=lambda u, y0=y_at: gD + (y0 - gD) * (1 - u) ** 2, blend=0.02)
    t_floor = t3 + 0.12
    crater(c, D.x_at(t_floor), D.z_at(t_floor), t_floor, 1.5 * power(A), g=gD)
    w.at(t_floor, lambda world: world.cue('floor_slam', who=D.name))
    w.mark('slam', t_floor)
    D.add(t_floor, 'bounce', 0.42, dx=-8, air=arc(10, gD, gD), blend=0.03)
    smear_floor(c, D.name, t_floor, t_floor + 0.5)
    D.add(t_floor + 0.42, 'lie', 0.6)
    D.add(t_floor + 1.02, 'get_up', 0.9)
    return t_floor + 1.92


@beat('climax', weight=0.0, pool=False)
def ice_spikes_launch(c, A, D, t):
    """Sub-Zero's climax: the floor freezes, a forest of spikes throws D high, A leaps after him for an axe kick that
    slams him down."""
    t = ice_spikes(c, A, D, t)
    return big_combo(c, A, D, engage(c, A, D, t))


@beat('climax', weight=0.0, pool=False)
def shield_combo(c, A, D, t):
    """Captain America's climax: the shield thrown off the scenery into D's head, a leaping kick, the rim under the
    chin, and the finishing string."""
    t = shield_throw(c, A, D, t)
    t = acrobat_dodge(c, A, D, t) if has(A, 'acrobat_dodge') else t
    return big_combo(c, A, D, engage(c, A, D, t))


@beat('climax', weight=0.0, pool=False)
def leap_smash_climax(c, A, D, t):
    t = roar(c, A, D, t)
    return leap_smash(c, A, D, t)


# ================================================================== the end
def finish(c, A, D, t, finisher):
    """The end: FINISH HIM (MK fatalities) and the finisher, or a knockout; then the win card."""
    fn = FINISHERS.get(finisher, knockout)
    return fn(c, A, D, t)


def finish_him(c, A, D, t, line='FINISH HIM'):
    w = c.w
    D.add(t, 'stagger', 30.0, blend=0.2)
    t_fh = t + 0.1
    w.words.append(T.Word(line, t_fh, t_fh + 1.8, 240, 52, 'big', T.BLOOD, drop=10))
    w.shake(t_fh, 2.0, 0.3)
    w.cue('say', t=t_fh, text=line, voice='announcer')
    w.cue('music', t=t_fh, section='tension')
    w.mark('finish_him', t_fh)
    walk_from = t + 0.3
    s = 1 if A.x_at(walk_from) < D.x_at(walk_from) else -1
    A.add(walk_from, 'stalk', 1.6, to=(D.x_at(walk_from) - s * 27 * (A.K + D.K) / 1.5, D.z_at(walk_from)))
    guard(A, walk_from + 1.6)
    return walk_from + 1.75


def fatality_word(c, t):
    w = c.w
    w.words.append(T.Word('FATALITY', t, 999.0, 240, 42, 'big', T.BLOOD, drop=14))
    w.at(t, lambda world: world.shake(t, 3.0, 0.4))
    w.cue('say', t=t, text='FATALITY', voice='announcer')
    w.mark('fatality', t)


def fire_breath(c, A, D, t):
    """The mask comes off: a burning skull breathes fire; D chars to a skeleton that collapses into bones."""
    w = c.w
    t_um = finish_him(c, A, D, t)
    A.add(t_um, 'unmask', 0.9)
    t_skull = t_um + 0.9 * moves.UNMASK_AT

    def skull(world):
        s = world.f[A.name]
        s.st.skull = True
        j = world.joints[A.name]
        x, y = j['Hc']
        fx.burst(world, fx.FLAME, x, y, s.z, 40, (20, 70), life=(0.3, 0.6), size=(1, 2), seed=c.seed(), up=20)
        fx.burst(world, fx.SMOKE, x, y, s.z, 10, (20, 60), life=(0.4, 0.9), seed=c.seed())
        world.flash(t_skull, 0.3, 0.1, 'ffb060')
        world.shake(t_skull, 2.0, 0.3)
        world.cue('skull', who=A.name)
        world.cue('music', section='fatality')
    w.at(t_skull, skull)
    w.mark('skull', t_skull)

    def skull_fire(world, tt, dt):
        j = world.joints[A.name]
        f = world.f[A.name]
        x, y = j['Hc']
        rng = np.random.default_rng(c.seed())
        for _ in range(2):
            world.particles.spawn(fx.FLAME, [x + rng.uniform(-2, 2)], [f.z], [floor_y(f.z) - y + 3],
                                  [rng.uniform(-5, 5)], [0.0], [rng.uniform(15, 30)], [rng.uniform(0.15, 0.3)],
                                  [rng.uniform(0.6, 1.0)], rng=rng)
    t_end = t_skull + 14.0
    w.effects.append(fx.Emitter(t_skull, t_end, skull_fire))
    t_br = t_um + 0.95
    A.add(t_br, 'breathe', 2.3, ease=moves.out)
    t_fire0, t_fire1 = t_br + 0.25, t_br + 2.0
    w.slowmo(t_fire0 - 0.05, t_fire0 + 1.6, 0.4)
    w.cue('slowmo', t=t_fire0 - 0.05)
    w.cue('fire_breath', t=t_fire0, dur=t_fire1 - t_fire0, who=A.name)
    w.mark('fire', t_fire0)

    def breath(world, tt, dt):
        j = world.joints[A.name]
        f = world.f[A.name]
        x = j['Hc'][0] + f.facing * 3.5 * f.K
        y = j['Hc'][1] + 2.0 * f.K
        rng = np.random.default_rng(c.seed())
        n = 14
        ang = (0.0 if f.facing > 0 else math.pi) + rng.uniform(-0.2, 0.2, n) + 0.05
        spd = rng.uniform(140, 210, n)
        h0 = floor_y(f.z) - y
        world.particles.spawn(fx.FLAME, x + rng.uniform(-1, 1, n), f.z + rng.uniform(-1, 1, n), h0 + rng.uniform(-1, 1, n),
                              np.cos(ang) * spd, rng.uniform(-12, 12, n), -np.sin(ang) * spd, rng.uniform(0.3, 0.55, n),
                              rng.uniform(1.4, 2.6, n), rng=rng)
        if rng.uniform() < 0.5:
            world.particles.spawn(fx.SMOKE, [x + f.facing * 26], [f.z], [h0 + 4], [f.facing * 30.0], [0.0], [20.0], [1.2],
                                  [1.5], rng=rng)
    w.effects.append(fx.Emitter(t_fire0, t_fire1, breath))

    def breath_glow(cv, add, tt, off, lights, world):
        import px
        j = world.joints[A.name]
        f = world.f[A.name]
        x = j['Hc'][0] + f.facing * 18 + off[0]
        y = j['Hc'][1] + 4 + off[1]
        px.glow(add, x, y, 36, hexc('ff7a20'), 0.5)
        lights.append((x, y, 110, hexc('ff8a3a'), 1.6))
    w.effects.append(fx.Glow(t_fire0, t_fire1 + 0.2, breath_glow))
    w.at(t_fire0 + 0.12, lambda world: (world.flash(t_fire0 + 0.12, 0.45, 0.08, 'fff0d0'),
                                        world.shake(t_fire0 + 0.12, 5.0, 0.6)))
    t_burn = t_fire0 + 0.2
    D.add(t_burn, 'burning', 2.3, blend=0.08)
    fire_body(c, D.name, t_burn, t_burn + 3.6, rate=9, big=True)

    def char(world, tt, dt):
        world.f[D.name].st.char = max(0.0, min(1.0, (tt - t_burn) / 1.3))
    w.effects.append(fx.Emitter(t_burn, t_burn + 1.4, char))
    fatality_word(c, t_burn + 0.25)
    t_bones = t_burn + 1.45

    def bones(world):
        z = world.f[D.name]
        z.st.skeleton = 1.0
        z.st.char = 0.0
        j = world.joints[D.name]
        fx.burst(world, fx.EMBER, j['C'][0], j['C'][1], z.z, 30, (20, 80), life=(0.6, 1.4), seed=c.seed(), up=30)
        fx.burst(world, fx.SMOKE, j['C'][0], j['C'][1] - 6, z.z, 12, (10, 30), life=(0.8, 1.6), size=(1.5, 2.5),
                 seed=c.seed())
        world.cue('bone_crunch', who=D.name)
    w.at(t_bones, bones)
    w.mark('bones', t_bones)
    t_col = t_burn + 2.3
    D.add(t_col, 'collapse', 0.9)
    bone_pile(c, D, t_col + 0.9, t_end)
    guard(A, t_br + 2.3)
    return win_card(c, A.name, t_col + 1.5)


def bone_pile(c, D, t, t_end):
    w = c.w

    def crumble(world):
        z = world.f[D.name]
        x = z.x
        g = world.stage.ground(z.x, z.z, world.t, z.y + 2)
        rng = np.random.default_rng(c.seed())
        n = 26
        world.particles.spawn(fx.BONE, x + rng.uniform(-8, 8, n), z.z + rng.uniform(-4, 4, n), g + rng.uniform(4, 16, n),
                              rng.uniform(-50, 50, n), rng.uniform(-30, 30, n), rng.uniform(20, 80, n),
                              rng.uniform(2, 3, n), rng.uniform(1, 2.2, n), rng=rng)
        for k in range(16):
            world.stage.stamp(x + rng.uniform(-9, 9), z.z + rng.uniform(-5, 5), hexc('d8ccb0') if k % 3 else hexc('6a5e4a'),
                              r=rng.uniform(0.7, 1.5), alpha=1.0, seed=c.seed(), h=g)
        for k in range(10):
            world.stage.stamp(x + rng.uniform(-12, 12), z.z + rng.uniform(-6, 6), hexc('1a1410'), r=rng.uniform(1.5, 3.0),
                              alpha=0.7, seed=c.seed(), h=g)
        fx.burst(world, fx.DUST, x, floor_y(z.z) - g - 3, z.z, 14, (10, 40), life=(0.6, 1.2), size=(2, 3), seed=c.seed())
        fx.burst(world, fx.SMOKE, x, floor_y(z.z) - g - 6, z.z, 16, (10, 30), life=(1.0, 2.0), size=(1.5, 3), seed=c.seed())
        world.hidden[D.name] = True
        world.shake(t, 2.5, 0.3)
        world.cue('bone_pile', who=D.name)
    w.at(t, crumble)
    w.mark('bone_pile', t)

    def embers(world, tt, dt):
        z = world.f[D.name]
        rng = np.random.default_rng(c.seed())
        g = world.stage.ground(z.x, z.z, tt, z.y + 2)
        world.particles.spawn(fx.FLAME, [z.x + rng.uniform(-7, 7)], [z.z + rng.uniform(-3, 3)], [g + 1.0],
                              [rng.uniform(-5, 5)], [0.0], [rng.uniform(10, 30)], [rng.uniform(0.2, 0.45)],
                              [rng.uniform(0.7, 1.3)], rng=rng)
        if rng.uniform() < 0.4:
            world.particles.spawn(fx.SMOKE, [z.x + rng.uniform(-5, 5)], [z.z], [g + 5.0], [4.0], [0.0], [18.0], [1.5],
                                  [1.6], rng=rng)
    w.effects.append(fx.Emitter(t, t_end, embers))


def ice_shatter(c, A, D, t):
    """Sub-Zero's fatality: the beaten foe frozen solid where he stands, then one uppercut shatters him into pieces
    of ice and blood."""
    w = c.w
    t_go = finish_him(c, A, D, t)
    A.add(t_go, 'ice_cast', 1.0, ease=moves.out)
    t_rel = t_go + 1.0 * moves.ICE_RELEASE
    element_hands(c, A.name, t_go, t_rel, 'ice')
    w.cue('music', t=t_rel, section='fatality')

    def freeze(world):
        f = world.f[D.name]
        f.st.frozen = 1.0
        j = world.joints[D.name]
        g = world.stage.ground(f.x, f.z, world.t, f.y + 2)
        pts = [v for k, v in j.items() if isinstance(v, tuple) and len(v) == 2 and k != 'up']
        box = (min(p[0] for p in pts) - 3, min(p[1] for p in pts) - 4, max(p[0] for p in pts) + 3, floor_y(f.z) - g + 1)
        blk = fx.IceBlock(box, t_rel, t_rel + 1.5, t_rel + 2.4, c.seed())
        blk.z = f.z
        world.effects.append(blk)
        fx.burst(world, fx.FROST, j['C'][0], j['C'][1], f.z, 80, (20, 100), life=(0.4, 1.0), seed=c.seed())
        world.flash(t_rel, 0.35, 0.12, 'cfefff')
        world.shake(t_rel, 3.0, 0.4)
        world.cue('freeze', who=D.name, big=True)
    w.at(t_rel, freeze)
    D.add(t_rel, 'hold', 3.0, pose=D.pose_at(t_rel), blend=0.0)
    w.mark('frozen_solid', t_rel + 0.2)
    t_up = t_rel + 1.2
    t_up = engage(c, A, D, t_up)
    th = strike(c, A.name, D.name, t_up, 'uppercut', 'chin', react=None, blood=0, size=14, spark='big', shake=6.0,
                flash=0.3, ice=True)
    w.slowmo(th - 0.02, th + 0.5, 0.3)

    def shatter(world):
        f = world.f[D.name]
        j = world.joints[D.name]
        cx, cy = j['C']
        g = world.stage.ground(f.x, f.z, world.t, f.y + 2)
        fx.burst(world, fx.ICE, cx, cy, f.z, 160, (60, 260), life=(1.0, 2.2), size=(1.2, 3.0), seed=c.seed(), up=40,
                 zs=0.9)
        fx.burst(world, fx.BLOOD, cx, cy, f.z, 50, (60, 220), life=(0.8, 1.8), size=(1, 2.6), seed=c.seed(), up=30)
        fx.burst(world, fx.GORE, cx, cy, f.z, 16, (60, 200), life=(0.8, 1.6), size=(1, 2.2), seed=c.seed(), up=30)
        fx.burst(world, fx.FROST, cx, cy, f.z, 80, (30, 140), life=(0.4, 1.0), seed=c.seed())
        world.effects.append(fx.Ring(cx, cy, th, 4, 70, 0.4, 'd8f4ff', thick=2))
        world.hidden[D.name] = True
        world.flash(th, 0.35, 0.08, 'e8f8ff')
        world.shake(th, 7.0, 0.6)
        world.cue('ice_shatter', who=D.name)
        world.cue('bone_crunch', who=D.name)
        for k in range(18):
            rng = np.random.default_rng(c.seed())
            world.stage.stamp(f.x + rng.uniform(-14, 14), f.z + rng.uniform(-6, 6), hexc('6e0807'),
                              r=rng.uniform(1.0, 2.4), alpha=0.9, seed=c.seed(), h=g)
            mg.ice_coat(world, f.x + rng.uniform(-16, 16), f.z + rng.uniform(-8, 8), 5.0, c.seed(), h=g, k=0.5)
    w.at(th + 0.01, shatter)
    w.mark('shatter', th + 0.05)
    fatality_word(c, th + 0.3)
    guard(A, th + 0.5)
    return win_card(c, A.name, th + 2.0)


def knockout(c, A, D, t):
    """No fatality: a last huge blow; D goes down, tries to rise, falls back; the winner stands over him."""
    w = c.w
    t = engage(c, A, D, t)
    th = strike(c, A.name, D.name, t, choose_blow(c, A, 'heavy'), 'head', react=None, blood=18, size=14, spark='big',
                shake=6.0, flash=0.3, wound=wound('head', 'stain', 2.4, c.rng))
    w.slowmo(th - 0.02, th + 0.6, 0.3)
    s = sd(A, D, th)
    to = (clamp_x(c, D.x_at(th) + s * 70 * power(A), 6), D.z_at(th))
    tumble_to(c, D, th, to, 0.8, 30)
    t_d = th + 0.8
    g = floor_at(c, to[0], to[1], t_d)
    D.add(t_d, 'bounce', 0.4, air=arc(8, g, g))
    floor_hit(c, t_d, D.name, 1.5)
    D.add(t_d + 0.4, 'lie_front', 1.2, air=hold_y(g))
    D.add(t_d + 1.6, 'kneel_up', 1.0, air=hold_y(g))
    D.add(t_d + 2.6, 'fall_back', 0.8, air=hold_y(g))
    D.add(t_d + 3.4, 'lie_front', 30.0, air=hold_y(g))
    w.mark('knockout', th)
    w.mark('down', t_d + 3.0)
    A.add(th + 0.6, 'stalk', 1.2, to=(to[0] - s * 30, to[1]))
    guard(A, th + 1.8)
    return win_card(c, A.name, t_d + 3.6)


FINISHERS = {'fire_breath': fire_breath, 'ice_shatter': ice_shatter, 'knockout': knockout}
CLIMAXES = {'spear': spear, 'ice_spikes_launch': ice_spikes_launch, 'shield_combo': shield_combo,
            'leap_smash': leap_smash_climax, 'big_combo': big_combo}
