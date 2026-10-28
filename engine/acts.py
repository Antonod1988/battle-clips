"""Choreography primitives: one blow, one clash, a leap, a fall, a burning body. Every beat in beats.py is built from
these. They schedule actions on the fighters' timelines and events on the world (sparks, blood, wounds, squash, shake,
slow motion, sound cues) at the right scene times.

`strike` places the attacker so his fist, foot, blade or hammer meets the target point at the move's hit frame; the
reaction, knockback and effects scale with the attacker's power tier (see power()).
"""
import math
import numpy as np

import moves
import rig
import body
import fx
import magic as mg
from scene import arc, hold_y
from proj import floor_y
from px import hexc

HIT = {'jab': moves.JAB_HIT, 'cross': moves.CROSS_HIT, 'hook': moves.HOOK_HIT, 'uppercut': moves.UPPER_HIT,
       'roundhouse': moves.KICK_HIT, 'front_kick': 0.45, 'knee': moves.KNEE_HIT, 'sweep': moves.SWEEP_HIT,
       'jump_kick': moves.JUMP_HIT, 'air_kick': moves.AIRKICK_HIT, 'air_punch': moves.AIRPUNCH_HIT,
       'axe_kick': moves.AXE_HIT, 'elbow': moves.ELBOW_HIT, 'slash_down': moves.SLASH_HIT, 'slash_up': moves.SLASH_HIT,
       'slash_across': moves.SLASH_HIT, 'thrust': moves.THRUST_HIT, 'x_slash': moves.SLASH_HIT,
       'hammer_smash': moves.HAMMER_HIT, 'hammer_swing': moves.HAMMER_HIT, 'hammer_upper': moves.HAMMER_HIT,
       'ice_slide': moves.SLIDE_HIT, 'spin_slash': 0.55, 'bash': moves.BASH_HIT, 'up_bash': moves.UPBASH_HIT,
       'flip_kick': moves.FLIPKICK_HIT, 'base_slide': moves.BASESLIDE_HIT, 'swat': moves.SWAT_HIT,
       'haymaker': moves.HAY_HIT, 'hulk_jab': moves.JABH_HIT, 'double_fist': moves.DFIST_HIT,
       'double_fist_high': moves.DFISTHI_HIT, 'headbutt': moves.HEADBUTT_HIT, 'stomp': moves.STOMP_HIT}
LIMB = {'jab': 'FW', 'cross': 'NW', 'hook': 'NW', 'uppercut': 'NW', 'roundhouse': 'NT', 'front_kick': 'NT',
        'knee': 'NK', 'sweep': 'NT', 'jump_kick': 'FT', 'air_kick': 'NT', 'air_punch': 'NW', 'axe_kick': 'NT',
        'elbow': 'NE', 'ice_slide': 'NT', 'bash': 'FW', 'up_bash': 'FW', 'flip_kick': 'FT', 'base_slide': 'FT',
        'swat': 'NW', 'haymaker': 'NW', 'hulk_jab': 'FW', 'double_fist': 'FW', 'double_fist_high': 'FW',
        'headbutt': 'Hc', 'stomp': 'NT'}
DUR = {'jab': 0.24, 'cross': 0.28, 'hook': 0.32, 'uppercut': 0.42, 'roundhouse': 0.48, 'front_kick': 0.42,
       'knee': 0.36, 'sweep': 0.5, 'jump_kick': 0.7, 'air_kick': 0.34, 'air_punch': 0.3, 'axe_kick': 0.4,
       'elbow': 0.3, 'slash_down': 0.4, 'slash_up': 0.4, 'slash_across': 0.38, 'thrust': 0.36, 'x_slash': 0.46,
       'hammer_smash': 0.62, 'hammer_swing': 0.56, 'hammer_upper': 0.56, 'ice_slide': 0.7, 'spin_slash': 0.6,
       'bash': 0.3, 'up_bash': 0.42, 'flip_kick': 0.62, 'base_slide': 0.6, 'swat': 0.46, 'haymaker': 0.62,
       'hulk_jab': 0.32, 'double_fist': 0.64, 'double_fist_high': 0.6, 'headbutt': 0.4, 'stomp': 0.5}
AIR = {'air_kick', 'air_punch', 'axe_kick', 'jump_kick', 'spin_slash'}
BLADES = {'slash_down', 'slash_up', 'slash_across', 'thrust', 'x_slash', 'hammer_smash', 'hammer_swing',
          'hammer_upper', 'spin_slash'}
PUNCHES = {'jab', 'cross', 'hook', 'uppercut', 'elbow', 'air_punch', 'bash', 'up_bash', 'swat', 'haymaker', 'hulk_jab',
           'double_fist', 'double_fist_high', 'headbutt'}
SHIELD_BLOWS = {'bash', 'up_bash'}


class Ctx:
    """What the beats share: the world, the random stream, the stage, the counters and the log."""

    def __init__(self, world, rng, seed0=400):
        self.w = world
        self.rng = rng
        self.seed_n = seed0
        self.log = []
        self.count = {'hits': 0, 'blocks': 0}

    def seed(self):
        self.seed_n += 1
        return self.seed_n


def power(f):
    """How much a fighter's blows move the world: 1.0 for a magic warrior (tier 3), less for a trained human, far more
    for a titan. Drives knockback, shake, sparks, leap heights, magic and destruction sizes."""
    return {1: 0.75, 2: 0.88, 3: 1.0, 4: 1.35, 5: 1.8}.get(f.tier, 1.0)


def leap_k(f):
    """Leap height and length factor by tier."""
    return {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.3, 5: 1.6}.get(f.tier, 1.0)


# ------------------------------------------------------------------ geometry helpers
def target_local(j, where):
    up = j['up']
    if where == 'head':
        return (j['Hc'][0] + 2.5, j['Hc'][1] + 0.5)
    if where == 'chin':
        return (j['Hc'][0] + 2.0, j['Hc'][1] + 3.5)
    if where == 'chest':
        return (j['C'][0] + 3.5, j['C'][1] + 4.0)
    if where == 'gut':
        return (j['P'][0] + up[0] * 7 + 4.0, j['P'][1] + up[1] * 7)
    if where == 'legs':
        return ((j['FA'][0] + j['NA'][0]) / 2 + 2.0, (j['FA'][1] + j['NA'][1]) / 2 - 4.0)
    if where == 'guard':
        return (j['FW'][0] + 2.5, j['FW'][1])
    if where == 'shoulder':
        return (j['FS'][0] + 1.0, j['FS'][1])
    if where == 'back':
        return (j['C'][0] - 3.0, j['C'][1] + 3.0)
    raise KeyError(where)


def world_point(f, t, where):
    j = rig.solve(f.pose_at(t), f.look.dims)
    lx, ly = target_local(j, where)
    return (f.x_at(t) + f.facing_at(t) * f.K * lx, floor_y(f.z_at(t)) - f.y_at(t) + f.K * ly)


def joint_point(f, t, name):
    j = rig.solve(f.pose_at(t), f.look.dims)
    return (f.x_at(t) + f.facing_at(t) * f.K * j[name][0], floor_y(f.z_at(t)) - f.y_at(t) + f.K * j[name][1])


def tip_local(f, move, hu, limb, params, wep=None):
    j = rig.solve(move(hu, **params), f.look.dims)
    if limb in ('NB', 'FB'):
        old = body.SC[0]
        body.SC[0] = 1.0
        p = body.weapon_points(j, limb[0].lower(), wep, 1, 1.0, frac=0.72)
        body.SC[0] = old
        return p
    p = j[limb]
    if limb in ('FW', 'NW'):
        e = j[limb[0] + 'E']
        d = (p[0] - e[0], p[1] - e[1])
        L = math.hypot(*d) + 1e-6
        p = (p[0] + d[0] / L * 2.2, p[1] + d[1] / L * 2.2)
    return p


def gy(w, f, t):
    """The height of the surface under a fighter at t (he may be in the air)."""
    return w.stage.ground(f.x_at(t), f.z_at(t), t, f.y_at(t) + 1.0)


def rel(f, t, h, y1=None):
    y0 = f.y_at(t)
    return arc(h, y0, y0 if y1 is None else y1)


def hand_pt(f, t, side='n'):
    return joint_point(f, t, side.upper() + 'W')


def keep_in(c, D, t, away, knock):
    """Shortens a knockback so it does not carry D off the area he fights in (over a cliff, off a roof)."""
    st = c.w.stage
    if not hasattr(c, 'area') or c.area not in getattr(st, 'AREAS', {}):
        return knock
    a = st.AREAS[c.area]
    x = D.x_at(t)
    room = (a['x'][1] + 18 - x) if away > 0 else (x - a['x'][0] + 18)
    return max(0.0, min(knock, room))


def guard_move(f, t):
    """The guard that fits what he holds."""
    n = f.wep_at(t, 'n')[0]
    fw = f.wep_at(t, 'f')[0]
    if n == 'hammer':
        return 'hammer_guard'
    if n == 'sword' and fw == 'sword':
        return 'sword_guard'
    if n in ('ice_sword', 'sword'):
        return 'blade_guard'
    return 'guard'


def guard(f, t, dur=9.0):
    f.add(t, guard_move(f, t), dur, blend=0.1)


# ------------------------------------------------------------------ blows
def strike(c, att, dfn, t, name, where='head', dur=None, react='hit_head', rdur=0.36, knock=6.0, kz=0.0,
           spark='hit', size=7.0, blood=6, shake=1.5, wound=None, block=False, flash=0.0, react_air=None,
           settle=True, extra=None, hitstop=True, land=None, ice=False, mp=None, wep=None, settle_move='guard',
           at=None, quiet=False, scale=True):
    """A blow that lands where it should. wep=(side, kind) for a blade or a hammer; at=(x, y, z) aims at a fixed point
    instead of a fighter (an ice clone, the floor). Knockback, shake and spark size scale with the attacker's power."""
    w = c.w
    A = w.f[att]
    D = w.f[dfn] if dfn else None
    mv = moves.MOVES[name]
    mp = mp or {}
    dur = dur or DUR[name]
    hu = HIT[name]
    th = t + dur * hu
    pw = power(A) if scale else 1.0
    if D is not None and scale:
        pw = pw * (power(D) ** -0.5) * 1.0          # a titan barely moves when a man hits him
    knock *= pw
    shake *= min(2.2, pw ** 0.8)
    size *= min(1.6, pw ** 0.5)
    limb = (wep[0].upper() + 'B') if wep else LIMB[name]
    tip = tip_local(A, mv, hu, limb, mp, wep[1] if wep else None)
    if at is not None:
        tx, ty, zD = at
        react, block = None, False
    else:
        tx, ty = world_point(D, th, where)
        zD = D.z_at(th)
    xa0, za0 = A.x_at(t), A.z_at(t)
    face = 1 if (D.x_at(th) if D else tx) > xa0 else -1
    e = moves.out(hu)
    dx = ((tx - xa0) * face - A.K * tip[0]) / e
    dz = (zD - za0) / e
    air = None
    if name in AIR:
        need = floor_y(zD) + A.K * tip[1] - ty            # his height at the hit so the tip meets the target
        y0 = A.y_at(t)
        if land is None:
            land = w.stage.ground(xa0 + face * dx, zD, t + dur, y0 + 2.0)
            if land < -900:
                land = y0
        hh = (need - y0 - (land - y0) * hu) / (4 * hu * (1 - hu))
        air = arc(hh, y0, land)
    c.log.append(f'{att} {name} at {t:.2f} hit {th:.2f} step {dx:+.1f} dz {dz:+.1f}')
    A.add(t, mv, dur, dx=dx, dz=dz, ease=moves.out, air=air, face=face, **mp)
    if settle:
        A.add(t + dur, guard_move(A, t + dur) if settle_move == 'guard' else settle_move, 9.0, blend=0.1, face=face)
    shield_block = False
    if block:
        cur = D.find(th - 0.05)[0]
        bm = D.mv('block')
        already = cur is not None and cur.move in (moves.block, moves.block_hold, moves.shield_block) and \
            th - 0.05 < cur.t0 + cur.dur
        shield_block = bm is moves.shield_block
        D.add(th - 0.05, 'block_hold' if already else 'block', 0.36, dx=-knock * 0.4, ease=moves.out, blend=0.04)
        D.add(th + 0.31, guard_move(D, th + 0.31), 9.0, blend=0.1)
    elif react:
        knock = keep_in(c, D, th, face, knock)
        D.add(th, react, rdur, dx=-knock, dz=kz, ease=moves.out, air=react_air, blend=0.03)
        if react in ('hit_head', 'hit_body'):
            D.add(th + rdur, guard_move(D, th + rdur), 9.0, blend=0.1)
    weight = 2 if (spark == 'big' or size >= 11) else 1 if size >= 8 else 0
    blade = wep is not None and wep[1] in ('sword', 'ice_sword', 'axe')
    limb_k = 'blade' if blade else 'hammer' if wep else ('punch' if name in PUNCHES else 'kick')
    w.at(t + dur * hu * 0.55, lambda world: world.cue('swish' if blade else 'whoosh', who=att,
                                                      speed=1.0 + 0.3 * weight, limb=limb_k))

    def ev(world, A=A, D=D):
        x, y = tx, ty
        if quiet:
            if extra:
                extra(world)
            return
        if block and shield_block:
            world.cue('shield_block', who=dfn, weight=weight, big=weight >= 2)
        elif block:
            world.cue('block', who=dfn, weight=weight)
        elif ice:
            world.cue('ice_hit', who=dfn, weight=weight)
        elif blade:
            world.cue('cut', who=dfn, weight=weight, blood=blood)
        elif name in SHIELD_BLOWS:
            world.cue('shield_block', who=dfn, weight=weight, big=False)
            world.cue('hit', who=dfn, weight=weight, limb='punch', blood=blood)
        else:
            world.cue('hit', who=dfn, weight=weight, limb=limb_k, blood=blood)
        k = 'block' if (block or ice) else ('cut' if blade and spark != 'big' else spark)
        world.effects.append(fx.HitSpark(x - face * 1.0, y, th, size * (0.7 if block else 1.0), k, c.seed()))
        if block or ice:
            fx.burst(world, fx.SPARK if block else fx.ICE, x, y, zD, 7, (40, 120), angle=-math.pi / 2, spread=1.4,
                     life=(0.15, 0.4) if block else (0.6, 1.2), size=(1, 1.8), seed=c.seed())
            if block:
                c.count['blocks'] += 1
                world.count['blocks'] += 1
            else:
                c.count['hits'] += 1
                world.count['hits'] += 1
        else:
            c.count['hits'] += 1
            world.count['hits'] += 1
            if blade:
                world.count['cuts'] += 1
            D.flash_t = th
            world.jolt(dfn, 5.0)
            if blood:
                ang = -0.55 if face > 0 else -(math.pi - 0.55)
                fx.burst(world, fx.BLOOD, x, y, zD, blood, (60, 170), angle=ang, spread=0.7, life=(0.8, 1.6),
                         size=(0.8, 2.2), seed=c.seed())
                fx.burst(world, fx.BLOOD, x, y, zD, max(1, blood // 3), (15, 50), angle=math.pi / 2, spread=1.2,
                         life=(0.6, 1.2), size=(0.8, 1.6), seed=c.seed())
                if blood >= 10:
                    fx.burst(world, fx.MIST, x, y, zD, 6, (10, 40), angle=ang, spread=0.8, life=(0.2, 0.45),
                             size=(1.2, 2.0), seed=c.seed())
            if wound:
                for wd in (wound if isinstance(wound, list) else [wound]):
                    D.st.wounds.append(dict(wd, seed=c.seed()))
        world.shake(th, shake, 0.25)
        if flash:
            world.flash(th, flash, 0.1)
        if extra:
            extra(world)
    w.at(th, ev)
    if hitstop and not block:
        w.slowmo(th, th + (0.012 if spark == 'big' else 0.007), 0.06, ramp=0.0)
    return th


def whiff(c, att, dfn, t, name, where='head', dur=None, mp=None, wep=None):
    """A blow aimed where he was (he gets out of the way): placed like a strike, nothing lands."""
    w = c.w
    A, D = w.f[att], w.f[dfn]
    mv = moves.MOVES[name]
    mp = mp or {}
    dur = dur or DUR[name]
    hu = HIT[name]
    th = t + dur * hu
    limb = (wep[0].upper() + 'B') if wep else LIMB[name]
    tip = tip_local(A, mv, hu, limb, mp, wep[1] if wep else None)
    tx, ty = world_point(D, th, where)
    xa0 = A.x_at(t)
    face = 1 if D.x_at(th) > xa0 else -1
    dx = ((tx - xa0) * face - A.K * tip[0]) / moves.out(hu)
    A.add(t, mv, dur, dx=dx, dz=(D.z_at(th) - A.z_at(t)), ease=moves.out, face=face, **mp)
    A.add(t + dur, guard_move(A, t + dur), 9.0, blend=0.1, face=face)
    blade = wep is not None and wep[1] in ('sword', 'ice_sword')
    w.at(t + dur * hu * 0.55, lambda world: world.cue('swish' if blade else 'whoosh', who=att, speed=1.4,
                                                      limb='blade' if blade else 'kick'))
    return th


def clash(c, att, dfn, t, name='slash_down', ah='n', dh='n', dist=24.0, big=False, push=4.0, high=True, dz=0.0,
          settle=True):
    """Blade meets blade: the attacker cuts, the defender's parry catches it; sparks fly where the two blades cross."""
    w = c.w
    A, D = w.f[att], w.f[dfn]
    dur = DUR[name]
    th = t + dur * HIT[name]
    xa, xd = A.x_at(t), D.x_at(th)
    face = 1 if xd > xa else -1
    step = (xd - xa) * face - dist * (A.K + D.K) / 1.5
    A.add(t, name, dur, dx=step / moves.out(HIT[name]) * 0.9, dz=(D.z_at(th) - A.z_at(t) + dz) * 0.8, ease=moves.out,
          face=face, hand=ah)
    if settle:
        A.add(t + dur, guard_move(A, t + dur), 9.0, blend=0.1, face=face)
    D.add(th - 0.13, 'parry', 0.42, dx=-push * power(A), ease=moves.out, blend=0.04, hand=dh, high=high)
    D.add(th + 0.29, guard_move(D, th + 0.29), 9.0, blend=0.1)
    w.at(t + dur * 0.25, lambda world: world.cue('swish', who=att, speed=1.3, limb='blade'))

    def ev(world):
        pa = world.wtips.get(att, {}).get(ah)
        pd = world.wtips.get(dfn, {}).get(dh)
        if pa and pd:
            a1 = (pa[0][0] * 0.3 + pa[1][0] * 0.7, pa[0][1] * 0.3 + pa[1][1] * 0.7)
            d1 = (pd[0][0] * 0.35 + pd[1][0] * 0.65, pd[0][1] * 0.35 + pd[1][1] * 0.65)
            x, y = (a1[0] + d1[0]) / 2, (a1[1] + d1[1]) / 2
        else:
            j = world.joints[dfn]
            x, y = j['Hc'][0] - face * 6, j['Hc'][1] + 4
        z = (world.f[att].z + world.f[dfn].z) / 2
        mg.clash_fx(world, x, y, z, c.seed(), big)
        world.cue('clash', x=x, big=big, ice=('ice' in (pd[2] if pd else '')) or ('ice' in (pa[2] if pa else '')) or
                  ('hammer' in (pd[2] if pd else '')))
        world.jolt(dfn, 2.5)
        world.jolt(att, 1.5)
    w.at(th, ev)
    w.slowmo(th, th + (0.014 if big else 0.008), 0.06, ramp=0.0)
    return th


def dash(c, f, t, to, dur=0.24, move='dash'):
    f.add(t, move, dur, to=to, ease=moves.ease, blend=0.05)
    c.w.trail(f.name, t, t + dur)
    f.add(t + dur, guard_move(f, t + dur), 9.0, blend=0.08)
    return t + dur


def jump_to(c, f, t, to, dur, h, y1=None, move='jump', settle=True, face=None, trail=False):
    """A leap from where he stands to `to` (x, z); lands on the highest surface there unless y1 is given."""
    w = c.w
    y0 = f.y_at(t)
    if y1 is None:
        y1 = w.stage.ground(to[0], to[1], t + dur)
    f.add(t, move, dur, to=to, ease=lambda u: u, air=arc(h, y0, y1), blend=0.04, face=face)
    if trail:
        w.trail(f.name, t, t + dur)
    w.cue('fly', t=t, dur=dur, who=f.name)
    if settle:
        f.add(t + dur, guard_move(f, t + dur), 9.0, blend=0.1)
    return t + dur


def floor_hit(c, t, who, power_=1.0):
    def ev(world):
        f = world.f[who]
        g = world.stage.ground(f.x, f.z, t, f.y + 2)
        world.cue('bodyfall', who=who, power=power_)
        fx.burst(world, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, int(10 * power_), (10, 45), angle=-math.pi / 2,
                 spread=1.5, life=(0.4, 0.9), size=(1.5, 2.6), seed=c.seed(), zs=0.7)
        world.shake(t, 2.0 * power_, 0.25)
    c.w.at(t, ev)


def fall_lie_up(c, f, t, bounce_dx=-10.0, lie=0.3, kip=0.5, h=9.0, g=None, slow=False):
    """Hit the ground: bounce once, lie a moment, kip up (a heavy fighter or a hurt one gets up instead)."""
    w = c.w
    lie *= 0.8
    kip *= 0.9
    if g is None:
        g = w.stage.ground(f.x_at(t), f.z_at(t), t, f.y_at(t) + 2)
        if g < -900:
            g = f.y_at(t)
    f.add(t, 'bounce', 0.36, dx=bounce_dx, air=arc(h, g, g), blend=0.03)
    floor_hit(c, t, f.name, 1.0 * (1.5 if f.heavy else 1.0))
    f.add(t + 0.36, 'lie', lie, air=hold_y(g))
    if slow or f.heavy:
        f.add(t + 0.36 + lie, 'get_up', kip + 0.3, air=hold_y(g), blend=0.04)
        kip = kip + 0.3
    else:
        f.add(t + 0.36 + lie, 'kip_up', kip, air=arc(10, g, g), blend=0.04)
    f.add(t + 0.36 + lie + kip, guard_move(f, t + 0.36 + lie + kip), 9.0, blend=0.1)
    return t + 0.36 + lie + kip


def tumble_to(c, f, t, to, dur, h, y0=None, y1=None, trail=True):
    w = c.w
    yy = f.y_at(t) if y0 is None else y0
    if y1 is None:
        y1 = w.stage.ground(to[0], to[1], t + dur, yy + 3.0)
        if y1 < -900:
            y1 = yy
    f.add(t, 'tumble', dur, to=to, ease=lambda u: u, air=arc(h, yy, y1), blend=0.03)
    w.cue('fly', t=t, dur=dur, who=f.name)
    if trail:
        w.trail(f.name, t, t + dur)
    return y1


def wep(f, t, side, kind, grow=0.0):
    f.weps.append((t, side, kind, grow))


def drip(c, who, t0, t1, every=0.5, where='mouth'):
    state = {'next': t0}

    def fn(world, t, dt):
        if t < state['next'] or world.is_hidden(who, t):
            return
        state['next'] = t + every * (0.6 + 0.8 * ((c.seed() * 37) % 10) / 10.0)
        f = world.f[who]
        j = world.joints[who]
        if where == 'mouth':
            x, y = j['Hc'][0] + f.facing * 3 * f.K, j['Hc'][1] + 3 * f.K
        elif where == 'shoulder':
            x, y = j['FS']
        elif where == 'arm':
            x, y = j['NE']
        else:
            x, y = j['C'][0], j['C'][1] + 5 * f.K
        h = floor_y(f.z) - y
        world.particles.spawn(fx.BLOOD, [x], [f.z], [h], [0.0], [0.0], [-8.0], [3.0], [1.0])
    c.w.effects.append(fx.Emitter(t0, t1, fn))


def smear_floor(c, who, t0, t1):
    def fn(world, t, dt):
        f = world.f[who]
        g = world.stage.ground(f.x, f.z, t, f.y + 1)
        if g > -900 and f.y - g < 3:
            world.stage.stamp(f.x, f.z + 1, fx.BLOOD_C[1], r=1.3, alpha=0.8, seed=c.seed(), h=g)
    c.w.effects.append(fx.Emitter(t0, t1, fn))


def element_hands(c, who, t0, t1, element):
    """Frost (or fire, or crackling energy) on both hands while he gathers a power."""
    w = c.w
    if element == 'ice':
        w.cue('frost', t=t0, dur=t1 - t0, who=who)

        def fn(world, t, dt):
            j = world.joints[who]
            z = world.f[who].z
            for k in ('FW', 'NW'):
                fx.burst(world, fx.FROST, j[k][0], j[k][1], z, 3, (8, 28), life=(0.25, 0.6), seed=c.seed())
        col, col2 = hexc('5ab4ff'), hexc('7ac8ff')
    elif element == 'fire':
        def fn(world, t, dt):
            j = world.joints[who]
            z = world.f[who].z
            for k in ('FW', 'NW'):
                x, y = j[k]
                world.particles.spawn(fx.FLAME, [x], [z], [floor_y(z) - y + 1], [0.0], [0.0], [25.0], [0.25], [0.8])
        col, col2 = hexc('ff7a20'), hexc('ff8a3a')
    else:
        def fn(world, t, dt):
            j = world.joints[who]
            z = world.f[who].z
            for k in ('FW', 'NW'):
                fx.burst(world, fx.ELEC, j[k][0], j[k][1], z, 2, (20, 60), life=(0.1, 0.3), seed=c.seed())
        col, col2 = hexc('9ac8ff'), hexc('c8e8ff')
    w.effects.append(fx.Emitter(t0, t1, fn))

    def glow(cv, add, t, off, lights, world):
        import px
        j = world.joints[who]
        x = (j['FW'][0] + j['NW'][0]) / 2 + off[0]
        y = (j['FW'][1] + j['NW'][1]) / 2 + off[1]
        px.glow(add, x, y, 14, col, 0.45)
        lights.append((x, y, 36, col2, 0.6))
    w.effects.append(fx.Glow(t0, t1, glow))


def fire_body(c, who, t0, t1, rate=6, big=False):
    w = c.w
    w.cue('burning', t=t0, dur=t1 - t0, who=who, big=big)

    def fn(world, t, dt):
        j = world.joints[who]
        z = world.f[who].z
        names = ['P', 'C', 'Hc', 'FE', 'NE', 'FW', 'NW', 'FK', 'NK', 'FS', 'NS']
        rng = np.random.default_rng(c.seed())
        for _ in range(rate):
            a = j[names[rng.integers(len(names))]]
            x, y = a[0] + rng.uniform(-2, 2), a[1] + rng.uniform(-2, 2)
            world.particles.spawn(fx.FLAME, [x], [z + rng.uniform(-1, 1)], [floor_y(z) - y],
                                  [rng.uniform(-10, 10)], [0.0], [rng.uniform(10, 40)], [rng.uniform(0.22, 0.45)],
                                  [rng.uniform(0.7, 1.6 if big else 1.2)], rng=rng)
        if rng.uniform() < 0.3:
            world.particles.spawn(fx.EMBER, [j['C'][0]], [z], [floor_y(z) - j['C'][1]], [rng.uniform(-20, 20)],
                                  [0.0], [rng.uniform(20, 50)], [rng.uniform(0.6, 1.2)], [1.0], rng=rng)
    w.effects.append(fx.Emitter(t0, t1, fn))

    def glow(cv, add, t, off, lights, world):
        j = world.joints[who]
        lights.append((j['C'][0] + off[0], j['C'][1] + off[1], 56 if big else 36, hexc('ff8a3a'), 0.9 if big else 0.6))
    w.effects.append(fx.Glow(t0, t1, glow))


def glint(c, who, t, side):
    """The steel catching the light as it leaves the scabbard."""
    def ev(world):
        tips = world.wtips.get(who, {})
        if side in tips:
            hand, tip, _ = tips[side]
            world.effects.append(fx.HitSpark(tip[0], tip[1], world.t, 5.0, 'clash', c.seed()))
            world.effects.append(fx.Ring(tip[0], tip[1], world.t, 1, 9, 0.15, 'ffffff'))
    c.w.at(t, ev)


def crater(c, x, z, t, power_=1.0, ice=False, g=0.0):
    """A heavy impact on the floor: cracks run out, chips fly, a ring of dust; ice coats it if the weapon was ice."""
    def ev(world):
        rng = np.random.default_rng(c.seed())
        for k in range(int(8 + 4 * power_)):
            ang = rng.uniform(0, 2 * math.pi)
            xx, zz = x, z
            for s_ in range(int(rng.uniform(6, 14) * power_ ** 0.7)):
                xx += math.cos(ang) * 1.5
                zz += math.sin(ang) * 2.0
                ang += rng.uniform(-0.4, 0.4)
                world.stage.stamp(xx, zz, hexc('120e16'), r=0.6, alpha=0.9, seed=c.seed(), h=g)
        if ice:
            mg.ice_coat(world, x, z, 11.0 * power_ ** 0.6, c.seed(), h=g, k=0.45)
            fx.burst(world, fx.ICE, x, floor_y(z) - g - 2, z, int(20 * power_), (40, 120), angle=-math.pi / 2,
                     spread=1.3, life=(0.6, 1.2), seed=c.seed())
        fx.burst(world, fx.CHIP, x, floor_y(z) - g - 1, z, int(18 * power_), (40, 120), angle=-math.pi / 2, spread=1.3,
                 life=(1, 2), size=(1, 2), seed=c.seed(), zs=0.8)
        world.effects.append(fx.Ring(x, floor_y(z) - g, t, 4, 32 * power_ ** 0.7, 0.3, 'd8f0ff' if ice else 'd8d0e0',
                                     squash=0.6, thick=2 if power_ > 1.5 else 1))
        world.shake(t, min(7.0, 4.0 * power_), 0.35 + 0.1 * power_)
        if hasattr(world.stage, 'smash_floor'):
            world.stage.smash_floor(world, x, z, power_, c.seed(), g)
    c.w.at(t, ev)
