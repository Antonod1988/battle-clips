"""The fighters' skeleton (shared by every fighter; a broad build only spreads the shoulders and hips): a pose is a few targets (pelvis, lean, hands, feet); two-bone IK places elbows and knees.

Local coordinates face right (+x towards the foe), y grows downwards, the origin is the ground under the pelvis.
Hands are given relative to their own shoulder, feet relative to the ground origin.
"""
import math
import numpy as np

TORSO, NECK, HEAD = 17.0, 2.0, 4.6
UPPER, FORE = 10.0, 9.0
THIGH, SHIN = 15.5, 15.5
ANKLE = 2.0

# shoulder and hip offsets in the torso's frame (x forward, y down); far = the side towards the foe
F_SHOULDER, N_SHOULDER = (4.2, 1.4), (-3.8, 1.6)
F_HIP, N_HIP = (2.0, 1.2), (-2.0, 1.2)

BASE = {
    'hip': (0.0, -33.0), 'lean': 0.0, 'head': 0.0,
    'fh': (6.0, 12.0), 'nh': (4.0, 13.0),           # relaxed hands
    'ff': (5.0, -ANKLE), 'nf': (-5.0, -ANKLE),      # feet (ankle points)
    'ftoe': 0.0, 'ntoe': 0.0,                       # toe angle, + points the toes down
    'fopen': 0.0, 'nopen': 0.0,                     # 1: open hand
    'fbend': 1.0, 'nbend': 1.0,                     # elbow bend side
    'fkb': -1.0, 'nkb': -1.0,                       # knee bend side
    'twist': 0.0,                                   # 0..1 turns the chest towards the camera (shoulders spread)
    'spin': 0.0,                                    # whole-body rotation about the belly (flips, rolls), + = forward
    'fwa': 0.0, 'nwa': 0.0,                         # weapon angle in each hand, relative to its usual grip
}

VEC_KEYS = ('hip', 'fh', 'nh', 'ff', 'nf')


def pose(**kw):
    p = dict(BASE)
    p.update(kw)
    return p


def lerp_pose(a, b, w):
    out = {}
    for k in BASE:
        va, vb = a.get(k, BASE[k]), b.get(k, BASE[k])
        if k in VEC_KEYS:
            out[k] = (va[0] + (vb[0] - va[0]) * w, va[1] + (vb[1] - va[1]) * w)
        elif k in ('fbend', 'nbend', 'fkb', 'nkb'):
            out[k] = va if w < 0.5 else vb
        else:
            out[k] = va + (vb - va) * w
    return out


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def scale(a, k):
    return (a[0] * k, a[1] * k)


def length(a):
    return math.hypot(a[0], a[1])


def ik(s, t, a, b, bend):
    d = sub(t, s)
    dist = length(d)
    if dist < 1e-4:
        d, dist = (1.0, 0.0), 1e-4
    u = scale(d, 1.0 / dist)
    dist = min(max(dist, abs(a - b) + 0.05), a + b - 0.05)
    ca = (a * a + dist * dist - b * b) / (2 * a * dist)
    ang = math.acos(max(-1.0, min(1.0, ca)))
    elbow = add(s, scale(rot(u, bend * ang), a))
    end = add(s, scale(u, dist))
    return elbow, end


def solve(p, dims=None):
    """Local joint positions for a pose (facing right). dims: {'sh': shoulder spread, 'hip': hip spread} multipliers
    for a broader build (a brute, a titan)."""
    dims = dims or {}
    shk, hipk = dims.get('sh', 1.0), dims.get('hip', 1.0)
    j = {}
    P = p['hip']
    lean = p['lean']
    up = rot((0.0, -1.0), lean)            # torso axis
    j['P'] = P
    C = add(P, scale(up, TORSO))
    j['C'] = C
    j['up'] = up
    j['lean'] = lean
    ha = lean + p['head']
    j['N'] = add(C, scale(rot((0.0, -1.0), ha), NECK))
    j['Hc'] = add(j['N'], scale(rot((0.0, -1.0), ha), HEAD))
    j['head_a'] = ha
    tw = p.get('twist', 0.0)
    fs = (F_SHOULDER[0] * shk - 2.7 * tw, F_SHOULDER[1])          # the chest turns: the far shoulder goes back,
    ns = (N_SHOULDER[0] * shk + 5.8 * tw, N_SHOULDER[1] - 0.5 * tw)  # the near one comes forward (a cross reaches)
    j['FS'] = add(C, rot(fs, lean))
    j['NS'] = add(C, rot(ns, lean))
    j['FHip'] = add(P, rot((F_HIP[0] * hipk, F_HIP[1]), lean * 0.3))
    j['NHip'] = add(P, rot((N_HIP[0] * hipk, N_HIP[1]), lean * 0.3))
    for side, S, T in (('f', j['FS'], p['fh']), ('n', j['NS'], p['nh'])):
        e, w = ik(S, add(S, T), UPPER, FORE, p[side + 'bend'])
        j[side.upper() + 'E'] = e
        j[side.upper() + 'W'] = w
    for side, Hp, T in (('f', j['FHip'], p['ff']), ('n', j['NHip'], p['nf'])):
        k, a = ik(Hp, T, THIGH, SHIN, p[side + 'kb'])
        j[side.upper() + 'K'] = k
        j[side.upper() + 'A'] = a
        toe = p[side + 'toe']
        j[side.upper() + 'T'] = add(a, rot((5.0, 0.0), toe))
    j['fopen'] = p['fopen']
    j['nopen'] = p['nopen']
    j['fwa'] = p.get('fwa', 0.0)
    j['nwa'] = p.get('nwa', 0.0)
    sp = p.get('spin', 0.0)
    if abs(sp) > 1e-4:
        c = add(P, scale(up, 8.0))
        for k, v in list(j.items()):
            if isinstance(v, tuple) and len(v) == 2 and k != 'up':
                j[k] = add(c, rot(sub(v, c), sp))
        j['up'] = rot(j['up'], sp)
        j['lean'] = lean + sp
        j['head_a'] = j['head_a'] + sp
    j['spin'] = sp
    return j


def to_world(j, root, facing):
    """Mirror and move local joints into the world (root = (x, ground_y))."""
    out = {}
    for k, v in j.items():
        if isinstance(v, tuple) and len(v) == 2 and k not in ('up',):
            out[k] = (root[0] + v[0] * facing, root[1] + v[1])
        else:
            out[k] = v
    out['facing'] = facing
    out['up'] = (j['up'][0] * facing, j['up'][1])
    return out
