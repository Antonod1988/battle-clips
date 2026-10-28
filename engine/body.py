"""Draws a fighter from his solved joints: shaded limbs with outlines, the costume from his look data, wounds, tears,
bruises and special states (frozen, burnt, skull, skeleton, electrocuted).

Every limb is a capsule with an anatomical radius profile, shaded per pixel from a top-left light and quantised to a
five-tone ramp, so the figure reads as hand-shaded pixel art. Parts are drawn back to front in groups; each group gets
a one-pixel outline, which also separates a near arm from the chest behind it.

A costume is data (fighters/<name>.py, the 'look' dict): materials (five-tone ramps), bands of material along each
limb, the torso's belly, belt, chest, neckline and emblem, the head (ninja hood and mask, helmet, bare head with hair,
full mask), tabards, a cape, and gear (scabbards on the back, a rope coil, a round shield on the lead arm). Look.from_data
turns that dict into what the painter below uses; nothing here knows any fighter by name.
"""
import math
import numpy as np
from scipy.ndimage import binary_dilation

from px import ramp, hexc, hash_noise

# the drawn size: every radius and body length is multiplied by SC[0] (the scene sets it per fighter per frame:
# his size, times the squash and stretch); the joints arrive already scaled.
SC = [1.0]
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
LIGHT = np.array([-0.45, -0.75, 0.55], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
RIM = np.array([0.8, -0.35, -0.15], np.float32)
RIM /= np.linalg.norm(RIM)

BLOOD = ramp('220202', '4e0606', '820c0a', 'b8180f', 'e0402a')
BONE = ramp('4a4030', '8a7c62', 'c8b996', 'e6dcc0', 'fffaf0')
CHAR = ramp('050303', '120a08', '20120c', '2e1a10', '3e2414')
BRUISE = ramp('120a18', '26142e', '3c2046', '502c5a', '62386c')
MOUTH_C = ramp('1a0204', '3a0608', '5a0c10', '7a1418', '9a2024')

# limb radius profiles (t from the joint nearest the body to the far one), per build
PROF_HUMAN = {
    'upper': ([0, 0.3, 0.7, 1], [3.0, 3.1, 2.6, 2.2]),
    'fore': ([0, 0.25, 0.8, 1], [2.3, 2.6, 2.4, 2.1]),
    'thigh': ([0, 0.3, 0.75, 1], [3.7, 3.9, 3.0, 2.6]),
    'shin': ([0, 0.25, 0.7, 1], [2.6, 3.0, 2.2, 1.9]),
    'foot': ([0, 1], [2.0, 1.5]),
    'abs': ([0, 0.5, 1], [4.6, 4.3, 5.0]),
}
PROF_BRUTE = {
    'upper': ([0, 0.2, 0.5, 0.8, 1], [3.1, 3.5, 3.4, 2.8, 2.4]),     # a bulging biceps
    'fore': ([0, 0.2, 0.55, 1], [2.5, 3.0, 2.8, 2.6]),               # forearms that widen to huge hands
    'thigh': ([0, 0.3, 0.75, 1], [3.8, 4.0, 3.2, 2.7]),
    'shin': ([0, 0.25, 0.7, 1], [2.8, 3.1, 2.5, 2.2]),
    'foot': ([0, 1], [2.2, 1.8]),
    'abs': ([0, 0.45, 1], [4.4, 4.2, 5.4]),
}
# a build: radius profiles, bulk multipliers per part, and the skeleton's shoulder/hip spread
BUILDS = {
    'human': (PROF_HUMAN, {}, {}),
    'lean': (PROF_HUMAN, {'upper': 0.92, 'fore': 0.92, 'thigh': 0.94, 'shin': 0.94, 'chest_c': 0.94, 'abs': 0.94},
             {'sh': 0.96}),
    'athletic': (PROF_HUMAN, {'upper': 1.06, 'chest_c': 1.08, 'deltoid': 1.05}, {'sh': 1.08}),
    'heavy': (PROF_BRUTE, {'upper': 1.3, 'fore': 1.25, 'thigh': 1.2, 'shin': 1.15, 'abs': 1.3, 'chest_a': 1.12,
                           'chest_c': 1.3, 'neck': 1.5, 'fist': 1.3, 'deltoid': 1.25}, {'sh': 1.3, 'hip': 1.2}),
    'brute': (PROF_BRUTE, {'upper': 1.82, 'fore': 1.74, 'thigh': 1.6, 'shin': 1.5, 'foot': 1.45, 'abs': 1.75,
                           'chest_a': 1.3, 'chest_c': 1.7, 'neck': 2.1, 'fist': 1.75, 'deltoid': 1.62, 'head': 1.1},
              {'sh': 1.75, 'hip': 1.5}),
}


class Look:
    """A fighter's costume, ready for the painter (see fighters/_template.py for the data it is built from)."""

    def __init__(self, name, mats, limbs, torso, head, gear, build='human', rim='ffffff', eye='ffffff',
                 outline='07060a', bulk=None, flaps=None, cape=None, smear='ffffff'):
        self.name = name
        self.names = list(mats.keys()) + ['blood', 'bone', 'bruise', 'mouth']
        self.index = {n: i for i, n in enumerate(self.names)}
        self.ramps = [mats[n] for n in mats] + [BLOOD, BONE, BRUISE, MOUTH_C]
        self.limbs = limbs
        self.torso = torso
        self.head = head
        self.gear = gear
        prof, bk, dims = BUILDS[build]
        self.prof = prof
        self.bulk = dict(bk)
        self.bulk.update(bulk or {})
        self.dims = dims
        self.rim = hexc(rim)
        self.eye = hexc(eye)
        self.outline = hexc(outline)
        self.flaps = flaps
        self.cape = cape
        self.smear = hexc(smear)

    def mi(self, name):
        return self.index[name]

    def ramp_of(self, name):
        return self.ramps[self.index[name]]

    @staticmethod
    def from_data(name, d):
        mats = {k: ramp(*v) for k, v in d['palette'].items()}
        return Look(name, mats, d['limbs'], d['torso'], d['head'], d.get('gear', {}), build=d.get('build', 'human'),
                    rim=d.get('rim', 'ffffff'), eye=d.get('eye', 'ffffff'), outline=d.get('outline', '07060a'),
                    bulk=d.get('bulk'), flaps=d.get('flaps'), cape=d.get('cape'), smear=d.get('smear', 'ffffff'))


class State:
    """What has happened to a fighter's body (filled by the choreography)."""

    def __init__(self):
        self.wounds = []        # dicts: part, t, s, r, kind ('tear' | 'stain' | 'cut' | 'bruise'), seed
        self.flash = 0.0        # white hit flash 0..1
        self.frozen = 0.0       # icy tint 0..1
        self.char = 0.0         # burnt 0..1
        self.elec = 0.0         # electrocuted 0..1
        self.xray = False       # the skeleton shows (electrocution, alternate frames)
        self.skull = False      # mask off: a burning skull
        self.skeleton = 0.0     # 0 body, 1 bones only
        self.glow_eyes = 0.0
        self.flap = 0.0         # tabard/cape swing (radians), from the body's motion
        self.mask_torn = 0.0    # 0..1, the lower mask ripped (his mouth shows)
        self.roar = 0.0         # mouth open 0..1 (bare heads)
        self.anger = 0.0        # glowing eyes and veins 0..1 (bare heads)
        self.dirt = 0.0         # dust and grime 0..1
        self.hide = False
        self.weps = {}          # side ('f' or 'n') -> (kind, grow 0..1): what each hand holds
        self.back = 0           # swords still sheathed on his back
        self.shield = True      # the shield is on his arm (not thrown)
        self.alpha = 1.0        # see-through (an ice clone)
        self.skull_only = False
        self.view = 0.0         # -1 three-quarter back .. 0 profile .. +1 three-quarter front (set by the scene)


def dilate(m):
    return binary_dilation(m, CROSS)


class Canvas:
    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        ys, xs = np.mgrid[0:h, 0:w]
        self.xs = (xs + x0 + 0.5).astype(np.float32)
        self.ys = (ys + y0 + 0.5).astype(np.float32)
        self.ix = np.floor(self.xs).astype(int)
        self.iy = np.floor(self.ys).astype(int)
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def rgba(self):
        return np.concatenate([self.rgb, self.a[..., None]], axis=2)


def capsule(cv, A, B, ra, rb, prof=None, mult=1.0):
    ax, ay = A
    bx, by = B
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby + 1e-6
    L = math.sqrt(L2)
    t = ((cv.xs - ax) * abx + (cv.ys - ay) * aby) / L2
    tc = np.clip(t, 0.0, 1.0)
    dx = cv.xs - (ax + tc * abx)
    dy = cv.ys - (ay + tc * aby)
    k = SC[0] * mult
    if prof is None:
        r = (ra + (rb - ra) * tc) * k
    else:
        r = np.interp(tc, prof[0], prof[1]).astype(np.float32) * k
    inside = dx * dx + dy * dy <= r * r
    nx, ny = -aby / L, abx / L
    rr = np.maximum(r, 1e-3)
    s = (dx * nx + dy * ny) / rr
    return inside, tc, s, dx / rr, dy / rr, L, r


def shade_tone(nx, ny, far=False, bias=0.0):
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2] + bias
    tone = np.digitize(lam, [0.05, 0.38, 0.70, 0.93])
    if far:
        tone = np.maximum(tone - 1, 0)
    rim = (nx * RIM[0] + ny * RIM[1] + nz * RIM[2]) > 0.62
    return tone, rim


class Part:
    """One shaded shape: its mask, the coordinates along (t) and across (s) it, its normals."""

    def __init__(self, cv, A, B, ra, rb, prof=None, mult=1.0):
        self.inside, self.t, self.s, self.nx, self.ny, self.L, self.r = capsule(cv, A, B, ra, rb, prof, mult)
        self.rm = (float(np.mean(prof[1])) if prof is not None else (ra + rb) * 0.5) * SC[0] * mult


class Ellipse:
    """An ellipse part along an axis (centre, axis unit vector, radius along it, radius across)."""

    def __init__(self, cv, c, axis, ra, rc):
        ra, rc = ra * SC[0], rc * SC[0]
        dx = cv.xs - c[0]
        dy = cv.ys - c[1]
        a = dx * axis[0] + dy * axis[1]
        b = dx * -axis[1] + dy * axis[0]
        u = a / ra
        v = b / rc
        self.inside = u * u + v * v <= 1
        self.t = np.clip(0.5 + u * 0.5, 0, 1)
        self.s = v
        self.nx = (u * axis[0] - v * axis[1]) * 0.95
        self.ny = (u * axis[1] + v * axis[0]) * 0.95
        self.L = ra * 2
        self.rm = rc
        self.u, self.v = u, v


def paint(cv, look, st, part, mats, far=False, name='', facing=1, bias=0.0, rim_on=True, under='skin'):
    """Shade a part from its material map (indices into look.ramps), then lay its wounds on it."""
    tone, rim = shade_tone(part.nx, part.ny, far, bias)
    col = np.zeros((cv.h, cv.w, 3), np.float32)
    ins = part.inside
    for mi in np.unique(mats[ins]):
        if mi < 0:
            rp = look.ramps[-mi - 1]
            m = (mats == mi) & ins
            col[m] = rp[np.clip(tone[m] - 1, 0, 4)]           # a darker wrap of that material
            continue
        rp = look.ramps[int(mi)]
        m = (mats == mi) & ins
        col[m] = rp[tone[m]]
    if rim_on:
        r = rim & ins & (tone >= 1)
        col[r] = col[r] * 0.5 + look.rim * 0.5
    edge_c = look.ramps[0][0] * 0.8
    under_r = look.ramp_of(under) if under in look.index else BLOOD
    for w in st.wounds:
        if w['part'] != name:
            continue
        dt = (part.t - w['t']) * part.L
        ds = (part.s * facing - w['s']) * part.rm
        n = hash_noise(cv.ix, cv.iy, w['seed'])
        wr = w['r'] * 1.35 * SC[0]
        kind = w['kind']
        if kind == 'stain':
            d = np.sqrt((dt * 0.6) ** 2 + ds ** 2) + n * 1.2 - wr
            m = ins & (d < 0)
            col[m] = BLOOD[np.clip(tone[m], 1, 3)]
        elif kind == 'bruise':
            d = np.sqrt((dt * 0.7) ** 2 + ds ** 2) + n * 1.0 - wr
            m = ins & (d < 0)
            col[m] = col[m] * 0.35 + BRUISE[np.clip(tone[m], 1, 4)] * 0.65
            core = m & (d < -wr * 0.45)
            col[core] = BRUISE[np.clip(tone[core], 0, 2)]
        else:
            d = np.sqrt(dt ** 2 + (ds * 1.2) ** 2) + n * 1.6 - wr
            edge = ins & (d < 1.0) & (d >= 0)
            hole = ins & (d < 0)
            col[edge] = edge_c
            ur = under_r if kind == 'tear' else BLOOD
            col[hole] = ur[np.clip(tone[hole], 1, 4)]
            core = hole & (d < -wr * 0.4)
            col[core] = BLOOD[np.clip(tone[core], 1, 3)]
        if kind in ('tear', 'cut') or (kind == 'stain' and w['r'] >= 1.8):
            down = -dt if name in ('torso', 'flap') else dt
            run = ins & (np.abs(ds - 0.4) < 0.7) & (down > 0) & (down < wr * 3.6) & (n > 0.2)
            col[run] = BLOOD[np.clip(tone[run], 1, 2)]
    return col


def group(cv, look, items, outline=True):
    union = np.zeros((cv.h, cv.w), bool)
    col = np.zeros((cv.h, cv.w, 3), np.float32)
    for m, c in items:
        col[m] = c[m]
        union |= m
    if not union.any():
        return
    if outline:
        ol = dilate(union) & ~union
        cv.rgb[ol] = look.outline
        cv.a[ol] = 1
    cv.rgb[union] = col[union]
    cv.a[union] = 1


# ------------------------------------------------------------------ limbs
def zone_mats(cv, look, p, kind, far):
    """The material map of a limb from its bands: [(material, t0, t1, pattern), ...]. Patterns: 'wraps' (darker
    stripes across), 'wraps_diag', 'rag' (a torn lower edge)."""
    zones = look.limbs.get(kind) or look.limbs.get('default') or [('suit', 0, 1)]
    mats = np.full((cv.h, cv.w), look.mi(zones[0][0]), np.int16)
    t, Lp = p.t, p.L
    for z in zones:
        mat, t0, t1 = z[0], z[1], z[2]
        pat = z[3] if len(z) > 3 else None
        if pat == 'rag':
            rag = t1 + 0.08 * np.sin(p.s * 6.0 + (3.0 if far else 0.0)) + 0.07 * hash_noise(cv.ix // 2, cv.iy // 2, 41 + far)
            band = (t >= t0) & (t < rag)
        else:
            band = (t >= t0) & (t < t1) if t1 < 1.0 else (t >= t0)
        mats[band] = look.mi(mat)
        if pat == 'wraps':
            mats[band & (((t * Lp) % 2.6) < 0.7)] = -look.mi(mat) - 1
        elif pat == 'wraps_diag':
            mats[band & ((((t * Lp) + p.s * 0.8) % 3.0) < 0.75)] = -look.mi(mat) - 1
    return mats, zones


def zone_marks(cv, look, st, p, col, kind, zones, far, facing):
    """Lines drawn over a painted limb: cuffs at band edges, seams, torn edges, veins, the biceps line."""
    t = p.t
    for z in zones:
        opts = z[4] if len(z) > 4 else {}
        pat = z[3] if len(z) > 3 else None
        if opts.get('cuff'):
            cuff = p.inside & (np.abs(t - z[1] - 0.03) < 0.035)
            col[cuff] = col[cuff] * 0.62
        if pat == 'rag':
            edge = p.inside & (np.abs(t - (z[2] + 0.08 * np.sin(p.s * 6.0 + (3.0 if far else 0.0)))) < 0.05)
            col[edge] = col[edge] * 0.55
    marks = look.limbs.get('marks', {})
    if kind in marks.get('seam', ()):
        seam = p.inside & (np.abs(t - 0.12) < 0.04)
        col[seam] = col[seam] * 0.7
    if kind in marks.get('slash', ()):
        slash = p.inside & (np.abs(t - 0.62 - p.s * 0.18) < 0.04) & (p.s > -0.4)
        col[slash] = col[slash] * 0.55
    if kind in marks.get('veins', ()) and st.anger > 0.45:
        vein = p.inside & (np.abs(p.s * facing + 0.25 + 0.12 * np.sin(t * 9.0)) < 0.09) & (t > 0.15) & (t < 0.9)
        col[vein] = col[vein] * 0.78
    if kind in marks.get('muscle', ()):
        line = p.inside & (np.abs(p.s * facing - 0.15) < 0.08) & (t > 0.25) & (t < 0.7)
        col[line] = col[line] * 0.8
    return col


def limb(cv, look, st, A, B, kind, name, far, facing):
    p = Part(cv, A, B, 0, 0, look.prof[kind], look.bulk.get(kind, 1.0))
    mats, zones = zone_mats(cv, look, p, kind, far)
    col = paint(cv, look, st, p, mats, far, name, facing, under=look.limbs.get('under', 'skin'))
    return p.inside, zone_marks(cv, look, st, p, col, kind, zones, far, facing)


def ball(cv, look, st, c, r, mat, far, facing, name='deltoid', bias=0.0):
    p = Part(cv, c, (c[0] + 0.01, c[1]), r, r, None, look.bulk.get('deltoid', 1.0))
    mats = np.full((cv.h, cv.w), look.mi(mat), np.int16)
    return p.inside, paint(cv, look, st, p, mats, far, name, facing, bias)


def fist(cv, look, st, W, E, opened, far, facing):
    d = (W[0] - E[0], W[1] - E[1])
    L = math.hypot(*d) + 1e-6
    u = (d[0] / L, d[1] / L)
    fb = look.bulk.get('fist', 1.0)
    k = SC[0] * fb
    if opened > 0.5:
        p = Part(cv, W, (W[0] + u[0] * 3.4 * k, W[1] + u[1] * 3.4 * k), 1.8, 1.2, None, fb)
    else:
        p = Part(cv, (W[0] + u[0] * 0.5 * k, W[1] + u[1] * 0.5 * k), (W[0] + u[0] * 1.9 * k, W[1] + u[1] * 1.9 * k),
                 2.4, 2.2, None, fb)
    mats = np.full((cv.h, cv.w), look.mi(look.limbs.get('fist', 'suit')), np.int16)
    col = paint(cv, look, st, p, mats, far, 'fist', facing, bias=0.1)
    knuck = p.inside & (p.t > 0.7) & (np.abs(p.s) < 0.7)
    col[knuck] = col[knuck] * 1.13
    return p.inside, col


# ------------------------------------------------------------------ the torso
def star_mask(a, b, r):
    """A five-pointed star of radius r (point up) in local coordinates a (right), b (down)."""
    rr = np.sqrt(a * a + b * b)
    th = np.arctan2(a, -b)
    seg = (th % (2 * math.pi / 5)) / (math.pi / 5)
    lim = r * (0.42 + 0.58 * np.abs(seg - 1.0))
    return rr <= lim


def emblem_mask(kind, a, b, r):
    """Chest emblems in local coordinates (a across, b down, r radius): star, spider, ring, bolt, diamond."""
    if kind == 'star':
        return star_mask(a, b, r)
    if kind == 'spider':
        bodym = ((a / (r * 0.28)) ** 2 + ((b - r * 0.1) / (r * 0.55)) ** 2) <= 1
        head = ((a / (r * 0.2)) ** 2 + ((b + r * 0.55) / (r * 0.2)) ** 2) <= 1
        legs = np.zeros_like(bodym)
        for sgn in (-1, 1):
            for k, (dy, slope) in enumerate(((-0.35, -0.8), (-0.05, -0.25), (0.25, 0.25), (0.5, 0.8))):
                aa = a * sgn
                on = (aa > r * 0.2) & (aa < r * 1.0) & (np.abs(b - (r * dy + (aa - r * 0.2) * slope)) < r * 0.09)
                legs |= on
        return bodym | head | legs
    if kind == 'ring':
        rr = np.sqrt(a * a + b * b)
        return (rr <= r) & (rr >= r * 0.62)
    if kind == 'bolt':
        return (np.abs(a - (-b * 0.45)) < r * 0.28) & (np.abs(b) < r) & ~((b > 0) & (a > r * 0.1))
    if kind == 'diamond':
        return (np.abs(a) / r + np.abs(b) / (r * 1.2)) <= 1
    return np.zeros(a.shape, bool)


def torso(cv, look, st, j, facing):
    P, C, up = j['P'], j['C'], j['up']
    T = look.torso
    items = []
    k = SC[0]
    bk = look.bulk
    # the pelvis and the belly up to the chest
    ab_top = (P[0] + up[0] * 11.0 * k, P[1] + up[1] * 11.0 * k)
    pa = Part(cv, P, ab_top, 0, 0, look.prof['abs'], bk.get('abs', 1.0))
    sfa = pa.s * facing
    mats = np.full((cv.h, cv.w), look.mi(T['belly']), np.int16)
    if T.get('stripes'):
        stripes = (np.floor((sfa + 1.0) * 2.6) % 2) == 1
        mats[stripes] = look.mi(T['stripes'])
    belt = T.get('belt')
    wb = 0.22
    if T.get('belt_ragged'):
        wb = 0.24 + 0.04 * np.sin(sfa * 9.0)
    if belt:
        mats[pa.t < wb] = look.mi(belt)
    col = paint(cv, look, st, pa, mats, False, 'torso', facing, bias=0.03)
    if T.get('seam'):
        seam = pa.inside & (pa.t > 0.22) & (np.abs(sfa - 0.2 - (pa.t - 0.22) * 0.5) < 0.13)
        col[seam] = col[seam] * 0.6
    if T.get('belt_knot'):
        knot = pa.inside & (pa.t < 0.22) & (np.abs(sfa - 0.45) < 0.22)
        col[knot] = look.ramp_of(belt)[3]
        hi = pa.inside & (pa.t < 0.22) & (pa.t > 0.17)
        col[hi] = look.ramp_of(belt)[2]
    if T.get('buckle'):
        buckle = pa.inside & (pa.t < 0.2) & (np.abs(sfa - 0.35) < 0.22)
        col[buckle] = hexc(T['buckle'])
    if T.get('abs'):
        absl = pa.inside & (pa.t > wb + 0.06) & ((np.abs(sfa - 0.25) < 0.07) |
                                                  ((np.abs((pa.t * 4.2) % 1.0 - 0.5) < 0.09) & (np.abs(sfa - 0.25) < 0.6)))
        col[absl] = col[absl] * 0.72
    items.append((pa.inside, col))
    # the chest: a broad ellipse across the shoulders
    ca, cc = 5.6 * bk.get('chest_a', 1.0), 6.4 * bk.get('chest_c', 1.0)
    cen = (P[0] + up[0] * 12.6 * bk.get('chest_a', 1.0) * k, P[1] + up[1] * 12.6 * bk.get('chest_a', 1.0) * k)
    ch = Ellipse(cv, cen, up, ca, cc)
    sf = ch.v * facing                                           # across, + towards the foe
    mats = np.full((cv.h, cv.w), look.mi(T['chest']), np.int16)
    back = st.view < -0.35
    if T.get('vee') and not back:
        vee = (ch.u > 0.25) & (np.abs(sf - 0.3) < (ch.u - 0.25) * 1.1)
        mats[vee] = look.mi(T['vee'])
    colc = paint(cv, look, st, ch, mats, False, 'torso', facing, bias=0.05)
    if T.get('lapel') and not back:
        lap = ch.inside & (ch.u <= 0.3) & (np.abs(sf - 0.3 + (0.3 - ch.u) * 0.55) < 0.09)
        colc[lap] = colc[lap] * 0.6
    if T.get('pecs') and not back:
        if T.get('pecs') == 'big':
            pec = ch.inside & (np.abs(ch.u + 0.05 + 0.25 * np.abs(sf - 0.2)) < 0.08) & (sf > -0.8)
            colc[pec] = colc[pec] * 0.72
            mid = ch.inside & (np.abs(sf - 0.2) < 0.06) & (ch.u < 0.1) & (ch.u > -0.7)
            colc[mid] = colc[mid] * 0.78
        else:
            pec = ch.inside & (np.abs(ch.u + 0.05) < 0.07) & (sf < 0.25) & (sf > -0.7)
            colc[pec] = colc[pec] * 0.8
    em = T.get('emblem')
    if em and not back:
        shift = 0.3 + 0.25 * max(0.0, st.view)
        m = ch.inside & emblem_mask(em['kind'], (sf - shift) * cc, -(ch.u - 0.05) * ca, em.get('size', 2.6))
        er = look.ramp_of(em['mat'])
        colc[m] = er[3]
        colc[m & (sf < 0.2)] = er[4]
    if T.get('low_shade'):
        low = ch.inside & (ch.u < -0.55)
        colc[low] = colc[low] * 0.92
    items.append((ch.inside, colc))
    nk = Part(cv, C, j['N'], 2.5, 2.3, None, bk.get('neck', 1.0))
    items.append((nk.inside, paint(cv, look, st, nk, np.full((cv.h, cv.w), look.mi(T.get('neck', T['chest'])), np.int16),
                                   False, 'neck', facing)))
    return items


def flap(cv, look, st, j, facing, back=False):
    """A tabard hanging from the belt, front (or back), swinging with the body."""
    P = j['P']
    spec = look.flaps
    ang = st.flap * (0.7 if back else 1.0)
    k = SC[0]
    off = (-2.4 if back else 2.2) * k
    top = (P[0] + facing * off, P[1] + 0.5 * k)
    dv = (math.sin(ang) * facing, math.cos(ang))
    ln = (10.0 if back else 12.0) * k * spec.get('length', 1.0)
    bot = (top[0] + dv[0] * ln, top[1] + dv[1] * ln)
    p = Part(cv, top, bot, 1.25, 1.75)
    mats = np.full((cv.h, cv.w), look.mi(spec['mat']), np.int16)
    col = paint(cv, look, st, p, mats, back, 'flap', facing, bias=-0.05)
    hem = p.inside & (p.t > 0.86)
    col[hem] = col[hem] * 0.7
    return p.inside, col


def cape(cv, look, st, j, facing):
    """A cape from the shoulders, streaming back with the motion (drawn behind the body)."""
    spec = look.cape
    k = SC[0]
    C, P = j['C'], j['P']
    sh = ((j['FS'][0] + j['NS'][0]) / 2, (j['FS'][1] + j['NS'][1]) / 2)
    ang = 0.25 + max(-0.3, st.flap) * 0.9
    ln = 26.0 * k * spec.get('length', 1.0)
    dv = (-math.sin(ang) * facing, math.cos(ang))
    bot = (sh[0] + dv[0] * ln, sh[1] + dv[1] * ln)
    p = Part(cv, (sh[0] - facing * 1.5 * k, sh[1]), bot, 3.6, 6.0)
    mats = np.full((cv.h, cv.w), look.mi(spec['mat']), np.int16)
    col = paint(cv, look, st, p, mats, True, 'cape', facing, bias=-0.1)
    fold = p.inside & (np.abs(((p.s + 1.0) * 2.2) % 1.0 - 0.5) < 0.12) & (p.t > 0.25)
    col[fold] = col[fold] * 0.72
    return p.inside, col


# ------------------------------------------------------------------ the head
def head(cv, look, st, j, facing, skull=False):
    Hc = j['Hc']
    a = j['head_a']
    ca, sa = math.cos(a), math.sin(a)
    hb = look.bulk.get('head', 1.0)
    lx = (cv.xs - Hc[0]) * facing
    ly = cv.ys - Hc[1]
    u = (lx * ca + ly * sa) / (SC[0] * hb)
    v = (-lx * sa + ly * ca) / (SC[0] * hb)
    H = look.head
    kind = H.get('kind', 'ninja')
    if kind == 'bare':
        e1 = (u / 4.1) ** 2 + ((v + 0.2) / 4.5) ** 2
        e2 = ((u - 1.0) / 3.7) ** 2 + ((v - 1.9) / 3.0) ** 2
    else:
        e1 = (u / 4.0) ** 2 + (v / 4.9) ** 2
        e2 = ((u - 1.2) / 3.1) ** 2 + ((v - 1.8) / 3.3) ** 2
    inside = (e1 <= 1) | (e2 <= 1)
    nx = np.clip(u / 4.0, -1, 1) * facing
    ny = np.clip(v / 4.9, -1, 1)
    tone, rim = shade_tone(nx * 0.9, ny * 0.9, False, 0.05)
    col = np.zeros((cv.h, cv.w, 3), np.float32)
    if skull or st.skull:
        col[inside] = BONE[np.clip(tone[inside], 1, 4)]
        sock = inside & ((((u - 1.3) / 1.3) ** 2 + ((v + 0.5) / 1.1) ** 2 <= 1) |
                         (((u - 3.3) / 0.8) ** 2 + ((v + 0.5) / 1.1) ** 2 <= 1))
        col[sock] = hexc('1a0804')
        if st.skull:
            burn = sock & (np.abs(v + 0.5) < 0.55)
            col[burn] = hexc('ffb030')
        nose = inside & (np.abs(u - 3.0) < 0.6) & (np.abs(v - 1.1) < 0.6)
        col[nose] = hexc('2a1810')
        teeth = inside & (v > 2.4) & (v < 3.5) & (u > 0.4)
        col[teeth] = BONE[3]
        col[teeth & ((cv.ix % 2) == 0) & (v > 2.8)] = BONE[1]
        cheek = inside & (np.abs(u - 0.0) < 0.5) & (v > 0.5) & (v < 2.2)
        col[cheek] = BONE[1]
        return inside, col
    view = st.view
    back = view < -0.35
    fwd = max(0.0, view)                     # the face turns towards the camera: its features slide back
    if kind == 'ninja':
        hood = look.ramp_of(H['hood'])
        col[inside] = hood[tone[inside]]
        r = rim & inside & (tone >= 1)
        col[r] = col[r] * 0.5 + look.rim * 0.5
        if back:
            knot = inside & (np.abs(u + 3.2) < 0.9) & (np.abs(v + 0.8) < 0.7)
            col[knot] = hood[1]
        else:
            seam = inside & (np.abs(u + 0.4 - v * 0.2) < 0.33) & (v < -1.6)
            col[seam] = col[seam] * 0.72
            f0 = -0.5 - 1.6 * fwd
            face = inside & (u > f0) & (v > -2.0) & (v < 0.35)
            col[face] = look.ramp_of(H.get('skin', 'skin'))[np.clip(tone[face], 1, 3)]
            mask = inside & (u > -1.6 - 1.2 * fwd) & (v >= 0.35)
            col[mask] = look.ramp_of(H['mask'])[np.clip(tone[mask] + 1, 1, 4)]
            if H.get('mask_line'):
                ml = mask & (np.abs(v - 1.6 - 0.5 * np.sign(np.sin(u * 2.2))) < 0.36)
                col[ml] = hexc(H['mask_line'])
            if st.mask_torn > 0:
                torn = mask & (u > 0.6) & (v > 1.5) & (v < 1.5 + 2.2 * st.mask_torn)
                col[torn] = look.ramp_of(H.get('skin', 'skin'))[np.clip(tone[torn], 1, 3)]
                mouth = torn & (np.abs(v - 2.5) < 0.5) & (u > 1.6)
                col[mouth] = BLOOD[2]
            brow = face & (v < -1.2)
            col[brow] = col[brow] * 0.68
            eye_c = look.eye if st.glow_eyes <= 0 else hexc('fff2c0')
            for ex, w in ((1.2 - 1.4 * fwd, 1.3), (3.2 - 0.8 * fwd, 0.7 + 0.6 * fwd)):
                e = inside & (np.abs(u - ex) < w * 0.75) & (np.abs(v + 0.8) < 0.55)
                col[e] = eye_c
    elif kind == 'helmet':
        hm = look.ramp_of(H['helmet'])
        col[inside] = hm[tone[inside]]
        r = rim & inside & (tone >= 1)
        col[r] = col[r] * 0.5 + look.rim * 0.5
        if not back:
            sk = look.ramp_of(H.get('skin', 'skin'))
            face = inside & (u > 0.0 - 1.4 * fwd) & (v > 0.05)
            col[face] = sk[np.clip(tone[face], 1, 4)]
            chin = face & (v > 2.6)
            col[chin] = sk[np.clip(tone[chin] - 1, 0, 4)]
            mouth = face & (np.abs(v - 2.0) < 0.26) & (u > 1.6 - 1.2 * fwd)
            col[mouth] = sk[0]
            for ex in ((2.1,) if fwd < 0.3 else (2.1 - 1.2 * fwd, 0.1 - 1.0 * fwd)):
                eye = inside & (np.abs(u - ex) < 0.75) & (np.abs(v + 0.85) < 0.42)
                col[eye] = look.eye
            if H.get('letter'):
                lc = look.ramp_of(H.get('letter_mat', 'white'))[4]
                lu = 1.4 - 1.0 * fwd
                letter = inside & (v > -3.6) & (v < -1.8) & (
                    (np.abs(np.abs(u - lu) - (v + 3.6) * 0.38) < 0.42) | ((np.abs(u - lu) < 0.75) & (np.abs(v + 2.45) < 0.28)))
                col[letter] = lc
        if H.get('wings'):
            wr = look.ramp_of(H.get('letter_mat', 'white'))
            wing = inside & (u < 0.6) & (u > -2.4) & (np.abs(v + 1.7 - (u + 0.7) * 0.42) < 0.42 + (0.6 - u) * 0.12)
            col[wing] = wr[np.clip(tone[wing] + 1, 2, 4)]
    elif kind == 'mask':
        mk = look.ramp_of(H['mask'])
        col[inside] = mk[tone[inside]]
        r = rim & inside & (tone >= 1)
        col[r] = col[r] * 0.5 + look.rim * 0.5
        if H.get('web') and True:
            rr = np.sqrt((u - 1.5) ** 2 + (v + 0.2) ** 2)
            th = np.arctan2(v + 0.2, u - 1.5)
            web = inside & ((np.abs(((th / (math.pi / 5)) % 1.0) - 0.5) > 0.44) | (np.abs((rr % 1.6) - 0.8) > 0.72))
            col[web] = hexc(H['web'])
        if not back:
            lens = look.eye
            for ex, ey, w in ((2.4 - 1.4 * fwd, -0.8, 1.1 + 0.3 * fwd),) + (((0.0 - 1.3 * fwd, -0.8, 0.9),)
                                                                         if fwd > 0.25 else ()):
                e = inside & ((((u - ex) / w) ** 2 + ((v - ey) / 1.0) ** 2) <= 1)
                rim_e = inside & ((((u - ex) / (w + 0.5)) ** 2 + ((v - ey) / 1.45) ** 2) <= 1) & ~e
                col[rim_e] = hexc(H.get('lens_rim', '101014'))
                col[e] = lens
    else:  # 'bare': a face, a brow, hair
        sk = look.ramp_of(H.get('skin', 'skin'))
        col[inside] = sk[tone[inside]]
        r = rim & inside & (tone >= 1)
        col[r] = col[r] * 0.5 + look.rim * 0.5
        if not back:
            brow = inside & (u > 0.2 - 1.2 * fwd) & (np.abs(v + 1.25) < 0.5)
            col[brow] = sk[np.clip(tone[brow] - 1, 0, 4)]
            eye_c = look.eye if st.anger < 0.5 else hexc(H.get('angry_eye', 'd8ff70'))
            eyes = inside & (np.abs(u - 2.15 + 1.3 * fwd) < 0.8) & (np.abs(v + 0.55) < 0.34)
            col[eyes] = eye_c
            nose = inside & (np.abs(u - 3.5 + 1.6 * fwd) < 0.5) & (np.abs(v - 0.6) < 0.5)
            col[nose] = sk[1]
            if st.roar > 0.05:
                mo = inside & (u > 0.7 - fwd) & (v > 1.1) & (v < 1.1 + 2.6 * st.roar)
                col[mo] = MOUTH_C[np.clip(tone[mo], 0, 2)]
                teeth = mo & ((v < 1.55) | (v > 1.1 + 2.6 * st.roar - 0.45)) & (u > 1.3 - fwd)
                col[teeth] = hexc('f0f0d8')
            else:
                mo = inside & (u > 1.3 - fwd) & (np.abs(v - 2.1 + (u - 2.5) * 0.12) < 0.3)
                col[mo] = sk[0]
        if H.get('hair'):
            hr = look.ramp_of(H['hair'])
            hair = inside & ((v < -2.5 + 0.55 * np.sin(u * 2.6 + 1.0)) | ((u < -2.3 + 0.4 * np.sin(v * 3.0)) & (v < 1.2)))
            if back:
                hair = inside & (v < 1.0)
            col[hair] = hr[np.clip(tone[hair] + 1, 0, 4)]
            tips = hair & (hash_noise(cv.ix, cv.iy, 7) > 0.8)
            col[tips] = hr[4]
    for wd in st.wounds:
        if wd['part'] != 'head':
            continue
        n = hash_noise(cv.ix, cv.iy, wd['seed'])
        d = np.sqrt((u - wd['t']) ** 2 + ((v - wd['s']) * 0.8) ** 2) + n * 1.1 - wd['r'] * 1.3
        m = inside & (d < 0)
        if wd['kind'] == 'bruise':
            col[m] = col[m] * 0.35 + BRUISE[np.clip(tone[m], 1, 4)] * 0.65
        else:
            col[m] = BLOOD[np.clip(tone[m], 1, 3)]
    return inside, col


def skeleton(cv, look, st, j, facing):
    """Bones on the same joints (a burnt body, an electrocution's flash)."""
    items = []
    bmat = np.full((cv.h, cv.w), look.mi('bone'), np.int16)

    def bone(A, B, r=1.0, knob=True):
        p = Part(cv, A, B, r, r * 0.9)
        items.append((p.inside, paint(cv, look, st, p, bmat, False, 'bone', facing, rim_on=False)))
        if knob:
            for Q in (A, B):
                q = Part(cv, Q, (Q[0] + 0.01, Q[1]), 1.5, 1.5)
                items.append((q.inside, paint(cv, look, st, q, bmat, False, 'bone', facing, rim_on=False)))

    for s in ('F', 'N'):
        bone(j[s + 'Hip'], j[s + 'K'])
        bone(j[s + 'K'], j[s + 'A'])
        bone(j[s + 'A'], j[s + 'T'], 0.9, False)
        bone(j[s + 'S'], j[s + 'E'])
        bone(j[s + 'E'], j[s + 'W'])
    P, C, up = j['P'], j['C'], j['up']
    bone(P, C, 0.8, False)
    k = SC[0]
    pel = Part(cv, (P[0] - 2.5 * k, P[1] + 0.5 * k), (P[0] + 2.5 * k, P[1] + 0.5 * k), 1.8, 1.8)
    items.append((pel.inside, paint(cv, look, st, pel, bmat, False, 'bone', facing, rim_on=False)))
    side = (-up[1], up[0])
    for i in range(4):
        tt = 0.45 + i * 0.14
        c = (P[0] + (C[0] - P[0]) * tt, P[1] + (C[1] - P[1]) * tt)
        w = (4.6 - abs(i - 1.5) * 0.6) * SC[0]
        A = (c[0] - side[0] * w, c[1] - side[1] * w + SC[0])
        B = (c[0] + side[0] * w, c[1] + side[1] * w + SC[0])
        rib = Part(cv, A, B, 0.7, 0.7)
        items.append((rib.inside, paint(cv, look, st, rib, bmat, False, 'bone', facing, rim_on=False)))
    items.append(head(cv, look, st, j, facing, skull=True))
    group(cv, look, items, outline=True)


# ------------------------------------------------------------------ weapons
STEEL = ramp('1a1e24', '444c56', '86929e', 'c6d0da', 'ffffff')
ICEW = ramp('1f4fa8', '3f8ee0', '86ccf4', 'c8f0ff', 'ffffff')
WRAP = ramp('0e0606', '2a1010', '4a1c16', '6e2a1c', '8e4430')
LACQ = ramp('060406', '140a0c', '2a1214', '461c1c', '6a2a26')
GOLDW = ramp('3a2006', '7a5010', 'b8862a', 'e8c060', 'fff0b0')
WOODW = ramp('1a0e06', '3a2410', '5a3a1c', '7c5630', 'a07a4a')
# grip (from, to), blade (from, to, r0, r1) or head (at, half width, radius), base angle off the forearm (rig units);
# 'ice' weapons are drawn as faceted ice and glow; 'grows' weapons are formed in the hand (grow 0..1)
WEAPONS = {'sword': dict(grip=(-3.0, 3.0), blade=(3.6, 24.0, 1.25, 0.55), off=-0.45),
           'ice_sword': dict(grip=(-2.5, 3.0), blade=(3.4, 26.0, 1.8, 0.5), off=-0.45, ice=True, grows=True),
           'hammer': dict(grip=(-5.0, 20.0), head=(20.0, 6.5, 3.6), off=-1.0, ice=True, grows=True),
           'staff': dict(grip=(-14.0, 22.0), off=-0.6, wood=True),
           'axe': dict(grip=(-4.0, 16.0), head=(15.0, 4.0, 3.0), off=-0.9, steel_head=True)}


def weapon_axis(j, side, kind, facing, grow=1.0):
    """The weapon's line in world coordinates: (hand point, unit direction, tip point, scale)."""
    S = side.upper()
    W, E = j[S + 'W'], j[S + 'E']
    d = (W[0] - E[0], W[1] - E[1])
    L = math.hypot(*d) + 1e-6
    a = math.atan2(d[1], d[0]) + facing * (WEAPONS[kind]['off'] + j.get(side + 'wa', 0.0))
    u = (math.cos(a), math.sin(a))
    k = SC[0]
    hand = (W[0] + d[0] / L * 1.2 * k, W[1] + d[1] / L * 1.2 * k)
    spec = WEAPONS[kind]
    reach = spec['blade'][1] if 'blade' in spec else spec['head'][0] + spec['head'][2] if 'head' in spec else spec['grip'][1]
    g = grow if spec.get('grows') else 1.0
    tip = (hand[0] + u[0] * reach * k * g, hand[1] + u[1] * reach * k * g)
    return hand, u, tip, k


def weapon_points(j, side, kind, facing, grow=1.0, frac=0.7):
    """A point a fraction along the striking part (the blade, or the hammer's head)."""
    hand, u, tip, k = weapon_axis(j, side, kind, facing, grow)
    spec = WEAPONS[kind]
    g = grow if spec.get('grows') else 1.0
    if 'blade' in spec:
        b0, b1 = spec['blade'][0], spec['blade'][1] * g
        d = b0 + (b1 - b0) * frac
    elif 'head' in spec:
        d = spec['head'][0] * min(1.0, g)
    else:
        d = spec['grip'][1] * frac
    return (hand[0] + u[0] * d * k, hand[1] + u[1] * d * k)


def weapon(cv, look, st, j, side, kind, grow, facing):
    hand, u, tip, k = weapon_axis(j, side, kind, facing, grow)
    spec = WEAPONS[kind]
    n = (-u[1], u[0])
    items = []

    def seg(a, b, ra, rb, rp, bias=0.0):
        A = (hand[0] + u[0] * a * k, hand[1] + u[1] * a * k)
        B = (hand[0] + u[0] * b * k, hand[1] + u[1] * b * k)
        p = Part(cv, A, B, ra, rb)
        tone, rim = shade_tone(p.nx, p.ny, False, bias)
        col = rp[np.clip(tone, 0, 4)]
        return p, col
    icy = spec.get('ice', False)
    g0, g1 = spec['grip']
    if spec.get('grows') and 'head' in spec:
        g1 = g0 + (g1 - g0) * min(1.0, grow / 0.6)
    grip_r = ICEW if icy else WOODW if spec.get('wood') or 'head' in spec else WRAP
    p, col = seg(g0, g1, 0.85, 0.85, grip_r, 0.1 if icy else 0.0)
    if not icy and not spec.get('wood'):
        band = p.inside & (((p.t * p.L) % 1.8) < 0.6)
        col[band] = col[band] * 0.55
    if icy:
        facet = p.inside & ((((cv.ix * 2 + cv.iy) // 3) % 3) == 0)
        col[facet] = col[facet] * 0.8 + 0.2
    items.append((p.inside, col))
    if 'blade' in spec:
        b0, b1, r0, r1 = spec['blade']
        if spec.get('grows'):
            b1 = b0 + (b1 - b0) * grow
        if b1 > b0 + 0.5:
            br = ICEW if icy else STEEL
            p, col = seg(b0, b1, r0, r1, br, 0.15)
            edge = p.inside & ((p.s * facing) < -0.35)
            col[edge] = br[4]
            if icy:
                # faceted ice: flat planes split along the blade, a bright ridge, dark inner cracks
                plane = (p.s * facing > 0.05)
                col[p.inside & plane] = br[2]
                col[p.inside & ~plane] = br[3]
                ridge = p.inside & (np.abs(p.s * facing - 0.05) < 0.16)
                col[ridge] = br[4]
                crack = p.inside & (np.abs(((p.t * p.L * 0.45) % 1.0) - 0.5 - p.s * 0.3) < 0.07) & (p.s * facing > 0.2)
                col[crack] = br[1]
            items.append((p.inside, col))
        gd = Part(cv, (hand[0] + u[0] * 3.1 * k - n[0] * 1.9 * k, hand[1] + u[1] * 3.1 * k - n[1] * 1.9 * k),
                  (hand[0] + u[0] * 3.1 * k + n[0] * 1.9 * k, hand[1] + u[1] * 3.1 * k + n[1] * 1.9 * k), 0.9, 0.9)
        tone, _ = shade_tone(gd.nx, gd.ny, False, 0.1)
        items.append((gd.inside, (ICEW if icy else GOLDW)[np.clip(tone, 0, 4)]))
    elif 'head' in spec:
        hd, hw, hr = spec['head']
        gh = max(0.0, min(1.0, (grow - 0.45) / 0.55)) if spec.get('grows') else 1.0
        if gh > 0:
            c = (hand[0] + u[0] * hd * k, hand[1] + u[1] * hd * k)
            A = (c[0] - n[0] * hw * k * gh, c[1] - n[1] * hw * k * gh)
            B = (c[0] + n[0] * hw * k * gh, c[1] + n[1] * hw * k * gh)
            p = Part(cv, A, B, hr * gh, hr * gh * 0.9)
            tone, _ = shade_tone(p.nx, p.ny, False, 0.12)
            hrp = ICEW if icy else STEEL
            col = hrp[np.clip(tone, 0, 4)]
            if icy:
                # a faceted block: three flat faces by direction of the normal, a lit top edge, cracks
                face = np.where(p.ny < -0.3, 4, np.where(p.nx * facing > 0.2, 2, 3))
                col = hrp[np.clip(face - (p.t > 0.75) * 1, 0, 4)]
                col[p.inside & (np.abs(p.s) > 0.82)] = hrp[1]
                crack = p.inside & (np.abs(p.s) < 0.12) & (p.t > 0.2) & (p.t < 0.8)
                col[crack] = hrp[4]
                crack2 = p.inside & (np.abs(p.t - 0.45 - p.s * 0.3) < 0.04)
                col[crack2] = hrp[1]
            items.append((p.inside, col))
    return items


def back_swords(cv, look, j, facing, n):
    """Swords in their scabbards, crossed on the back, hilts over the shoulders."""
    k = SC[0]
    up = j['up']
    fwd = (-up[1] * facing, up[0] * facing)
    C, P = j['C'], j['P']
    items = []
    for i in range(2):
        top = (C[0] + up[0] * (2.5 + i * 1.2) * k - fwd[0] * (3.0 + i) * k,
               C[1] + up[1] * (2.5 + i * 1.2) * k - fwd[1] * (3.0 + i) * k)
        bot = (P[0] - fwd[0] * (5.5 - 2.5 * i) * k + up[0] * (-1.5 + 2.0 * i) * k,
               P[1] - fwd[1] * (5.5 - 2.5 * i) * k + up[1] * (-1.5 + 2.0 * i) * k)
        sc = Part(cv, top, bot, 1.3, 1.1)
        tone, _ = shade_tone(sc.nx, sc.ny, False, 0.0)
        items.append((sc.inside, LACQ[np.clip(tone, 0, 4)]))
        if i < n:
            d = (top[0] - bot[0], top[1] - bot[1])
            L = math.hypot(*d) + 1e-6
            h1 = (top[0] + d[0] / L * 6.0 * k, top[1] + d[1] / L * 6.0 * k)
            hp = Part(cv, top, h1, 0.95, 0.95)
            tone, _ = shade_tone(hp.nx, hp.ny, False, 0.0)
            col = WRAP[np.clip(tone, 0, 4)]
            band = hp.inside & (((hp.t * hp.L) % 1.8) < 0.6)
            col[band] = col[band] * 0.55
            items.append((hp.inside, col))
            gd = Part(cv, (top[0] - fwd[0] * 1.6 * k, top[1] - fwd[1] * 1.6 * k),
                      (top[0] + fwd[0] * 1.6 * k, top[1] + fwd[1] * 1.6 * k), 0.8, 0.8)
            tone, _ = shade_tone(gd.nx, gd.ny, False, 0.1)
            items.append((gd.inside, GOLDW[np.clip(tone, 0, 4)]))
    return items


def rope_coil(cv, look, j, facing):
    """A coiled rope at the hip."""
    P = j['P']
    k = SC[0]
    c = (P[0] - 3.4 * k * facing, P[1] + 1.8 * k)
    d = np.sqrt((cv.xs - c[0]) ** 2 + ((cv.ys - c[1]) * 1.3) ** 2) / k
    ring = (d <= 3.0) & (d >= 1.4)
    cv.rgb[ring] = hexc('5a4630')
    cv.rgb[ring & (((cv.ix + cv.iy) % 2) == 0)] = hexc('8a7050')
    cv.a[ring] = 1
    o = (d > 3.0) & (d <= 3.9) & (cv.a > 0)
    cv.rgb[o] = look.outline


# ------------------------------------------------------------------ the round shield
def shield_paint(xs, ys, cx, cy, rx, ry, rot=0.0, tone_k=1.0, dent=0.0, colors=None):
    """A round shield as an ellipse (rx across, ry along): rings, a centre disc and a star (data: colors = (outer ring,
    middle ring, centre, star)). Returns (inside, rgb). rot turns the star (it spins in flight)."""
    colors = colors or ('c8161c', 'eef0f4', '2448b8', 'eef0f4')
    a = (xs - cx) / max(rx, 0.5)
    b = (ys - cy) / max(ry, 0.5)
    rr = np.sqrt(a * a + b * b)
    inside = rr <= 1.0
    col = np.zeros(xs.shape + (3,), np.float32)
    red, white, blue, star_c = (hexc(c) for c in colors)
    col[inside] = red
    col[inside & (rr < 0.78)] = white
    col[inside & (rr < 0.58)] = red
    col[inside & (rr < 0.38)] = blue
    ca, sa = math.cos(rot), math.sin(rot)
    sa_ = a * ca - b * sa
    sb_ = a * sa + b * ca
    stm = inside & star_mask(sa_, sb_, 0.36)
    col[stm] = star_c
    lit = np.clip(0.75 - (a * 0.55 + b * 0.75) * 0.35, 0.45, 1.15) * tone_k
    col *= lit[..., None]
    rim = inside & (rr > 0.9) & ((a + b) < -0.3)
    col[rim] = np.minimum(1.0, col[rim] * 1.5 + 0.15)
    if dent > 0:
        n = hash_noise(np.floor(xs).astype(int), np.floor(ys).astype(int), 5)
        scr = inside & (n > 1 - 0.15 * dent)
        col[scr] = col[scr] * 0.6
    return inside, col


def shield_on_arm(cv, look, st, j, facing):
    W, E = j['FW'], j['FE']
    k = SC[0]
    d = (W[0] - E[0], W[1] - E[1])
    L = math.hypot(*d) + 1e-6
    cx = W[0] - d[0] / L * 1.5 * k + facing * 1.6 * k
    cy = W[1] - d[1] / L * 1.5 * k
    spec = look.gear['shield']
    R = spec.get('radius', 8.6) * k
    wide = 0.52 + 0.3 * max(0.0, st.view)
    return shield_paint(cv.xs, cv.ys, cx, cy, R * wide, R, rot=0.0, dent=min(1.0, len(st.wounds) / 10.0),
                        colors=spec.get('colors'))


# ------------------------------------------------------------------ the whole figure
def bbox(j, pad):
    pts = [v for k, v in j.items() if isinstance(v, tuple) and len(v) == 2 and k not in ('up',)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(math.floor(min(xs) - pad)), int(math.floor(min(ys) - pad)),
            int(math.ceil(max(xs) + pad)), int(math.ceil(max(ys) + pad)))


def draw(j, look, st, facing, pad=15):
    """Render a fighter (world joints) into an RGBA patch; returns (rgba, x0, y0)."""
    x0, y0, x1, y1 = bbox(j, pad + (8 if look.cape else 0))
    for side, (kind, grow) in st.weps.items():
        if kind:
            _, _, tip, _ = weapon_axis(j, side, kind, facing, grow)
            x0, y0 = min(x0, int(math.floor(tip[0] - 9))), min(y0, int(math.floor(tip[1] - 9)))
            x1, y1 = max(x1, int(math.ceil(tip[0] + 9))), max(y1, int(math.ceil(tip[1] + 9)))
    cv = Canvas(x0, y0, x1 - x0, y1 - y0)
    if st.skeleton >= 1.0 or st.xray:
        skeleton(cv, look, st, j, facing)
        return finish(cv, st), x0, y0
    gear = look.gear
    back_view = st.view < -0.35
    arm_mat = look.limbs.get('deltoid', 'suit')
    if look.cape and not back_view:
        group(cv, look, [cape(cv, look, st, j, facing)])
    if gear.get('scabbards') and not st.skull_only and not back_view:
        group(cv, look, back_swords(cv, look, j, facing, st.back))
    if look.flaps and look.flaps.get('back'):
        group(cv, look, [flap(cv, look, st, j, facing, back=True)])
    group(cv, look, [limb(cv, look, st, j['FHip'], j['FK'], 'thigh', 'fleg_u', True, facing),
                     limb(cv, look, st, j['FK'], j['FA'], 'shin', 'fleg_l', True, facing),
                     limb(cv, look, st, j['FA'], j['FT'], 'foot', 'foot', True, facing)])
    group(cv, look, [ball(cv, look, st, j['FS'], 3.1, arm_mat, True, facing),
                     limb(cv, look, st, j['FS'], j['FE'], 'upper', 'farm_u', True, facing),
                     limb(cv, look, st, j['FE'], j['FW'], 'fore', 'farm_l', True, facing),
                     fist(cv, look, st, j['FW'], j['FE'], j['fopen'], True, facing)])
    fw = st.weps.get('f')
    if fw and fw[0]:
        group(cv, look, weapon(cv, look, st, j, 'f', fw[0], fw[1], facing))
    group(cv, look, torso(cv, look, st, j, facing))
    if gear.get('shield') and st.shield and not back_view:
        group(cv, look, [shield_on_arm(cv, look, st, j, facing)])
    group(cv, look, [limb(cv, look, st, j['NHip'], j['NK'], 'thigh', 'nleg_u', False, facing),
                     limb(cv, look, st, j['NK'], j['NA'], 'shin', 'nleg_l', False, facing),
                     limb(cv, look, st, j['NA'], j['NT'], 'foot', 'foot', False, facing)])
    if look.flaps and look.flaps.get('front', True):
        group(cv, look, [flap(cv, look, st, j, facing)])
    if gear.get('rope'):
        rope_coil(cv, look, j, facing)
    group(cv, look, [head(cv, look, st, j, facing)])
    group(cv, look, [ball(cv, look, st, j['NS'], 3.2, arm_mat, False, facing),
                     limb(cv, look, st, j['NS'], j['NE'], 'upper', 'narm_u', False, facing),
                     limb(cv, look, st, j['NE'], j['NW'], 'fore', 'narm_l', False, facing),
                     fist(cv, look, st, j['NW'], j['NE'], j['nopen'], False, facing)])
    nw = st.weps.get('n')
    if nw and nw[0]:
        group(cv, look, weapon(cv, look, st, j, 'n', nw[0], nw[1], facing))
    if back_view:
        # seen from behind: the cape, the scabbards and the shield on the back are in front of everything
        if gear.get('scabbards') and not st.skull_only:
            group(cv, look, back_swords(cv, look, j, facing, st.back))
        if gear.get('shield') and st.shield:
            group(cv, look, [shield_on_arm(cv, look, st, j, facing)])
        if look.cape:
            group(cv, look, [cape(cv, look, st, j, facing)])
    return finish(cv, st), x0, y0


def finish(cv, st):
    rgba = cv.rgba()
    m = rgba[..., 3] > 0
    if st.dirt > 0:
        n = hash_noise(cv.ix, cv.iy, 91)
        d = m & (n < 0.35 * st.dirt)
        rgba[d, :3] *= 0.72
    if st.char > 0:
        lum = rgba[..., :3].mean(axis=2)
        charred = CHAR[np.clip((lum * 9).astype(int), 0, 4)]
        k = min(1.0, st.char)
        rgba[..., :3] = rgba[..., :3] * (1 - k) + charred * k
        n = hash_noise(cv.ix, cv.iy, 77)
        emb = m & (n > 1 - 0.2 * k)
        rgba[emb, :3] = hexc('ff7a1a') * (0.6 + 0.4 * n[emb, None])
    if st.frozen > 0:
        k = st.frozen
        lum = rgba[..., :3].mean(axis=2, keepdims=True)
        ice = np.array([0.55, 0.82, 1.0], np.float32) * (0.45 + lum * 0.9)
        rgba[..., :3] = rgba[..., :3] * (1 - 0.75 * k) + ice * 0.75 * k
    if st.elec > 0:
        lum = rgba[..., :3].mean(axis=2, keepdims=True)
        zap = np.array([0.7, 0.85, 1.0], np.float32) * (0.5 + lum)
        rgba[..., :3] = rgba[..., :3] * (1 - 0.7 * st.elec) + zap * 0.7 * st.elec
    if st.flash > 0:
        rgba[..., :3] = rgba[..., :3] * (1 - st.flash) + st.flash
    if st.alpha < 1.0:
        rgba[..., 3] *= st.alpha
    return rgba
