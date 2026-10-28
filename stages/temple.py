"""The temple (version 4's stage): a cliff-top temple bigger than one screen, seen from slightly above.

World: x along the screen, z into depth (0 at the hall's facade), y height above the courtyard floor (negative below it).
Projection (oblique, as in v3): stage_x = x, stage_y = Y0 + z * FS - y. The camera crops a 480x270 window that travels.

From left to right: pines and the low wall; the courtyard (four pillars with flat caps, four braziers, the fighting ring)
in front of the temple hall whose roof can be stood on; the balustrade along the cliff and, below it, a ledge over the
drop; on the right a broken rope bridge over a chasm to a rock pinnacle with a gong. Surfaces carry height functions so
fighters stand on them; pillars, braziers, the balustrade sections and the bridge are sprites that break.

For the choreographer: AREAS (where fights happen), ROUTES (how the fight travels between them), targets() (what a body
can be thrown into) and the stage protocol the engine draws through (see stages/README in the docs).
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

import px
import fx
from px import hexc, hash_noise, fbm
from proj import set_projection

W, H = px.W, px.H
M = 8                               # canvas margin (shake)
EW, EH = W + 2 * M, H + 2 * M
SW, SH = 1000, 540                  # stage size (stage coords)
PAD = 12                            # stage image margin
Y0, FS = 239.0, 0.6                 # stage y of (z=0, y=0), and the depth foreshortening
CAM_X, CAM_Y = (0.0, float(SW - W)), (0.0, float(SH - H))
FLOOR_X = (40, 600)
FLOOR_Z = (0.0, 236.0)
RING = (320.0, 118.0, 82.0)
# the hall and its roof (front slope from the eave to the ridge)
HALL_X = (170, 470)
ROOF_X = (150, 490)
EAVE_Z, EAVE_Y, RIDGE_Z, RIDGE_Y = 6.0, 58.0, -52.0, 92.0
RAIL_Z, RAIL_Y = 239.0, 16.0        # balustrade centre line and its top
WALL_Z, DROP = 242.0, 56.0          # the courtyard's front wall and how far it drops to the ledge
LEDGE_X, LEDGE_Z = (70, 470), (242.0, 284.0)
BRIDGE_X, BRIDGE_Z, BRIDGE_GAP = (600, 760), (106.0, 130.0), (664, 692)
PEAK_X, PEAK_Z = (760, 960), (60.0, 190.0)
PILLAR_TOP = 70.0
VOID = -999.0


def floor_y(z):
    return Y0 + z * FS


def proj(x, z, y=0.0):
    return x, Y0 + z * FS - y


def roof_y(z):
    return EAVE_Y + (EAVE_Z - z) * (RIDGE_Y - EAVE_Y) / (EAVE_Z - RIDGE_Z)


def sag(x):
    return -5.0 * math.sin(math.pi * min(1.0, max(0.0, (x - BRIDGE_X[0]) / (BRIDGE_X[1] - BRIDGE_X[0]))))


def col255(h, a=255):
    c = hexc(h)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def mix_arr(a, b, k):
    k = np.clip(np.asarray(k, np.float32), 0, 1)
    return a + (b - a) * k[..., None]


def rot_blit(dst, img, pivot_src, pivot_dst, ang, alpha=1.0):
    """Blit an RGBA image rotated by `ang` (radians, screen clockwise) about pivot_src, placing that pivot at pivot_dst.
    Nearest-neighbour, so the pixels stay pixels."""
    h, w = img.shape[:2]
    c, s = math.cos(ang), math.sin(ang)
    corners = [(-pivot_src[0], -pivot_src[1]), (w - pivot_src[0], -pivot_src[1]), (-pivot_src[0], h - pivot_src[1]),
               (w - pivot_src[0], h - pivot_src[1])]
    rc = [(x * c - y * s, x * s + y * c) for x, y in corners]
    x0 = int(math.floor(min(p[0] for p in rc) + pivot_dst[0])) - 1
    x1 = int(math.ceil(max(p[0] for p in rc) + pivot_dst[0])) + 1
    y0 = int(math.floor(min(p[1] for p in rc) + pivot_dst[1])) - 1
    y1 = int(math.ceil(max(p[1] for p in rc) + pivot_dst[1])) + 1
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if x1c <= x0c or y1c <= y0c:
        return
    ys, xs = np.mgrid[y0c:y1c, x0c:x1c]
    u = xs + 0.5 - pivot_dst[0]
    v = ys + 0.5 - pivot_dst[1]
    sx = u * c + v * s + pivot_src[0]
    sy = -u * s + v * c + pivot_src[1]
    ix = np.floor(sx).astype(int)
    iy = np.floor(sy).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    src = np.zeros(ix.shape + (4,), np.float32)
    src[ok] = img[iy[ok], ix[ok]]
    a = src[..., 3:4] * alpha
    dst[y0c:y1c, x0c:x1c] = dst[y0c:y1c, x0c:x1c] * (1 - a) + src[..., :3] * a


# ------------------------------------------------------------------ sprites
class Pillar:
    """A stone pillar with a flat cap a fighter can stand on; it keeps every crack, and it can be toppled."""
    R, HT = 9, int(PILLAR_TOP)

    def __init__(self, x, z, seed):
        self.x, self.z, self.seed = x, z, seed
        self.hits = []
        self.img = None
        self.fall_t = None          # when it starts to topple
        self.fall_dir = 1           # +1 falls to the right (screen), -1 to the left
        self.fall_dur = 0.75

    def standing(self, t):
        return self.fall_t is None or t < self.fall_t

    def angle(self, t):
        if self.fall_t is None or t < self.fall_t:
            return 0.0
        k = (t - self.fall_t) / self.fall_dur
        if k < 1.0:
            a = (math.pi / 2) * k * k
        else:                        # a little bounce on the floor
            b = k - 1.0
            a = math.pi / 2 - 0.12 * math.sin(min(1.0, b * 4.0) * math.pi) * math.exp(-b * 2)
        return a * self.fall_dir

    def hit(self, h, power, seed, out):
        self.hits.append((h, power, seed, out))
        self.img = None

    def bake(self):
        R, HT = self.R, self.HT
        w = 2 * (R + 7) + 1
        h = HT + 18
        c0 = R + 7
        base = h - 4                    # the row of the floor contact
        ys, xs = np.mgrid[0:h, 0:w]
        dx = xs + 0.5 - c0
        row = ys[:, :1]
        hy = base - row                 # height above the floor
        rr = np.full((h, 1), float(R))
        band = ((hy % 24) <= 2) & (hy > 8) & (hy < HT - 8)
        rr = rr + band * 1.0
        rr = np.where(hy < 7, R + 3.0, rr)
        rr = np.where((hy >= 7) & (hy < 10), R + 1.5, rr)
        rr = np.where((hy >= HT - 8) & (hy < HT - 4), R + 1.5, rr)
        inside = (np.abs(dx) <= rr) & (hy >= -1) & (hy < HT - 3)
        s = np.clip(dx / rr, -1, 1)
        lam = -0.55 * s + 0.62 * np.sqrt(1 - s * s)
        stone = np.where(band[..., None], hexc('8a6a32')[None, None, :], hexc('6e687a')[None, None, :])
        n = hash_noise(xs, ys, self.seed)
        rgb = stone * (0.36 + 0.82 * lam[..., None]) * (0.9 + 0.16 * n[..., None])
        flute = (~band) & (np.floor(dx + 100).astype(int) % 4 == 0) & (np.abs(s) < 0.8) & (hy > 10) & (hy < HT - 8)
        rgb[flute] *= 0.82
        edge = inside & (np.abs(np.abs(dx) - rr) < 1.0)
        rgb[edge] *= 0.35
        a = inside.astype(np.float32)
        # the cap: a square slab seen from slightly above (its top face is where a fighter stands)
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        top = base - HT
        d.rectangle([c0 - 12, top - 1, c0 + 12, top + 4], fill=col255('4e4858'))
        d.line([(c0 - 12, top + 4), (c0 + 12, top + 4)], fill=col255('2a2430'))
        d.polygon([(c0 - 12, top - 1), (c0 + 12, top - 1), (c0 + 10, top - 8), (c0 - 10, top - 8)], fill=col255('7a7486'))
        d.line([(c0 - 10, top - 8), (c0 + 10, top - 8)], fill=col255('9a94a6'))
        d.line([(c0 - 12, top - 1), (c0 + 12, top - 1)], fill=col255('8a8496'))
        lt = np.asarray(img).astype(np.float32) / 255.0
        la = lt[..., 3:4]
        rgb = rgb * (1 - la) + lt[..., :3] * la
        a = np.maximum(a, la[..., 0])
        for (hh, power, sd, out) in self.hits:
            rng = np.random.default_rng(sd)
            cy = base - hh
            cx = c0 + out * R * 0.5
            if power > 1.2:
                depth = 2.5 + power
                span = 5 + power * 2.5
                for yy in range(int(cy - span), int(cy + span) + 1):
                    if not 0 <= yy < h:
                        continue
                    dd = depth * (1 - abs(yy - cy) / span) + rng.uniform(-0.8, 0.8)
                    for k in range(int(max(0, dd))):
                        xx = int(round(c0 + out * (R - k)))
                        if 0 <= xx < w:
                            a[yy, xx] = 0.0
                    xx = int(round(c0 + out * (R - max(0, dd))))
                    if 0 <= xx < w and dd > 0:
                        rgb[yy, xx] = hexc('2a2430')
            cr = 1.3 + power * 0.8
            m = ((xs + 0.5 - cx) ** 2 + ((ys + 0.5 - cy) * 1.2) ** 2) <= cr * cr
            rgb[m & inside] *= 0.35
            branches = [(cx, cy, rng.uniform(0, 2 * math.pi), 5 + power * 8) for _ in range(5 + int(power * 3))]
            while branches:
                bx, by, ang, ln = branches.pop()
                for _ in range(int(ln)):
                    ang += rng.uniform(-0.5, 0.5)
                    bx += math.cos(ang)
                    by += math.sin(ang)
                    xi, yi = int(round(bx)), int(round(by))
                    if 0 <= xi < w and 0 <= yi < h and a[yi, xi] > 0 and inside[yi, xi]:
                        rgb[yi, xi] = hexc('0c0a0e')
                        if xi + 1 < w and yi + 1 < h:
                            rgb[yi + 1, xi + 1] = np.minimum(1.0, rgb[yi + 1, xi + 1] * 1.3)
                    if rng.uniform() < 0.08 and ln > 5:
                        branches.append((bx, by, ang + rng.choice([-0.9, 0.9]), ln * 0.45))
            for _ in range(int(4 + power * 3)):
                bx = cx + rng.uniform(-3.5, 3.5)
                by = cy + rng.uniform(-4, 4)
                for k in range(int(rng.uniform(2, 9))):
                    xi, yi = int(round(bx)), int(round(by + k))
                    if 0 <= xi < w and 0 <= yi < h and a[yi, xi] > 0 and inside[yi, xi]:
                        rgb[yi, xi] = hexc('6e0807') if k else hexc('9c0f0b')
        self.img = np.concatenate([rgb.astype(np.float32), a[..., None]], axis=2)
        self.anchor = (c0, base)

    def draw(self, cv, ox, oy, t):
        if self.img is None:
            self.bake()
        ang = self.angle(t)
        bx, by = self.x + ox, floor_y(self.z) + oy
        if ang == 0.0:
            px.blit(cv, self.img, int(round(bx - self.anchor[0])), int(round(by - self.anchor[1])))
        else:
            rot_blit(cv, self.img, (self.anchor[0] + 0.5, self.anchor[1]), (bx + 0.5, by), ang)


class Brazier:
    """A tall bronze brazier on three legs; a body can knock it over and spill its coals."""

    def __init__(self, x, z):
        self.x, self.z = x, z
        self.down_t = None
        self.dir = 1

    def lit(self, t):
        return self.down_t is None or t < self.down_t

    def draw(self, cv, ox, oy, t):
        """Draws at canvas offset (the stage->canvas shift)."""
        x = int(round(self.x + ox))
        yb = int(round(floor_y(self.z) + oy))
        bronze, bronze_d, bronze_l, hi = hexc('6a4a24'), hexc('2e200c'), hexc('a07838'), hexc('e0b060')
        h, w = cv.shape[:2]

        def put(xx, yy, c):
            if 0 <= xx < w and 0 <= yy < h:
                cv[yy, xx] = c
        if self.down_t is not None and t >= self.down_t:
            k = min(1.0, (t - self.down_t) / 0.25)
            ang = k * 1.35 * self.dir
            L = 30
            for s in range(L):
                for dxl, c in ((-4, bronze_d), (4, bronze_d), (0, bronze)):
                    put(x + int(round(dxl * (1 - s / L) + math.sin(ang) * s)), yb - int(round(math.cos(ang) * s)), c)
            bx = x + int(round(math.sin(ang) * (L + 3)))
            by = yb - int(round(math.cos(ang) * (L + 3)))
            for r in range(7):
                for dxp in range(-int(3 + r * 0.8), int(3 + r * 0.8) + 1):
                    put(bx + int(round(dxp * math.cos(ang) + r * math.sin(ang))),
                        by + int(round(dxp * math.sin(ang) - r * math.cos(ang) * 0.4)), bronze if dxp < 0 else bronze_d)
            return None
        L = 30
        for k in range(L):
            for dxl, c in ((-5, bronze_d), (5, bronze_d), (0, bronze)):
                put(x + int(round(dxl * (1 - k / L) ** 2)), yb - k, c)
        for row in range(6):
            yy = yb - L - 1 - row
            half = 4 + row * 0.9
            for dxp in range(-int(half), int(half) + 1):
                s = dxp / max(half, 1)
                c = bronze_l if s < -0.2 else (bronze if s < 0.45 else bronze_d)
                if row == 5:
                    c = hi if dxp < 0 else bronze_l
                put(x + dxp, yy, c)
        fire_shape(cv, x, yb - L - 7, t + self.x * 0.1, 7, 18)
        return (x, yb - L - 14, 60, hexc('ff8a3a'), 0.7)


class RailSection:
    """One span of the balustrade between two posts: coping on little balusters. It breaks into stubs and rubble."""

    def __init__(self, x0, x1, seed):
        self.x0, self.x1, self.seed = x0, x1, seed
        self.z = RAIL_Z
        self.break_t = None
        self.img = None
        self.broken_img = None

    def intact(self, t):
        return self.break_t is None or t < self.break_t

    def _bake(self, broken):
        w = self.x1 - self.x0 + 1
        h = 30
        base = h - 6                     # the row of the centre line at the floor
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        st, st_l, st_d, st_dd = col255('5e5768'), col255('7e7688'), col255('3a3444'), col255('221e28')
        rng = np.random.default_rng(self.seed)
        top = base - int(RAIL_Y)
        d.rectangle([0, base - 2, w - 1, base + 2], fill=st_d)          # the plinth
        d.line([(0, base - 2), (w - 1, base - 2)], fill=st)
        if not broken:
            d.rectangle([0, top - 1, w - 1, top + 3], fill=st)          # the coping (top face, then front face)
            d.line([(0, top - 1), (w - 1, top - 1)], fill=st_l)
            d.line([(0, top + 3), (w - 1, top + 3)], fill=st_dd)
            for k in range(3, w - 2, 5):
                d.rectangle([k, top + 4, k + 2, base - 3], fill=st)
                d.point([(k, top + 4)], fill=st_l)
                d.line([(k + 2, top + 4), (k + 2, base - 3)], fill=st_d)
        else:
            for k in range(3, w - 2, 5):
                hgt = int(rng.integers(0, 6))
                if hgt:
                    d.rectangle([k, base - 3 - hgt, k + 2, base - 3], fill=st)
                    d.line([(k + 2, base - 3 - hgt), (k + 2, base - 3)], fill=st_d)
                    d.point([(k + int(rng.integers(0, 3)), base - 3 - hgt)], fill=st_l)
            for _ in range(5):
                xx = int(rng.integers(0, w - 4))
                d.rectangle([xx, base - 4, xx + int(rng.integers(2, 5)), base - 2], fill=st_d)
        arr = np.asarray(img).astype(np.float32) / 255.0
        n = hash_noise(*np.meshgrid(np.arange(w) + self.x0, np.arange(h)), seed=self.seed)
        arr[..., :3] *= (0.92 + 0.14 * n[..., None])
        return arr

    def draw(self, cv, ox, oy, t):
        broken = not self.intact(t)
        if broken:
            if self.broken_img is None:
                self.broken_img = self._bake(True)
            img = self.broken_img
        else:
            if self.img is None:
                self.img = self._bake(False)
            img = self.img
        base = img.shape[0] - 6
        px.blit(cv, img, int(round(self.x0 + ox)), int(round(floor_y(self.z) + oy)) - base)


class RailPost:
    def __init__(self, x):
        self.x, self.z = x, RAIL_Z + 0.2
        self.break_t = None

    def draw(self, cv, ox, oy, t):
        x = int(round(self.x + ox))
        yb = int(round(floor_y(self.z) + oy))
        st, st_l, st_d = hexc('5e5768'), hexc('8a8296'), hexc('2e2a36')
        hgt = 21 if (self.break_t is None or t < self.break_t) else 7
        y0 = max(0, yb - hgt)
        y1 = min(cv.shape[0], yb + 3)
        x0, x1 = max(0, x - 3), min(cv.shape[1], x + 4)
        if x1 <= x0 or y1 <= y0:
            return
        cv[y0:y1, x0:x1] = st
        cv[y0:y1, x0] = st_l
        cv[y0:y1, min(cv.shape[1] - 1, x + 3)] = st_d
        if hgt > 10:
            ya = max(0, yb - hgt - 2)
            cv[ya:ya + 2, max(0, x - 4):min(cv.shape[1], x + 5)] = st_l


class Bridge:
    """The old rope bridge over the chasm, with planks already missing in the middle; it can snap and hang."""

    def __init__(self):
        self.snap_t = None
        self.layers = None

    def plank_at(self, x, t):
        if not (BRIDGE_X[0] <= x < BRIDGE_X[1]):
            return False
        if BRIDGE_GAP[0] <= x < BRIDGE_GAP[1]:
            return False
        return self.snap_t is None or t < self.snap_t

    def _bake(self):
        """Two layers (back: posts, rope, planks; front: front rope and posts) for each half, in stage pixels."""
        x0, x1 = BRIDGE_X
        mid = (BRIDGE_GAP[0] + BRIDGE_GAP[1]) // 2
        top = int(floor_y(BRIDGE_Z[0]) - 30)
        bot = int(floor_y(BRIDGE_Z[1]) + 12)
        h = bot - top
        out = {}
        rng = np.random.default_rng(77)
        wood, wood_l, wood_d = hexc('6a4a2a'), hexc('94704a'), hexc('3a2614')
        rope, rope_l = hexc('7a6448'), hexc('b0966c')
        for half, (a, b) in (('L', (x0 - 4, mid)), ('R', (mid, x1 + 4))):
            w = b - a + 1
            back = np.zeros((h, w, 4), np.float32)
            front = np.zeros((h, w, 4), np.float32)

            def put(arr, xx, yy, c, al=1.0):
                xi, yi = int(round(xx)) - a, int(round(yy)) - top
                if 0 <= xi < w and 0 <= yi < h:
                    arr[yi, xi, :3] = c
                    arr[yi, xi, 3] = al
            for x in range(max(a, x0), min(b, x1)):
                y_ = sag(x)
                broken = BRIDGE_GAP[0] <= x < BRIDGE_GAP[1]
                plank = (x - x0) % 5 != 4
                if not broken and plank:
                    shade = 0.85 + 0.3 * hash_noise(np.array([x // 5]), np.array([3]), 11)[0]
                    for z in np.arange(BRIDGE_Z[0], BRIDGE_Z[1] + 0.01, 0.5):
                        sx, sy = proj(x, z, y_)
                        put(back, sx, sy, (wood_l if z < BRIDGE_Z[0] + 1.5 else wood) * shade)
                    sx, sy = proj(x, BRIDGE_Z[1], y_)
                    put(back, sx, sy + 1, wood_d)
                elif broken and rng.uniform() < 0.12:
                    # a dangling broken plank end
                    for k in range(int(rng.integers(3, 8))):
                        sx, sy = proj(x, BRIDGE_Z[0] + 2 + k * 1.4, y_ - k * 1.2)
                        put(back, sx, sy, wood_d)
                # the rope under the planks and the hand ropes
                for z, arr in ((BRIDGE_Z[0], back), (BRIDGE_Z[1], front)):
                    sx, sy = proj(x, z, y_ + 1)
                    put(back, sx, sy, rope)
                    hr = y_ + 13 - 2.0 * math.sin(math.pi * (x - x0) / (x1 - x0))
                    sx, sy = proj(x, z, hr)
                    put(arr, sx, sy, rope_l)
                    put(arr, sx, sy + 1, rope)
                    if (x - x0) % 10 == 0:
                        for k in range(0, 13):
                            sx2, sy2 = proj(x, z, y_ + 1 + k * (hr - y_ - 1) / 12)
                            put(arr, sx2, sy2, rope)
            # posts at the ends
            for px_ in ((x0,) if half == 'L' else (x1,)):
                for z, arr in ((BRIDGE_Z[0], back), (BRIDGE_Z[1], front)):
                    for k in range(-6, 18):
                        sx, sy = proj(px_, z, k)
                        for dxp in (-2, -1, 0, 1):
                            put(arr, sx + dxp, sy, wood_l if dxp == -2 else wood if dxp < 1 else wood_d)
            out[half] = (back, front, a)
        self.layers = (out, top)

    def draw(self, cv, ox, oy, t, layer):
        """layer 'back' or 'front' while whole; once snapped both halves hang (drawn with the back layer)."""
        if self.layers is None:
            self._bake()
        out, top = self.layers
        snapped = self.snap_t is not None and t >= self.snap_t
        if not snapped:
            for half in ('L', 'R'):
                back, front, a = out[half]
                px.blit(cv, back if layer == 'back' else front, int(round(a + ox)), int(round(top + oy)))
            return
        if layer != 'back':
            return
        k = t - self.snap_t
        swing = (math.pi * 0.47) * min(1.0, (k / 0.45) ** 2) + 0.18 * math.sin(k * 7.0) * math.exp(-k * 2.2) * min(1.0, k / 0.45)
        for half, sgn, anchor_x in (('L', 1, BRIDGE_X[0]), ('R', -1, BRIDGE_X[1])):
            back, front, a = out[half]
            img = np.where(front[..., 3:4] > 0, front, back)
            ax, ay = proj(anchor_x, (BRIDGE_Z[0] + BRIDGE_Z[1]) / 2, 0)
            rot_blit(cv, img, (ax - a, ay - top), (ax + ox, ay + oy), sgn * swing)


# ------------------------------------------------------------------ the stage
class Stage:
    def __init__(self):
        set_projection(Y0, FS)
        self.img = np.zeros((SH + 2 * PAD, SW + 2 * PAD, 3), np.float32)
        self.build_sky()
        self.build_abyss()
        self.build_hall()
        self.build_sides()
        self.build_floor()
        self.build_front()
        self.build_peak()
        self.pillars = [Pillar(136, 30, 3), Pillar(504, 30, 9), Pillar(100, 206, 13), Pillar(540, 206, 17)]
        self.braziers = [Brazier(190, 56), Brazier(450, 56), Brazier(180, 186), Brazier(460, 186)]
        xs = list(range(FLOOR_X[0], FLOOR_X[1] + 1, 40))
        self.rails = [RailSection(a, b, 300 + i) for i, (a, b) in enumerate(zip(xs, xs[1:]))]
        self.posts = [RailPost(x) for x in xs]
        self.bridge = Bridge()

    # --------------------------------------------------------- surfaces
    def surfaces(self, x, z, t):
        """Every surface under (x, z) at scene time t: [(name, height)]."""
        out = []
        if FLOOR_X[0] <= x < FLOOR_X[1] and FLOOR_Z[0] <= z < FLOOR_Z[1]:
            out.append(('floor', 0.0))
        if FLOOR_X[0] <= x < FLOOR_X[1] and FLOOR_Z[1] <= z < WALL_Z:
            r = self.rail_at(x)
            out.append(('rail', RAIL_Y if (r is not None and r.intact(t)) else 0.0))
        if ROOF_X[0] <= x < ROOF_X[1] and RIDGE_Z <= z < EAVE_Z:
            out.append(('roof', roof_y(z)))
        for i, p in enumerate(self.pillars):
            if p.standing(t) and abs(x - p.x) <= 11 and abs(z - p.z) <= 9:
                out.append(('pillar%d' % i, PILLAR_TOP))
        if LEDGE_X[0] <= x < LEDGE_X[1] and LEDGE_Z[0] <= z < LEDGE_Z[1]:
            out.append(('ledge', -DROP))
        if BRIDGE_Z[0] <= z < BRIDGE_Z[1] and self.bridge.plank_at(x, t):
            out.append(('bridge', sag(x)))
        if PEAK_X[0] <= x < PEAK_X[1] and PEAK_Z[0] <= z < PEAK_Z[1]:
            out.append(('peak', 0.0))
        return out

    def ground(self, x, z, t, y=None):
        """The height of the highest surface under (x, z) at or below y (all of them if y is None), or VOID."""
        best = VOID
        for _, hgt in self.surfaces(x, z, t):
            if (y is None or hgt <= y + 3.0) and hgt > best:
                best = hgt
        return best

    def surface_fn(self, x, z, y, t):
        """The height function of the surface a fighter stands on at (x, z, y) at time t (he keeps to it)."""
        best, bh = None, VOID
        for name, hgt in self.surfaces(x, z, t):
            if hgt <= y + 3.0 and hgt > bh:
                best, bh = name, hgt
        if best is None:
            return lambda X, Z, y=y: y
        if best == 'roof':
            return lambda X, Z: roof_y(min(EAVE_Z, max(RIDGE_Z, Z)))
        if best == 'bridge':
            return lambda X, Z: sag(X)
        return lambda X, Z, h=bh: h

    def rail_at(self, x):
        for r in self.rails:
            if r.x0 <= x < r.x1:
                return r
        return None

    def ground_arr(self, x, z, h, t):
        """Vectorised for particles: (height of the surface they would land on, stampable)."""
        g = np.full(x.shape, VOID, np.float32)
        stamp = np.zeros(x.shape, bool)

        def take(m, hgt, st):
            hg = np.broadcast_to(np.asarray(hgt, np.float32), x.shape)
            ok = m & (hg <= h + 3.0) & (hg > g)
            g[ok] = hg[ok]
            stamp[ok] = st
        fl = (x >= FLOOR_X[0]) & (x < FLOOR_X[1])
        take(fl & (z >= FLOOR_Z[0]) & (z < WALL_Z), 0.0, True)
        take((x >= ROOF_X[0]) & (x < ROOF_X[1]) & (z >= RIDGE_Z) & (z < EAVE_Z),
             EAVE_Y + (EAVE_Z - z) * (RIDGE_Y - EAVE_Y) / (EAVE_Z - RIDGE_Z), True)
        take((x >= LEDGE_X[0]) & (x < LEDGE_X[1]) & (z >= LEDGE_Z[0]) & (z < LEDGE_Z[1]), -DROP, True)
        take((x >= PEAK_X[0]) & (x < PEAK_X[1]) & (z >= PEAK_Z[0]) & (z < PEAK_Z[1]), 0.0, True)
        for p in self.pillars:
            if p.standing(t):
                take((np.abs(x - p.x) <= 11) & (np.abs(z - p.z) <= 9), PILLAR_TOP, False)
        if self.bridge.snap_t is None or t < self.bridge.snap_t:
            gap = (x >= BRIDGE_GAP[0]) & (x < BRIDGE_GAP[1])
            take((x >= BRIDGE_X[0]) & (x < BRIDGE_X[1]) & ~gap & (z >= BRIDGE_Z[0]) & (z < BRIDGE_Z[1]),
                 -5.0 * np.sin(np.pi * np.clip((x - BRIDGE_X[0]) / (BRIDGE_X[1] - BRIDGE_X[0]), 0, 1)), False)
        return g, stamp

    # stage coords -> image index
    def put_img(self, arr, x, y):
        px.blit(self.img, arr, x + PAD, y + PAD)

    # --------------------------------------------------------- building
    def build_sky(self):
        h, w = self.img.shape[:2]
        y = np.arange(h, dtype=np.float32)[:, None]
        k = np.clip((y - PAD) / 230.0, 0, 1)
        sky = np.where((k < 0.6)[..., None], mix_arr(hexc('05040e'), hexc('150d28'), k / 0.6),
                       mix_arr(hexc('150d28'), hexc('3a2148'), (k - 0.6) / 0.4))
        sky = np.repeat(sky, w, axis=1)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        n = hash_noise(X, Y, seed=5)
        stars = (n > 0.9955) & (Y < 200)
        sky[stars] = hexc('cfd8ff') * (0.5 + 0.5 * hash_noise(X, Y, seed=6)[stars, None])
        mx, my, mr = 640 + PAD, 70 + PAD, 18.0
        d = np.sqrt((X + 0.5 - mx) ** 2 + (Y + 0.5 - my) ** 2)
        sky += (np.clip(1 - (d - mr) / 60.0, 0, 1) ** 2.2)[..., None] * hexc('5a4a7a') * 0.55
        lit = np.clip(((X - mx) * -0.4 + (Y - my) * -0.6) / mr * 0.5 + 0.75, 0, 1)
        moon = mix_arr(hexc('b9b4cc'), hexc('f4f0ff'), lit) * (0.86 + 0.18 * fbm(w, h, 4, seed=21)[..., None])
        sky[d <= mr] = moon[d <= mr]
        cl = fbm(w, h, 24, seed=31, octaves=4)
        band = np.exp(-((Y - (66 + PAD)) / 10.0) ** 2) + 0.7 * np.exp(-((Y - (104 + PAD)) / 7.0) ** 2)
        cloud = np.clip((cl - 0.48) * 3.0, 0, 1) * band
        sky = sky * (1 - cloud[..., None] * 0.7) + hexc('2a1d3e') * cloud[..., None] * 0.7
        # three mountain ranges, the nearest the darkest, fading into the mist of the valley
        for base, amp, cell, seed, fill, lit_c in ((176, 34, 40, 41, '231c3c', '4a3d6c'), (196, 26, 26, 47, '1a1430', '3a2f5a'),
                                                   (214, 18, 18, 53, '120e22', '2a2244')):
            nn = fbm(w, 1, cell, seed=seed, octaves=3)[0]
            peaks = np.abs(np.sin(np.arange(w) / (cell * 0.9) + seed)) ** 3
            ridge = base + PAD - (nn - 0.3) * amp * 2.0 - peaks * amp * 0.45
            m = Y >= ridge[None, :]
            slope = np.gradient(ridge)
            facing = np.clip(slope * 1.6, -1, 1)[None, :]
            depth = np.clip((Y - ridge[None, :]) / 24.0, 0, 1)
            kk = np.clip(0.55 + 0.45 * facing, 0, 1) * (1 - depth)
            rgb = mix_arr(hexc(fill), hexc(lit_c), kk) * (0.9 + 0.18 * fbm(w, h, 6, seed=seed + 3)[..., None])
            fog = np.clip((Y - (base + PAD - 6)) / 26.0, 0, 1)
            rgb = rgb * (1 - fog[..., None] * 0.6) + hexc('3a2c50') * fog[..., None] * 0.6
            sky[m] = rgb[m]
        self.img[:] = sky

    def build_abyss(self):
        """Below the mountains: the valley mist, then the dark drop (seen in the chasm and under the cliffs)."""
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        yy = Y - PAD
        m = yy > 232
        k = np.clip((yy - 232) / 300.0, 0, 1)
        base = mix_arr(hexc('2e2444'), hexc('08070e'), k ** 0.7)
        mist = fbm(w, h, 20, seed=61, octaves=3)
        bands = np.clip(np.sin(yy / 23.0 + mist * 4.0) * 0.5 + 0.2, 0, 1) * (1 - k) * 0.35
        col = base * (1 - bands[..., None]) + hexc('4a3c66') * bands[..., None]
        self.img[m] = col[m]
        # distant rock spires down in the chasm
        rng = np.random.default_rng(12)
        for cx, tw, ht in ((622, 9, 120), (648, 6, 80), (706, 12, 150), (742, 7, 95), (690, 5, 60)):
            topy = 520 - ht
            for row in range(int(topy), 560):
                k2 = (row - topy) / ht
                half = tw * (0.35 + 0.65 * min(1.0, k2 * 1.6))
                for xx in range(int(cx - half), int(cx + half) + 1):
                    if 0 <= row + PAD < h and 0 <= xx + PAD < w:
                        fog = 0.55 + 0.35 * (1 - min(1.0, k2))
                        c = hexc('1a1528') * (0.8 + 0.4 * (xx < cx)) * (1 - fog * 0.4) + hexc('2e2444') * fog * 0.4
                        self.img[row + PAD, xx + PAD] = c

    def build_hall(self):
        w, h = self.img.shape[1], self.img.shape[0]
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        P = PAD

        def S(x, z, y):
            return (x + P, Y0 + z * FS - y + P)
        wood, beam = col255('5a1a14'), col255('241210')
        roof, roof_l, roof_d, gold = col255('1c1a24'), col255('34324a'), col255('100e16'), col255('b58a3a')
        stone, stone_l, stone_d = col255('4a4352'), col255('6a6274'), col255('2a2430')
        # side wings: lower halls behind the courtyard's corners
        for (a, b) in ((FLOOR_X[0] - 30, HALL_X[0] + 6), (HALL_X[1] - 6, FLOOR_X[1] + 30)):
            fy0, fy1 = S(0, -2, 34)[1], S(0, -2, 0)[1]
            d.rectangle([a + P, fy0, b + P, fy1], fill=col255('3a0f0c'))
            for k in range(a, b, 20):
                d.rectangle([k + P, fy0, k + 2 + P, fy1], fill=beam)
            for k in range(a + 7, b - 8, 20):
                d.rectangle([k + P, fy0 + 6, k + 8 + P, fy1 - 8], fill=col255('d8883a'))
                for gx in range(k, k + 9, 3):
                    d.line([(gx + P, fy0 + 6), (gx + P, fy1 - 8)], fill=col255('5a2a10'))
            ey, ry = S(0, 4, 36)[1], S(0, -30, 54)[1]
            d.polygon([(a - 6 + P, ey), (b + 6 + P, ey), (b + P, ry), (a + P, ry)], fill=roof)
            for k in range(int(ry), int(ey), 2):
                d.line([(a + P, k), (b + P, k)], fill=roof_l if (k // 2) % 2 == 0 else roof)
            d.line([(a - 6 + P, ey), (b + 6 + P, ey)], fill=gold)
        # the main hall's facade (z = 0, 56 high)
        L, R = HALL_X
        fy0, fy1 = S(0, 0, 56)[1], S(0, 0, 0)[1]
        d.rectangle([L + P, fy0, R + P, fy1], fill=wood)
        for i, x in enumerate(range(L + 6, R, 26)):
            lit = 3 <= i <= 7
            if x + 22 < R:
                c = col255('ffb050') if lit else col255('7a3a1c')
                d.rectangle([x + 5 + P, fy0 + 8, x + 21 + P, fy1 - 6], fill=c)
                for gx in range(x + 5, x + 22, 3):
                    d.line([(gx + P, fy0 + 8), (gx + P, fy1 - 6)], fill=col255('5a2a10'))
                for gy in range(int(fy0) + 8, int(fy1) - 5, 4):
                    d.line([(x + 5 + P, gy), (x + 21 + P, gy)], fill=col255('5a2a10'))
            d.rectangle([x + P, fy0, x + 3 + P, fy1], fill=col255('8a2018'))
            d.line([(x + P, fy0), (x + P, fy1)], fill=col255('b03a24'))
        d.rectangle([L - 2 + P, fy0 - 4, R + 2 + P, fy0], fill=beam)
        d.line([(L - 2 + P, fy0 - 3), (R + 2 + P, fy0 - 3)], fill=gold)
        d.rectangle([294 + P, fy0 + 1, 346 + P, fy0 + 8], fill=col255('1a1014'), outline=gold)
        d.rectangle([L - 8 + P, fy1 - 4, R + 8 + P, fy1], fill=stone)
        d.line([(L - 8 + P, fy1 - 4), (R + 8 + P, fy1 - 4)], fill=stone_l)
        arr = np.asarray(img).astype(np.float32) / 255.0
        lit = (arr[..., 0] > 0.9) & (arr[..., 1] > 0.55)
        glow = gaussian_filter(lit.astype(np.float32), 3.0)
        arr[..., :3] += glow[..., None] * hexc('ff8a30') * 0.55 * (arr[..., 3:4] > 0)
        px.blit(self.img, arr, 0, 0)
        # the roof's front slope, tile by tile (it is a surface: fighters stand on it, tiles break off it)
        X, Y = np.meshgrid(np.arange(w) - PAD, np.arange(h) - PAD)
        # invert the projection on the slope: y = roof_y(z), sy = Y0 + z FS - roof_y(z)
        kslope = (RIDGE_Y - EAVE_Y) / (EAVE_Z - RIDGE_Z)
        # sy = Y0 + z FS - EAVE_Y - (EAVE_Z - z) k  =>  z = (sy - Y0 + EAVE_Y + EAVE_Z k) / (FS + k)
        Z = (Y - Y0 + EAVE_Y + EAVE_Z * kslope) / (FS + kslope)
        flare = np.clip((Z - (EAVE_Z - 10)) / 10.0, 0, 1) * 8.0          # the corners turn up at the eave
        on = (Z >= RIDGE_Z) & (Z < EAVE_Z) & (X >= ROOF_X[0] - flare) & (X < ROOF_X[1] + flare)
        row = (EAVE_Z - Z) / 4.0
        irow = np.floor(row).astype(int)
        frow = row - irow
        colw = 6.0
        cx_ = (X - ROOF_X[0] + (irow % 2) * 3) / colw
        icol = np.floor(cx_).astype(int)
        fcol = cx_ - icol
        bump = np.sin(fcol * math.pi)                                   # round tiles: lit on the left of each
        tone = 0.62 + 0.28 * bump - 0.35 * (frow < 0.22) + 0.08 * hash_noise(icol, irow, 9)
        tone += (fcol < 0.18) * -0.22
        k_ = np.clip((Z - RIDGE_Z) / (EAVE_Z - RIDGE_Z), 0, 1)
        base_c = mix_arr(hexc('3a3850'), hexc('262434'), k_)
        rc = base_c * tone[..., None] + hexc('5a5478') * (1 - k_[..., None]) * 0.12
        self.img[on] = rc[on]
        self.roof_mask = on
        # the eave's edge (a gold line), the ridge beam with its curled ends, and the dark under the eave
        for x in range(ROOF_X[0] - 8, ROOF_X[1] + 9):
            sx, sy = S(x, EAVE_Z, EAVE_Y)
            yi = int(round(sy))
            self.img[yi, x + P] = hexc('b58a3a')
            self.img[yi + 1, x + P] = hexc('100e16')
            self.img[yi + 2, x + P] = hexc('100e16') * 0.6 + self.img[yi + 2, x + P] * 0.4
        rx0, rx1 = ROOF_X[0] + 10, ROOF_X[1] - 10
        ry = int(round(S(0, RIDGE_Z, RIDGE_Y)[1]))
        self.img[ry - 4 + P - P:ry + 1, rx0 + P:rx1 + P] = hexc('100e16')
        self.img[ry - 4, rx0 + P:rx1 + P] = hexc('4a4660')
        for ex, sg in ((rx0, -1), (rx1, 1)):
            for k in range(9):
                xx = ex + sg * k // 2 + P
                self.img[ry - 4 - k, xx:xx + 3] = hexc('100e16')
        # hanging lanterns under the eave
        ly = int(round(S(0, EAVE_Z, EAVE_Y)[1])) + 3
        for lx in (190, 240, 400, 450):
            lantern(self.img, lx + PAD, ly)

    def build_sides(self):
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        # dark rock and pines beyond the low wall on the left
        x0, x1 = 0, FLOOR_X[0] - 6 + PAD
        region = (X >= x0) & (X < x1) & (Y >= 200 + PAD) & (Y < floor_y(WALL_Z) + PAD)
        n = fbm(w, h, 7, seed=101)
        col = hexc('121a16') * (0.7 + 0.6 * n[..., None]) + hexc('0a0c14') * 0.3
        self.img[region] = col[region]
        rng = np.random.default_rng(55)
        for _ in range(8):
            tx = rng.uniform(x0 + 2, x1 - 2)
            ty = rng.uniform(250, 380) + PAD
            th = rng.uniform(18, 32)
            for k in range(int(th)):
                half = (k / th) * th * 0.32
                yy = int(ty - th + k)
                for xx in range(int(tx - half), int(tx + half) + 1):
                    if x0 <= xx < x1 and 0 <= yy < h:
                        sh = 0.75 if xx < tx else 0.55
                        self.img[yy, xx] = hexc('1c3024') * sh * (0.85 + 0.3 * ((xx * 7 + yy * 3) % 5 == 0))
        # the low wall along the left side: a coping strip seen from above
        x = FLOOR_X[0] - 6
        xs = slice(x + PAD, x + 5 + PAD)
        ys = slice(int(Y0) - 14 + PAD, int(floor_y(WALL_Z)) + PAD)
        self.img[ys, xs] = hexc('5a5466')
        self.img[ys, x + PAD] = hexc('7a7486')
        self.img[ys, x + 4 + PAD] = hexc('2e2a36')
        for yy in range(int(Y0) - 14, int(floor_y(WALL_Z)), 9):
            self.img[yy + PAD, xs] = hexc('3a3644')
        # the gate posts at the head of the bridge (stone, with a lantern each)
        for (bx, bz) in ((FLOOR_X[1] - 6, BRIDGE_Z[0] - 6), (FLOOR_X[1] - 6, BRIDGE_Z[1] + 8)):
            sx, sy = proj(bx, bz, 0)
            x0_, y0_ = int(sx) + PAD, int(sy) + PAD
            self.img[y0_ - 34:y0_, x0_ - 3:x0_ + 4] = hexc('4e4858')
            self.img[y0_ - 34:y0_, x0_ - 3] = hexc('7a7486')
            self.img[y0_ - 34:y0_, x0_ + 3] = hexc('2a2430')
            self.img[y0_ - 38:y0_ - 34, x0_ - 5:x0_ + 6] = hexc('3a3646')
            self.img[y0_ - 33:y0_ - 28, x0_ - 2:x0_ + 3] = hexc('ffb050')

    def build_floor(self):
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w) - PAD, np.arange(h) - PAD)
        Z = (Y - Y0) / FS
        on = (X >= FLOOR_X[0]) & (X < FLOOR_X[1]) & (Z >= FLOOR_Z[0]) & (Z <= WALL_Z)
        cx, cz, rr = RING
        R = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)
        ang = np.arctan2(Z - cz, X - cx)
        RW = 17.0
        ir = np.floor(R / RW).astype(int)
        fr = R / RW - ir
        segs = np.maximum(6, np.round(2 * np.pi * (ir + 0.5) * RW / 24.0)).astype(int)
        sa = (ang + np.pi) / (2 * np.pi) * segs + (ir % 2) * 0.5
        isg = np.floor(sa).astype(int)
        fs = sa - isg
        circ = R < 7 * RW
        T = 24.0
        tx, tz = (X - FLOOR_X[0]) / T, Z / T
        ix, iz = np.floor(tx).astype(int), np.floor(tz).astype(int)
        v = np.where(circ, hash_noise(ir, isg, 3), hash_noise(ix + 500, iz, 4))
        n = fbm(w, h, 5, seed=71)
        base = hexc('423d4e')
        col = base * (0.78 + 0.3 * v[..., None]) * (0.88 + 0.22 * n[..., None])
        seg_len = 2 * np.pi * np.maximum(R, 1) / np.maximum(segs, 1)
        joint_c = (fr * RW < 0.9) | (fs * seg_len < 0.9)
        joint_s = ((tx - ix) * T < 0.9) | ((tz - iz) * T * FS < 0.9)
        joint = np.where(circ, joint_c, joint_s)
        col[joint] *= 0.66
        wear = (hash_noise(X, Y, seed=77) > 0.93) & ~joint
        col[wear] *= 1.12
        r = np.sqrt(((X - cx) / rr) ** 2 + ((Z - cz) / rr) ** 2)
        ring = (r > 0.86) & (r < 1.0)
        col[ring] = col[ring] * 0.55 + hexc('5a2420') * 0.45
        line = (np.abs(r - 0.9) < 0.012) | (np.abs(r - 0.35) < 0.012)
        col[line] = col[line] * 0.4 + hexc('b08a3c') * 0.6
        rays = (r < 0.33) & (np.abs(np.sin(ang * 4)) < (0.33 - r) * 1.6)
        col[rays] = col[rays] * 0.7 + hexc('2a2232') * 0.3
        moss = (fbm(w, h, 5, seed=81) > 0.66) & (hash_noise(X, Y, seed=82) > 0.7)
        col[moss] = col[moss] * 0.7 + hexc('2c3a28') * 0.3
        rng = np.random.default_rng(5)
        for _ in range(50):
            x = rng.uniform(FLOOR_X[0], FLOOR_X[1])
            y = rng.uniform(Y0 + 2, floor_y(FLOOR_Z[1]) - 2)
            for _ in range(rng.integers(6, 18)):
                xi, yi = int(x) + PAD, int(y) + PAD
                if 0 <= xi < w and 0 <= yi < h:
                    col[yi, xi] *= 0.55
                x += rng.uniform(-1.5, 2.5)
                y += rng.uniform(-0.6, 0.6)
        k = np.clip(Z / 240.0, 0, 1)[..., None]
        col = col * (1.04 - 0.2 * k) + hexc('2a2440') * (1 - k) * 0.3
        edge = np.minimum(X - FLOOR_X[0], FLOOR_X[1] - X)
        col *= (0.7 + 0.3 * np.clip(edge / 50.0, 0, 1))[..., None]
        pool = np.clip(1 - np.sqrt(((X - cx) / 170.0) ** 2 + ((Z - cz) / 130.0) ** 2), 0, 1) ** 1.5
        col = col * (0.82 + 0.38 * pool[..., None]) + hexc('3a3a5a') * pool[..., None] * 0.12
        self.img[on] = col[on]
        # the porch's shadow under the hall
        sh = on & (Z < 10) & (X >= HALL_X[0] - 10) & (X < HALL_X[1] + 10)
        self.img[sh] *= (0.6 + 0.04 * np.clip(Z[sh], 0, 10))[..., None]
        # the right edge of the courtyard over the chasm: a rim of rock
        for yy in range(int(Y0) - 2, int(floor_y(WALL_Z)) + 2):
            self.img[yy + PAD, FLOOR_X[1] + PAD - 1] = hexc('6a6274')
            self.img[yy + PAD, FLOOR_X[1] + PAD:FLOOR_X[1] + PAD + 2] = hexc('1a1622')

    def build_front(self):
        """The courtyard's front wall down to the ledge, the ledge itself, and the cliff below everything."""
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w) - PAD, np.arange(h) - PAD)
        top = floor_y(WALL_Z)                   # y = 0 on the wall
        rock_n = fbm(w, h, 6, seed=131, octaves=3)
        rock_n2 = fbm(w, h, 2.5, seed=133, octaves=2)
        # the cliff under the courtyard (where there is no ledge it falls straight into the mist)
        cliff = (X >= FLOOR_X[0] - 40) & (X < FLOOR_X[1]) & (Y >= top)
        k = np.clip((Y - top) / 160.0, 0, 1)
        rc = hexc('2a2432') * (0.65 + 0.6 * rock_n[..., None]) * (0.9 + 0.2 * rock_n2[..., None])
        strata = (np.sin((Y + rock_n * 30) / 5.0) > 0.75)
        rc[strata] *= 0.7
        rc = rc * (1 - k[..., None] * 0.75) + hexc('140f1e') * k[..., None] * 0.75
        self.img[cliff] = rc[cliff]
        # the dressed wall: big blocks from the floor down DROP px
        wall = (X >= FLOOR_X[0]) & (X < FLOOR_X[1]) & (Y >= top) & (Y < top + DROP)
        rowi = np.floor((Y - top) / 8.0).astype(int)
        bx = (X + (rowi % 2) * 9) / 18.0
        bi = np.floor(bx).astype(int)
        joint = ((Y - top) % 8 < 1) | ((bx - bi) * 18 < 1)
        wc = hexc('4a4454') * (0.82 + 0.26 * hash_noise(bi, rowi, 21)[..., None]) * (0.9 + 0.2 * rock_n2[..., None])
        wc[joint] *= 0.55
        shade = 1.0 - 0.35 * np.clip((Y - top) / DROP, 0, 1)
        wc *= shade[..., None]
        self.img[wall] = wc[wall]
        yt = int(round(top))
        self.img[yt + PAD, FLOOR_X[0] + PAD:FLOOR_X[1] + PAD] = hexc('7a7486')
        # the ledge: flagstones over the drop, then its own rock face
        ly0 = floor_y(LEDGE_Z[0]) + DROP
        ly1 = floor_y(LEDGE_Z[1]) + DROP
        ledge = (X >= LEDGE_X[0]) & (X < LEDGE_X[1]) & (Y >= ly0) & (Y < ly1)
        Zl = (Y - Y0 - DROP) / FS
        lt = (X - LEDGE_X[0]) / 16.0
        lz = (Zl - LEDGE_Z[0]) / 12.0
        li, lzi = np.floor(lt).astype(int), np.floor(lz).astype(int)
        lc = hexc('545064') * (0.8 + 0.28 * hash_noise(li, lzi, 41)[..., None]) * (0.9 + 0.2 * rock_n2[..., None])
        lj = ((lt - li) * 16 < 0.9) | ((lz - lzi) * 12 * FS < 0.9)
        lc[lj] *= 0.62
        mossl = (rock_n > 0.62)
        lc[mossl] = lc[mossl] * 0.7 + hexc('2c3a28') * 0.3
        self.img[ledge] = lc[ledge]
        face = (X >= LEDGE_X[0]) & (X < LEDGE_X[1]) & (Y >= ly1) & (Y < ly1 + 90)
        kf = np.clip((Y - ly1) / 90.0, 0, 1)
        fc = hexc('342e3c') * (0.7 + 0.5 * rock_n[..., None])
        fc = fc * (1 - kf[..., None] * 0.7) + hexc('140f1e') * kf[..., None] * 0.7
        self.img[face] = fc[face]
        yl = int(round(ly1))
        self.img[yl + PAD - 1, LEDGE_X[0] + PAD:LEDGE_X[1] + PAD] = hexc('8a8496')
        self.img[yl + PAD, LEDGE_X[0] + PAD:LEDGE_X[1] + PAD] = hexc('5e5868')
        self.img[yl + PAD + 1:yl + PAD + 3, LEDGE_X[0] + PAD:LEDGE_X[1] + PAD] = hexc('1a1622')
        # the wall's shadow on the ledge, and grass in the cracks
        y0l = int(round(ly0))
        for k in range(4):
            self.img[y0l + PAD + k, LEDGE_X[0] + PAD:LEDGE_X[1] + PAD] *= 0.55 + 0.1 * k
        rng = np.random.default_rng(171)
        for _ in range(26):
            gx = int(rng.uniform(LEDGE_X[0] + 3, LEDGE_X[1] - 3)) + PAD
            gy = int(rng.uniform(ly0 + 5, ly1 - 1)) + PAD
            for k in range(int(rng.integers(2, 5))):
                self.img[gy - k, gx + int(rng.integers(-1, 2))] = hexc('3a5a30') * (0.7 + 0.1 * k)
        # mist drifting below everything
        mist = np.clip(1 - np.abs(Y - 500) / 40.0, 0, 1) * (0.35 + 0.3 * fbm(w, h, 30, seed=141)) * 0.8
        self.img[:] = self.img * (1 - mist[..., None]) + hexc('3a2c50') * mist[..., None]

    def build_peak(self):
        """The rock pinnacle beyond the chasm: a little shrine with a torii, a gong and a stone lantern."""
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w) - PAD, np.arange(h) - PAD)
        Z = (Y - Y0) / FS
        on = (X >= PEAK_X[0]) & (X < PEAK_X[1]) & (Z >= PEAK_Z[0]) & (Z < PEAK_Z[1])
        n = fbm(w, h, 5, seed=151)
        t_ = (X - PEAK_X[0]) / 20.0
        z_ = Z / 14.0
        ti, zi = np.floor(t_).astype(int), np.floor(z_).astype(int)
        c = hexc('403b4c') * (0.78 + 0.3 * hash_noise(ti, zi, 52)[..., None]) * (0.88 + 0.22 * n[..., None])
        j = ((t_ - ti) * 20 < 0.9) | ((z_ - zi) * 14 * FS < 0.9)
        c[j] *= 0.62
        edge = np.minimum(X - PEAK_X[0], PEAK_X[1] - X)
        c *= (0.75 + 0.25 * np.clip(edge / 30.0, 0, 1))[..., None]
        self.img[on] = c[on]
        # its rock face towards us
        fy = floor_y(PEAK_Z[1])
        face = (X >= PEAK_X[0]) & (X < PEAK_X[1]) & (Y >= fy)
        kf = np.clip((Y - fy) / 200.0, 0, 1)
        rn = fbm(w, h, 6, seed=153, octaves=3)
        fc = hexc('2e2836') * (0.65 + 0.6 * rn[..., None])
        fc[np.sin((Y + rn * 30) / 5.0) > 0.75] *= 0.7
        fc = fc * (1 - kf[..., None] * 0.8) + hexc('100c18') * kf[..., None] * 0.8
        self.img[face] = fc[face]
        self.img[int(fy) + PAD, PEAK_X[0] + PAD:PEAK_X[1] + PAD] = hexc('6a6274')
        for yy in range(int(floor_y(PEAK_Z[0])), int(fy) + 60):
            self.img[yy + PAD, PEAK_X[0] + PAD] = hexc('6a6274')
            self.img[yy + PAD, PEAK_X[0] + PAD - 2:PEAK_X[0] + PAD] = hexc('1a1622')
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        P = PAD

        def S(x, z, y):
            return (x + P, Y0 + z * FS - y + P)
        red, red_d, black = col255('a8281c'), col255('6a140e'), col255('1a1014')
        # the torii at the back of the pinnacle
        for tx in (800, 920):
            x0, y0 = S(tx, 66, 62)
            x1, y1 = S(tx, 66, 0)
            d.rectangle([x0 - 3, y0, x0 + 3, y1], fill=red)
            d.line([(x0 + 3, y0), (x0 + 3, y1)], fill=red_d)
        a, b = S(788, 66, 66), S(932, 66, 62)
        d.polygon([(a[0] - 6, a[1] - 3), (b[0] + 6, a[1] - 3), (b[0] + 2, a[1] + 4), (a[0] - 2, a[1] + 4)], fill=black)
        a2 = S(792, 66, 54)
        d.rectangle([a2[0], a2[1], S(928, 66, 54)[0], a2[1] + 3], fill=red)
        # the gong in its frame
        gx, gz = 860, 80
        f0, f1 = S(gx - 18, gz, 44), S(gx + 18, gz, 0)
        d.rectangle([f0[0] - 2, f0[1], f0[0] + 1, f1[1]], fill=col255('4a2e18'))
        d.rectangle([f1[0] - 1, f0[1], f1[0] + 2, f1[1]], fill=col255('4a2e18'))
        d.rectangle([f0[0] - 4, f0[1] - 3, f1[0] + 4, f0[1]], fill=col255('3a2412'))
        gc = S(gx, gz, 26)
        d.ellipse([gc[0] - 13, gc[1] - 13, gc[0] + 13, gc[1] + 13], fill=col255('a07838'), outline=col255('3a2412'))
        d.ellipse([gc[0] - 9, gc[1] - 9, gc[0] + 9, gc[1] + 9], outline=col255('d8b060'))
        d.ellipse([gc[0] - 4, gc[1] - 4, gc[0] + 4, gc[1] + 4], fill=col255('c89a48'))
        # a stone lantern at the front corner
        lx, lz = 940, 176
        l0 = S(lx, lz, 30)
        l1 = S(lx, lz, 0)
        d.rectangle([l0[0] - 2, l0[1] + 10, l0[0] + 2, l1[1]], fill=col255('5a5466'))
        d.rectangle([l0[0] - 5, l0[1] + 2, l0[0] + 5, l0[1] + 10], fill=col255('5a5466'))
        d.rectangle([l0[0] - 3, l0[1] + 4, l0[0] + 3, l0[1] + 8], fill=col255('ffb050'))
        d.polygon([(l0[0] - 8, l0[1] + 2), (l0[0] + 8, l0[1] + 2), (l0[0], l0[1] - 5)], fill=col255('3a3646'))
        arr = np.asarray(img).astype(np.float32) / 255.0
        px.blit(self.img, arr, 0, 0)
        self.gong = (gx, gz, 26)

    # ------------------------------------------------------------ per frame
    def view(self, cam):
        cx, cy = int(round(cam[0])), int(round(cam[1]))
        y0 = cy - M + PAD
        x0 = cx - M + PAD
        return self.img[y0:y0 + EH, x0:x0 + EW].copy(), (M - cx, M - cy)

    def stamp(self, x, z, color, r=1.0, alpha=1.0, seed=0, h=0.0, flat=True):
        """Paints a splat into the stage image at (x, z) height h (blood, scorch, ice, cracks stay for good)."""
        cx = x + PAD
        cy = floor_y(z) - h + PAD
        rx = max(0.6, r)
        ry = max(0.5, r * (FS if flat else 1.0))
        for yy in range(int(math.floor(cy - ry - 1)), int(math.ceil(cy + ry + 1)) + 1):
            for xx in range(int(math.floor(cx - rx - 1)), int(math.ceil(cx + rx + 1)) + 1):
                dd = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2
                if dd <= 1.0 and 0 <= yy < self.img.shape[0] and 0 <= xx < self.img.shape[1]:
                    n = (hash(((xx * 73856093) ^ (yy * 19349663) ^ seed)) & 255) / 255.0
                    if dd > 0.6 and n < 0.4:
                        continue
                    a = alpha * (0.8 + 0.2 * n)
                    self.img[yy, xx] = self.img[yy, xx] * (1 - a) + np.asarray(color, np.float32) * a

    def tint(self, x, z, r, color, k, seed=0, h=0.0, gloss=True):
        """Ice over the stone: a translucent coat with glints (stays)."""
        cx = x + PAD
        cy = floor_y(z) - h + PAD
        rx, ry = max(1.0, r), max(0.8, r * FS)
        y0, y1 = int(cy - ry - 1), int(cy + ry + 2)
        x0, x1 = int(cx - rx - 1), int(cx + rx + 2)
        y0c, y1c = max(0, y0), min(self.img.shape[0], y1)
        x0c, x1c = max(0, x0), min(self.img.shape[1], x1)
        if y1c <= y0c or x1c <= x0c:
            return
        ys, xs = np.mgrid[y0c:y1c, x0c:x1c]
        dd = ((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / ry) ** 2
        nn = hash_noise(xs, ys, seed)
        m = dd + nn * 0.35 <= 1.0
        sub = self.img[y0c:y1c, x0c:x1c]
        c = np.asarray(color, np.float32)
        lum = sub.mean(axis=2, keepdims=True)
        sub[m] = (sub * (1 - k) + (c * (0.3 + lum * 1.0)) * k)[m]
        if gloss:
            g = m & (hash_noise(xs, ys, seed + 1) > 0.965)
            sub[g] = np.minimum(1.0, sub[g] * 0.4 + 0.7)


def fire_shape(cv, x, y, t, wdt, hgt):
    for row in range(hgt):
        k = row / hgt
        half = wdt * (1 - k) ** 0.8 * (0.8 + 0.2 * math.sin(t * 11 + row * 0.7))
        cxo = math.sin(t * 6 + row * 0.35) * k * 2.5
        yy = int(y - row)
        if not (0 <= yy < cv.shape[0]):
            continue
        for dx in range(-int(half) - 1, int(half) + 2):
            d = abs(dx - cxo) / max(half, 0.5)
            if d > 1:
                continue
            n = (hash((int(t * 12), row, dx)) & 255) / 255.0
            if k > 0.55 and n > 1.2 - k:
                continue
            heat = (1 - d) * (1 - k * 0.8)
            c = hexc('fff2b0') if heat > 0.55 else hexc('ffc040') if heat > 0.35 else hexc('ff7a1c') if heat > 0.18 else hexc('c43a10')
            xx = int(round(x + dx))
            if 0 <= xx < cv.shape[1]:
                cv[yy, xx] = c


def lantern(cv, x, y):
    x, y = int(round(x)), int(round(y))
    body_c, dark, glow_c = hexc('d8402a'), hexc('7a1a12'), hexc('ffb060')
    for dy in range(-1, 8):
        half = 2 if 1 <= dy <= 5 else 1
        for dx in range(-half, half + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                if dy in (-1, 7):
                    cv[yy, xx] = hexc('1a1418')
                elif abs(dx) == half:
                    cv[yy, xx] = dark
                else:
                    cv[yy, xx] = glow_c if abs(dx) <= 0 else body_c


# ------------------------------------------------------------------ bodies that turn in 3D (FLOW-1: no flat sliding)
def affine_blit(dst, img, src_origin, dst_origin, ax, pv, alpha=1.0):
    """Blit an RGBA texture through an affine map: texture point (c, r) = src_origin + u * (0, -1) + v * (1, 0) lands at
    dst_origin + u * ax + v * pv (ax: the screen step per texel up the texture, pv: per texel across). Nearest
    neighbour, so the pixels stay pixels; the map comes from the body's real 3D axis, so it foreshortens as it turns."""
    h, w = img.shape[:2]
    c0, r0 = src_origin
    corners = []
    for (c, r) in ((0, 0), (w, 0), (0, h), (w, h)):
        u, v = r0 - r, c - c0
        corners.append((dst_origin[0] + u * ax[0] + v * pv[0], dst_origin[1] + u * ax[1] + v * pv[1]))
    x0 = int(math.floor(min(p[0] for p in corners))) - 1
    x1 = int(math.ceil(max(p[0] for p in corners))) + 1
    y0 = int(math.floor(min(p[1] for p in corners))) - 1
    y1 = int(math.ceil(max(p[1] for p in corners))) + 1
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if x1c <= x0c or y1c <= y0c:
        return
    det = ax[0] * pv[1] - ax[1] * pv[0]
    if abs(det) < 1e-6:
        return
    ys, xs = np.mgrid[y0c:y1c, x0c:x1c]
    X = xs + 0.5 - dst_origin[0]
    Y = ys + 0.5 - dst_origin[1]
    u = (X * pv[1] - Y * pv[0]) / det
    v = (ax[0] * Y - ax[1] * X) / det
    ix = np.floor(c0 + v).astype(int)
    iy = np.floor(r0 - u).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    src = np.zeros(ix.shape + (4,), np.float32)
    src[ok] = img[iy[ok], ix[ok]]
    a = src[..., 3:4] * alpha
    dst[y0c:y1c, x0c:x1c] = dst[y0c:y1c, x0c:x1c] * (1 - a) + src[..., :3] * a


def _pillar_fall_vec(self):
    """The fall's direction on the floor: sideways, and a little towards the camera so it turns as a body."""
    ph = 0.42
    return (self.fall_dir * math.cos(ph), math.sin(ph) / FS * 0.6)


def _pillar_draw(self, cv, ox, oy, t):
    if self.img is None:
        self.bake()
    ang = abs(self.angle(t))
    bx, by = self.x + ox, floor_y(self.z) + oy
    if ang == 0.0:
        px.blit(cv, self.img, int(round(bx - self.anchor[0])), int(round(by - self.anchor[1])))
        return None
    dx, dz = self.fall_vec()
    s, c = math.sin(ang), math.cos(ang)
    # the axis in 3D: from the base up the shaft (one texel = one unit of height), projected
    top = proj(self.x + dx * s, self.z + dz * s, c)
    base = proj(self.x, self.z, 0.0)
    ax = (top[0] - base[0], top[1] - base[1])
    # across the shaft: a cylinder's outline keeps its width whichever way it turns, so the screen perpendicular to
    # the projected axis, one pixel per texel, turned so the lit side faces up-left
    m = math.hypot(*ax) + 1e-6
    pv = (-ax[1] / m, ax[0] / m)
    if pv[0] + pv[1] > 0:
        pv = (-pv[0], -pv[1])
    # the round foot first (a rim of the base), then the shaft through the affine map
    affine_blit(cv, self.img, (self.anchor[0] + 0.5, self.anchor[1]), (bx + 0.5, by), ax, pv)
    return None


Pillar.fall_vec = _pillar_fall_vec
Pillar.draw = _pillar_draw


def _bridge_draw(self, cv, ox, oy, t, layer):
    """Whole: the baked back and front layers. Snapped: each half turns down about its anchor in 3D; every plank and
    rope point is re-projected, so the halves hang with their depth (two ropes and the planks between them)."""
    if self.layers is None:
        self._bake()
    out, top = self.layers
    snapped = self.snap_t is not None and t >= self.snap_t
    if not snapped:
        for half in ('L', 'R'):
            back, front, a = out[half]
            px.blit(cv, back if layer == 'back' else front, int(round(a + ox)), int(round(top + oy)))
        return
    k = t - self.snap_t
    swing = (math.pi * 0.44) * min(1.0, (k / 0.45) ** 2) + 0.16 * math.sin(k * 7.0) * math.exp(-k * 2.2) * min(1.0, k / 0.45)
    twist = 0.25 * math.sin(k * 3.1) * math.exp(-k * 0.8)
    wood, wood_l, wood_d = hexc('6a4a2a'), hexc('94704a'), hexc('3a2614')
    rope, rope_l = hexc('7a6448'), hexc('b0966c')
    hgt, wdt = cv.shape[:2]
    z0, z1 = BRIDGE_Z
    zm = (z0 + z1) / 2

    def put(x, z, y, c):
        sx, sy = proj(x, z, y)
        xi, yi = int(round(sx + ox)), int(round(sy + oy))
        if 0 <= xi < wdt and 0 <= yi < hgt:
            cv[yi, xi] = c
    for half, sgn, ax_ in (('L', 1, BRIDGE_X[0]), ('R', -1, BRIDGE_X[1])):
        if layer == 'back':
            zs = (z0,)
        else:
            zs = (z1,)
        end = BRIDGE_GAP[0] if half == 'L' else BRIDGE_GAP[1]
        n = int(abs(end - ax_))
        for i in range(n):
            x_orig = ax_ + sgn * i
            s_ = i * 1.0
            sw = swing * (1.0 - 0.15 * (i / max(1, n)))
            X = ax_ + sgn * s_ * math.cos(sw)
            Y = -s_ * math.sin(sw)
            dzt = twist * s_ * 0.3
            plank = (x_orig - BRIDGE_X[0]) % 5 != 4
            if layer == 'back' and plank:
                shade = 0.85 + 0.3 * hash_noise(np.array([x_orig // 5]), np.array([3]), 11)[0]
                for z in np.arange(z0, z1 + 0.01, 0.75):
                    put(X, z + dzt, Y, (wood_l if z < z0 + 1.5 else wood) * shade)
                put(X, z1 + dzt, Y - 1, wood_d)
            for z in zs:
                put(X, z + dzt, Y + 1, rope)
                hr = Y + 9 * math.cos(sw)
                put(X - sgn * 9 * math.sin(sw) * 0.6, z + dzt, hr, rope_l)
        # posts at the anchor
        for z in zs:
            for kk in range(-6, 18):
                for dxp in (-2, -1, 0, 1):
                    put(ax_ + dxp, z, kk, wood_l if dxp == -2 else wood if dxp < 1 else wood_d)


Bridge.draw = _bridge_draw


# ================================================================== the stage protocol (what the engine and the beats use)
class _BridgeLayer:
    def __init__(self, bridge, layer, z):
        self.bridge, self.layer, self.z = bridge, layer, z

    def draw(self, cv, ox, oy, t):
        self.bridge.draw(cv, ox, oy, t, self.layer)


def _cam_limits(self, world, t):
    return CAM_X, CAM_Y


def _floor_lights(self, cv, off, t, world):
    ox, oy = off
    for b in self.braziers:
        if b.lit(t):
            fl = 0.8 + 0.12 * math.sin(t * 9 + b.x) + 0.06 * math.sin(t * 23 + b.z)
            px.boost(cv, b.x + ox, floor_y(b.z) + oy, 54, hexc('ff9a4a'), 0.5 * fl, squash=0.6)


def _sprites(self, t, world):
    items = [(p.z - 0.01, p) for p in self.pillars] + [(b.z, b) for b in self.braziers]
    items += [(r.z - 0.5, r) for r in self.rails] + [(p.z - 0.4, p) for p in self.posts]
    items += [(105.5, _BridgeLayer(self.bridge, 'back', 105.5)), (130.5, _BridgeLayer(self.bridge, 'front', 130.5))]
    return items


def _front(self, cv, off, t, world):
    return None


def _update(self, world, t, dt):
    return None


Stage.cam_limits = _cam_limits
Stage.floor_lights = _floor_lights
Stage.sprites = _sprites
Stage.front = _front
Stage.update = _update


# ------------------------------------------------------------------ areas and routes (data for the choreographer)
# An area: where a fight can take place. x, z: the bounds to keep the fighters in; y: its height (a number, or 'roof'
# for the slope); center: where an exchange gravitates to; size: 'wide' areas take long exchanges, 'narrow' short ones.
AREAS = {
    'courtyard': dict(x=(90.0, 560.0), z=(40.0, 205.0), y=0.0, center=(320.0, 118.0), size='wide',
                      floor='stone', label='the courtyard'),
    'roof': dict(x=(190.0, 450.0), z=(-24.0, -16.0), y='roof', center=(320.0, -20.0), size='narrow',
                 floor='tiles', label='the temple roof'),
    'ledge': dict(x=(100.0, 440.0), z=(256.0, 274.0), y=-DROP, center=(260.0, 264.0), size='narrow',
                  floor='stone', label='the ledge over the drop'),
    'peak': dict(x=(790.0, 920.0), z=(90.0, 160.0), y=0.0, center=(850.0, 122.0), size='medium',
                 floor='stone', label='the pinnacle'),
}
# A route: how the fight gets from one area to another. how: 'leap' (both jump there), 'knocked_off' (a big blow
# throws the loser of the exchange there, the other follows), 'thrown_through' (through a breakable on the way, then
# down), 'run_across' (along a path with gaps to leap; the other chases). 'needs' lists what must still stand.
ROUTES = [
    dict(a='courtyard', b='roof', how='leap', land_z=-20.0),
    dict(a='roof', b='courtyard', how='knocked_off', path='eave'),
    dict(a='courtyard', b='ledge', how='thrown_through', target='rail'),
    dict(a='ledge', b='courtyard', how='leap', land_z=200.0),
    dict(a='courtyard', b='peak', how='run_across', via='bridge', needs=('bridge',),
         path=[(600.0, 118.0), (656.0, 118.0), ('gap', 704.0), (760.0, 120.0), (806.0, 124.0)]),
    dict(a='peak', b='courtyard', how='run_across', via='bridge', needs=('bridge',),
         path=[(760.0, 120.0), (704.0, 118.0), ('gap', 656.0), (600.0, 118.0), (560.0, 118.0)]),
]
# set pieces a beat can use where they stand: perches to spring onto (pillar caps, the balustrade's top)
PERCHES = {'courtyard': [('pillar', 0), ('pillar', 1), ('pillar', 2), ('pillar', 3)], 'courtyard_rail': True}
TITLE_PAN = dict(start=(520.0, 120.0), end=(80.0, 45.0))     # the camera's pan over the stage under the title
START = {'courtyard': [(260.0, 120.0), (392.0, 118.0)]}
PROJ = (Y0, FS)


def _area_y(self, area, x, z):
    a = AREAS[area]
    if a['y'] == 'roof':
        return roof_y(z)
    return a['y']


def _area_of(self, x, z, y, t):
    """The area a fighter at (x, z, y) is in (None in the air between areas)."""
    for name, a in AREAS.items():
        if a['x'][0] - 30 <= x <= a['x'][1] + 30 and a['z'][0] - 30 <= z <= a['z'][1] + 30:
            ay = _area_y(self, name, x, z)
            if abs(y - ay) < 14:
                return name
    return None


def _route_open(self, route, t):
    if 'bridge' in route.get('needs', ()) and self.bridge.snap_t is not None and t >= self.bridge.snap_t - 0.01:
        return False
    return True


Stage.area_y = _area_y
Stage.area_of = _area_of
Stage.route_open = _route_open
Stage.AREAS = AREAS
Stage.ROUTES = ROUTES
Stage.START = START
Stage.TITLE_PAN = TITLE_PAN
Stage.CAM_X = CAM_X
Stage.CAM_Y = CAM_Y


# ------------------------------------------------------------------ targets: what a body or a weapon can smash
class Target:
    """Something in an area a fighter can be thrown into or a weapon can hit. smash(world, t, power, out, seed)
    schedules its breaking at t; power 1 is a hard blow, 2 a heavy weapon, 3+ a titan."""

    def __init__(self, stage, kind, area, x, z, h, ref=None):
        self.stage, self.kind, self.area, self.x, self.z, self.h, self.ref = stage, kind, area, x, z, h, ref
        self.used = 0

    def ok(self, t):
        if self.kind == 'brazier':
            return self.ref.lit(t)
        if self.kind == 'pillar':
            return self.ref.standing(t)
        if self.kind == 'rail':
            return self.ref.intact(t)
        return True

    def smash(self, world, t, power, out, seed):
        self.used += 1
        if self.kind == 'brazier':
            brazier_crash(world, self.ref, t, out, seed)
        elif self.kind == 'pillar':
            idx = self.stage.pillars.index(self.ref)
            pillar_hit(world, t, idx, self.h, power, out=-out)
            if power >= 2.0:
                return topple(world, idx, t + 0.3, out, seed)
        elif self.kind == 'rail':
            idx = self.stage.rails.index(self.ref)
            rail_break(world, idx, t, seed, out=1.0, power=power)
        elif self.kind == 'gong':
            gong(world, t, seed)
            world.at(t, lambda w: (w.shake(t, 4.0, 0.4),
                                   w.effects.append(fx.Ring(self.x, floor_y(self.z) - self.h, t, 4, 30, 0.35, 'ffd070')),
                                   fx.burst(w, fx.SPARK, self.x, floor_y(self.z) - self.h, self.z, 20, (40, 140),
                                            life=(0.2, 0.5), seed=seed)))
        return t


def _targets(self, area, t):
    if not hasattr(self, '_targets_cache'):
        T = []
        for i, b in enumerate(self.braziers):
            T.append(Target(self, 'brazier', 'courtyard', b.x, b.z, 30, b))
        for p in self.pillars:
            T.append(Target(self, 'pillar', 'courtyard', p.x, p.z, 30, p))
        for r in self.rails:
            T.append(Target(self, 'rail', 'courtyard', (r.x0 + r.x1) / 2, RAIL_Z, RAIL_Y, r))
        gx, gz, gh = self.gong
        T.append(Target(self, 'gong', 'peak', gx, gz, gh))
        self._targets_cache = T
    return [x for x in self._targets_cache if x.area == area and x.ok(t)]


Stage.targets = _targets


# ------------------------------------------------------------------ the temple breaking (stage-specific set pieces)
def _rng(seed):
    return np.random.default_rng(seed)


def brazier_crash(world, br, t, out, seed):
    """A body knocks a brazier over: it topples, its coals spill and burn to the end."""
    def crash(w):
        w.cue('brazier_crash', x=br.x)
        w.cue('coals', x=br.x - 14 * out, dur=90.0)
        br.down_t = w.t
        br.dir = out
        x, z = br.x + 14 * out, br.z + 2
        fx.burst(w, fx.FLAME, br.x, floor_y(br.z) - 30, br.z, 30, (30, 110), life=(0.3, 0.6), size=(1, 2), seed=seed)
        fx.burst(w, fx.EMBER, br.x, floor_y(br.z) - 30, br.z, 20, (30, 120), life=(0.6, 1.4), seed=seed + 1)
        rng = _rng(seed)
        for k in range(22):
            w.stage.stamp(x + rng.uniform(-14, 10), z + rng.uniform(-8, 8), hexc('1a120c'), r=rng.uniform(1, 2.5),
                          alpha=0.8, seed=seed + 10 + k)
        w.shake(w.t, 3.0, 0.3)
    world.at(t, crash)

    def coals(w, tt, dt):
        rng = _rng(int(tt * 1000) + seed)
        x, z = br.x + 14 * out + rng.uniform(-12, 8), br.z + 2 + rng.uniform(-7, 7)
        w.particles.spawn(fx.FLAME, [x], [z], [0.5], [rng.uniform(-5, 5)], [0.0], [rng.uniform(15, 35)],
                          [rng.uniform(0.2, 0.45)], [rng.uniform(0.8, 1.5)], rng=rng)
        if rng.uniform() < 0.25:
            w.particles.spawn(fx.EMBER, [x], [z], [1.0], [rng.uniform(-10, 10)], [0.0], [rng.uniform(20, 50)],
                              [rng.uniform(0.5, 1.0)], [1.0], rng=rng)

    def coal_glow(cv, add, tt, off, lights, w):
        x, y = br.x + 14 * out + off[0], floor_y(br.z + 2) + off[1]
        px.boost(cv, x, y, 40, hexc('ff7a2a'), 0.55, squash=0.6)
    world.effects.append(fx.Emitter(t, 900.0, coals))
    world.effects.append(fx.Glow(t, 900.0, coal_glow))


def pillar_hit(world, t, idx, h, power=1.0, out=-1):
    def ev(w):
        p = w.stage.pillars[idx]
        w.cue('stone_crash', x=p.x, power=power)
        p.hit(h, power, int(t * 1000) + idx, out)
        face = p.x + out * (p.R + 1)
        y = floor_y(p.z) - h
        fx.burst(w, fx.CHIP, face, y, p.z + 2, int(10 * power), (40, 150), angle=-math.pi / 2 + out * 0.9,
                 spread=0.9, life=(1.0, 2.5), size=(1, 2.2), seed=int(t * 999), zs=0.6)
        fx.burst(w, fx.DUST, face, y, p.z + 2, 8, (8, 30), life=(0.5, 1.0), size=(1.5, 2.5), seed=int(t * 997))
        w.shake(t, 3.0 * power, 0.35)
        w.effects.append(fx.Ring(face, y, t, 3, 14 + 6 * power, 0.25, 'cfc8d8'))
    world.at(t, ev)


def roof_break(world, x, z, power=1.0, seed=0):
    """A heavy landing or a blow on the roof: tiles burst out of it (a dark hole stays) and slide off the eave."""
    rng = _rng(seed)
    g = roof_y(z)
    for k in range(int(5 + 6 * power)):
        world.stage.stamp(x + rng.uniform(-6, 6) * power, z + rng.uniform(-4, 4), hexc('0c0a10'),
                          r=rng.uniform(1.0, 2.2), alpha=0.95, seed=int(rng.integers(1e6)), h=g)
    for k in range(int(3 + 3 * power)):
        world.stage.stamp(x + rng.uniform(-8, 8) * power, z + rng.uniform(-5, 5), hexc('3a2414'), r=rng.uniform(0.6, 1.2),
                          alpha=0.9, seed=int(rng.integers(1e6)), h=g)
    n = int(14 + 16 * power)
    world.particles.spawn(fx.TILE, x + rng.uniform(-6, 6, n), z + rng.uniform(-3, 3, n), g + rng.uniform(1, 4, n),
                          rng.uniform(-60, 60, n), rng.uniform(20, 70, n), rng.uniform(40, 130, n), rng.uniform(1.5, 2.5, n),
                          rng.uniform(1.2, 2.4, n), rng=rng)
    fx.burst(world, fx.DUST, x, floor_y(z) - g - 2, z, int(8 * power), (10, 40), life=(0.4, 0.9), size=(1.5, 2.5),
             seed=seed + 1)
    world.cue('roof_tiles', x=x, power=power)


def rail_break(world, idx, t, seed, out=1.0, power=1.0):
    """A balustrade section bursts: chunks fly out over the drop, dust, the stubs stay."""
    r = world.stage.rails[idx]
    r.break_t = t

    def ev(w):
        rng = _rng(seed)
        n = int(22 * power)
        xs = rng.uniform(r.x0 + 2, r.x1 - 2, n)
        w.particles.spawn(fx.CHIP, xs, RAIL_Z + rng.uniform(-2, 2, n), rng.uniform(4, RAIL_Y, n),
                          rng.uniform(-50, 50, n), out * rng.uniform(20, 90, n), rng.uniform(30, 110, n),
                          rng.uniform(1.5, 3.0, n), rng.uniform(1.4, 3.0, n), rng=rng)
        fx.burst(w, fx.DUST, (r.x0 + r.x1) / 2, floor_y(RAIL_Z) - 8, RAIL_Z, 14, (10, 40), life=(0.5, 1.0),
                 size=(1.5, 2.8), seed=seed + 1)
        for k in range(8):
            w.stage.stamp(rng.uniform(r.x0, r.x1), rng.uniform(FLOOR_Z[1] - 6, FLOOR_Z[1]), fx.STONE_C[1 + k % 3],
                          r=rng.uniform(0.6, 1.4), alpha=1.0, seed=seed + 10 + k)
        w.shake(t, 2.5 * power, 0.25)
        w.cue('stone_crash', x=(r.x0 + r.x1) / 2, power=power)
    world.at(t, ev)


def topple(world, idx, t, direction, seed):
    """A pillar cracked through at the base goes over, turning as a body (towards the camera a little); where it
    lands the floor cracks and dust rolls out. Returns when it lands."""
    p = world.stage.pillars[idx]
    p.fall_t = t
    p.fall_dir = direction
    tl = t + p.fall_dur

    def start(w):
        fx.burst(w, fx.CHIP, p.x, floor_y(p.z) - 6, p.z, 16, (40, 120), angle=-math.pi / 2, spread=1.2,
                 life=(0.8, 1.6), size=(1, 2.4), seed=seed)
        w.cue('stone_crack', x=p.x)

    def land(w):
        rng = _rng(seed + 1)
        dx, dz = p.fall_vec()
        for k in range(int(p.HT)):
            x, z = p.x + dx * k, p.z + dz * k
            if rng.uniform() < 0.4:
                w.stage.stamp(x + rng.uniform(-1, 1), z + rng.uniform(-6, 8), hexc('120e16'), r=0.7, alpha=0.9,
                              seed=seed + k)
            if k % 6 == 0:
                fx.burst(w, fx.DUST, x, floor_y(z) - 4, z, 3, (10, 50), life=(0.6, 1.2), size=(1.6, 3.0),
                         seed=seed + 100 + k)
        n = 30
        w.particles.spawn(fx.CHIP, p.x + dx * rng.uniform(0, p.HT, n), p.z + dz * rng.uniform(0, p.HT, n) + rng.uniform(-6, 6, n),
                          rng.uniform(2, 14, n), rng.uniform(-60, 60, n), rng.uniform(-40, 40, n), rng.uniform(40, 120, n),
                          rng.uniform(1.0, 2.0, n), rng.uniform(1.2, 2.6, n), rng=rng)
        w.shake(tl, 5.0, 0.45)
        w.cue('pillar_fall', x=p.x + dx * p.HT * 0.6)
    world.at(t, start)
    world.at(tl, land)
    return tl


def bridge_snap(world, t, seed):
    br = world.stage.bridge
    br.snap_t = t

    def ev(w):
        rng = _rng(seed)
        n = 40
        x = rng.uniform(640, 720, n)
        w.particles.spawn(fx.WOOD, x, rng.uniform(106, 130, n), rng.uniform(-6, 4, n), rng.uniform(-60, 60, n),
                          rng.uniform(-20, 20, n), rng.uniform(10, 90, n), rng.uniform(1.5, 2.5, n),
                          rng.uniform(1.0, 2.0, n), rng=rng)
        fx.burst(w, fx.DUST, 679, floor_y(118) + 2, 118, 12, (10, 40), life=(0.5, 1.0), size=(1.5, 2.5), seed=seed + 1)
        w.shake(t, 3.0, 0.3)
        w.cue('bridge_snap', x=679)
    world.at(t, ev)


def gong(world, t, seed):
    """Something hits the gong on the pinnacle: it rings (a glow pulses on it)."""
    gx, gz, gh = world.stage.gong

    def glow(cv, add, tt, off, lights, w):
        k = max(0.0, 1 - (tt - t) / 1.6)
        x, y = gx + off[0], floor_y(gz) - gh + off[1]
        px.glow(add, x, y, 16, hexc('ffd070'), 0.5 * k * (0.6 + 0.4 * math.sin(tt * 40)))
        lights.append((x, y, 40, hexc('ffc060'), 0.7 * k))
    world.effects.append(fx.Glow(t, t + 1.6, glow))
    world.at(t, lambda w: w.cue('gong', x=gx))


def roof_hook(world, f, g):
    """Landings on the roof burst tiles."""
    if abs(g - roof_y(f.z)) < 1.5 and RIDGE_Z <= f.z <= EAVE_Z and ROOF_X[0] <= f.x <= ROOF_X[1]:
        roof_break(world, f.x, f.z, 0.8, int(world.t * 1000) + 7)


def setup(world):
    """Called once when the world is built: hooks that belong to this stage."""
    world.land_hooks.append(roof_hook)


def smash_floor(world, x, z, power, seed, g=None):
    """A heavy impact on whatever surface is at (x, z): the roof's tiles burst, the bridge's planks splinter, stone
    cracks into a crater (bigger with power)."""
    if g is None:
        g = world.stage.ground(x, z, world.t, 999.0)
    if ROOF_X[0] <= x <= ROOF_X[1] and RIDGE_Z <= z <= EAVE_Z and abs(g - roof_y(z)) < 2:
        roof_break(world, x, z, 1.0 + power * 0.6, seed)
        return 'roof'
    if BRIDGE_X[0] <= x <= BRIDGE_X[1] and BRIDGE_Z[0] - 4 <= z <= BRIDGE_Z[1] + 4:
        fx.burst(world, fx.WOOD, x, floor_y(z) - g + 2, z, int(14 + 8 * power), (40, 120), angle=-math.pi / 2,
                 spread=1.2, life=(0.8, 1.5), seed=seed)
        return 'bridge'
    return 'stone'


def _smash_floor(self, world, x, z, power, seed, g=None):
    return smash_floor(world, x, z, power, seed, g)


def _eave_fall(self, c, D, th, to):
    """Thrown down the roof's slope, off the eave, onto the courtyard (tiles burst on the way)."""
    from acts import tumble_to  # noqa: F401
    w = c.w
    x0, z0 = D.x_at(th), D.z_at(th)
    s = 1 if to[0] > x0 else -1
    D.add(th, 'tumble', 0.3, to=(x0 + 22 * s, 2.0), ease=lambda u: u,
          air=lambda u, y0=roof_y(z0): y0 + (roof_y(2.0) - y0) * u + 8 * u * (1 - u), blend=0.03)
    w.trail(D.name, th, th + 0.9)
    w.at(th + 0.15, lambda world: roof_break(world, x0 + 10 * s, -8.0, 1.2, c.seed()))
    D.add(th + 0.3, 'tumble', 0.55, to=to, ease=lambda u: u, air=lambda u, y0=roof_y(2.0): y0 * (1 - u * u), blend=0.02)
    w.cue('fly', t=th, dur=0.85, who=D.name)
    return th + 0.85


def _set_piece(self, c, A, D, t, piece):
    """Set pieces only this stage has. 'bridge_snap': out on the bridge A's heavy blow throws D back over the planks,
    A leaps after him and smashes them; the bridge snaps and both leap for their own side of the chasm."""
    import beats as B
    from acts import strike, tumble_to, fall_lie_up, jump_to, guard, guard_move
    from scene import arc
    w = c.w
    if piece != 'bridge_snap' or not self.route_open({'needs': ('bridge',)}, t):
        return t
    k = A.wep_at(t, 'n')[0]
    t = B.engage(c, A, D, t)
    D.add(t + 0.05, 'x_block' if D.wep_at(t, 'f')[0] else 'block', 0.5)
    nm, wp = ('hammer_upper', ('n', 'hammer')) if k == 'hammer' else ('uppercut', None)
    th = strike(c, A.name, D.name, t, nm, 'chest', wep=wp, react=None, blood=14, size=11, spark='big', shake=4.0,
                settle_move=guard_move(A, t), wound=B.wound('head', 'stain', 1.6, c.rng))
    tumble_to(c, D, th, (722.0, 118.0), 0.6, 30, y1=sag(722))
    t_up = fall_lie_up(c, D, th + 0.6, -6, 0.2, 0.45, h=7, g=sag(716))
    w.mark('thrown_back', th + 0.3)
    tsj = th + 0.75
    jump_to(c, A, tsj, (D.x_at(t_up) + 40, 118.0), 0.5, 26, y1=sag(D.x_at(t_up) + 40), move='jump', settle=False)
    guard(A, tsj + 0.5)
    tsm = tsj + 0.62
    if k == 'hammer':
        th2 = strike(c, A.name, None, tsm, 'hammer_smash', at=(D.x_at(tsm) + 2, floor_y(118) - sag(D.x_at(tsm)), 118.0),
                     wep=('n', 'hammer'), settle_move='hammer_guard', quiet=True)
    else:
        th2 = strike(c, A.name, None, tsm, 'axe_kick', at=(D.x_at(tsm) + 2, floor_y(118) - sag(D.x_at(tsm)), 118.0),
                     quiet=True)
    D.add(th2 - 0.3, 'back_flip', 0.5, to=(646.0, 118.0), air=arc(22, D.y_at(th2 - 0.3), sag(646)), face=1)
    t_snap = th2 + 0.25
    bridge_snap(w, t_snap, c.seed())
    w.at(th2, lambda world: (fx.burst(world, fx.WOOD, A.x_at(th2) - 18, floor_y(118) + 4, 118, 20, (40, 120),
                                      angle=-math.pi / 2, spread=1.2, life=(0.8, 1.5), seed=c.seed()),
                             world.shake(th2, 3.5, 0.3), world.cue('hammer_floor', x=A.x_at(th2) - 18, wood=True)))
    w.mark('snap', t_snap + 0.1)
    jump_to(c, D, t_snap + 0.05, (564.0, 48.0), 0.6, 26, y1=0.0, move='flip', face=1)
    jump_to(c, A, t_snap + 0.08, (800.0, 120.0), 0.55, 24, y1=0.0, move='flip', face=-1)
    A.add(t_snap + 0.63, guard_move(A, t_snap), 0.4, face=-1)
    c.area = 'courtyard'
    return t_snap + 1.0


Stage.smash_floor = _smash_floor
Stage.eave_fall = _eave_fall
Stage.set_piece = _set_piece
