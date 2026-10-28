"""The courtyard (version 3's stage, the look the owner approved): a cliff-top temple courtyard seen from slightly above (three-quarter view), wide enough to show it all.

World: x along the screen, z into depth (0 at the back edge of the floor), y height above the floor. The projection is
oblique: stage_x = x, stage_y = Y0 + z * FS - y. The stage image (with a margin) is cropped by the camera; pillars and
braziers are sprites sorted by depth with the fighters; blood and debris are stamped into the floor for good.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

import px
from px import hexc, hash_noise, fbm
from proj import set_projection

W, H = px.W, px.H
M = 8                               # canvas margin (shake)
EW, EH = W + 2 * M, H + 2 * M
SW, SH = 520, 300                   # stage size (stage coords)
PAD = 12                            # stage image margin
Y0, FS = 127.0, 0.6                 # floor: stage y of z=0, and the depth foreshortening
CAM_X, CAM_Y = (0.0, 40.0), (0.0, 30.0)
FLOOR_X = (36, 484)
FLOOR_Z = (0.0, 232.0)


def floor_y(z):
    return Y0 + z * FS


def col255(h, a=255):
    c = hexc(h)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def mix_arr(a, b, k):
    k = np.clip(np.asarray(k, np.float32), 0, 1)
    return a + (b - a) * k[..., None]


class Pillar:
    """A stone pillar topped with a stone lantern; it keeps every crack a body puts in it."""
    R, HT = 9, 82

    def __init__(self, x, z, seed):
        self.x, self.z, self.seed = x, z, seed
        self.hits = []
        self.img = None

    def hit(self, h, power, seed, out):
        self.hits.append((h, power, seed, out))
        self.img = None

    def bake(self):
        R, HT = self.R, self.HT
        w = 2 * (R + 7) + 1
        h = HT + 30
        c0 = R + 7
        base = h - 4                    # the row of the floor contact
        ys, xs = np.mgrid[0:h, 0:w]
        dx = xs + 0.5 - c0
        row = ys[:, :1]
        hy = base - row                 # height above the floor
        rr = np.full((h, 1), float(R))
        band = ((hy % 30) <= 2) & (hy > 8) & (hy < HT - 6)
        rr = rr + band * 1.0
        rr = np.where(hy < 7, R + 3.0, rr)
        rr = np.where((hy >= 7) & (hy < 10), R + 1.5, rr)
        rr = np.where((hy >= HT - 6) & (hy < HT), R + 2.0, rr)
        inside = (np.abs(dx) <= rr) & (hy >= -1) & (hy < HT)
        s = np.clip(dx / rr, -1, 1)
        lam = -0.55 * s + 0.62 * np.sqrt(1 - s * s)
        stone = np.where(band[..., None], hexc('8a6a32')[None, None, :], hexc('6e687a')[None, None, :])
        n = hash_noise(xs, ys, self.seed)
        rgb = stone * (0.36 + 0.82 * lam[..., None]) * (0.9 + 0.16 * n[..., None])
        flute = (~band) & (np.floor(dx + 100).astype(int) % 4 == 0) & (np.abs(s) < 0.8) & (hy > 10) & (hy < HT - 6)
        rgb[flute] *= 0.82
        edge = inside & (np.abs(np.abs(dx) - rr) < 1.0)
        rgb[edge] *= 0.35
        a = inside.astype(np.float32)
        # the stone lantern on top: a lit box under a little roof
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        top = base - HT
        d.rectangle([c0 - 6, top - 12, c0 + 6, top], fill=col255('5a5466'))
        d.rectangle([c0 - 4, top - 10, c0 + 4, top - 3], fill=col255('ffb050'))
        d.line([(c0, top - 10), (c0, top - 3)], fill=col255('6a3a14'))
        d.polygon([(c0 - 11, top - 12), (c0 + 11, top - 12), (c0 + 6, top - 19), (c0 - 6, top - 19)], fill=col255('3a3646'))
        d.line([(c0 - 11, top - 12), (c0 + 11, top - 12)], fill=col255('8a8496'))
        d.rectangle([c0 - 1, top - 22, c0 + 1, top - 19], fill=col255('3a3646'))
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

    def sprite(self):
        if self.img is None:
            self.bake()
        return self.img, self.anchor


class Brazier:
    """A tall bronze brazier on three legs; a body can knock it over and spill its coals."""

    def __init__(self, x, z):
        self.x, self.z = x, z
        self.down_t = None
        self.dir = 1

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
        return (x, yb - L - 14)


class Stage:
    def __init__(self):
        self.img = np.zeros((SH + 2 * PAD, SW + 2 * PAD, 3), np.float32)
        self.build_sky()
        self.build_hall()
        self.build_sides()
        self.build_floor()
        self.build_front()
        self.pillars = [Pillar(60, 8, 3), Pillar(460, 8, 9), Pillar(60, 224, 13), Pillar(460, 224, 17)]
        self.braziers = [Brazier(112, 34), Brazier(408, 34), Brazier(104, 200), Brazier(416, 200)]
        self.mist = fbm(SW + 2 * PAD + 128, 40, 16, seed=91, octaves=3)

    # stage coords -> image index
    def put_img(self, arr, x, y):
        px.blit(self.img, arr, x + PAD, y + PAD)

    def build_sky(self):
        h, w = self.img.shape[:2]
        y = np.arange(h, dtype=np.float32)[:, None]
        k = np.clip((y - PAD) / 110.0, 0, 1)
        sky = np.where((k < 0.6)[..., None], mix_arr(hexc('05040e'), hexc('150d28'), k / 0.6),
                       mix_arr(hexc('150d28'), hexc('3a2148'), (k - 0.6) / 0.4))
        sky = np.repeat(sky, w, axis=1)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        n = hash_noise(X, Y, seed=5)
        stars = (n > 0.9955) & (Y < 95)
        sky[stars] = hexc('cfd8ff') * (0.5 + 0.5 * hash_noise(X, Y, seed=6)[stars, None])
        mx, my, mr = 412 + PAD, 36 + PAD, 16.0
        d = np.sqrt((X + 0.5 - mx) ** 2 + (Y + 0.5 - my) ** 2)
        sky += (np.clip(1 - (d - mr) / 50.0, 0, 1) ** 2.2)[..., None] * hexc('5a4a7a') * 0.55
        lit = np.clip(((X - mx) * -0.4 + (Y - my) * -0.6) / mr * 0.5 + 0.75, 0, 1)
        moon = mix_arr(hexc('b9b4cc'), hexc('f4f0ff'), lit) * (0.86 + 0.18 * fbm(w, h, 4, seed=21)[..., None])
        sky[d <= mr] = moon[d <= mr]
        cl = fbm(w, h, 24, seed=31, octaves=4)
        band = np.exp(-((Y - (34 + PAD)) / 9.0) ** 2) + 0.7 * np.exp(-((Y - (58 + PAD)) / 6.0) ** 2)
        cloud = np.clip((cl - 0.48) * 3.0, 0, 1) * band
        sky = sky * (1 - cloud[..., None] * 0.7) + hexc('2a1d3e') * cloud[..., None] * 0.7
        # two mountain ranges
        for base, amp, cell, seed, fill, lit_c in ((92, 26, 34, 41, '231c3c', '4a3d6c'), (104, 20, 22, 47, '140f24', '2e2548')):
            nn = fbm(w, 1, cell, seed=seed, octaves=3)[0]
            peaks = np.abs(np.sin(np.arange(w) / (cell * 0.9) + seed)) ** 3
            ridge = base + PAD - (nn - 0.3) * amp * 2.0 - peaks * amp * 0.45
            m = Y >= ridge[None, :]
            slope = np.gradient(ridge)
            facing = np.clip(slope * 1.6, -1, 1)[None, :]
            depth = np.clip((Y - ridge[None, :]) / 24.0, 0, 1)
            kk = np.clip(0.55 + 0.45 * facing, 0, 1) * (1 - depth)
            rgb = mix_arr(hexc(fill), hexc(lit_c), kk) * (0.9 + 0.18 * fbm(w, h, 6, seed=seed + 3)[..., None])
            fog = np.clip((Y - (base + PAD - 6)) / 22.0, 0, 1)
            rgb = rgb * (1 - fog[..., None] * 0.55) + hexc('3a2c50') * fog[..., None] * 0.55
            sky[m] = rgb[m]
        self.img[:] = sky

    def build_hall(self):
        w, h = self.img.shape[1], self.img.shape[0]
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        P = PAD
        wood, wood_d, beam = col255('5a1a14'), col255('3a0f0c'), col255('241210')
        roof, roof_l, roof_d, gold = col255('1c1a24'), col255('34324a'), col255('100e16'), col255('b58a3a')
        stone, stone_l, stone_d = col255('4a4352'), col255('6a6274'), col255('2a2430')
        L, R = 116 + P, 404 + P
        # side wings, lower, behind the main hall
        for (a, b) in ((-10 + P, L + 10), (R - 10, SW + 10 + P)):
            d.rectangle([a, 78 + P, b, 116 + P], fill=wood_d)
            for k in range(a, b, 20):
                d.rectangle([k, 78 + P, k + 2, 116 + P], fill=beam)
            for k in range(a + 7, b - 8, 20):
                d.rectangle([k, 86 + P, k + 8, 104 + P], fill=col255('d8883a'))
                for gx in range(k, k + 9, 3):
                    d.line([(gx, 86 + P), (gx, 104 + P)], fill=col255('5a2a10'))
            d.polygon([(a - 6, 80 + P), (b + 6, 80 + P), (b, 66 + P), (a, 66 + P)], fill=roof)
            for k in range(66 + P, 80 + P, 2):
                d.line([(a, k), (b, k)], fill=roof_l if (k // 2) % 2 == 0 else roof)
            d.line([(a - 6, 80 + P), (b + 6, 80 + P)], fill=gold)
        # the main hall's facade
        d.rectangle([L, 70 + P, R, 108 + P], fill=wood)
        for i, x in enumerate(range(L + 6, R, 26)):
            lit = 3 <= i <= 7
            if x + 22 < R:
                c = col255('ffb050') if lit else col255('7a3a1c')
                d.rectangle([x + 5, 76 + P, x + 21, 104 + P], fill=c)
                for gx in range(x + 5, x + 22, 3):
                    d.line([(gx, 76 + P), (gx, 104 + P)], fill=col255('5a2a10'))
                for gy in range(76 + P, 105 + P, 4):
                    d.line([(x + 5, gy), (x + 21, gy)], fill=col255('5a2a10'))
            d.rectangle([x, 70 + P, x + 3, 108 + P], fill=col255('8a2018'))
            d.line([(x, 70 + P), (x, 108 + P)], fill=col255('b03a24'))
        d.rectangle([L - 2, 66 + P, R + 2, 70 + P], fill=beam)
        d.line([(L - 2, 67 + P), (R + 2, 67 + P)], fill=gold)
        d.rectangle([236 + P, 71 + P, 284 + P, 78 + P], fill=col255('1a1014'), outline=gold)
        # the roof: a broad tiled slope with upturned corners and a ridge
        d.polygon([(L - 22, 66 + P), (R + 22, 66 + P), (R + 30, 58 + P), (R - 52, 30 + P), (L + 52, 30 + P), (L - 30, 58 + P)],
                  fill=roof)
        for k in range(31, 66, 3):
            t_ = (k - 30) / 36.0
            xl = L + 52 - (82 * t_)
            xr = R - 52 + (82 * t_)
            d.line([(xl + P * 0, k + P), (xr, k + P)], fill=roof_l if (k // 3) % 2 == 0 else roof_d)
        d.line([(L - 30, 58 + P), (L - 22, 66 + P), (R + 22, 66 + P), (R + 30, 58 + P)], fill=gold)
        d.rectangle([L + 50, 26 + P, R - 50, 31 + P], fill=roof_d)
        d.line([(L + 50, 26 + P), (R - 50, 26 + P)], fill=col255('4a4660'))
        for ex in (L + 50, R - 50):
            d.polygon([(ex - 4, 31 + P), (ex + 4, 31 + P), (ex + (6 if ex > 260 else -6), 18 + P)], fill=roof_d)
        # the platform and the steps down to the courtyard
        d.rectangle([L - 14, 108 + P, R + 14, 116 + P], fill=stone)
        d.line([(L - 14, 108 + P), (R + 14, 108 + P)], fill=stone_l)
        for i, (a, b) in enumerate(((206, 314), (198, 322), (190, 330))):
            y0 = 116 + i * 4
            d.rectangle([a + P, y0 + P, b + P, y0 + 4 + P], fill=stone if i % 2 == 0 else stone_d)
            d.line([(a + P, y0 + P), (b + P, y0 + P)], fill=stone_l)
        arr = np.asarray(img).astype(np.float32) / 255.0
        lit = (arr[..., 0] > 0.9) & (arr[..., 1] > 0.55)
        glow = gaussian_filter(lit.astype(np.float32), 3.0)
        arr[..., :3] += glow[..., None] * hexc('ff8a30') * 0.55 * (arr[..., 3:4] > 0)
        px.blit(self.img, arr, 0, 0)
        # hanging lanterns under the eaves
        for lx in (128, 176, 344, 392):
            lantern(self.img, lx + PAD, 69 + PAD)

    def build_sides(self):
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        # dark rock and pines beyond the low side walls
        for side in (0, 1):
            x0, x1 = (0, FLOOR_X[0] - 6 + PAD) if side == 0 else (FLOOR_X[1] + 6 + PAD, w)
            region = (X >= x0) & (X < x1) & (Y >= 112 + PAD)
            n = fbm(w, h, 7, seed=101 + side)
            col = hexc('121a16') * (0.7 + 0.6 * n[..., None]) + hexc('0a0c14') * 0.3
            self.img[region] = col[region]
            rng = np.random.default_rng(55 + side)
            for _ in range(9):
                tx = rng.uniform(x0 + 2, x1 - 2)
                ty = rng.uniform(130, 290) + PAD
                th = rng.uniform(16, 30)
                for k in range(int(th)):
                    half = (k / th) * th * 0.32
                    yy = int(ty - th + k)
                    for xx in range(int(tx - half), int(tx + half) + 1):
                        if x0 <= xx < x1 and 0 <= yy < h:
                            sh = 0.75 if xx < tx else 0.55
                            self.img[yy, xx] = hexc('1c3024') * sh * (0.85 + 0.3 * ((xx * 7 + yy * 3) % 5 == 0))
        # the low walls along the floor's sides: a coping strip seen from above
        for x in (FLOOR_X[0] - 6, FLOOR_X[1] + 1):
            xs = slice(x + PAD, x + 5 + PAD)
            ys = slice(int(Y0) - 10 + PAD, int(floor_y(FLOOR_Z[1])) + 8 + PAD)
            self.img[ys, xs] = hexc('5a5466')
            self.img[ys, x + PAD] = hexc('7a7486')
            self.img[ys, x + 4 + PAD] = hexc('2e2a36')
            for yy in range(int(Y0) - 10, int(floor_y(FLOOR_Z[1])) + 8, 9):
                self.img[yy + PAD, xs] = hexc('3a3644')

    def build_floor(self):
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w) - PAD, np.arange(h) - PAD)
        Z = (Y - Y0) / FS
        on = (X >= FLOOR_X[0]) & (X < FLOOR_X[1]) & (Z >= FLOOR_Z[0]) & (Z <= FLOOR_Z[1])
        # paving: concentric rings of stones round the fighting ring, square slabs further out
        cx, cz = 260.0, 116.0
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
        # the fighting ring inlaid in the floor
        r = np.sqrt(((X - cx) / 80.0) ** 2 + ((Z - cz) / 80.0) ** 2)
        ring = (r > 0.86) & (r < 1.0)
        col[ring] = col[ring] * 0.55 + hexc('5a2420') * 0.45
        line = (np.abs(r - 0.9) < 0.012) | (np.abs(r - 0.35) < 0.012)
        col[line] = col[line] * 0.4 + hexc('b08a3c') * 0.6
        ang = np.arctan2(Z - cz, X - cx)
        rays = (r < 0.33) & (np.abs(np.sin(ang * 4)) < (0.33 - r) * 1.6)
        col[rays] = col[rays] * 0.7 + hexc('2a2232') * 0.3
        moss = (fbm(w, h, 5, seed=81) > 0.66) & (hash_noise(X, Y, seed=82) > 0.7)
        col[moss] = col[moss] * 0.7 + hexc('2c3a28') * 0.3
        rng = np.random.default_rng(5)
        for _ in range(40):
            x = rng.uniform(FLOOR_X[0], FLOOR_X[1])
            y = rng.uniform(Y0 + 2, floor_y(FLOOR_Z[1]) - 2)
            for _ in range(rng.integers(6, 18)):
                xi, yi = int(x) + PAD, int(y) + PAD
                if 0 <= xi < w and 0 <= yi < h:
                    col[yi, xi] *= 0.55
                x += rng.uniform(-1.5, 2.5)
                y += rng.uniform(-0.6, 0.6)
        # moonlight sheen towards the back, a little darker at the front
        k = np.clip((Z - 0) / 240.0, 0, 1)[..., None]
        col = col * (1.04 - 0.2 * k) + hexc('2a2440') * (1 - k) * 0.3
        # darker towards the side walls, a cool pool of moonlight on the ring
        edge = np.minimum(X - FLOOR_X[0], FLOOR_X[1] - X)
        col *= (0.7 + 0.3 * np.clip(edge / 50.0, 0, 1))[..., None]
        pool = np.clip(1 - np.sqrt(((X - cx) / 150.0) ** 2 + ((Z - cz) / 120.0) ** 2), 0, 1) ** 1.5
        col = col * (0.82 + 0.38 * pool[..., None]) + hexc('3a3a5a') * pool[..., None] * 0.12
        self.img[on] = col[on]
        # the floor's back edge
        yy = int(Y0) + PAD
        self.img[yy - 1, FLOOR_X[0] + PAD:FLOOR_X[1] + PAD] = hexc('6a6274')

    def build_front(self):
        """The balustrade at the cliff's edge, in front of everything, and the drop below it."""
        h, w = self.img.shape[:2]
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        yb = int(floor_y(FLOOR_Z[1])) + PAD
        st, st_l, st_d = col255('4e4758'), col255('6e6678'), col255('2a2430')
        d.rectangle([0, yb + 14, w, h], fill=col255('07060c'))
        d.rectangle([0, yb + 2, w, yb + 14], fill=st_d)
        d.rectangle([0, yb - 12, w, yb - 8], fill=st)
        d.line([(0, yb - 12), (w, yb - 12)], fill=st_l)
        for k in range(0, w, 6):
            d.rectangle([k + 1, yb - 7, k + 3, yb + 2], fill=st)
            d.point([(k + 1, yb - 7)], fill=st_l)
        for k in range(8, w, 52):
            d.rectangle([k, yb - 18, k + 6, yb + 6], fill=st)
            d.rectangle([k - 1, yb - 20, k + 7, yb - 18], fill=st_l)
            d.line([(k + 6, yb - 18), (k + 6, yb + 6)], fill=st_d)
        arr = np.asarray(img).astype(np.float32) / 255.0
        # pine tops and mist rising from the drop
        Y = np.arange(h)[:, None]
        mist = np.clip(1 - np.abs(Y - (yb + 22)) / 10.0, 0, 1) * 0.5
        arr[..., :3] = arr[..., :3] * (1 - mist[..., None]) + hexc('3a2c50') * mist[..., None]
        self.front = arr

    # ------------------------------------------------------------ per frame
    def view(self, cam):
        cx, cy = int(round(cam[0])), int(round(cam[1]))
        y0 = cy - M + PAD
        x0 = cx - M + PAD
        return self.img[y0:y0 + EH, x0:x0 + EW].copy(), (M - cx, M - cy)

    def front_view(self, cam):
        cx, cy = int(round(cam[0])), int(round(cam[1]))
        y0 = cy - M + PAD
        x0 = cx - M + PAD
        return self.front[y0:y0 + EH, x0:x0 + EW]

    def stamp(self, x, z, color, r=1.0, alpha=1.0, seed=0):
        if not (FLOOR_X[0] <= x < FLOOR_X[1] and FLOOR_Z[0] <= z <= FLOOR_Z[1]):
            return
        cx = x + PAD
        cy = floor_y(z) + PAD
        rx = max(0.6, r)
        ry = max(0.5, r * FS)
        for yy in range(int(math.floor(cy - ry - 1)), int(math.ceil(cy + ry + 1)) + 1):
            for xx in range(int(math.floor(cx - rx - 1)), int(math.ceil(cx + rx + 1)) + 1):
                dd = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2
                if dd <= 1.0 and 0 <= yy < self.img.shape[0] and 0 <= xx < self.img.shape[1]:
                    n = (hash(((xx * 73856093) ^ (yy * 19349663) ^ seed)) & 255) / 255.0
                    if dd > 0.6 and n < 0.4:
                        continue
                    a = alpha * (0.8 + 0.2 * n)
                    self.img[yy, xx] = self.img[yy, xx] * (1 - a) + np.asarray(color, np.float32) * a


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


# ================================================================== the stage protocol (what the engine and the beats use)
# A stage module gives the engine a Stage class with:
#   view(cam) -> (canvas, offset)          the background crop for the camera (canvas = 480x270 plus an 8 px margin)
#   cam_limits(world, t) -> (xrange, yrange)
#   ground(x, z, t, y) -> height           the highest surface at (x, z) not above y (VOID = -999 if none)
#   surface_fn(x, z, y, t) -> fn(X, Z)     the height function of the surface a fighter stands on (he keeps to it)
#   ground_arr(xs, zs, hs, t) -> (heights, stampable)   the same for particles, vectorised
#   stamp(x, z, color, r, alpha, seed, h)  paints a splat that stays (blood, scorch, chips); tint(...) coats ice
#   floor_lights(cv, off, t, world)        light pools on the floor (braziers, street lamps)
#   sprites(t, world) -> [(z, obj)]        things sorted in depth with the fighters; obj.draw(cv, ox, oy, t) may
#                                          return a light (x, y, radius, colour, strength)
#   overlay(cv, off, t, world)             a foreground layer over everything (mist, a cutaway)
#   update(world, t, dt)                   per-frame stage logic (usually nothing)
#   AREAS, ROUTES, START, TITLE_PAN        data for the choreographer (see below); area_y, area_of, route_open
#   targets(area, t) -> [Target]           what can be smashed: Target.kind, x, z, h, ok(t), smash(world, t, power,
#                                          out, seed)
#   optional: smash_floor(world, x, z, power, seed, g), smash_through(c, A, D, t, route), set_piece(c, A, D, t, name),
#   eave_fall(...); and a module-level setup(world) for hooks. The projection: set_projection(Y0, FS) in __init__.
import fx  # noqa: E402

VOID = -999.0


class _PillarSprite:
    def __init__(self, p):
        self.p = p

    def draw(self, cv, ox, oy, t):
        img, (ax, ay) = self.p.sprite()
        px.blit(cv, img, int(self.p.x + ox - ax), int(floor_y(self.p.z) + oy - ay))


class _BrazierSprite:
    def __init__(self, b):
        self.b = b

    def draw(self, cv, ox, oy, t):
        top = self.b.draw(cv, ox, oy, t)
        if top is not None:
            return (top[0], top[1], 60, hexc('ff8a3a'), 0.7)
        return None


_orig_init = Stage.__init__


def _init(self):
    set_projection(Y0, FS)
    _orig_init(self)


Stage.__init__ = _init


def _cam_limits(self, world, t):
    return CAM_X, CAM_Y


def _ground(self, x, z, t, y=None):
    if FLOOR_X[0] - 4 <= x <= FLOOR_X[1] + 4 and FLOOR_Z[0] - 4 <= z <= FLOOR_Z[1] + 4:
        return 0.0
    return VOID


def _surface_fn(self, x, z, y, t):
    return lambda X, Z: 0.0


def _ground_arr(self, x, z, h, t):
    ok = (x >= FLOOR_X[0]) & (x < FLOOR_X[1]) & (z >= FLOOR_Z[0]) & (z <= FLOOR_Z[1])
    g = np.where(ok, 0.0, VOID).astype(np.float32)
    return g, ok


def _floor_lights(self, cv, off, t, world):
    ox, oy = off
    for b in self.braziers:
        if b.down_t is None or t < b.down_t:
            fl = 0.8 + 0.12 * math.sin(t * 9 + b.x) + 0.06 * math.sin(t * 23 + b.z)
            px.boost(cv, b.x + ox, floor_y(b.z) + oy, 54, hexc('ff9a4a'), 0.5 * fl, squash=0.6)
    for p in self.pillars:
        px.boost(cv, p.x + ox, floor_y(p.z) + oy, 30, hexc('ffb060'), 0.3, squash=0.6)


def _sprites(self, t, world):
    return [(p.z, _PillarSprite(p)) for p in self.pillars] + [(b.z, _BrazierSprite(b)) for b in self.braziers]


def _front(self, cv, off, t, world):
    px.blit(cv, self.front_view(world.cam), 0, 0)


def _update(self, world, t, dt):
    return None


_orig_stamp = Stage.stamp


def _stamp(self, x, z, color, r=1.0, alpha=1.0, seed=0, h=0.0, **_):
    return _orig_stamp(self, x, z, color, r, alpha, seed)


def _tint(self, x, z, r, color, k, seed=0, h=0.0, gloss=True):
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


Stage.cam_limits = _cam_limits
Stage.ground = _ground
Stage.surface_fn = _surface_fn
Stage.ground_arr = _ground_arr
Stage.floor_lights = _floor_lights
Stage.sprites = _sprites
Stage.overlay = _front
Stage.update = _update
Stage.stamp = _stamp
Stage.tint = _tint

# one area, no routes: the whole fight stays in the courtyard
AREAS = {'courtyard': dict(x=(80.0, 440.0), z=(30.0, 200.0), y=0.0, center=(260.0, 116.0), size='wide', floor='stone',
                           label='the courtyard')}
ROUTES = []
START = {'courtyard': [(180.0, 118.0), (340.0, 114.0)]}
TITLE_PAN = None


def _area_y(self, area, x, z):
    return 0.0


def _area_of(self, x, z, y, t):
    return 'courtyard'


def _route_open(self, route, t):
    return False


Stage.area_y = _area_y
Stage.area_of = _area_of
Stage.route_open = _route_open
Stage.AREAS = AREAS
Stage.ROUTES = ROUTES
Stage.START = START
Stage.TITLE_PAN = TITLE_PAN
Stage.CAM_X = CAM_X
Stage.CAM_Y = CAM_Y


class Target:
    """A pillar (it keeps every crack) or a brazier (it topples and its coals burn)."""

    def __init__(self, stage, kind, area, x, z, h, ref=None):
        self.stage, self.kind, self.area, self.x, self.z, self.h, self.ref = stage, kind, area, x, z, h, ref
        self.used = 0

    def ok(self, t):
        if self.kind == 'brazier':
            return self.ref.down_t is None or t < self.ref.down_t
        return True

    def smash(self, world, t, power, out, seed):
        self.used += 1
        r = self.ref
        if self.kind == 'brazier':
            def crash(w):
                w.cue('brazier_crash', x=r.x)
                w.cue('coals', x=r.x + 14 * out, dur=90.0)
                r.down_t = w.t
                r.dir = out
                fx.burst(w, fx.FLAME, r.x, floor_y(r.z) - 30, r.z, 30, (30, 110), life=(0.3, 0.6), size=(1, 2),
                         seed=seed)
                fx.burst(w, fx.EMBER, r.x, floor_y(r.z) - 30, r.z, 20, (30, 120), life=(0.6, 1.4), seed=seed + 1)
                w.shake(w.t, 3.0, 0.3)
            world.at(t, crash)
        else:
            def hit(w):
                r.hit(self.h, power, seed, -out)
                face = r.x - out * (r.R + 1)
                y = floor_y(r.z) - self.h
                fx.burst(w, fx.CHIP, face, y, r.z + 2, int(10 * power), (40, 150), angle=-math.pi / 2 - out * 0.9,
                         spread=0.9, life=(1.0, 2.5), size=(1, 2.2), seed=seed, zs=0.6)
                w.shake(w.t, 3.0 * power, 0.35)
                w.cue('stone_crash', x=r.x, power=power)
            world.at(t, hit)
        return t


def _targets(self, area, t):
    if not hasattr(self, '_targets_cache'):
        self._targets_cache = ([Target(self, 'brazier', 'courtyard', b.x, b.z, 30, b) for b in self.braziers] +
                               [Target(self, 'pillar', 'courtyard', p.x, p.z, 30, p) for p in self.pillars])
    return [x for x in self._targets_cache if x.ok(t)]


Stage.targets = _targets


def setup(world):
    return None
