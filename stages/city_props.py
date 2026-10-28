"""The city's props (from the HULK vs CAPTAIN AMERICA clip): the things that get thrown, crushed, knocked down and smashed through.

Every prop is a sprite sorted by depth with the fighters (its `z`), drawn at the canvas offset `off` (stage -> canvas).
A prop can move along a path (a thrown car or table, a falling lamp), carry a rotation, and change state (dented,
wrecked, flipped, broken); lamps also light the floor while they work.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

import px
from px import hexc, hash_noise
from city import floor_y, wall_x, ZM, FS, Y0, DS, PLAT, PLAT_H, SLOPE, WALL_T, WALL_H, CEIL1

OUTLINE = hexc('07060a')


def rgba_of(im):
    return np.asarray(im).astype(np.float32) / 255.0


def outline(arr, c=OUTLINE):
    a = arr[..., 3] > 0.5
    ol = binary_dilation(a, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)) & ~a
    arr[ol, :3] = c
    arr[ol, 3] = 1.0
    return arr


def C(h, k=1.0, a=255):
    c = np.clip(hexc(h) * k, 0, 1)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def rotate(arr, deg, pivot):
    """Rotates an RGBA float sprite about pivot (in the sprite); returns (arr, new pivot)."""
    if abs(deg) < 0.5:
        return arr, pivot
    h, w = arr.shape[:2]
    R = int(math.ceil(math.hypot(max(pivot[0], w - pivot[0]), max(pivot[1], h - pivot[1])))) + 2
    big = np.zeros((2 * R, 2 * R, 4), np.float32)
    ox, oy = R - int(round(pivot[0])), R - int(round(pivot[1]))
    big[oy:oy + h, ox:ox + w] = arr
    im = Image.fromarray((np.clip(big, 0, 1) * 255).astype(np.uint8), 'RGBA')
    im = im.rotate(-deg, resample=Image.NEAREST, center=(R, R))
    return rgba_of(im), (R, R)


class Prop:
    z = 0.0
    x = 0.0
    h = 0.0
    hidden = False
    path = None          # (t0, t1, fn(u) -> (x, z, h, deg)) while it flies or falls

    def place(self, t):
        if self.path is not None:
            t0, t1, fn = self.path
            if t0 <= t <= t1:
                return fn((t - t0) / max(1e-6, t1 - t0))
            if t > t1:
                x, z, h, deg = fn(1.0)
                self.x, self.z, self.h, self.deg = x, z, h, deg
                self.path = None
        return self.x, self.z, self.h, getattr(self, 'deg', 0.0)

    def key(self, t):
        return self.place(t)[1]

    def lights(self, t):
        return []

    def draw(self, cv, off, t):
        pass


def blit_sprite(cv, arr, anchor, x, y, deg=0.0, alpha=1.0):
    if deg:
        arr, anchor = rotate(arr, deg, anchor)
    px.blit(cv, arr, int(round(x - anchor[0])), int(round(y - anchor[1])), alpha)


# ------------------------------------------------------------------ cars
def car_sprite(color, state='ok', taxi=False, seed=0):
    Wd, Ht = 64, 36
    im = Image.new('RGBA', (Wd, Ht), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(seed)
    body, dark, light = C(color), C(color, 0.55), C(color, 1.3)
    glass, glass_hi = C('22304a'), C('5a7aa0')
    crushed = state in ('wreck', 'flipped')
    jit = (lambda: int(rng.integers(-1, 2))) if crushed else (lambda: 0)
    d.polygon([(4, 18 + jit()), (58, 18 + jit()), (60, 21), (60, 27), (57, 29), (6, 29), (3, 27), (3, 21)], fill=body)
    d.polygon([(4, 18), (58, 18), (59, 20), (4, 20)], fill=light)
    d.polygon([(45, 15 if not crushed else 16), (59, 15), (60, 18), (45, 18)], fill=light)       # the hood from above
    d.polygon([(3, 16), (15, 16), (15, 18), (3, 18)], fill=light)                                 # the trunk
    roof = {'ok': 5, 'dent': 8, 'wreck': 12, 'flipped': 12}[state]
    dip = 2 if state == 'dent' else 0
    d.polygon([(18, roof + 3), (28, roof + 3 + dip), (40, roof + 3), (47, 16), (14, 16)], fill=glass)
    d.polygon([(20, roof), (28, roof + dip), (39, roof), (40, roof + 3), (28, roof + 3 + dip), (18, roof + 3)], fill=light)
    d.line([(28, roof + 3 + dip), (29, 16)], fill=body, width=2)
    d.line([(18, roof + 3), (14, 16)], fill=body)
    d.line([(40, roof + 3), (47, 16)], fill=body)
    if state == 'ok':
        d.line([(20, roof + 5), (25, roof + 5)], fill=glass_hi)
        d.line([(31, roof + 5), (38, roof + 5)], fill=glass_hi)
    elif state == 'dent':
        for _ in range(7):
            x0 = rng.uniform(16, 44)
            y0 = rng.uniform(roof + 3, 15)
            d.line([(x0, y0), (x0 + rng.uniform(-4, 4), y0 + rng.uniform(-3, 3))], fill=C('d8e8ff'))
    else:
        d.polygon([(18, roof + 3), (40, roof + 3), (46, 16), (15, 16)], fill=C('0a0a10'))
        for _ in range(6):
            x0 = rng.uniform(16, 44)
            d.point([(x0, 16 - rng.integers(0, 2))], fill=C('c8e0ff'))
    d.line([(30, 18), (30, 28)], fill=dark)
    d.line([(32, 21), (34, 21)], fill=light)
    d.line([(5, 21), (58, 21)], fill=C(color, 0.85))
    d.rectangle([58, 20, 59, 22], fill=C('fff2c0'))
    d.rectangle([3, 20, 4, 22], fill=C('ff2a2a'))
    d.rectangle([59, 24, 61, 26], fill=C('8a8a94'))
    d.rectangle([2, 24, 4, 26], fill=C('8a8a94'))
    if taxi:
        if state == 'ok':
            d.rectangle([25, roof - 3, 32, roof - 1], fill=C('fff080'))
        for xx in range(6, 57, 2):
            d.point([(xx, 23)], fill=C('101010') if (xx // 2) % 2 else C('f0f0f0'))
    for wx in (15, 48):
        dy = jit() if crushed else 0
        d.ellipse([wx - 6, 22, wx + 6, 34], fill=C('141418'))
        d.ellipse([wx - 4, 26 + dy, wx + 4, 34 + dy], fill=C('0c0c0e'))
        d.ellipse([wx - 2, 28 + dy, wx + 1, 31 + dy], fill=C('8a8a90'))
    arr = rgba_of(im)
    if state != 'ok':
        n = hash_noise(*np.meshgrid(np.arange(Wd), np.arange(Ht)), seed + 5)
        dents = (arr[..., 3] > 0) & (n > (0.86 if state == 'dent' else 0.72))
        arr[dents, :3] *= 0.6
    if crushed:
        sc = (arr[..., 3] > 0) & (hash_noise(*np.meshgrid(np.arange(Wd) // 3, np.arange(Ht) // 3), seed + 9) > 0.7)
        arr[sc, :3] *= 0.5
    arr = outline(arr)
    if state == 'flipped':
        arr = arr[::-1].copy()
    return arr


class Car(Prop):
    def __init__(self, x, z, color, face=1, taxi=False, seed=0):
        self.x, self.z, self.h, self.deg = x, z, 0.0, 0.0
        self.color, self.face, self.taxi, self.seed = color, face, taxi, seed
        self.state = 'ok'
        self.cache = {}
        self.alarm_t = None
        self.hop = None          # (t0, height): a bounce on the spot
        self.burn_t = None

    def sprite(self):
        if self.state not in self.cache:
            a = car_sprite(self.color, self.state, self.taxi, self.seed)
            if self.face < 0:
                a = a[:, ::-1].copy()
            self.cache[self.state] = a
        return self.cache[self.state]

    def place(self, t):
        x, z, h, deg = Prop.place(self, t)
        if self.hop is not None and self.path is None:
            t0, hh = self.hop
            u = (t - t0) / 0.45
            if 0 <= u <= 1:
                h = h + hh * 4 * u * (1 - u)
                deg = deg + 6 * math.sin(u * math.pi) * self.face
        return x, z, h, deg

    def draw(self, cv, off, t):
        if self.hidden:
            return
        x, z, h, deg = self.place(t)
        arr = self.sprite()
        hh, ww = arr.shape[:2]
        sx, sy = x + off[0], floor_y(z) - h + off[1]
        if deg:
            blit_sprite(cv, arr, (ww / 2, hh / 2), sx, sy - hh / 2 + 2, deg)
        else:
            px.blit(cv, arr, int(round(sx - ww / 2)), int(round(sy - hh + 2)))
        if self.alarm_t is not None and t >= self.alarm_t and int((t - self.alarm_t) * 3) % 2 == 0 and self.path is None:
            for ex in (-27, 27):
                xx, yy = int(round(sx + ex)), int(round(sy - 12))
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    cv[yy - 1:yy + 1, max(0, xx - 1):xx + 1] = hexc('ffb020')

    def lights(self, t):
        out = []
        if self.alarm_t is not None and t >= self.alarm_t and int((t - self.alarm_t) * 3) % 2 == 0 and self.path is None:
            x, z, h, _ = self.place(t)
            out.append((x, floor_y(z) - 12, 30, hexc('ff9a20'), 0.45))
        if self.burn_t is not None and t >= self.burn_t:
            x, z, h, _ = self.place(t)
            fl = 0.8 + 0.2 * math.sin(t * 13 + self.x)
            out.append((x, floor_y(z) - 10, 60, hexc('ff7a2a'), 0.9 * fl))
        return out


# ------------------------------------------------------------------ lamp posts and the traffic light
def lamp_sprite(kind='lamp', face=1, broken=False):
    Wd, Ht = 40, 84
    im = Image.new('RGBA', (Wd, Ht), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bx = 10
    pole, hi = C('1c2a24'), C('4a6a5a')
    d.rectangle([bx - 3, Ht - 6, bx + 3, Ht - 1], fill=pole)
    d.rectangle([bx - 2, Ht - 8, bx + 2, Ht - 6], fill=pole)
    d.rectangle([bx - 1, 12, bx + 1, Ht - 8], fill=pole)
    d.line([(bx - 1, 12), (bx - 1, Ht - 8)], fill=hi)
    if kind == 'lamp':
        d.line([(bx, 12), (bx + 3, 8), (bx + 12, 6), (bx + 18, 7)], fill=pole, width=2)
        d.polygon([(bx + 15, 7), (bx + 23, 7), (bx + 25, 11), (bx + 13, 11)], fill=pole)
        if not broken:
            d.rectangle([bx + 16, 11, bx + 22, 12], fill=C('fff2c8'))
    else:
        d.line([(bx, 14), (bx + 26, 14)], fill=pole, width=2)
        d.rectangle([bx + 20, 15, bx + 26, 33], fill=C('16181c'))
        d.rectangle([bx - 4, 30, bx - 1, 40], fill=C('16181c'))
    arr = outline(rgba_of(im))
    if face < 0:
        arr = arr[:, ::-1].copy()
        return arr, (Wd - 1 - bx, Ht - 1)
    return arr, (bx, Ht - 1)


class Lamp(Prop):
    """A street lamp or a traffic light; it can be bent, or knocked down to lie across the road."""

    def __init__(self, x, z, face=1, kind='lamp'):
        self.x, self.z, self.h, self.face, self.kind = x, z, 0.0, face, kind
        self.fall_t = None
        self.fall_dir = 1
        self.fall_to = 86.0
        self.bend_t = None
        self.bend_deg = 0.0
        self.on = True
        self.off_t = None
        self.flicker_t = None
        self.spr = {}

    def sprite(self, broken):
        if broken not in self.spr:
            self.spr[broken] = lamp_sprite(self.kind, self.face, broken)
        return self.spr[broken]

    def angle(self, t):
        if self.fall_t is not None and t >= self.fall_t:
            u = min(1.0, (t - self.fall_t) / 0.5)
            a = self.fall_to * u * u
            if u >= 1.0:
                b = (t - self.fall_t - 0.5)
                a = self.fall_to - 6 * math.exp(-b * 9) * abs(math.sin(b * 25))
            return a * self.fall_dir
        if self.bend_t is not None and t >= self.bend_t:
            u = min(1.0, (t - self.bend_t) / 0.15)
            return self.bend_deg * u * (1 + 0.15 * math.sin(min(1.0, (t - self.bend_t) / 0.6) * 9) * (1 - u))
        return 0.0

    def lit(self, t):
        if not self.on or (self.off_t is not None and t >= self.off_t):
            return 0.0
        if self.flicker_t is not None and t >= self.flicker_t:
            return 1.0 if int(t * 17) % 3 else 0.0
        return 1.0

    def head(self, t):
        """The lamp's bulb (stage coords)."""
        a = math.radians(self.angle(t))
        lx, ly = (16 * self.face, -72) if self.kind == 'lamp' else (23 * self.face, -60)
        rx = lx * math.cos(a) - ly * math.sin(a)
        ry = lx * math.sin(a) + ly * math.cos(a)
        return self.x + rx, floor_y(self.z) - self.h + ry

    def draw(self, cv, off, t):
        if self.hidden:
            return
        broken = self.fall_t is not None and t >= self.fall_t + 0.5
        arr, anc = self.sprite(broken)
        blit_sprite(cv, arr, anc, self.x + off[0], floor_y(self.z) - self.h + off[1], self.angle(t))
        L = self.lit(t)
        hx, hy = self.head(t)
        if self.kind == 'lamp' and L > 0:
            px.glow(cv, hx + off[0], hy + off[1] + 1, 7, hexc('fff0c0'), 0.5 * L)
        if self.kind == 'traffic':
            k = int(t / 2.2) % 3 if L > 0 else -1
            a = math.radians(self.angle(t))
            for i, c in enumerate(('ff3020', 'ffc020', '30ff70')):
                lx, ly = 23 * self.face, -64 + 5 * i
                xx = self.x + lx * math.cos(a) - ly * math.sin(a) + off[0]
                yy = floor_y(self.z) - self.h + lx * math.sin(a) + ly * math.cos(a) + off[1]
                col = hexc(c) if (2 - i) == k else hexc(c) * 0.2
                xi, yi = int(round(xx)), int(round(yy))
                if 1 <= xi < cv.shape[1] - 1 and 1 <= yi < cv.shape[0] - 1:
                    cv[yi - 1:yi + 1, xi - 1:xi + 1] = col

    def lights(self, t):
        L = self.lit(t)
        if L <= 0 or self.kind != 'lamp':
            return []
        hx, hy = self.head(t)
        if self.fall_t is not None and t >= self.fall_t:
            return []
        return [(hx, floor_y(self.z) - self.h + 2, 52, hexc('ffd890'), 0.55 * L, 0.55), (hx, hy, 12, hexc('fff0c0'), 0.6 * L, 1.0)]


# ------------------------------------------------------------------ small street things
class Hydrant(Prop):
    def __init__(self, x, z):
        self.x, self.z, self.h = x, z, 0.0
        self.broken_t = None

    def draw(self, cv, off, t):
        x, y = int(round(self.x + off[0])), int(round(floor_y(self.z) + off[1]))
        top = 3 if (self.broken_t is not None and t >= self.broken_t) else 9
        for dy in range(top):
            half = 2 if dy < 7 else 1
            for dx in range(-half, half + 1):
                xx, yy = x + dx, y - dy
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    cv[yy, xx] = hexc('c81e1a') if dx < 1 else hexc('8a1210')
        if top == 9:
            for dx in (-3, 3):
                if 0 <= x + dx < cv.shape[1] and 0 <= y - 5 < cv.shape[0]:
                    cv[y - 5, x + dx] = hexc('e8c040')


class Table(Prop):
    """A diner table (a round white top on a chrome stem); thrown, it spins and breaks."""

    def __init__(self, x, z, seed=0):
        self.x, self.z, self.h, self.deg = x, z, 0.0, 0.0
        im = Image.new('RGBA', (22, 18), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse([3, 14, 15, 17], fill=C('8a8a94'))
        d.rectangle([8, 5, 10, 15], fill=C('b8b8c4'))
        d.ellipse([1, 1, 19, 7], fill=C('dcd8cc'))
        d.ellipse([1, 4, 19, 8], fill=C('8a2a20'))
        d.ellipse([1, 1, 19, 6], fill=C('e8e4d8'))
        self.arr = outline(rgba_of(im))
        self.anchor = (10, 17)

    def draw(self, cv, off, t):
        if self.hidden:
            return
        x, z, h, deg = self.place(t)
        blit_sprite(cv, self.arr, self.anchor, x + off[0], floor_y(z) - h + off[1], deg)


class Stool(Prop):
    def __init__(self, x, z):
        self.x, self.z, self.h = x, z, 0.0

    def draw(self, cv, off, t):
        if self.hidden:
            return
        x, y = int(round(self.x + off[0])), int(round(floor_y(self.z) + off[1]))
        h_, w_ = cv.shape[:2]
        for dy in range(10):
            yy = y - dy
            if 0 <= yy < h_:
                if dy >= 8:
                    for dx in range(-3, 4):
                        if 0 <= x + dx < w_:
                            cv[yy, x + dx] = hexc('c02a24') if dy == 9 else hexc('6a1410')
                elif 0 <= x < w_:
                    cv[yy, x] = hexc('b8b8c4')


class Counter(Prop):
    """The diner's long counter; smashed, it breaks in two with a gap of splinters."""

    def __init__(self, x0, x1, z0, z1, ht=16):
        self.x0, self.x1, self.z0, self.z1, self.ht = x0, x1, z0, z1, ht
        self.x, self.z, self.h = (x0 + x1) / 2, z1, 0.0
        self.breaks = []
        self.arr = None

    def bake(self):
        w = self.x1 - self.x0
        top_rows = int(round((self.z1 - self.z0) * FS))
        Ht = self.ht + top_rows + 2
        arr = np.zeros((Ht, w, 4), np.float32)
        arr[:top_rows, :, :3] = hexc('d8d4c8')
        arr[:top_rows, :, :3] *= np.linspace(0.85, 1.0, top_rows)[:, None, None]
        arr[top_rows - 1, :, :3] = hexc('c8c8d4')
        arr[top_rows:top_rows + 2, :, :3] = hexc('d0d0dc')
        arr[top_rows + 2:Ht - 2, :, :3] = hexc('a8201c')
        for xx in range(0, w, 6):
            arr[top_rows + 2:Ht - 2, xx, :3] = hexc('7a1410')
        arr[Ht - 2:, :, :3] = hexc('3a3a44')
        arr[..., 3] = 1.0
        for (bx, bw) in self.breaks:
            for yy in range(Ht):
                gap = bw * (0.3 + 0.7 * (1 - yy / Ht))
                a0, a1 = int(bx - self.x0 - gap), int(bx - self.x0 + gap)
                arr[yy, max(0, a0):max(0, a1), 3] = 0
                for e in (a0 - 1, a1):
                    if 0 <= e < w:
                        arr[yy, e, :3] = hexc('5a3a20')
        self.arr = outline(arr)
        self.top_rows = top_rows

    def draw(self, cv, off, t):
        if self.arr is None:
            self.bake()
        Ht = self.arr.shape[0]
        y = floor_y(self.z1) + off[1] - Ht + 2
        px.blit(cv, self.arr, int(round(self.x0 + off[0])), int(round(y)))


class Pendant(Prop):
    """A diner lamp hanging on its cord from the ceiling; a shock sets it swinging, a hit brings it down."""

    def __init__(self, x, z, h, top_row):
        self.x, self.z, self.h, self.top_row = x, z, h, top_row
        self.swings = []
        self.drop_t = None
        self.broken = False
        self.on = True

    def swing(self, t):
        a = 0.0
        for (t0, amp) in self.swings:
            if t >= t0:
                a += amp * math.exp(-(t - t0) * 0.9) * math.sin((t - t0) * 5.0)
        return a

    def pos(self, t):
        sy = floor_y(self.z) - self.h
        L = sy - self.top_row
        a = self.swing(t)
        x = self.x + L * math.sin(a)
        y = self.top_row + L * math.cos(a)
        if self.drop_t is not None and t >= self.drop_t:
            u = t - self.drop_t
            fall = min(self.h, 0.5 * 420 * u * u)
            return x, y + fall, fall >= self.h
        return x, y, False

    def draw(self, cv, off, t):
        x, y, down = self.pos(t)
        if self.drop_t is None or t < self.drop_t:
            for p in px.line_points(self.x + off[0], self.top_row + off[1], x + off[0], y + off[1] - 3):
                if 0 <= p[0] < cv.shape[1] and 0 <= p[1] < cv.shape[0]:
                    cv[p[1], p[0]] = hexc('101012')
        xi, yi = int(round(x + off[0])), int(round(y + off[1]))
        shade = hexc('1e4a3a') if not down else hexc('163428')
        for dy in range(-3, 1):
            half = 2 + (dy + 3)
            for dx in range(-half, half + 1):
                xx, yy = xi + dx, yi + dy
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    cv[yy, xx] = shade if abs(dx) < half else hexc('0a1a14')
        if self.lit(t) > 0 and 0 <= yi + 1 < cv.shape[0]:
            cv[yi + 1, max(0, xi - 2):xi + 3] = hexc('fff2c0')

    def lit(self, t):
        return 0.0 if (not self.on or (self.drop_t is not None and t >= self.drop_t)) else 1.0

    def lights(self, t):
        if self.lit(t) <= 0:
            return []
        x, y, _ = self.pos(t)
        return [(x, floor_y(self.z) + 2, 46, hexc('ffc070'), 0.5, 0.55), (x, y + 2, 10, hexc('fff0c0'), 0.5, 1.0)]


# ------------------------------------------------------------------ columns
class Column(Prop):
    """A round concrete column of the diner (v3's pillar redrawn in concrete); it keeps every crater."""
    R, HT = 7, 86

    def __init__(self, x, z, seed, base_h=0.0):
        self.x, self.z, self.seed, self.h = x, z, seed, base_h
        self.hits = []
        self.img = None

    def hit(self, hh, power, seed, out):
        self.hits.append((hh, power, seed, out))
        self.img = None

    def bake(self):
        R, HT = self.R, self.HT
        w = 2 * (R + 7) + 1
        h = HT + 8
        c0 = R + 7
        base = h - 4
        ys, xs = np.mgrid[0:h, 0:w]
        dx = xs + 0.5 - c0
        hy = base - ys[:, :1]
        rr = np.full((h, 1), float(R))
        rr = np.where(hy < 5, R + 2.0, rr)
        rr = np.where(hy > HT - 5, R + 2.0, rr)
        inside = (np.abs(dx) <= rr) & (hy >= -1) & (hy < HT)
        s = np.clip(dx / rr, -1, 1)
        lam = -0.55 * s + 0.62 * np.sqrt(1 - s * s)
        n = hash_noise(xs, ys, self.seed)
        rgb = hexc('7a7680')[None, None, :] * (0.4 + 0.8 * lam[..., None]) * (0.9 + 0.16 * n[..., None])
        edge = inside & (np.abs(np.abs(dx) - rr) < 1.0)
        rgb[edge] *= 0.35
        top = inside & (hy >= HT - 2)
        rgb[top] = hexc('9a96a4')
        a = inside.astype(np.float32)
        for (hh, power, sd, out) in self.hits:
            rng = np.random.default_rng(sd)
            cy = base - hh
            cx = c0 + out * R * 0.5
            depth = 2.0 + power
            span = 4 + power * 2.5
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

    def draw(self, cv, off, t):
        if self.img is None:
            self.bake()
        ax, ay = self.anchor
        px.blit(cv, self.img, int(round(self.x + off[0] - ax)), int(round(floor_y(self.z) - self.h + off[1] - ay)))


class Beam(Prop):
    """A blue steel column of the subway platform; hit hard, it buckles."""

    def __init__(self, x, z, base_h, top_row):
        self.x, self.z, self.h, self.top_row = x, z, base_h, top_row
        self.bend_t = None
        self.bend = 0.0
        self.bend_dir = 1

    def draw(self, cv, off, t):
        yb = floor_y(self.z) - self.h + off[1]
        yt = self.top_row + off[1]
        b = 0.0
        if self.bend_t is not None and t >= self.bend_t:
            b = self.bend * min(1.0, (t - self.bend_t) / 0.12) * self.bend_dir
        hgt = yb - yt
        h_, w_ = cv.shape[:2]
        for yy in range(int(yt), int(yb) + 1):
            k = (yy - yt) / max(1, hgt)
            kink = b * (1 - abs(k - 0.5) * 2) if b else 0.0
            x0 = int(round(self.x + off[0] + kink))
            if not 0 <= yy < h_:
                continue
            for dx in range(-3, 4):
                xx = x0 + dx
                if 0 <= xx < w_:
                    c = hexc('3a5a9a') if abs(dx) == 3 else hexc('20386a') if abs(dx) == 2 else hexc('2a4a86')
                    if dx == -3:
                        c = hexc('6a8ac8')
                    if abs(dx) <= 1 and (yy % 7) == 0:
                        c = hexc('8aa0d0')
                    cv[yy, xx] = c
            for xx in (x0 - 4, x0 + 4):
                if 0 <= xx < w_:
                    cv[yy, xx] = OUTLINE


class Fluo(Prop):
    """A fluorescent fitting hanging in the subway on two chains; it can fall."""

    def __init__(self, x, z, h, top_row):
        self.x, self.z, self.h, self.top_row = x, z, h, top_row
        self.drop_t = None
        self.flicker_t = None

    def pos(self, t):
        y = floor_y(self.z) - self.h
        if self.drop_t is not None and t >= self.drop_t:
            u = t - self.drop_t
            fall = min(self.h + DS, 0.5 * 420 * u * u) if self.h < 0 else min(self.h, 0.5 * 420 * u * u)
            fall = min(fall, (floor_y(self.z) + DS) - y)
            return y + fall, fall >= (floor_y(self.z) + DS) - y - 0.5
        return y, False

    def lit(self, t):
        if self.drop_t is not None and t >= self.drop_t:
            return 0.0
        if self.flicker_t is not None and t >= self.flicker_t:
            return 1.0 if int(t * 13) % 4 else 0.2
        return 1.0

    def draw(self, cv, off, t):
        y, down = self.pos(t)
        x = self.x + off[0]
        yy = int(round(y + off[1]))
        h_, w_ = cv.shape[:2]
        if self.drop_t is None or t < self.drop_t:
            for cx in (x - 7, x + 7):
                for r in range(int(self.top_row + off[1]), yy - 1):
                    if 0 <= r < h_ and 0 <= int(cx) < w_:
                        cv[r, int(cx)] = hexc('2a2a30')
        L = self.lit(t)
        for dy in range(3):
            r = yy + dy
            if not 0 <= r < h_:
                continue
            for dx in range(-10, 11):
                xx = int(round(x + dx))
                if 0 <= xx < w_:
                    if dy == 2 and abs(dx) < 9:
                        cv[r, xx] = hexc('f0fff8') * (0.35 + 0.65 * L) if not down else hexc('6a7a78')
                    else:
                        cv[r, xx] = hexc('4a4e56')

    def lights(self, t):
        L = self.lit(t)
        if L <= 0:
            return []
        y, _ = self.pos(t)
        return [(self.x, y + 3, 16, hexc('e8fff8'), 0.6 * L, 0.6)]


# ------------------------------------------------------------------ the diner's slanted side walls
class CrossWall:
    """A side wall of the diner running into the depth on a slant, seen from slightly above: its lit face, its top and
    its cut front end. Breaches knock holes through it (in depth z and height h); the pixels behind show through."""

    def __init__(self, xb, style, seed):
        self.xb, self.style, self.seed = xb, style, seed
        self.breaches = []
        self.arr = None
        self.x0 = int(math.floor(xb)) - 2
        self.x1 = int(math.ceil(xb + SLOPE + WALL_T)) + 3
        self.r0 = int(floor_y(0) - WALL_H) - 2
        self.r1 = int(floor_y(ZM)) + 2

    def breach(self, zc, hc, rz, rh, seed):
        self.breaches.append((zc, hc, rz, rh, seed))
        self.arr = None

    def holed(self, z, h):
        z = np.asarray(z, np.float32)
        h = np.asarray(h, np.float32)
        out = np.zeros(np.broadcast(z, h).shape, bool)
        for (zc, hc, rz, rh, sd) in self.breaches:
            n = hash_noise((z * 0.5).astype(int), (h * 0.5).astype(int), sd)
            out |= ((z - zc) / rz) ** 2 + ((h - hc) / rh) ** 2 + (n - 0.5) * 0.5 < 1.0
        return out

    def bake(self):
        w, hg = self.x1 - self.x0, self.r1 - self.r0
        rgb = np.zeros((hg, w, 3), np.float32)
        a = np.zeros((hg, w), np.float32)
        hs = np.arange(0, WALL_H + 1, dtype=np.float32)
        if self.style == 'brick':
            base, top_c, end_c = hexc('6a3428'), hexc('8a7a70'), hexc('5a3a30')
        else:
            base, top_c, end_c = hexc('8a8468'), hexc('9a948a'), hexc('6a6458')
        for z in np.arange(0.0, ZM + 0.01, 0.25):
            col = int(round(wall_x(self.xb, z))) - self.x0
            fy = floor_y(z) - self.r0
            rows = np.round(fy - hs).astype(int)
            keep = ~self.holed(np.full_like(hs, z), hs)
            n = hash_noise(np.full(hs.shape, int(z)), hs.astype(int), self.seed)
            c = np.tile(base, (len(hs), 1)) * (0.88 + 0.14 * n[:, None])
            if self.style == 'brick':
                mortar = ((hs.astype(int) % 3) == 0) | (((int(z * 0.5) + (hs.astype(int) // 3) * 2) % 5) == 0)
                c[mortar] *= 0.72
                c *= (0.85 + 0.25 * (hs / WALL_H))[:, None]
            else:
                low = hs < 22
                c[low] = hexc('5a2016') * (0.9 + 0.1 * n[low, None])
                c[(hs >= 22) & (hs < 23.5)] = hexc('d0d0d8')
                c[(hs.astype(int) % 6 == 0) & ~low] *= 0.85
            ok = keep & (rows >= 0) & (rows < hg)
            rgb[rows[ok], col] = c[ok]
            a[rows[ok], col] = 1.0
            tr = int(round(fy - WALL_H))
            if not self.holed(z, WALL_H) and 0 <= tr < hg:
                rgb[tr, col:col + WALL_T] = top_c * (0.9 + 0.1 * (z % 2))
                a[tr, col:col + WALL_T] = 1.0
        # the cut front end
        col = int(round(wall_x(self.xb, ZM))) - self.x0
        fy = floor_y(ZM) - self.r0
        for hh in range(WALL_H + 1):
            if self.holed(ZM, hh):
                continue
            r = int(round(fy - hh))
            if 0 <= r < hg:
                rgb[r, col:col + WALL_T] = end_c * (0.8 if hh % 3 == 0 else 1.0)
                a[r, col:col + WALL_T] = 1.0
        # broken edges around the holes: exposed brick, darker
        if self.breaches:
            m = a > 0
            ring = binary_dilation(~m, iterations=2) & m
            yy, xx = np.nonzero(ring)
            n = hash_noise(xx, yy, self.seed + 3)
            rgb[yy, xx] = np.where((n > 0.5)[:, None], hexc('8a3a2a')[None, :], hexc('2a1a16')[None, :])
        arr = np.concatenate([rgb, a[..., None]], axis=2)
        self.arr = outline(arr)

    def slices(self):
        """Column strips with their depth, for sorting with the fighters."""
        if self.arr is None:
            self.bake()
        out = []
        w = self.arr.shape[1]
        for c in range(0, w, 2):
            x = self.x0 + c + 1
            z = max(0.0, min(ZM, (x - self.xb) * ZM / SLOPE))
            out.append((z, c))
        return out

    def draw_slice(self, cv, off, c):
        strip = self.arr[:, c:c + 2]
        px.blit(cv, strip, int(round(self.x0 + c + off[0])), int(round(self.r0 + off[1])))


class Chunk(Prop):
    """A slab of road (or wall) torn out and thrown."""

    def __init__(self, x, z, w=30, h=12, seed=0, kind='asphalt'):
        self.x, self.z, self.h, self.deg = x, z, 0.0, 0.0
        self.hidden = True
        rng = np.random.default_rng(seed)
        im = Image.new('RGBA', (w + 4, h + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        pts = []
        for k in range(12):
            a = k / 12 * 2 * math.pi
            pts.append((w / 2 + 2 + math.cos(a) * w / 2 * rng.uniform(0.75, 1.0), h / 2 + 2 + math.sin(a) * h / 2 * rng.uniform(0.7, 1.0)))
        top, side = ('3a3a42', '22222a') if kind == 'asphalt' else ('8a3a2a', '5a2418')
        d.polygon(pts, fill=C(side))
        d.polygon([(px_, py_ - 3) for (px_, py_) in pts if py_ < h / 2 + 3] + [(w + 2, h / 2), (2, h / 2)], fill=C(top))
        if kind == 'asphalt':
            d.line([(6, h / 2 - 1), (w - 4, h / 2 - 2)], fill=C('b8962a'))
        arr = rgba_of(im)
        n = hash_noise(*np.meshgrid(np.arange(w + 4), np.arange(h + 4)), seed)
        arr[(arr[..., 3] > 0) & (n > 0.8), :3] *= 0.7
        self.arr = outline(arr)
        self.anchor = ((w + 4) / 2, h + 2)

    def draw(self, cv, off, t):
        if self.hidden:
            return
        x, z, h, deg = self.place(t)
        blit_sprite(cv, self.arr, self.anchor, x + off[0], floor_y(z) - h + off[1], deg)
