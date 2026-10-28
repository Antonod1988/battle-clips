"""The city block (from the HULK vs CAPTAIN AMERICA clip): at night, seen from slightly above (VERSUS-1 v3's three-quarter view), long
enough for the fight to travel through it: MAIN ST (shops), the diner's ground floor (a cutaway: its front wall is gone,
a slanted side wall on each end), 5TH AVE (bank, cinema, the subway entrance) and under it the 5TH AVE subway station.

World: x along the screen, z into depth (0 at the facades), y height above the street. The projection is v3's:
stage_x = x, stage_y = Y0 + z * FS - y. The subway floor lies DS below the street; the soil between them is drawn as a
cutaway band in the front layer (with the water main and the cables), cut open when the road caves in.

For the choreographer: AREAS (MAIN ST, the diner, 5TH AVE), ROUTES (thrown through the diner's side walls), targets()
(cars, lamps, the traffic light, the hydrant, tables, the counter, columns, shop windows) and the stage protocol.
The subway is drawn but no route reaches it yet.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_dilation

import px
from px import hexc, hash_noise, fbm
from proj import set_projection

W, H = px.W, px.H
M = 8                               # canvas margin (shake)
EW, EH = W + 2 * M, H + 2 * M
SW, SH = 1960, 560                  # the stage (stage coords)
PAD = 12
Y0, FS = 118.0, 0.6
ZM = 215.0                          # the front edge of the street
DS = 230.0                          # the subway floor, below the street
WALK1, ROAD0, ROAD1 = 28.0, 30.0, 186.0
WEST, EAST, SLOPE, WALL_T, WALL_H = 630.0, 1104.0, 64.0, 12, 90
SUB = (1140.0, 1960.0)
PLAT, PLAT_H = 60.0, 10.0           # the subway platform's edge (z) and height
SOIL0, SOIL1 = 247, 272             # the cutaway soil band rows; the tunnel's ceiling slab below it
CEIL1 = 284
SUB_Y0 = Y0 + DS                    # the subway floor's back edge row
CAM_X = (0.0, float(SW - W))
CAM_Y = {'street': (0.0, 22.0), 'subway': (238.0, 276.0)}


def floor_y(z, h=0.0):
    return Y0 + z * FS - h


def wall_x(xb, z):
    return xb + SLOPE * z / ZM


def area(x, z):
    """Which part of the block a floor point is in."""
    if x < wall_x(WEST, z):
        return 'main'
    if x < wall_x(WEST, z) + WALL_T:
        return 'wall'
    if x < wall_x(EAST, z):
        return 'diner'
    if x < wall_x(EAST, z) + WALL_T:
        return 'wall'
    return 'fifth'


def col255(h, a=255):
    c = hexc(h)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def mix_arr(a, b, k):
    k = np.clip(np.asarray(k, np.float32), 0, 1)
    return a + (b - a) * k[..., None]


# ------------------------------------------------------------------ a 3x5 pixel font for signs
FONT = {
    'A': '010101111101101', 'B': '110101110101110', 'C': '011100100100011', 'D': '110101101101110',
    'E': '111100110100111', 'F': '111100110100100', 'G': '011100101101011', 'H': '101101111101101',
    'I': '111010010010111', 'J': '001001001101010', 'K': '101101110101101', 'L': '100100100100111',
    'M': '101111111101101', 'N': '110101101101101', 'O': '010101101101010', 'P': '110101110100100',
    'Q': '010101101110011', 'R': '110101110101101', 'S': '011100010001110', 'T': '111010010010010',
    'U': '101101101101111', 'V': '101101101101010', 'W': '101101111111101', 'X': '101101010101101',
    'Y': '101101010010010', 'Z': '111001010100111', '0': '111101101101111', '1': '010110010010111',
    '2': '110001010100111', '3': '110001010001110', '4': '101101111001001', '5': '111100110001110',
    '6': '011100111101111', '7': '111001010010010', '8': '111101111101111', '9': '111101111001110',
    ' ': '000000000000000', '.': '000000000000010', '!': '010010010000010', '-': '000000111000000',
}


def text_mask(text, scale=1):
    w = len(text) * 4 - 1
    m = np.zeros((5, max(1, w)), bool)
    for i, ch in enumerate(text.upper()):
        g = FONT.get(ch, FONT[' '])
        for r in range(5):
            for c in range(3):
                if g[r * 3 + c] == '1':
                    m[r, i * 4 + c] = True
    if scale > 1:
        m = np.kron(m, np.ones((scale, scale), bool))
    return m


def put_text(img, x, y, text, color, scale=1, glow=0.0, glow_c=None, vertical=False):
    """Writes pixel text into an RGB float image at stage coords (x, y) = top left (PAD applied by the caller)."""
    if vertical:
        for i, ch in enumerate(text):
            put_text(img, x, y + i * 6 * scale, ch, color, scale, glow, glow_c)
        return
    m = text_mask(text, scale)
    h, w = m.shape
    x, y = int(x), int(y)
    if glow > 0:
        g = np.zeros((h + 12, w + 12), np.float32)
        g[6:6 + h, 6:6 + w] = m
        g = gaussian_filter(g, 2.2) * glow
        gc = hexc(glow_c) if isinstance(glow_c, str) else (color if glow_c is None else glow_c)
        sub = img[y - 6:y - 6 + h + 12, x - 6:x - 6 + w + 12]
        sub += g[:sub.shape[0], :sub.shape[1], None] * np.asarray(gc, np.float32)
    sub = img[y:y + h, x:x + w]
    mm = m[:sub.shape[0], :sub.shape[1]]
    sub[mm] = color


class Stage:
    def __init__(self):
        h, w = SH + 2 * PAD, SW + 2 * PAD
        self.img = np.zeros((h, w, 3), np.float32)
        self.front = np.zeros((h, w, 4), np.float32)
        self.build_sky()
        self.build_main()
        self.build_diner()
        self.build_fifth()
        self.build_floor()
        self.build_subway()
        self.build_front()
        self.holes = []                 # (x0, x1, z0, z1): the road has caved in there

    # ------------------------------------------------------------ the sky and the skyline
    def build_sky(self):
        h, w = self.img.shape[:2]
        Y = (np.arange(h, dtype=np.float32) - PAD)[:, None]
        k = np.clip((Y + 12) / 130.0, 0, 1)
        sky = mix_arr(hexc('04050e'), hexc('1a1632'), k ** 1.3)
        sky = sky + hexc('3a1e34') * (np.clip((Y - 30) / 90.0, 0, 1) ** 2 * 0.7)[..., None]
        sky = np.repeat(sky, w, axis=1)
        X, YY = np.meshgrid(np.arange(w), np.arange(h))
        n = hash_noise(X, YY, seed=5)
        stars = (n > 0.9975) & (YY < 70)
        sky[stars] = hexc('b8c0e8') * (0.4 + 0.6 * hash_noise(X, YY, seed=6)[stars, None])
        mx, my, mr = 330 + PAD, 14 + PAD, 9.0
        d = np.sqrt((X + 0.5 - mx) ** 2 + (YY + 0.5 - my) ** 2)
        sky += (np.clip(1 - (d - mr) / 40.0, 0, 1) ** 2.2)[..., None] * hexc('4a4a7a') * 0.5
        moon = hexc('e8e4f4') * (0.82 + 0.2 * fbm(w, h, 3, seed=21)[..., None])
        sky[d <= mr] = moon[d <= mr]
        # two rows of far towers with a few lit windows
        rng = np.random.default_rng(3)
        for layer, (top_lo, top_hi, colr, win) in enumerate(((18, 60, '121428', 0.05), (34, 76, '0c0d1c', 0.08))):
            x = -PAD
            while x < w:
                bw = int(rng.uniform(14, 44))
                top = int(rng.uniform(top_lo, top_hi))
                x0, x1 = x + PAD, min(w, x + bw + PAD)
                sky[top + PAD:Y0.__int__() + PAD, x0:x1] = hexc(colr)
                if rng.uniform() < 0.3:
                    sky[top + PAD - 6:top + PAD, x0 + bw // 2:x0 + bw // 2 + 1] = hexc(colr)
                for yy in range(top + 3, int(Y0), 4):
                    for xx in range(x0 + 2, x1 - 2, 3):
                        if rng.uniform() < win:
                            sky[yy + PAD, xx] = hexc('7a6a3a') * rng.uniform(0.6, 1.2)
                x += bw + int(rng.uniform(0, 6))
        self.img[:] = sky

    # ------------------------------------------------------------ facades
    def building(self, x0, x1, top, wall, *, sign=None, sign_c='ffd84a', neon=None, awning=None, lit=0.35, seed=0,
                 fire_escape=False, vsign=None, cornice='8a8090', shop_glow='ffb050', door=None, tower=False):
        """A facade standing on the back edge of the sidewalk (stage rows top..Y0), drawn into the stage image."""
        rng = np.random.default_rng(seed)
        P = PAD
        img = self.img
        yb = int(Y0)
        base = hexc(wall)
        X, YY = np.meshgrid(np.arange(x0, x1), np.arange(top, yb))
        n = hash_noise(X, YY, seed + 11)
        brick = ((YY % 3) == 0) | (((X + (YY // 3) * 3) % 7) == 0)
        col = base[None, None, :] * (0.82 + 0.16 * n[..., None])
        col[brick] *= 0.8
        k = np.clip((X - x0) / max(1, x1 - x0), 0, 1)
        col *= (0.9 + 0.12 * np.sin(k * math.pi))[..., None]
        img[top + P:yb + P, x0 + P:x1 + P] = col
        # cornice and floor bands
        img[top + P:top + P + 3, x0 + P - 1:x1 + P + 1] = hexc(cornice) * 0.7
        img[top + P, x0 + P - 1:x1 + P + 1] = hexc(cornice)
        # windows of the upper floors
        for fy in range(top + 8, yb - 44, 15):
            img[fy + 11 + P, x0 + P:x1 + P] *= 0.75
            for wx in range(x0 + 5, x1 - 9, 13):
                on = rng.uniform() < lit
                c = hexc('ffcf70') * rng.uniform(0.75, 1.0) if on else hexc('1c2236') * rng.uniform(0.8, 1.2)
                img[fy + P:fy + 9 + P, wx + P:wx + 7 + P] = c
                img[fy + 4 + P, wx + P:wx + 7 + P] = c * 0.6
                img[fy + P:fy + 9 + P, wx + 3 + P] = c * 0.6
                if on and rng.uniform() < 0.5:                        # curtains
                    img[fy + P:fy + 9 + P, wx + P:wx + 2 + P] = hexc('a04a30') * 0.8
                img[fy + 9 + P, wx - 1 + P:wx + 8 + P] = hexc('aaa0b0') * 0.55
                if on:
                    img[fy + 10 + P:fy + 13 + P, wx + P:wx + 7 + P] += hexc('ffb060') * 0.08
        # the shop front
        sy0 = yb - 30
        img[sy0 + P:yb + P, x0 + P:x1 + P] *= 0.7
        for wx in range(x0 + 4, x1 - 10, 20):
            if door is not None and abs(wx - door) < 14:
                img[sy0 + 8 + P:yb + P, wx + 2 + P:wx + 12 + P] = hexc('2a1a10')
                img[sy0 + 10 + P:sy0 + 20 + P, wx + 4 + P:wx + 10 + P] = hexc(shop_glow) * 0.7
                continue
            img[sy0 + 6 + P:yb - 3 + P, wx + P:wx + 16 + P] = hexc(shop_glow) * rng.uniform(0.55, 0.8)
            for s in range(3):                                         # shelves and goods behind the glass
                yy = sy0 + 12 + s * 5
                img[yy + P, wx + 1 + P:wx + 15 + P] = hexc('5a3018')
                for gx in range(wx + 2, wx + 14, 2):
                    if rng.uniform() < 0.7:
                        img[yy - 2 + P:yy + P, gx + P] = hexc(['c84a30', '3a7ac8', 'e8d070', '4aa060'][rng.integers(4)]) * 0.8
            img[sy0 + 6 + P:yb - 3 + P, wx + 8 + P] *= 0.5
            img[sy0 + 6 + P, wx + P:wx + 16 + P] = hexc('2a2228')
        img[yb - 3 + P:yb + P, x0 + P:x1 + P] = hexc('3a3440')
        # glow spilling onto the sidewalk
        spill = np.zeros((40, x1 - x0), np.float32)
        spill[:] = np.linspace(1, 0, 40)[:, None] ** 2
        img[yb + P:yb + 40 + P, x0 + P:x1 + P] *= 1 + spill[..., None] * hexc(shop_glow) * 0.25
        if awning is not None:
            a1, a2 = hexc(awning[0]), hexc(awning[1])
            for yy in range(sy0 - 1, sy0 + 6):
                for xx in range(x0 + 1, x1 - 1):
                    img[yy + P, xx + P] = (a1 if ((xx - x0) // 4) % 2 == 0 else a2) * (0.75 + 0.05 * (yy - sy0))
            img[sy0 + 6 + P, x0 + 1 + P:x1 - 1 + P] = hexc('140c10')
        if sign is not None:
            sw_ = len(sign) * 8 - 2
            sx = (x0 + x1) // 2 - sw_ // 2
            syy = sy0 - 14
            img[syy - 2 + P:syy + 12 + P, sx - 4 + P:sx + sw_ + 4 + P] = hexc('140e18')
            img[syy - 2 + P, sx - 4 + P:sx + sw_ + 4 + P] = hexc('5a4a5a')
            put_text(img, sx + P, syy + P, sign, hexc(sign_c) if neon is None else hexc(neon), scale=2,
                     glow=0.9 if neon else 0.0, glow_c=neon)
        if vsign is not None:
            vx = x1 - 14
            vy = top + 14
            hh = len(vsign) * 12 + 6
            img[vy - 3 + P:vy + hh + P, vx - 3 + P:vx + 9 + P] = hexc('140e18')
            put_text(img, vx + P, vy + P, vsign, hexc(vsign[1]) if False else hexc('ff4a6a'), scale=2, glow=1.0,
                     glow_c='ff3a5a', vertical=True)
        if fire_escape:
            fx0 = x0 + (x1 - x0) // 2 - 14
            for fy in range(top + 20, yb - 40, 15):
                img[fy + P:fy + 2 + P, fx0 + P:fx0 + 30 + P] = hexc('0c0c10')
                img[fy - 5 + P, fx0 + P:fx0 + 30 + P] = hexc('1a1a20')
                for px_ in range(fx0, fx0 + 31, 5):
                    img[fy - 5 + P:fy + P, px_ + P] = hexc('1a1a20')
                for s in range(15):
                    xx = fx0 + 4 + s if (fy // 15) % 2 == 0 else fx0 + 26 - s
                    img[fy + 2 + s + P, xx + P] = hexc('16161c')
                img[fy + P, fx0 + P:fx0 + 30 + P] += 0.05
        if tower:
            tx = x0 + 18
            ty = top - 4
            img[ty - 22 + P:ty + P, tx + P:tx + 18 + P] = hexc('2a1e1a')
            for yy in range(ty - 22, ty, 3):
                img[yy + P, tx + P:tx + 18 + P] = hexc('1a1210')
            for k_ in range(7):
                img[ty - 29 + k_ + P, tx + 9 - k_ - 1 + P:tx + 9 + k_ + 1 + P] = hexc('1e1614')
            for lx in (tx + 2, tx + 15):
                img[ty + P:ty + 6 + P, lx + P] = hexc('120c0a')

    def build_main(self):
        self.building(-12, 120, 6, '6a2a22', sign='PIZZA', neon='ff4a3a', awning=('b02020', 'e8e0d0'), seed=1,
                      fire_escape=True, door=60)
        self.building(120, 300, -12, '6a5a48', sign='PAWN SHOP', sign_c='ffd84a', awning=('2a5a2a', '1a3a1a'), seed=2,
                      lit=0.25, door=200, tower=False)
        self.building(300, 468, 22, '4a2a2e', sign='PHARMACY', neon='4aff8a', seed=3, fire_escape=True,
                      shop_glow='d8f0ff', door=440, tower=True)
        self.building(468, int(WEST) + 4, 0, '545460', vsign='HOTEL', seed=4, lit=0.45, awning=('6a1a2a', '4a101a'),
                      door=520)

    def build_fifth(self):
        x0 = int(EAST) + WALL_T
        self.building(x0, 1330, -4, '7a7268', sign='BANK', sign_c='e8c860', seed=11, lit=0.15, shop_glow='c8b080',
                      door=1220, cornice='b0a890')
        img, P = self.img, PAD
        for cx in range(x0 + 14, 1330, 34):                              # the bank's pilasters
            if abs(cx + 2 - 1223) < 22:
                continue
            img[int(Y0) - 70 + P:int(Y0) - 3 + P, cx + P:cx + 5 + P] = hexc('9a9288') * 0.75
            img[int(Y0) - 70 + P:int(Y0) - 3 + P, cx + P] = hexc('c8c0b4') * 0.8
            img[int(Y0) - 72 + P, cx - 1 + P:cx + 6 + P] = hexc('c8c0b4') * 0.8
        self.building(1330, 1560, 10, '3a2e3a', seed=12, lit=0.3, awning=None, shop_glow='ffd070', door=1440)
        # the cinema's marquee with chasing bulbs (baked dim; the scene lights them)
        my0, my1 = int(Y0) - 46, int(Y0) - 32
        img[my0 + P:my1 + P, 1350 + P:1540 + P] = hexc('1a0c14')
        img[my0 + P, 1350 + P:1540 + P] = hexc('c89a3a')
        img[my1 - 1 + P, 1350 + P:1540 + P] = hexc('c89a3a')
        put_text(img, 1400 + P, my0 + 4 + P, 'CINEMA', hexc('ffe890'), scale=1, glow=0.6, glow_c='ffb040')
        put_text(img, 1440 + P, my0 + 4 + P, 'TONIGHT', hexc('ff9a5a'), scale=1, glow=0.4, glow_c='ff7040')
        put_text(img, 1356 + P, my0 - 30 + P, 'CINEMA', hexc('ffe0a0'), scale=3, glow=1.1, glow_c='ff9a30')
        self.bulbs = [(bx, my0 - 1) for bx in range(1352, 1540, 4)] + [(bx, my1) for bx in range(1352, 1540, 4)]
        self.building(1560, 1720, 18, '5a2a1e', sign='DELI', neon='ffe04a', awning=('1a6a3a', 'e8e0d0'), seed=13,
                      fire_escape=True, door=1600)
        self.building(1720, SW + 12, -8, '4a4a58', sign='BAR', neon='4ac8ff', seed=14, lit=0.4, vsign='HOTEL',
                      door=1800, tower=True)

    def build_diner(self):
        """The diner's ground floor in cutaway: its back wall, the ceiling slab, a dim flat above."""
        img, P = self.img, PAD
        x0, x1 = int(WEST) + WALL_T - 2, int(EAST) + 2
        yb, yt = int(Y0), int(Y0) - WALL_H
        X, YY = np.meshgrid(np.arange(x0, x1), np.arange(yt, yb))
        n = hash_noise(X, YY, 31)
        tiles = ((X % 8) == 0) | ((YY % 6) == 0)
        col = hexc('8a8468')[None, None, :] * (0.9 + 0.12 * n[..., None])
        col[tiles] *= 0.82
        img[yt + P:yb + P, x0 + P:x1 + P] = col
        # wainscot, a chrome rail
        img[yb - 22 + P:yb + P, x0 + P:x1 + P] = hexc('5a2016')
        for xx in range(x0, x1, 10):
            img[yb - 22 + P:yb + P, xx + P] = hexc('3a140e')
        img[yb - 23 + P, x0 + P:x1 + P] = hexc('d0d0d8')
        # the kitchen pass-through
        kx0, kx1, ky0, ky1 = 770, 870, yt + 22, yt + 48
        img[ky0 + P:ky1 + P, kx0 + P:kx1 + P] = hexc('e8b060') * 0.8
        img[ky0 + 16 + P:ky1 + P, kx0 + P:kx1 + P] = hexc('a07a50') * 0.8
        for hx in range(kx0 + 8, kx1 - 4, 14):
            img[ky0 + 2 + P:ky0 + 8 + P, hx + P] = hexc('2a2a2a')
            img[ky0 + 8 + P:ky0 + 11 + P, hx - 3 + P:hx + 4 + P] = hexc('3a3a3a')
        img[ky0 - 1 + P, kx0 - 1 + P:kx1 + 1 + P] = hexc('c8c8d0')
        img[ky1 + P:ky1 + 2 + P, kx0 - 2 + P:kx1 + 2 + P] = hexc('c8c8d0')
        # the menu board and a clock
        img[yt + 12 + P:yt + 40 + P, 890 + P:968 + P] = hexc('1a1a1c')
        put_text(img, 916 + P, yt + 15 + P, 'MENU', hexc('f0f0e0'))
        rng = np.random.default_rng(4)
        for r in range(4):
            yy = yt + 23 + r * 4
            for xx in range(896, 962, 2):
                if rng.uniform() < 0.55:
                    img[yy + P, xx + P] = hexc('c8c8b8') * 0.8
        cx_, cy_ = 712, yt + 22
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                d = math.hypot(dx, dy)
                if d <= 6:
                    img[cy_ + dy + P, cx_ + dx + P] = hexc('2a2a2a') if d > 5 else hexc('e8e4d8')
        img[cy_ - 4 + P:cy_ + 1 + P, cx_ + P] = hexc('1a1a1a')
        img[cy_ + P, cx_ + P:cx_ + 4 + P] = hexc('1a1a1a')
        # the neon «DINER» and «OPEN», framed photos
        put_text(img, 990 + P, yt + 10 + P, 'DINER', hexc('ff6a8a'), scale=2, glow=1.2, glow_c='ff3a6a')
        put_text(img, 1010 + P, yt + 34 + P, 'OPEN', hexc('6af0ff'), glow=0.8, glow_c='3ac8ff')
        for fx_ in (660, 735, 990, 1060):
            img[yt + 40 + P:yt + 52 + P, fx_ + P:fx_ + 12 + P] = hexc('2a1a10')
            img[yt + 42 + P:yt + 50 + P, fx_ + 2 + P:fx_ + 10 + P] = hexc(['6a8aa0', 'a08a6a', '8a6a8a', '6a9a6a'][fx_ % 4]) * 0.8
        # the jukebox against the wall
        jx = 1074
        for dy in range(26):
            half = 7 if dy > 6 else int(3 + dy * 0.6)
            img[yb - 26 + dy + P, jx - half + P:jx + half + 1 + P] = hexc('7a2a1a') if dy < 6 else hexc('5a1a10')
        img[yb - 18 + P:yb - 8 + P, jx - 5 + P:jx + 6 + P] = hexc('ffd070')
        for dy in range(6):
            img[yb - 26 + dy + P, jx - 1 + P:jx + 2 + P] = hexc('ff8a3a')
        # the ceiling slab and the flat above
        img[yt - 10 + P:yt + P, x0 - 4 + P:x1 + 4 + P] = hexc('3e3a46')
        img[yt - 1 + P, x0 - 4 + P:x1 + 4 + P] = hexc('24202a')
        img[yt - 10 + P, x0 - 4 + P:x1 + 4 + P] = hexc('5a5664')
        img[-PAD + P:yt - 10 + P, x0 + P:x1 + P] = hexc('241c2a')
        for wx in (700, 880, 1040):
            img[-6 + P:yt - 16 + P, wx + P:wx + 20 + P] = hexc('1a2a4a')
            img[-6 + P:yt - 16 + P, wx + 10 + P] = hexc('101828')
        # warm interior light on the wall
        lit = np.zeros((yb - yt, x1 - x0), np.float32)
        for lx in (740, 860, 980):
            xx = np.arange(x0, x1)[None, :]
            yy = np.arange(yt, yb)[:, None]
            lit += np.clip(1 - np.sqrt(((xx - lx) / 70.0) ** 2 + ((yy - yt - 30) / 50.0) ** 2), 0, 1) ** 1.5
        img[yt + P:yb + P, x0 + P:x1 + P] *= 1 + lit[..., None] * hexc('ffb060') * 0.35

    # ------------------------------------------------------------ floors
    def build_floor(self):
        h, w = self.img.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32) - PAD, np.arange(h, dtype=np.float32) - PAD)
        Z = (Y - Y0) / FS
        on = (Z >= 0) & (Z <= ZM) & (X >= -PAD) & (X < SW + PAD)
        xw = wall_x(WEST, Z)
        xe = wall_x(EAST, Z)
        diner = (X >= xw + WALL_T - 1) & (X < xe + 1)
        wall_ft = ((X >= xw - 1) & (X < xw + WALL_T)) | ((X >= xe - 1) & (X < xe + WALL_T))
        street = ~diner & ~wall_ft
        n = fbm(w, h, 4, seed=71)
        nn = hash_noise(X.astype(int), Y.astype(int), 72)
        col = np.zeros((h, w, 3), np.float32)
        # sidewalks: concrete slabs
        walk = (Z < WALK1) | (Z > ROAD1)
        slab = ((((X + 1000) % 16) < 1) | ((((Z % 16) * FS) < 0.7)))
        cw = hexc('4e4c58') * (0.85 + 0.2 * n[..., None]) * (0.92 + 0.12 * nn[..., None])
        cw[slab] *= 0.7
        col[walk] = cw[walk]
        curb = (np.abs(Z - WALK1 + 1) < 1.6) | (np.abs(Z - ROAD1 - 1) < 1.6)
        col[curb] = hexc('6e6c78')
        gut = ((Z > WALK1 + 0.6) & (Z < WALK1 + 3)) | ((Z > ROAD1 - 3) & (Z < ROAD1 - 0.6))
        col[gut] = hexc('1c1c22')
        # the road: asphalt, patches, markings
        road = (Z >= WALK1) & (Z <= ROAD1) & ~curb & ~gut
        a = hexc('26262e') * (0.8 + 0.35 * n[..., None]) * (0.9 + 0.18 * nn[..., None])
        patch = fbm(w, h, 18, seed=73) > 0.62
        a[patch] *= 0.82
        col[road] = a[road]
        yel = road & ((np.abs(Z - 104) < 1.0) | (np.abs(Z - 108) < 1.0))
        col[yel] = hexc('b8962a') * (0.8 + 0.2 * nn[yel, None])
        dash = road & ((np.abs(Z - 67) < 1.0) | (np.abs(Z - 146) < 1.0)) & (((X + 2000) % 26) < 14)
        col[dash] = hexc('a8a8b0') * (0.75 + 0.2 * nn[dash, None])
        zebra = road & (X > 1690) & (X < 1740) & ((np.floor(Z / 9) % 2) == 0) & (Z > 36) & (Z < 180)
        col[zebra] = hexc('9a9aa4') * (0.8 + 0.2 * nn[zebra, None])
        for mx_, mz_ in ((250, 90), (1250, 126), (1600, 80)):             # manholes
            d = np.sqrt(((X - mx_) / 8.0) ** 2 + ((Z - mz_) / 8.0) ** 2)
            mh = road & (d < 1)
            col[mh] = hexc('1a1a1e')
            col[road & (d < 0.75) & (((X.astype(int) + Y.astype(int)) % 3) == 0)] = hexc('3a3a40')
        # the diner's floor: black and white tiles
        T = 10.0
        ti = (np.floor((X - WEST) / T) + np.floor(Z / T)) % 2 == 0
        tc = np.where(ti[..., None], hexc('aaa494')[None, None, :], hexc('1e1a20')[None, None, :])
        tc = tc * (0.86 + 0.18 * nn[..., None])
        grout = (((X - WEST) % T) < 0.8) | (((Z % T) * FS) < 0.5)
        tc[grout] *= 0.75
        col[diner] = tc[diner]
        col[wall_ft] = hexc('3a3640') * (0.8 + 0.3 * nn[wall_ft, None])
        # night shading: darker towards the front of the street, warm light inside the diner
        kk = np.clip(Z / 240.0, 0, 1)[..., None]
        col = np.where(street[..., None], col * (1.0 - 0.18 * kk) + hexc('1a1a34') * (1 - kk) * 0.1, col)
        lit = np.zeros(Z.shape, np.float32)
        for lx in (720, 840, 960, 1060):
            lit += np.clip(1 - np.sqrt(((X - lx) / 60.0) ** 2 + ((Z - 100) / 80.0) ** 2), 0, 1) ** 1.3
        col = np.where(diner[..., None], col * (0.75 + 0.55 * np.minimum(lit, 1.2)[..., None]), col)
        self.img[on] = col[on]
        # the subway entrance on 5TH AVE's back sidewalk: railings and stairs going down, a lit sign
        P = PAD
        sx0, sx1 = 1600, 1650
        for z in range(4, 24):
            yy = int(floor_y(z))
            k = (z - 4) / 20.0
            self.img[yy + P, sx0 + 2 + P:sx1 - 2 + P] = hexc('1a1a1e') * (1.2 - k)
            if z % 3 == 0:
                self.img[yy + P, sx0 + 2 + P:sx1 - 2 + P] = hexc('3a3a44') * (1.1 - k)
        for xx in (sx0, sx1 - 2):
            self.img[int(floor_y(2)) - 6 + P:int(floor_y(24)) + P, xx + P:xx + 2 + P] = hexc('2a5a3a')
        self.img[int(floor_y(2)) - 7 + P, sx0 + P:sx1 + P] = hexc('3a7a4a')
        put_text(self.img, sx0 + 2 + P, int(Y0) - 14 + P, 'SUBWAY', hexc('f0f0f0'), glow=0.5, glow_c='4aff8a')

    # ------------------------------------------------------------ the subway station under 5TH AVE
    def build_subway(self):
        img, P = self.img, PAD
        h, w = img.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32) - PAD, np.arange(h, dtype=np.float32) - PAD)
        # the shaft's dark back (seen only once the road caves in) and the bedrock everywhere below the soil
        rock = Y >= SOIL0
        n = fbm(w, h, 6, seed=81)
        rc = hexc('1a1410') * (0.7 + 0.6 * n[..., None])
        layers = (np.sin(Y * 0.35 + n * 4) > 0.85)
        rc[layers] *= 0.7
        img[rock] = rc[rock]
        x0, x1 = int(SUB[0]), SW
        ceil0, ceil1 = SOIL1, CEIL1
        wall0, wall1 = CEIL1, int(SUB_Y0 - PLAT_H)
        nn = hash_noise(X.astype(int), Y.astype(int), 82)
        sub = (X >= x0) & (X < x1)
        # the ceiling slab
        m = sub & (Y >= ceil0) & (Y < ceil1)
        img[m] = hexc('3a3840') * (0.85 + 0.2 * nn[m, None])
        img[(ceil1 - 1 + P), x0 + P:x1 + P] = hexc('1a1820')
        # tiled back wall with a green band and the station's name
        m = sub & (Y >= wall0) & (Y < wall1)
        tl = ((X % 6) < 0.9) | ((Y % 4) < 0.9)
        tc = hexc('9a9480') * (0.85 + 0.15 * nn[..., None])
        tc[tl] *= 0.78
        img[m] = tc[m]
        band = sub & (Y >= wall0 + 10) & (Y < wall0 + 20)
        img[band] = hexc('1e4a36') * (0.9 + 0.1 * nn[band, None])
        lowt = sub & (Y >= wall1 - 16) & (Y < wall1)
        img[lowt] *= 0.78
        for sx in range(x0 + 40, x1 - 40, 150):
            put_text(img, sx + P, wall0 + 13 + P, '5TH AVE', hexc('f0f0e8'))
        rng = np.random.default_rng(9)
        for sx in range(x0 + 100, x1 - 40, 150):                       # posters
            c = hexc(['c84a30', '3a6ac8', 'e8c040', '8a3ac8'][rng.integers(4)]) * 0.7
            img[wall0 + 24 + P:wall0 + 44 + P, sx + P:sx + 16 + P] = c
            img[wall0 + 28 + P:wall0 + 34 + P, sx + 3 + P:sx + 13 + P] = hexc('e8e0d0') * 0.7
        # cables along the top of the wall
        for k, c in ((3, '101014'), (5, '2a1a10'), (6, '101014')):
            img[wall0 + k + P, x0 + P:x1 + P] = hexc(c)
        # the dark tunnel mouth on the left
        img[ceil0 + P:int(SUB_Y0 + ZM * FS) + 8 + P, x0 - 2 + P:x0 + 22 + P] = hexc('050508')
        img[ceil0 + P:int(SUB_Y0 + ZM * FS) + 8 + P, x0 + 22 + P] = hexc('2a2830')
        # the floor: the platform at the back (raised), then the track bed with sleepers and rails, the third rail
        Z = (Y - SUB_Y0) / FS
        plat = sub & (X > x0 + 22) & (Z + PLAT_H / FS >= 0) & (Z + PLAT_H / FS < PLAT)
        pc = hexc('6a6672') * (0.85 + 0.2 * nn[..., None])
        pj = ((X % 16) < 0.8)
        pc[pj] *= 0.8
        img[plat] = pc[plat]
        zp = Z + PLAT_H / FS
        strip = plat & (zp > PLAT - 7)
        img[strip] = hexc('c8a828') * (0.8 + 0.2 * nn[strip, None])
        edge = sub & (X > x0 + 22) & (Y >= SUB_Y0 + PLAT * FS - PLAT_H) & (Y < SUB_Y0 + PLAT * FS)
        img[edge] = hexc('4a4652') * (0.8 + 0.2 * nn[edge, None])
        img[int(SUB_Y0 + PLAT * FS) - 1 + P, x0 + 23 + P:x1 + P] = hexc('141218')
        bed = sub & (X > x0 + 22) & (Z >= PLAT) & (Z <= ZM)
        gc = hexc('3e3a38') * (0.65 + 0.6 * hash_noise(X.astype(int), Y.astype(int), 83)[..., None])
        img[bed] = gc[bed]
        tie = bed & (((X + 3000) % 10) < 4) & (Z > 92) & (Z < 168)
        img[tie] = hexc('3a2618') * (0.8 + 0.3 * nn[tie, None])
        for rz in (104.0, 156.0):
            r = bed & (np.abs(Z - rz) < 1.4)
            img[r] = hexc('5a5660')
            r2 = bed & (np.abs(Z - rz + 1.0) < 0.7)
            img[r2] = hexc('b8b8c8')
        third = bed & (np.abs(Z - 188) < 3.0)
        img[third] = hexc('4a3a20')
        img[bed & (np.abs(Z - 186.4) < 0.8)] = hexc('a89a50')
        # the cool fluorescent light (static part)
        lit = np.zeros(Z.shape, np.float32)
        for lx in (1260, 1440, 1620, 1800):
            lit += np.clip(1 - np.sqrt(((X - lx) / 90.0) ** 2 + ((Z - 110) / 110.0) ** 2), 0, 1) ** 1.2
        sm = sub & (Y >= wall0) & (Z <= ZM)
        img[sm] = img[sm] * (0.72 + 0.7 * np.minimum(lit[sm], 1.0)[..., None]) * hexc('e8f4ff')

    # ------------------------------------------------------------ the front layer: kerb, soil, the water main, the tunnel
    def build_front(self):
        fr = self.front
        h, w = fr.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32) - PAD, np.arange(h, dtype=np.float32) - PAD)
        n = fbm(w, h, 5, seed=91)
        nn = hash_noise(X.astype(int), Y.astype(int), 92)
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        # kerb face
        m = (Y >= SOIL0) & (Y < SOIL0 + 4)
        rgb[m] = hexc('4a4854') * (0.85 + 0.2 * nn[m, None])
        rgb[(Y >= SOIL0) & (Y < SOIL0 + 1)] = hexc('6e6c78')
        a[m] = 1
        # the soil with stones
        m = (Y >= SOIL0 + 4) & (Y < SOIL1)
        sc = hexc('2a1e16') * (0.7 + 0.6 * n[..., None])
        st = (nn > 0.94)
        sc[st] = hexc('4a4038')
        rgb[m] = sc[m]
        a[m] = 1
        # under the diner: its concrete footing
        dm = (X > WEST + 20) & (X < EAST + 10) & (Y >= SOIL0) & (Y < SOIL0 + 9)
        rgb[dm] = hexc('4a4652') * (0.85 + 0.2 * nn[dm, None])
        # the water main and two cables, all along the block
        pm = (Y >= 255) & (Y < 263)
        k = (Y - 255) / 8.0
        pc = mix_arr(hexc('6a7a88'), hexc('1a2028'), k) * (0.9 + 0.15 * nn[..., None])
        rgb[pm] = pc[pm]
        fl = pm & (((X + 4000) % 64) < 3)
        rgb[fl] = hexc('8a9aa8')
        rgb[(Y >= 254) & (Y < 255) & (((X + 4000) % 64) < 3)] = hexc('8a9aa8')
        for cy, c in ((266, '101014'), (268, 'b06a20')):
            rgb[(Y >= cy) & (Y < cy + 1)] = hexc(c)
        # the tunnel's ceiling slab under 5TH AVE, rock elsewhere
        m = (Y >= SOIL1) & (Y < CEIL1)
        sub = X >= SUB[0]
        rk = hexc('1e1814') * (0.7 + 0.6 * n[..., None])
        rgb[m & ~sub] = rk[m & ~sub]
        cc = hexc('4a4852') * (0.85 + 0.2 * nn[..., None])
        rebar = (nn > 0.9) & (((Y.astype(int)) % 4) == 0)
        cc[rebar] = hexc('2a2a30')
        rgb[m & sub] = cc[m & sub]
        a[m] = 1
        deep = (Y >= CEIL1) & ~sub
        rgb[deep] = rk[deep]
        a[deep] = 1
        # the subway's front cut
        f0 = SUB_Y0 + ZM * FS
        m = sub & (Y >= f0) & (Y < f0 + 6)
        rgb[m] = hexc('4a4852') * (0.85 + 0.2 * nn[m, None])
        rgb[sub & (Y >= f0) & (Y < f0 + 1)] = hexc('6a6874')
        a[m] = 1
        m = sub & (Y >= f0 + 6)
        rgb[m] = rk[m]
        a[m] = 1
        # the street's front edge light, and the tunnel mouth stays dark
        fr[..., :3] = rgb
        fr[..., 3] = a

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

    def stamp(self, x, z, color, r=1.0, alpha=1.0, seed=0, h=0.0, sq=FS):
        """A splat on a floor (h: its height; -DS is the subway) or, with sq=1, on a wall (z then is the row)."""
        cx = x + PAD
        cy = (floor_y(z) - h if sq != 1 else z) + PAD
        rx = max(0.6, r)
        ry = max(0.5, r * sq)
        ih, iw = self.img.shape[:2]
        y0, y1 = int(math.floor(cy - ry - 1)), int(math.ceil(cy + ry + 1)) + 1
        x0, x1 = int(math.floor(cx - rx - 1)), int(math.ceil(cx + rx + 1)) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= iw or y0 >= ih:
            return
        col = np.asarray(color, np.float32)
        for yy in range(max(0, y0), min(ih, y1)):
            for xx in range(max(0, x0), min(iw, x1)):
                dd = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2
                if dd <= 1.0:
                    nz = (((xx * 73856093) ^ (yy * 19349663) ^ seed) & 255) / 255.0
                    if dd > 0.6 and nz < 0.4:
                        continue
                    aa = alpha * (0.8 + 0.2 * nz)
                    self.img[yy, xx] = self.img[yy, xx] * (1 - aa) + col * aa

    def darken(self, x, row, r, k, sq=1.0, seed=0, layer='img'):
        """Multiplies a ragged ellipse of the stage (or the front layer) by k (scorch, wet, shadow)."""
        arr = self.img if layer == 'img' else self.front
        cx, cy = x + PAD, row + PAD
        ih, iw = arr.shape[:2]
        y0, y1 = max(0, int(cy - r * sq - 1)), min(ih, int(cy + r * sq + 2))
        x0, x1 = max(0, int(cx - r - 1)), min(iw, int(cx + r + 2))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = ((xx + 0.5 - cx) / r) ** 2 + ((yy + 0.5 - cy) / max(0.5, r * sq)) ** 2
        n = hash_noise(xx, yy, seed)
        m = d + n * 0.35 < 1.0
        arr[y0:y1, x0:x1, :3][m] *= k

    def crack_lines(self, x, row, n, length, seed, sq=FS, color='1a171e'):
        """A few thin forking cracks from a point (stage coords) along a floor (sq=FS) or a wall (sq=1)."""
        rng = np.random.default_rng(seed)
        ih, iw = self.img.shape[:2]
        c = hexc(color)
        branches = [(x, row, (i + rng.uniform(-0.3, 0.3)) * 2 * math.pi / n, length * rng.uniform(0.55, 1.1)) for i in range(n)]
        while branches:
            bx, by, ang, ln = branches.pop()
            for k in range(int(ln)):
                ang += rng.uniform(-0.35, 0.35)
                bx += math.cos(ang)
                by += math.sin(ang) * sq
                xi, yi = int(round(bx)) + PAD, int(round(by)) + PAD
                if 0 <= xi < iw - 1 and 0 <= yi < ih - 1:
                    self.img[yi, xi] = self.img[yi, xi] * 0.35 + c * 0.65
                if rng.uniform() < 0.025 and ln > 10 and k > 3:
                    branches.append((bx, by, ang + rng.choice([-0.8, 0.8]), (ln - k) * 0.5))

    def crater(self, x, z, r, seed, h=0.0, kind='asphalt'):
        """A smashed hollow in a floor: a dark bowl, a rim of broken chunks, cracks running out."""
        row = floor_y(z) - h
        self.darken(x, row, r, 0.55, FS, seed)
        self.darken(x, row + 0.5, r * 0.6, 0.6, FS, seed + 1)
        self.darken(x, row + 0.8, r * 0.3, 0.7, FS, seed + 2)
        rng = np.random.default_rng(seed)
        chunk = hexc({'asphalt': '4a4652', 'tile': 'c8c0b0', 'stone': '6a6672'}[kind])
        for k in range(int(r * 2.2)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * rng.uniform(0.85, 1.15)
            self.stamp(x + math.cos(a) * rr, z + math.sin(a) * rr, chunk * rng.uniform(0.6, 1.1),
                       r=rng.uniform(0.6, 1.6), alpha=1.0, seed=seed + k, h=h)
        self.crack_lines(x, row, 5 + int(r / 6), r * 1.7, seed + 7)

    def wall_crater(self, x, row, r, seed, hole=True, brick='6a2a22'):
        """A body or a car smashed into a facade (stage rows): a hole into the dark, broken bricks, cracks."""
        if hole:
            self.darken(x, row, r * 0.75, 0.12, 0.9, seed)
        self.darken(x, row, r * 1.15, 0.6, 0.9, seed + 1)
        rng = np.random.default_rng(seed)
        bc = hexc(brick)
        for k in range(int(r * 3)):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * rng.uniform(0.7, 1.05)
            self.stamp(x + math.cos(a) * rr, row + math.sin(a) * rr * 0.9, bc * rng.uniform(0.8, 1.4), r=rng.uniform(0.6, 1.3),
                       seed=seed + k, sq=1)
        self.crack_lines(x, row, 4 + int(r / 5), r * 1.6, seed + 3, sq=1)

    def blood_on_wall(self, x, row, seed, n=10):
        rng = np.random.default_rng(seed)
        for k in range(n):
            bx = x + rng.uniform(-5, 5)
            by = row + rng.uniform(-5, 4)
            for s in range(int(rng.uniform(2, 9))):
                self.stamp(bx, by + s, hexc('6e0807') if s else hexc('9c0f0b'), r=0.6, seed=seed + k * 13 + s, sq=1)

    def cave_in(self, x0, x1, z0, z1, seed):
        """The road gives way: a ragged pit in the street, and the soil band of the front layer opens into a shaft."""
        self.holes.append((x0, x1, z0, z1))
        P = PAD
        rng = np.random.default_rng(seed)
        r0, r1 = int(floor_y(z0)), int(floor_y(z1)) + 2
        for yy in range(r0 - 2, CEIL1 + 2):
            k = (yy - r0) / max(1, CEIL1 - r0)
            jag = rng.uniform(-3, 3)
            xa = int(x0 + 6 * math.sin(yy * 0.37) + jag)
            xb = int(x1 - 6 * math.sin(yy * 0.29 + 1) + rng.uniform(-3, 3))
            if yy < r1:
                # the pit in the road: ragged at its back edge
                kk = (yy - r0) / max(1, r1 - r0)
                half = 0.5 * (xb - xa) * min(1.0, 0.35 + kk * 1.3)
                cxm = (xa + xb) / 2
                xa, xb = int(cxm - half), int(cxm + half)
                row = np.linspace(-1, 1, max(1, xb - xa))
                c = hexc('0a0c12') * (0.7 + 0.5 * kk) + hexc('1a2a3a') * (1 - np.abs(row))[:, None] * 0.6 * kk
                self.img[yy + P, xa + P:xb + P] = c
                for e, cc in ((xa - 1, '6a6674'), (xa - 2, '4a4652'), (xb, '4a4652'), (xb + 1, '2e2a36')):
                    self.img[yy + P, e + P] = hexc(cc)
            # the shaft through the soil and the slab (in the front layer)
            if yy >= SOIL0:
                self.front[yy + P, xa + P:xb + P, 3] = 0.0
                self.front[yy + P, xa - 2 + P:xa + P, :3] *= 0.5
                self.front[yy + P, xb + P:xb + 2 + P, :3] *= 0.5
                self.img[yy + P, xa + P:xb + P] = hexc('0e0a08') * (1.0 + 0.6 * k)
        # bent rebar and the broken pipe ends
        for k in range(10):
            xx = rng.uniform(x0 + 4, x1 - 4)
            yy = rng.uniform(CEIL1 - 12, CEIL1)
            for s in range(int(rng.uniform(3, 8))):
                xi, yi = int(xx + s * rng.uniform(-0.6, 0.6)), int(yy + s)
                if 0 <= yi + P < self.front.shape[0]:
                    self.front[yi + P, xi + P, :3] = hexc('6a4a3a')
                    self.front[yi + P, xi + P, 3] = 1.0


# ================================================================== the stage protocol (what the engine and the beats use)
import fx  # noqa: E402
import city_props as pr  # noqa: E402

VOID = -999.0


class _PropSprite:
    """A prop or a slice of a slanted wall in the depth sort (the engine calls draw(cv, ox, oy, t))."""

    def __init__(self, obj, slice_=None):
        self.obj, self.slice_ = obj, slice_

    def draw(self, cv, ox, oy, t):
        if self.slice_ is None:
            self.obj.draw(cv, (ox, oy), t)
        else:
            self.obj.draw_slice(cv, (ox, oy), self.slice_)


def _setup_props(self):
    """The block's furniture (as in the Hulk clip): cars, street lamps, a traffic light, a hydrant, the diner's tables,
    stools, counter, columns and hanging lamps, the subway's beams and lights, the diner's two slanted walls."""
    self.cars = [pr.Car(430, 46, 'e8b818', face=-1, taxi=True, seed=1), pr.Car(215, 172, '2a4a9a', face=1, seed=2),
                 pr.Car(70, 46, 'a81c1c', face=1, seed=3), pr.Car(1290, 46, '9a9aa8', face=-1, seed=4),
                 pr.Car(1530, 174, '7a1a2a', face=1, seed=5), pr.Car(1840, 46, '2a6a3a', face=-1, seed=6)]
    self.lamps = [pr.Lamp(160, 200, face=1), pr.Lamp(520, 200, face=1), pr.Lamp(300, 14, face=1),
                  pr.Lamp(580, 14, face=-1), pr.Lamp(1230, 200, face=1), pr.Lamp(1640, 14, face=-1),
                  pr.Lamp(1880, 200, face=-1)]
    self.traffic = pr.Lamp(1666, 202, face=-1, kind='traffic')
    self.hydrant = pr.Hydrant(1478, 204)
    self.tables = [pr.Table(760, 150), pr.Table(900, 176), pr.Table(1010, 140)]
    self.counter = pr.Counter(720, 900, 16, 34)
    self.stools = [pr.Stool(x, 44) for x in range(736, 900, 22)]
    self.columns = [pr.Column(820, 72, 21), pr.Column(1000, 80, 22)]
    self.pendants = [pr.Pendant(x, z, 78, floor_y(0) - WALL_H) for x, z in ((735, 96), (850, 104), (965, 98), (1065, 108))]
    beams = [pr.Beam(x, PLAT - 3, -DS + PLAT_H, CEIL1) for x in (1200, 1360, 1520, 1680, 1840)]
    fluos = [pr.Fluo(x, 120, -DS + 64, CEIL1) for x in (1270, 1450, 1630, 1810)]
    self.props = (self.cars + self.lamps + [self.traffic, self.hydrant] + self.tables + [self.counter] + self.stools +
                  self.columns + self.pendants + beams + fluos)
    self.walls = {'west': pr.CrossWall(WEST, 'brick', 31), 'east': pr.CrossWall(EAST, 'diner', 32)}


_orig_init = Stage.__init__


def _init(self):
    set_projection(Y0, FS)
    _orig_init(self)
    _setup_props(self)


Stage.__init__ = _init


def _cam_limits(self, world, t):
    return CAM_X, CAM_Y['street']


def _ground(self, x, z, t, y=None):
    """The street (and the diner's floor) at height 0; the subway floor -DS for anything below the street."""
    if y is not None and y < -40:
        return -DS
    if -10 <= z <= ZM + 30 and -20 <= x <= SW + 20:
        return 0.0
    return VOID


def _surface_fn(self, x, z, y, t):
    h = -DS if y < -40 else 0.0
    return lambda X, Z, h=h: h


def _ground_arr(self, x, z, h, t):
    g = np.where(h < -40, -DS, 0.0).astype(np.float32)
    return g, np.ones(x.shape, bool)


def _floor_lights(self, cv, off, t, world):
    ox, oy = off
    cx0, cx1 = world.cam[0] - 140, world.cam[0] + W + 140
    for p in self.props:
        if not (cx0 < p.x < cx1):
            continue
        for L in p.lights(t):
            x, y, r, c, s = L[:5]
            sq = L[5] if len(L) > 5 else 0.8
            px.boost(cv, x + ox, y + oy, r, c, s, squash=sq)
    for i, (bx, by) in enumerate(getattr(self, 'bulbs', [])):
        if cx0 < bx < cx1 and (i + int(t * 8)) % 3 != 0:
            xx, yy = int(bx + ox), int(by + oy)
            if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                cv[yy, xx] = hexc('fff0a0')


def _sprites(self, t, world):
    cx0, cx1 = world.cam[0] - 140, world.cam[0] + W + 140
    out = []
    for p in self.props:
        if not p.hidden and cx0 < p.place(t)[0] < cx1:
            out.append((p.key(t), _PropSprite(p)))
    for wl in self.walls.values():
        if wl.x1 < cx0 or wl.x0 > cx1:
            continue
        for (z, c) in wl.slices():
            out.append((z, _PropSprite(wl, c)))
    return out


def _front(self, cv, off, t, world):
    px.blit(cv, self.front_view(world.cam), 0, 0)


def _update(self, world, t, dt):
    return None


def _view(self, cam):
    cx, cy = int(round(cam[0])), int(round(cam[1]))
    y0 = cy - M + PAD
    x0 = cx - M + PAD
    return self.img[y0:y0 + EH, x0:x0 + EW].copy(), (M - cx, M - cy)


Stage.cam_limits = _cam_limits
Stage.ground = _ground
Stage.surface_fn = _surface_fn
Stage.ground_arr = _ground_arr
Stage.floor_lights = _floor_lights
Stage.sprites = _sprites
Stage.overlay = _front
Stage.update = _update


def _tint(self, x, z, r, color, k, seed=0, h=0.0, gloss=True):
    """Ice over the road: a translucent coat with glints (stays)."""
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


Stage.tint = _tint


# ------------------------------------------------------------------ areas and routes (data for the choreographer)
AREAS = {
    'main': dict(x=(80.0, 580.0), z=(50.0, 180.0), y=0.0, center=(320.0, 116.0), size='wide', floor='asphalt',
                 label='MAIN ST'),
    'diner': dict(x=(740.0, 1050.0), z=(60.0, 180.0), y=0.0, center=(900.0, 118.0), size='medium', floor='tile',
                  label="the diner"),
    'fifth': dict(x=(1220.0, 1880.0), z=(50.0, 180.0), y=0.0, center=(1500.0, 116.0), size='wide', floor='asphalt',
                  label='5TH AVE'),
}
# smash_through: the stronger one throws the other through the diner's side wall (it stays holed); later trips go
# through the hole
ROUTES = [
    dict(a='main', b='diner', how='smash_through', wall='west'),
    dict(a='diner', b='fifth', how='smash_through', wall='east'),
    dict(a='diner', b='main', how='smash_through', wall='west'),
    dict(a='fifth', b='diner', how='smash_through', wall='east'),
]
START = {'main': [(170.0, 118.0), (340.0, 112.0)], 'fifth': [(1400.0, 118.0), (1580.0, 112.0)]}
TITLE_PAN = dict(start=(560.0, 6.0), end=(60.0, 14.0))


def _area_y(self, area, x, z):
    return 0.0


def _area_of(self, x, z, y, t):
    if y < -40:
        return None
    for name, a in AREAS.items():
        if a['x'][0] - 60 <= x <= a['x'][1] + 60:
            return name
    return None


def _route_open(self, route, t):
    return True


Stage.area_y = _area_y
Stage.area_of = _area_of
Stage.route_open = _route_open
Stage.AREAS = AREAS
Stage.ROUTES = ROUTES
Stage.START = START
Stage.TITLE_PAN = TITLE_PAN
Stage.CAM_X = CAM_X
Stage.CAM_Y = CAM_Y['street']


# ------------------------------------------------------------------ targets: what a body or a weapon can smash
def _rng(seed):
    return np.random.default_rng(seed)


def glass_burst(world, x, y, z, n, seed, up=0.0, angle=-math.pi / 2, spread=1.3):
    fx.burst(world, fx.GLASS, x, y, z, n, (40, 160), angle=angle, spread=spread, life=(0.8, 1.8), size=(1, 1.6),
             seed=seed, zs=0.9, up=up)


def sparks_at(world, x, y, z, n, seed, elec=False):
    fx.burst(world, fx.ELEC if elec else fx.SPARK, x, y, z, n, (40, 160), angle=-math.pi / 2, spread=1.5,
             life=(0.2, 0.6), size=(1, 1.6), seed=seed)


class Target:
    """Something in an area a fighter can be thrown into or a weapon can hit. smash(world, t, power, out, seed)
    schedules its breaking at t; power 1 is a hard blow, 2 a heavy weapon, 3+ a titan."""

    def __init__(self, stage, kind, area, x, z, h, ref=None):
        self.stage, self.kind, self.area, self.x, self.z, self.h, self.ref = stage, kind, area, x, z, h, ref
        self.used = 0

    def ok(self, t):
        r = self.ref
        if self.kind == 'car':
            return r.state != 'wreck' and r.path is None
        if self.kind == 'lamp':
            return r.fall_t is None or t < r.fall_t
        if self.kind == 'pole':
            return r.bend_t is None
        if self.kind == 'hydrant':
            return r.broken_t is None
        if self.kind == 'table':
            return not r.hidden
        if self.kind == 'window':
            return self.used < 1
        return True

    def smash(self, world, t, power, out, seed):
        self.used += 1
        r = self.ref
        k = self.kind
        if k == 'car':
            def ev(w):
                r.state = 'wreck' if (power >= 2.0 or r.state == 'dent') else 'dent'
                r.alarm_t = w.t
                r.hop = (w.t, 6 + 6 * power)
                glass_burst(w, r.x, floor_y(r.z) - 14, r.z, int(20 + 10 * power), seed, up=30)
                w.shake(w.t, 3.0 + power, 0.35)
                w.cue('car_hit', x=r.x, power=round(min(2.0, power), 2))
                w.cue('glass', x=r.x, power=1.0)
                w.cue('car_alarm', x=r.x, dur=40.0)
            world.at(t, ev)
        elif k == 'lamp':
            def down(w):
                r.fall_t = w.t
                r.fall_dir = out
                x, y = r.x, floor_y(r.z) - 4
                sparks_at(w, x, y, r.z, 24, seed, elec=True)
                sparks_at(w, x, y, r.z, 14, seed + 1)
                w.shake(w.t, 3.0, 0.3)
                w.cue('pole_hit', x=x)
                w.cue('sparks', x=x, dur=0.8)

            def lands(w):
                hx, hy = r.head(w.t)
                glass_burst(w, hx, hy, r.z, 22, seed + 2, up=20)
                sparks_at(w, hx, hy, r.z, 18, seed + 3, elec=True)
                w.cue('metal_crash', x=hx)
                w.cue('glass', x=hx, power=0.6)
            world.at(t, down)
            world.at(t + 0.5, lands)
        elif k == 'pole':
            def bend(w):
                r.bend_t = w.t
                r.bend_deg = -18.0 * out
                r.flicker_t = w.t
                x, y = r.head(w.t)
                sparks_at(w, x, y, r.z, 16, seed, elec=True)
                w.cue('pole_hit', x=x)
            world.at(t, bend)
        elif k == 'hydrant':
            def burst(w):
                r.broken_t = w.t
                w.cue('water_burst', x=r.x)
                w.cue('water', x=r.x, dur=200.0, source='geyser')
            world.at(t, burst)

            def geyser(w, tt, dt):
                rng = _rng(int(tt * 1000) + seed)
                n = 7
                w.particles.spawn(fx.WATER, r.x + rng.uniform(-1.5, 1.5, n), r.z + rng.uniform(-1, 1, n),
                                  rng.uniform(6, 9, n), rng.uniform(-14, 14, n), rng.uniform(-8, 8, n),
                                  rng.uniform(150, 215, n), rng.uniform(1.0, 1.6, n), rng.uniform(1.0, 1.8, n), rng=rng)
            world.effects.append(fx.Emitter(t, 900.0, geyser))
        elif k == 'table':
            def smashed(w):
                r.hidden = True
                fx.burst(w, fx.WOOD, r.x, floor_y(r.z) - 6, r.z, 26, (40, 150), angle=-math.pi / 2, spread=1.4,
                         life=(0.8, 1.6), size=(1, 2.2), seed=seed)
                fx.burst(w, fx.TILE, r.x, floor_y(r.z) - 6, r.z, 10, (40, 120), angle=-math.pi / 2, spread=1.3,
                         life=(0.8, 1.4), seed=seed + 1)
                w.shake(w.t, 2.5, 0.3)
                w.cue('table_break', x=r.x)
            world.at(t, smashed)
        elif k == 'counter':
            def broke(w):
                r.breaks.append((self.x, 8 + 6 * power))
                r.arr = None
                fx.burst(w, fx.WOOD, self.x, floor_y(r.z1) - 14, r.z1, 30, (40, 160), angle=-math.pi / 2, spread=1.4,
                         life=(0.8, 1.6), size=(1, 2.2), seed=seed)
                w.shake(w.t, 3.5, 0.35)
                w.cue('wood_crash', x=self.x)
            world.at(t, broke)
        elif k == 'column':
            def hit(w):
                r.hit(self.h, power, seed, -out)
                fx.burst(w, fx.CHIP, r.x - out * 6, floor_y(r.z) - self.h, r.z + 2, int(12 * power), (40, 150),
                         angle=-math.pi / 2 - out * 0.9, spread=0.9, life=(1.0, 2.2), size=(1, 2.2), seed=seed)
                w.shake(w.t, 3.0 * power, 0.35)
                w.cue('stone_crash', x=r.x, power=power)
            world.at(t, hit)
        elif k == 'window':
            def shop(w):
                row = floor_y(0) - self.h
                w.stage.wall_crater(self.x, row, 8 + 3 * power, seed, hole=True)
                w.stage.blood_on_wall(self.x, row, seed + 1)
                fx.burst(w, fx.BRICK, self.x, row, 2, int(16 * power), (40, 150), angle=math.pi / 2, spread=1.2,
                         life=(0.8, 1.8), size=(1, 2.4), seed=seed + 2, zs=1.4, up=40)
                glass_burst(w, self.x, row + 6, 3, int(18 * power), seed + 3, angle=math.pi / 2, spread=1.4, up=30)
                fx.burst(w, fx.DUST, self.x, row, 3, 16, (15, 60), life=(0.6, 1.4), size=(1.8, 3.0), seed=seed + 4,
                         zs=1.2)
                w.shake(w.t, 2.5 + 2 * power, 0.35)
                w.effects.append(fx.Ring(self.x, row, w.t, 3, 14 + 8 * power, 0.25, 'cfc8d8'))
                w.cue('wall_break', x=self.x, power=round(power, 2))
                w.cue('glass', x=self.x, power=round(power, 2))
            world.at(t, shop)
        return t


def _targets(self, area, t):
    if not hasattr(self, '_targets_cache'):
        T = []

        def ar(x):
            return 'main' if x < 680 else ('diner' if x < 1110 else 'fifth')
        for c in self.cars:
            T.append(Target(self, 'car', ar(c.x), c.x, c.z, 14, c))
        for L in self.lamps:
            T.append(Target(self, 'lamp', ar(L.x), L.x, L.z, 30, L))
        T.append(Target(self, 'pole', 'fifth', self.traffic.x, self.traffic.z, 50, self.traffic))
        T.append(Target(self, 'hydrant', 'fifth', self.hydrant.x, self.hydrant.z, 5, self.hydrant))
        for tb in self.tables:
            T.append(Target(self, 'table', 'diner', tb.x, tb.z, 8, tb))
        T.append(Target(self, 'counter', 'diner', 820.0, 34.0, 14, self.counter))
        for col in self.columns:
            T.append(Target(self, 'column', 'diner', col.x, col.z, 24, col))
        for x in (130.0, 270.0, 430.0, 560.0):
            T.append(Target(self, 'window', 'main', x, 6.0, 26))
        for x in (1260.0, 1420.0, 1600.0, 1780.0):
            T.append(Target(self, 'window', 'fifth', x, 6.0, 26))
        self._targets_cache = T
    return [x for x in self._targets_cache if x.area == area and x.ok(t)]


Stage.targets = _targets


def smash_floor(world, x, z, power, seed, g=None):
    """A heavy impact on the road or the diner's tiles: a crater with cracks and chunks."""
    kind = 'tile' if WEST + 60 < x < EAST + 40 else 'asphalt'
    world.stage.crater(x, z, 8 + 5 * power, seed, h=0.0, kind=kind)
    fx.burst(world, fx.TILE if kind == 'tile' else fx.ASPHALT, x, floor_y(z) - 1, z, int(12 * power), (40, 150),
             angle=-math.pi / 2, spread=1.2, life=(0.9, 2.0), size=(1, 2.4), seed=seed, zs=0.8)
    world.cue('smash', x=x, power=round(min(2.0, power), 2), floor=kind)
    return kind


def _smash_floor(self, world, x, z, power, seed, g=None):
    return smash_floor(world, x, z, power, seed, g)


Stage.smash_floor = _smash_floor


def _smash_through(self, c, A, D, t, route):
    """A throws D through the diner's side wall into the next room (a hole that stays, bricks, dust); A follows him
    through it. If the wall is already holed, the hole is torn wider."""
    import beats as B
    from acts import strike, tumble_to, fall_lie_up, guard, guard_move, power
    w = c.w
    wall = self.walls[route['wall']]
    dst = route['b']
    ad = AREAS[dst]
    s = 1 if ad['center'][0] > AREAS[route['a']]['center'][0] else -1
    zc = float(c.rng.uniform(90, 140))
    wx = wall_x(wall.xb, zc) + (WALL_T / 2)
    # D is driven back against the wall, A presses after him
    dpos = (wx - s * 16, zc)
    apos = (wx - s * (16 + B.fdist(A, D)), zc + 4)
    dur = max(0.5, min(1.2, abs(D.x_at(t) - dpos[0]) / 160.0))
    D.add(t, 'stalk', dur, to=dpos, face=-s)
    A.add(t, 'stalk', dur, to=apos, face=s)
    guard(D, t + dur)
    guard(A, t + dur)
    t += dur + 0.05
    hk, k = B.held(A, t)
    if k == 'hammer':
        nm, wp = 'hammer_swing', ('n', 'hammer')
    elif k:
        nm, wp = 'slash_across', (hk, k)
    else:
        nm, wp = B.choose_blow(c, A, 'heavy'), None
    th = strike(c, A.name, D.name, t, nm, 'chest' if wp else B.WHERE.get(nm, 'head'), wep=wp, react=None, blood=12,
                size=11, spark='big', shake=4.0, settle_move=guard_move(A, t), wound=B.wound('torso', 'tear', 2.0, c.rng),
                mp={'hand': wp[0]} if wp and nm == 'slash_across' else None)
    pw = power(A) / max(0.7, power(D) ** 0.5)
    land = (wx + s * (40 + 30 * pw), zc + float(c.rng.uniform(-10, 10)))
    tumble_to(c, D, th, land, 0.55, 14)
    t_wall = th + 0.18
    hc = 30.0

    def breach(world):
        wall.breach(zc, hc, 30 + 6 * pw, 24 + 6 * pw, c.seed())
        x = wall_x(wall.xb, zc)
        y = floor_y(zc) - hc
        fx.burst(world, fx.BRICK, x, y, zc, int(36 * pw), (60, 200), angle=0.0 if s > 0 else math.pi, spread=1.4,
                 life=(0.9, 2.2), size=(1, 2.6), seed=c.seed(), zs=1.2, up=40)
        fx.burst(world, fx.DUST, x, y, zc, int(40 * pw), (20, 90), life=(0.8, 1.8), size=(2.0, 3.4), seed=c.seed(),
                 zs=1.4)
        rng = _rng(c.seed())
        for kk in range(int(26 * pw)):
            world.stage.stamp(x + s * rng.uniform(0, 30), zc + rng.uniform(-20, 20),
                              fx.BRICK_C[rng.integers(1, 4)] * rng.uniform(0.7, 1.1), r=rng.uniform(1.0, 2.6),
                              seed=c.seed())
        world.shake(world.t, 4.0 + 2 * pw, 0.45)
        world.cue('wall_break', x=x, power=round(pw + 0.5, 2))
        world.count['hits'] += 0
    w.at(t_wall, breach)
    t_up = fall_lie_up(c, D, th + 0.55, -6 * s, 0.35, 0.5)
    # A comes through the hole after him
    ta = th + 0.6
    A.add(ta, 'run', 0.55, to=(land[0] - s * (B.fdist(A, D) + 10), land[1]), face=s)
    guard(A, ta + 0.55)
    c.area = dst
    w.mark(f'wall_{route["wall"]}', t_wall)
    return max(t_up, ta + 0.6) + 0.05


Stage.smash_through = _smash_through


def setup(world):
    return None
