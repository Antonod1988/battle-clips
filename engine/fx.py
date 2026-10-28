"""Effects: particles live on the stage (x, depth z, height h) and land on whatever surface is under them for good
(blood, chips, ice, glass, bricks, bones stay as stamps); the set pieces (hit sparks, rings, the ice ball, the ice block,
the spear on its rope, the flying shield, electric arcs) are drawn in stage coordinates, shifted to the canvas by `off`
(stage -> canvas)."""
import math
import numpy as np

import px
from px import hexc, ramp
from proj import floor_y

(BLOOD, SPARK, CHIP, DUST, ICE, EMBER, SMOKE, FLAME, FROST, GORE, BONE, MIST, TILE, WOOD, STEAM, GLASS, BRICK, ASPHALT,
 WATER, ELEC) = range(20)
GRAV = {BLOOD: 330.0, SPARK: 160.0, CHIP: 380.0, DUST: -6.0, ICE: 360.0, EMBER: -40.0, SMOKE: -22.0, FLAME: -95.0,
        FROST: -10.0, GORE: 360.0, BONE: 380.0, MIST: 20.0, TILE: 380.0, WOOD: 360.0, STEAM: -30.0, GLASS: 380.0,
        BRICK: 400.0, ASPHALT: 400.0, WATER: 330.0, ELEC: 120.0}
DRAG = {BLOOD: 0.6, SPARK: 2.0, CHIP: 0.4, DUST: 3.0, ICE: 0.5, EMBER: 1.2, SMOKE: 1.5, FLAME: 2.4, FROST: 2.0, GORE: 0.5,
        BONE: 0.4, MIST: 4.0, TILE: 0.4, WOOD: 0.5, STEAM: 1.8, GLASS: 0.5, BRICK: 0.35, ASPHALT: 0.35, WATER: 0.4,
        ELEC: 3.0}
TILE_C = ramp('141218', '262434', '3a3850', '585478')
WOOD_C = ramp('2a1a0c', '4a3018', '6a4a2a', '94704a')
_G = np.zeros(24, np.float32)
_D = np.zeros(24, np.float32)
for _k, _v in GRAV.items():
    _G[_k] = _v
for _k, _v in DRAG.items():
    _D[_k] = _v
BLOOD_C = ramp('3a0303', '6e0807', '9c0f0b', 'c81c12', 'e8412a')
FLAME_C = ramp('fffbe0', 'ffe070', 'ffb02a', 'ff6a14', 'd8320c', '7a1a0c', '3a1a14')
SPARK_C = ramp('ffffff', 'fff4b0', 'ffd050', 'ff8a20')
ICE_C = ramp('ffffff', 'd8f4ff', '9ad8fa', '58a8ec', '2c6cc0')
STONE_C = ramp('2a2530', '4a4452', '6a6474', '8c8698')
BONE_C = ramp('6a5e4a', 'b8ab8e', 'ddd2b6', 'f6efe0')
WATER_C = ramp('ffffff', 'd8f0ff', '9ad0f4', '5a9ad0')
GLASS_C = ramp('ffffff', 'd8f4ff', '8ac8e8', '5a8aa8')
BRICK_C = ramp('3a1610', '6a2a1e', '8a3a28', 'a85a40')
ASPHALT_C = ramp('16161a', '2a2a30', '3e3e46', '5a5a64')


class Particles:
    def __init__(self):
        self.a = {k: np.zeros(0, np.float32) for k in ('x', 'z', 'h', 'vx', 'vz', 'vh', 'age', 'life', 'size', 'seed')}
        self.kind = np.zeros(0, np.int16)

    def spawn(self, kind, x, z, h, vx, vz, vh, life, size=1.0, rng=None):
        x = np.atleast_1d(np.asarray(x, np.float32))
        n = len(x)

        def b(v):
            v = np.asarray(v, np.float32)
            return np.broadcast_to(v, (n,)).astype(np.float32) if v.ndim == 0 or v.shape[0] != n else v
        rng = rng or np.random.default_rng()
        new = {'x': x, 'z': b(z), 'h': b(h), 'vx': b(vx), 'vz': b(vz), 'vh': b(vh), 'age': np.zeros(n, np.float32),
               'life': b(life), 'size': b(size), 'seed': rng.uniform(0, 1, n).astype(np.float32)}
        for k in self.a:
            self.a[k] = np.concatenate([self.a[k], new[k]])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.int16)])

    def update(self, dt, world):
        if len(self.kind) == 0 or dt <= 0:
            return
        a, k = self.a, self.kind
        g = _G[k]
        dr = _D[k]
        st = world.stage
        gh, stampable = st.ground_arr(a['x'], a['z'], a['h'], world.t)
        a['age'] += dt
        a['vh'] -= g * dt
        a['vx'] *= np.maximum(0, 1 - dr * dt)
        a['vz'] *= np.maximum(0, 1 - dr * dt)
        a['vh'] *= np.maximum(0, 1 - dr * 0.5 * dt)
        a['x'] += a['vx'] * dt
        a['z'] += a['vz'] * dt
        a['h'] += a['vh'] * dt
        keep = a['age'] < a['life']
        land = (a['h'] <= gh) & (a['vh'] < 0) & (gh > -900)
        for i in np.nonzero(land & keep)[0]:
            kk = k[i]
            x, z = a['x'][i], a['z'][i]
            hh = float(gh[i])
            sd = int(a['seed'][i] * 1e6)
            if kk in (BLOOD, GORE):
                if stampable[i]:
                    sp = min(1.0, (abs(a['vx'][i]) + abs(a['vz'][i])) / 160.0)
                    r = 0.6 + a['size'][i] * 0.6 + (1.4 if kk == GORE else 0) + sp * 0.5
                    st.stamp(x, z, BLOOD_C[1 + int(a['seed'][i] * 2.99)], r=r, alpha=0.95, seed=sd, h=hh)
                    if a['size'][i] > 1.2 or kk == GORE:
                        dx = 1.0 if a['vx'][i] >= 0 else -1.0
                        st.stamp(x + dx * (r + 1.5), z + (a['seed'][i] - 0.5) * 3, BLOOD_C[1], r=0.6, alpha=0.9,
                                 seed=sd + 1, h=hh)
                keep[i] = False
            elif kk == WATER:
                if stampable[i]:
                    st.stamp(x, z, hexc('1a3a5a'), r=0.8 + a['size'][i] * 0.7, alpha=0.16, seed=sd, h=hh)
                keep[i] = False
            elif kk in (CHIP, ICE, BONE, TILE, WOOD, GLASS, BRICK, ASPHALT):
                if abs(a['vh'][i]) < 45:
                    if stampable[i]:
                        sz = a['size'][i]
                        if kk == CHIP:
                            c = STONE_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == BONE:
                            c = BONE_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == TILE:
                            c = TILE_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == WOOD:
                            c = WOOD_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == GLASS:
                            c = GLASS_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == BRICK:
                            c = BRICK_C[1 + int(a['seed'][i] * 2.99)]
                        elif kk == ASPHALT:
                            c = ASPHALT_C[1 + int(a['seed'][i] * 2.99)]
                        else:
                            c = ICE_C[2 + int(a['seed'][i] * 1.99)]
                        st.stamp(x, z, c, r=0.5 + sz * 0.45, alpha=0.8 if kk in (ICE, GLASS) else 1.0, seed=sd, h=hh)
                    keep[i] = False
                else:
                    a['h'][i] = hh
                    a['vh'][i] = -a['vh'][i] * 0.35
                    a['vx'][i] *= 0.55
                    a['vz'][i] *= 0.55
            elif kk in (SPARK, ELEC):
                a['h'][i] = hh
                a['vh'][i] = -a['vh'][i] * 0.3
        for key in a:
            a[key] = a[key][keep]
        self.kind = k[keep]

    def draw(self, cv, off, lights, add):
        if len(self.kind) == 0:
            return
        a, k = self.a, self.kind
        cx = a['x'] + off[0]
        cy = floor_y(a['z']) - a['h'] + off[1]
        h, w = cv.shape[:2]
        f = np.clip(a['age'] / np.maximum(a['life'], 1e-3), 0, 1)
        svx, svy = a['vx'], a['vz'] * 0.6 - a['vh']

        def put(sel, xs, ys, cols, alpha=None, additive=False):
            xs = np.round(xs).astype(int)
            ys = np.round(ys).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys, cols = xs[ok], ys[ok], cols[ok]
            if additive:
                np.add.at(add, (ys, xs), cols * (alpha[ok][:, None] if alpha is not None else 1.0))
            elif alpha is None:
                cv[ys, xs] = cols
            else:
                al = alpha[ok][:, None]
                cv[ys, xs] = cv[ys, xs] * (1 - al) + cols * al

        def discs(sel, rad, cols, alpha=None, additive=False):
            idx = np.nonzero(sel)[0]
            for r in (1, 2, 3, 4):
                ii = idx[np.clip(np.round(rad[idx]).astype(int), 1, 4) == r]
                if len(ii) == 0:
                    continue
                offs = [(dx, dy) for dx in range(-r + 1, r) for dy in range(-r + 1, r) if dx * dx + dy * dy < (r - 0.3) ** 2] or [(0, 0)]
                for dx, dy in offs:
                    put(None, cx[ii] + dx, cy[ii] + dy, cols[ii], None if alpha is None else alpha[ii], additive)

        # smoke and dust first (behind the bright stuff)
        sel = (k == SMOKE)
        if sel.any():
            col = np.tile(hexc('2a2228'), (len(k), 1)) * (1.0 + 0.2 * a['seed'][:, None])
            discs(sel, 1 + f * 3.2 * a['size'], col, alpha=(1 - f) * 0.55)
        sel = (k == MIST)
        if sel.any():
            col = np.tile(hexc('8a0c0a'), (len(k), 1))
            discs(sel, 1 + f * 2.6 * a['size'], col, alpha=(1 - f) * 0.5)
        sel = (k == DUST)
        if sel.any():
            col = np.tile(hexc('7a7080'), (len(k), 1))
            discs(sel, 1 + f * 3.0 * a['size'], col, alpha=(1 - f) * 0.5)
        # blood: drops with a short streak
        sel = (k == BLOOD) | (k == GORE)
        if sel.any():
            ci = np.clip((a['seed'] * 3).astype(int) + 1, 1, 4)
            col = BLOOD_C[ci]
            idx = np.nonzero(sel)[0]
            put(None, cx[idx], cy[idx], col[idx])
            sp = np.hypot(svx[idx], svy[idx])
            fast = idx[sp > 60]
            put(None, cx[fast] - svx[fast] * 0.012, cy[fast] - svy[fast] * 0.012, BLOOD_C[np.clip(ci[fast] - 1, 0, 4)])
            big = idx[a['size'][idx] > 1.5]
            put(None, cx[big] + 1, cy[big], col[big])
            put(None, cx[big], cy[big] + 1, BLOOD_C[np.clip(ci[big] - 1, 0, 4)])
            gore = np.nonzero(k == GORE)[0]
            for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 0)):
                put(None, cx[gore] + dx, cy[gore] + dy, BLOOD_C[np.clip(ci[gore] - (dx + dy) % 2, 0, 4)])
        sel = (k == CHIP)
        if sel.any():
            col = STONE_C[np.clip((a['seed'] * 3).astype(int) + 1, 1, 3)]
            discs(sel, np.maximum(1, a['size'] * 0.8), col)
        sel = (k == TILE)
        if sel.any():
            col = TILE_C[np.clip((a['seed'] * 3).astype(int) + 1, 1, 3)]
            discs(sel, np.maximum(1, a['size'] * 0.9), col)
        sel = (k == WOOD)
        if sel.any():
            col = WOOD_C[np.clip((a['seed'] * 3).astype(int) + 1, 1, 3)]
            idx = np.nonzero(sel)[0]
            put(None, cx[idx], cy[idx], col[idx])
            put(None, cx[idx] + 1, cy[idx], col[idx])
        for kk, rp in ((BRICK, BRICK_C), (ASPHALT, ASPHALT_C)):
            sel = (k == kk)
            if sel.any():
                col = rp[np.clip((a['seed'] * 3).astype(int) + 1, 1, 3)]
                discs(sel, np.maximum(1, a['size'] * 0.8), col)
        sel = (k == GLASS)
        if sel.any():
            idx = np.nonzero(sel)[0]
            tw = ((a['age'][idx] * 20 + a['seed'][idx] * 7).astype(int) % 3)
            put(None, cx[idx], cy[idx], GLASS_C[np.clip(tw, 0, 3)])
        sel = (k == WATER)
        if sel.any():
            idx = np.nonzero(sel)[0]
            col = WATER_C[np.clip((a['seed'][idx] * 3).astype(int), 0, 3)]
            put(None, cx[idx], cy[idx], col, np.full(len(idx), 0.8, np.float32))
        sel = (k == ELEC)
        if sel.any():
            idx = np.nonzero(sel)[0]
            col = np.tile(hexc('c8e8ff'), (len(idx), 1))
            for s_ in (0.0, 0.008, 0.016):
                put(None, cx[idx] - svx[idx] * s_, cy[idx] - svy[idx] * s_, col, (1 - f[idx]) * 1.6, True)
        sel = (k == STEAM)
        if sel.any():
            col = np.tile(hexc('d8e8f0'), (len(k), 1))
            discs(sel, 1 + f * 3.0 * a['size'], col, alpha=(1 - f) * 0.4)
        sel = (k == BONE)
        if sel.any():
            col = BONE_C[np.clip((a['seed'] * 3).astype(int) + 1, 1, 3)]
            discs(sel, np.maximum(1, a['size'] * 0.8), col)
        sel = (k == ICE)
        if sel.any():
            col = ICE_C[np.clip((a['seed'] * 3).astype(int), 0, 3)]
            idx = np.nonzero(sel)[0]
            put(None, cx[idx], cy[idx], col[idx])
            big = idx[a['size'][idx] > 1.4]
            put(None, cx[big] + 1, cy[big] + 1, ICE_C[np.clip((a['seed'][big] * 3).astype(int) + 1, 0, 4)])
            put(None, cx[big] - 1, cy[big], ICE_C[3][None, :].repeat(len(big), 0))
        # fire: discs whose colour runs from white-hot to red and smoke
        sel = (k == FLAME)
        if sel.any():
            ci = np.clip((f * 6.99).astype(int), 0, 6)
            col = FLAME_C[ci]
            rad = 1 + (0.4 + f * 1.4) * a['size']
            discs(sel, rad, col, alpha=np.clip(1.25 - f, 0, 1))
            hot = sel & (f < 0.55)
            discs(hot, rad * 0.6, FLAME_C[np.clip(ci + 1, 0, 6)] * 0.6, alpha=(0.55 - f) * 1.5, additive=True)
        sel = (k == SPARK)
        if sel.any():
            ci = np.clip((f * 3.99).astype(int), 0, 3)
            col = SPARK_C[ci]
            idx = np.nonzero(sel)[0]
            for s in (0.0, 0.006, 0.012):
                put(None, cx[idx] - svx[idx] * s, cy[idx] - svy[idx] * s, col[idx], (1 - f[idx]) * 1.2, True)
        sel = (k == EMBER)
        if sel.any():
            fl = 0.6 + 0.4 * np.sin(a['age'] * 30 + a['seed'] * 20)
            col = np.tile(hexc('ff8a2a'), (len(k), 1))
            idx = np.nonzero(sel)[0]
            put(None, cx[idx], cy[idx], col[idx], (1 - f[idx]) * fl[idx] * 1.3, True)
        sel = (k == FROST)
        if sel.any():
            col = np.tile(hexc('bfe8ff'), (len(k), 1))
            idx = np.nonzero(sel)[0]
            put(None, cx[idx], cy[idx], col[idx], (1 - f[idx]) * 0.9, True)


def burst(world, kind, x, y, z, n, speed, angle=None, spread=math.pi, life=(0.3, 0.8), size=(1, 1), up=0.0, seed=0,
          zs=0.35, vx0=0.0, vh0=0.0):
    """Spawn n particles from the stage point (x, y) at depth z: around `angle` (radians in the picture plane, screen
    coords) within +-spread, scattered in depth by zs times their speed."""
    rng = np.random.default_rng(seed)
    if angle is None:
        ang = rng.uniform(-math.pi, math.pi, n)
    else:
        ang = angle + rng.uniform(-spread, spread, n)
    sp = rng.uniform(speed[0], speed[1], n) if isinstance(speed, tuple) else np.full(n, speed)
    vx = np.cos(ang) * sp + vx0
    vh = -(np.sin(ang) * sp - up) + vh0
    vz = rng.uniform(-1, 1, n) * sp * zs
    h0 = floor_y(z) - y
    world.particles.spawn(kind, x + rng.uniform(-1, 1, n), z + rng.uniform(-1.5, 1.5, n), h0 + rng.uniform(-1, 1, n),
                          vx, vz, vh, rng.uniform(life[0], life[1], n), rng.uniform(size[0], size[1], n), rng=rng)


# --------------------------------------------------------------- set pieces
class Effect:
    t0 = 0.0
    t1 = 0.0

    def alive(self, t):
        return self.t0 <= t <= self.t1

    def update(self, t, dt, world):
        pass

    def draw(self, cv, add, t, off, lights, world):
        pass


class HitSpark(Effect):
    """A fighting-game hit flash: a white-hot core, thick spikes going yellow to orange to red at the tips; it plays in
    a few stepped frames like a sprite. 'block' is a cold blue-white star, 'big' adds more and longer spikes."""

    def __init__(self, x, y, t0, size=8.0, kind='hit', seed=0):
        self.x, self.y, self.t0, self.size, self.kind, self.seed = x, y, t0, size * 1.1, kind, seed
        self.t1 = t0 + (0.26 if kind == 'big' else 0.18)
        rng = np.random.default_rng(seed)
        n = 12 if kind == 'big' else 9
        base = rng.uniform(0, math.pi * 2)
        self.rays = [(base + i * 2 * math.pi / n + rng.uniform(-0.18, 0.18),
                      (1.0 if i % 2 == 0 else 0.55) * rng.uniform(0.8, 1.15)) for i in range(n)]
        if kind == 'block':
            self.cols = ramp('ffffff', 'e0f6ff', '9cd6ff', '4a90ec', '2a58b0')
        elif kind == 'clash':
            self.cols = ramp('ffffff', 'fffbe8', 'ffeaa0', 'ffc040', 'e07a10')
        elif kind == 'cut':
            self.cols = ramp('ffffff', 'ffe0d0', 'ff9070', 'e8301c', '8a0c08')
        else:
            self.cols = ramp('ffffff', 'fff6c0', 'ffd448', 'ff8a1c', 'c8280c')

    def draw(self, cv, add, t, off, lights, world):
        k = (t - self.t0) / (self.t1 - self.t0)
        if k < 0 or k > 1:
            return
        ks = math.floor(k * 5) / 5.0                      # stepped, like hand-drawn frames
        x = self.x + off[0]
        y = self.y + off[1]
        R = self.size * (0.55 + 0.75 * ks)
        r0 = int(math.ceil(R + 2))
        x0, y0 = int(round(x)) - r0, int(round(y)) - r0
        ys, xs = np.mgrid[y0:y0 + 2 * r0 + 1, x0:x0 + 2 * r0 + 1]
        dx = xs + 0.5 - x
        dy = ys + 0.5 - y
        d = np.sqrt(dx * dx + dy * dy)
        th = np.arctan2(dy, dx)
        best = np.full(d.shape, 9.0, np.float32)
        for (a, ln) in self.rays:
            L = R * ln
            da = np.abs((th - a + math.pi) % (2 * math.pi) - math.pi)
            half = 0.28 * (1 - d / L) * (1.0 - 0.5 * ks)
            inside = (da < half) & (d < L)
            q = np.where(inside, d / L, 9.0)
            best = np.minimum(best, q)
        core_r = self.size * 0.42 * (1 - ks)
        hole = self.size * 0.5 * ks                       # the burst opens out as it fades
        m = (best < 9.0) & (d >= hole)
        idx = np.clip((best * 4.2 + ks * 1.2).astype(int), 0, 4)
        ok = (ys >= 0) & (ys < cv.shape[0]) & (xs >= 0) & (xs < cv.shape[1])
        sel = m & ok
        cv[ys[sel], xs[sel]] = self.cols[idx[sel]]
        core = (d <= core_r) & ok
        cv[ys[core], xs[core]] = self.cols[0]
        px.glow(add, x, y, self.size * 2.6, self.cols[2], 0.8 * (1 - k))
        lights.append((x, y, self.size * 6, self.cols[2], 1.0 * (1 - k)))


class Ring(Effect):
    def __init__(self, x, y, t0, r0=3, r1=20, dur=0.22, color='ffffff', squash=1.0, thick=1):
        self.x, self.y, self.t0, self.r0, self.r1, self.squash, self.thick = x, y, t0, r0, r1, squash, thick
        self.t1 = t0 + dur
        self.c = hexc(color)

    def draw(self, cv, add, t, off, lights, world):
        k = (t - self.t0) / (self.t1 - self.t0)
        if not 0 <= k <= 1:
            return
        x = self.x + off[0]
        y = self.y + off[1]
        for th in range(self.thick):
            r = self.r0 + (self.r1 - self.r0) * (1 - (1 - k) ** 2) - th
            if r <= 0:
                continue
            n = int(r * 7) + 8
            for i in range(n):
                a = i * 2 * math.pi / n
                xx = int(round(x + math.cos(a) * r))
                yy = int(round(y + math.sin(a) * r * self.squash))
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    add[yy, xx] += self.c * (1 - k) * (0.9 - 0.25 * th)


class IceBall(Effect):
    def __init__(self, a, b, t0, t1, seed=0, za=0.0, zb=0.0):
        self.a, self.b, self.t0, self.t1, self.seed = a, b, t0, t1, seed
        self.za, self.zb = za, zb

    def pos(self, t):
        k = (t - self.t0) / (self.t1 - self.t0)
        return (self.a[0] + (self.b[0] - self.a[0]) * k, self.a[1] + (self.b[1] - self.a[1]) * k + math.sin(k * 9) * 0.6)

    def update(self, t, dt, world):
        x, y = self.pos(t)
        k = min(1.0, max(0.0, (t - self.t0) / (self.t1 - self.t0)))
        z = self.za + (self.zb - self.za) * k
        burst(world, FROST, x, y, z, 6, (5, 30), life=(0.2, 0.5), seed=int(t * 1000) + 3)
        burst(world, ICE, x, y, z, 1, (10, 40), angle=math.pi / 2, spread=0.8, life=(0.6, 1.2), seed=int(t * 1000) + 7)

    def draw(self, cv, add, t, off, lights, world):
        x, y = self.pos(t)
        x = x + off[0]
        y = y + off[1]
        R = 5.0
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                d = math.hypot(dx, dy)
                if d > R:
                    continue
                xx, yy = int(round(x)) + dx, int(round(y)) + dy
                if not (0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]):
                    continue
                ang = math.atan2(dy, dx) + t * 9
                swirl = math.sin(ang * 3 + d * 1.3)
                if d > R - 1:
                    c = ICE_C[3]
                elif d < 1.6:
                    c = ICE_C[0]
                else:
                    c = ICE_C[1] if swirl > 0.2 else ICE_C[2]
                cv[yy, xx] = c
        px.glow(add, x, y, 22, hexc('5ab4ff'), 0.6)
        lights.append((x, y, 44, hexc('7ac8ff'), 0.9))


class IceBlock(Effect):
    """A block of ice around a frozen fighter; cracks spread before it bursts."""
    zsort = True
    z = 0.0

    def __init__(self, box, t0, t_crack, t1, seed=0):
        self.box, self.t0, self.tc, self.t1, self.seed = box, t0, t_crack, t1, seed
        x0, y0, x1, y1 = box
        rng = np.random.default_rng(seed)
        self.poly = [(x0 - 2, y1), (x0 - 4, y0 + (y1 - y0) * 0.55), (x0 + 1, y0 + 2), (x0 + (x1 - x0) * 0.45, y0 - 4),
                     (x1 + 2, y0 + 3), (x1 + 4, y0 + (y1 - y0) * 0.5), (x1 + 1, y1)]
        self.cracks = []
        cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
        for i in range(7):
            pts = [(cxm + rng.uniform(-3, 3), cym + rng.uniform(-6, 6))]
            ang = rng.uniform(0, 2 * math.pi)
            for s in range(14):
                ang += rng.uniform(-0.6, 0.6)
                pts.append((pts[-1][0] + math.cos(ang) * 1.6, pts[-1][1] + math.sin(ang) * 1.6))
            self.cracks.append(pts)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = self.box
        ox = int(math.floor(x0 - 8 + off[0]))
        oy = int(math.floor(y0 - 8 + off[1]))
        w = int(x1 - x0 + 18)
        h = int(y1 - y0 + 18)
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        poly = [(px_ + off[0] - ox, py_ + off[1] - oy) for px_, py_ in self.poly]
        d.polygon(poly, fill=255)
        m = np.asarray(im) > 0
        grow = min(1.0, (t - self.t0) / 0.12)
        yy = np.arange(h)[:, None]
        m = m & (yy >= h - h * grow - 1)
        sub = cv[max(0, oy):oy + h, max(0, ox):ox + w]
        mm = m[max(0, -oy):max(0, -oy) + sub.shape[0], max(0, -ox):max(0, -ox) + sub.shape[1]]
        tint = np.array([0.62, 0.86, 1.0], np.float32)
        sub[mm] = sub[mm] * 0.45 + tint * 0.42
        xx = np.arange(w)[None, :]
        facet = m & ((((xx * 3 + yy * 2) // 7) % 3) == 0)
        ff = facet[max(0, -oy):max(0, -oy) + sub.shape[0], max(0, -ox):max(0, -ox) + sub.shape[1]]
        sub[ff] = sub[ff] * 0.85 + 0.15
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(m) & ~m
        ee = edge[max(0, -oy):max(0, -oy) + sub.shape[0], max(0, -ox):max(0, -ox) + sub.shape[1]]
        sub[ee] = hexc('e8fbff')
        # cracks spreading
        if t > self.tc:
            k = min(1.0, (t - self.tc) / max(0.05, self.t1 - self.tc))
            for pts in self.cracks:
                for (px_, py_) in pts[:int(len(pts) * k)]:
                    xi = int(round(px_ + off[0]))
                    yi = int(round(py_ + off[1]))
                    if 0 <= xi < cv.shape[1] and 0 <= yi < cv.shape[0]:
                        cv[yi, xi] = [1.0, 1.0, 1.0]
        lights.append(((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], 50, hexc('8ad0ff'), 0.6))


class Spear(Effect):
    """Scorpion's kunai on a rope: thrown, it bites into the target point; the rope stays taut while he hauls."""

    def __init__(self, thrower, target, t0, t_hit, t_out, t1, sag=0.0):
        self.thrower, self.target = thrower, target
        self.t0, self.t_hit, self.t_out, self.t1 = t0, t_hit, t_out, t1
        self.hit_point = None

    def hand(self, world):
        j = world.joints[self.thrower]
        return j['NW']

    def target_point(self, world):
        j = world.joints[self.target]
        s = j['FS']
        c = j['C']
        return (s[0] * 0.75 + c[0] * 0.25, s[1] * 0.8 + c[1] * 0.2 + 1)

    def tip(self, t, world):
        h = self.hand(world)
        tp = self.target_point(world)
        if t < self.t_hit:
            k = (t - self.t0) / (self.t_hit - self.t0)
            k = k ** 0.8
            return (h[0] + (tp[0] - h[0]) * k, h[1] + (tp[1] - h[1]) * k - math.sin(k * math.pi) * 3), False
        if t < self.t_out:
            return tp, True
        k = min(1.0, (t - self.t_out) / max(0.05, self.t1 - self.t_out))
        return (tp[0] + (h[0] - tp[0]) * k, tp[1] + (h[1] - tp[1]) * k), False

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        h = self.hand(world)
        tip, stuck = self.tip(t, world)
        dx, dy = tip[0] - h[0], tip[1] - h[1]
        L = math.hypot(dx, dy) + 1e-6
        n = int(L) + 2
        wave = 0.0
        if t < self.t_hit:
            wave = 2.5 * (1 - (t - self.t0) / (self.t_hit - self.t0))
        slack = 0.0
        pts = []
        for i in range(n + 1):
            k = i / n
            x = h[0] + dx * k
            y = h[1] + dy * k + math.sin(k * math.pi * 3 + t * 40) * wave * math.sin(k * math.pi) + slack * math.sin(k * math.pi)
            pts.append((x + off[0], y + off[1]))
        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < cv.shape[1] and 1 <= yi < cv.shape[0]:
                cv[yi, xi] = hexc('3a2a1a')
                cv[yi - 1, xi] = cv[yi - 1, xi] * 0.4 + hexc('a88a5a') * 0.6
        # the kunai: a leaf blade along the rope, a ring at its back
        ux, uy = dx / L, dy / L
        if stuck:
            depth = 2.5
        else:
            depth = 0.0
        tx, ty = tip[0] + off[0] + ux * 1.5, tip[1] + off[1] + uy * 1.5
        for s in range(-1, 7):
            if s > 7 - depth:
                break
            wdt = [0, 1, 1.6, 1.6, 1.2, 0.8, 0.4, 0][min(7, s + 1)]
            cxp, cyp = tx - ux * (s - 3), ty - uy * (s - 3)
            for o in np.arange(-wdt, wdt + 0.01, 0.5):
                xi = int(round(cxp - uy * o))
                yi = int(round(cyp + ux * o))
                if 0 <= xi < cv.shape[1] and 0 <= yi < cv.shape[0]:
                    cv[yi, xi] = hexc('dfe6ee') if o < 0 else hexc('7c8894')
        rx, ry = int(round(tx - ux * 6)), int(round(ty - uy * 6))
        for (ox, oy) in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            if 0 <= rx + ox < cv.shape[1] and 0 <= ry + oy < cv.shape[0]:
                cv[ry + oy, rx + ox] = hexc('5a6068')
        if not stuck and t < self.t_hit:
            add[max(0, min(cv.shape[0] - 1, int(ty))), max(0, min(cv.shape[1] - 1, int(tx)))] += 0.6


class Emitter(Effect):
    """Calls fn(world, t, dt) every frame between t0 and t1."""

    def __init__(self, t0, t1, fn):
        self.t0, self.t1, self.fn = t0, t1, fn

    def update(self, t, dt, world):
        if self.alive(t):
            self.fn(world, t, dt)


class Glow(Effect):
    def __init__(self, t0, t1, fn):
        self.t0, self.t1, self.fn = t0, t1, fn

    def draw(self, cv, add, t, off, lights, world):
        if self.alive(t):
            self.fn(cv, add, t, off, lights, world)


class Shield(Effect):
    """The shield in flight: it spins flat through its waypoints (each a ricochet), with a pale streak behind it.
    Waypoints: [(t, (x, z, h)) or (t, fn(world) -> (x, z, h))]; the last is usually his hand."""

    def __init__(self, points, seed=0, colors=None):
        self.points = points
        self.colors = colors
        self.t0, self.t1 = points[0][0], points[-1][0]
        self.seed = seed
        self.hist = []

    def where(self, t, world):
        pts = self.points
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                pa = pa(world) if callable(pa) else pa
                pb = pb(world) if callable(pb) else pb
                u = (t - ta) / max(1e-6, tb - ta)
                return tuple(pa[i] + (pb[i] - pa[i]) * u for i in range(3))
        p = pts[-1][1]
        return p(world) if callable(p) else p

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        import body
        p = self.where(t, world)
        x, y = p[0], floor_y(p[1]) - p[2]
        self.hist.append((x, y))
        del self.hist[:-5]
        for k, (gx, gy) in enumerate(self.hist[:-1]):
            al = 0.14 + 0.1 * k
            for q in px.line_points(gx + off[0], gy + off[1], x + off[0], y + off[1]):
                if 0 <= q[0] < cv.shape[1] and 0 <= q[1] < cv.shape[0]:
                    add[q[1], q[0]] += np.array([0.5, 0.55, 0.7], np.float32) * al
        R = 6.6
        x0, y0 = int(math.floor(x + off[0] - R - 2)), int(math.floor(y + off[1] - R - 2))
        ys, xs = np.mgrid[y0:y0 + int(2 * R + 5), x0:x0 + int(2 * R + 5)]
        inside, col = body.shield_paint(xs + 0.5, ys + 0.5, x + off[0], y + off[1], R, R * 0.62, rot=t * 22.0,
                                        colors=self.colors)
        rr = np.sqrt(((xs + 0.5 - x - off[0]) / (R + 1)) ** 2 + ((ys + 0.5 - y - off[1]) / (R * 0.62 + 1)) ** 2)
        inb = (ys >= 0) & (ys < cv.shape[0]) & (xs >= 0) & (xs < cv.shape[1])
        ring = (rr <= 1.0) & ~inside & inb
        cv[ys[ring], xs[ring]] = hexc('07060a')
        ok = inside & inb
        cv[ys[ok], xs[ok]] = col[ok]
        lights.append((x + off[0], y + off[1], 24, hexc('c8d8ff'), 0.35))


class Arc(Effect):
    """An electric arc: a jagged bolt between two stage points, redrawn every frame."""

    def __init__(self, a, b, t0, t1, seed=0, width=1):
        self.a, self.b, self.t0, self.t1, self.seed, self.width = a, b, t0, t1, seed, width

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        a = self.a(world) if callable(self.a) else self.a
        b = self.b(world) if callable(self.b) else self.b
        rng = np.random.default_rng(int(t * 60) + self.seed)
        if rng.uniform() < 0.2:
            return
        n = 7
        pts = [a]
        for i in range(1, n):
            u = i / n
            pts.append((a[0] + (b[0] - a[0]) * u + rng.uniform(-4, 4), a[1] + (b[1] - a[1]) * u + rng.uniform(-4, 4)))
        pts.append(b)
        for (p0, p1) in zip(pts, pts[1:]):
            for q in px.line_points(p0[0] + off[0], p0[1] + off[1], p1[0] + off[0], p1[1] + off[1]):
                if 0 <= q[0] < cv.shape[1] and 0 <= q[1] < cv.shape[0]:
                    cv[q[1], q[0]] = hexc('f0f8ff')
                    if self.width > 1 and q[1] + 1 < cv.shape[0]:
                        add[q[1] + 1, q[0]] += hexc('6ab0ff') * 0.6
        mx, my = (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]
        px.glow(add, mx, my, 18, hexc('6ab0ff'), 0.6)
        lights.append((mx, my, 70, hexc('9ac8ff'), 1.1))


class ShieldRoll(Effect):
    """The dropped shield rolls on its rim along the road, wobbles in smaller and smaller circles and falls flat."""

    def __init__(self, a, b, t0, t_flat, t1, h=0.0, colors=None):
        self.a, self.b, self.t0, self.tf, self.t1, self.h = a, b, t0, t_flat, t1, h
        self.colors = colors

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        import body
        u = min(1.0, (t - self.t0) / max(1e-6, self.tf - self.t0))
        e = 1 - (1 - u) ** 2
        x = self.a[0] + (self.b[0] - self.a[0]) * e
        z = self.a[1] + (self.b[1] - self.a[1]) * e
        R = 6.6
        if t < self.tf:
            wob = math.sin(t * (14 + 30 * u)) * (0.15 + 0.55 * u)
            rx = R * (0.3 + 0.25 * abs(wob))
            ry = R
            cy = floor_y(z) - self.h - R
        else:
            k = min(1.0, (t - self.tf) / 0.25)
            bounce = math.sin(min(1.0, (t - self.tf) / 0.5) * math.pi * 3) * (1 - k) * 0.15
            rx = R * (0.55 + 0.45 * k)
            ry = R * (1.0 - 0.38 * k + bounce)
            cy = floor_y(z) - self.h - ry
        X = x + off[0]
        Y = cy + off[1]
        x0, y0 = int(math.floor(X - R - 2)), int(math.floor(Y - R - 2))
        ys, xs = np.mgrid[y0:y0 + int(2 * R + 5), x0:x0 + int(2 * R + 5)]
        inside, col = body.shield_paint(xs + 0.5, ys + 0.5, X, Y, rx, ry, rot=0.0, dent=0.6, colors=self.colors)
        rr = np.sqrt(((xs + 0.5 - X) / (rx + 1)) ** 2 + ((ys + 0.5 - Y) / (ry + 1)) ** 2)
        inb = (ys >= 0) & (ys < cv.shape[0]) & (xs >= 0) & (xs < cv.shape[1])
        ring = (rr <= 1.0) & ~inside & inb
        cv[ys[ring], xs[ring]] = hexc('07060a')
        ok = inside & inb
        cv[ys[ok], xs[ok]] = col[ok]
