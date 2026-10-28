"""Magic and set pieces shared by every fighter (by element), and weapons out of the hand.

Fire: the teleport's fire, hellfire columns out of the floor (a wall of them), the fire dash's trail, a fire tornado.
Ice: the clone, the slide's frozen track, the shower from above, the freezing ground wave, the wall, giant spikes, the
whole floor freezing. Weapons in flight and stuck in the floor; blade clashes. Sizes are parameters: the beats scale
them by the caster's power tier. The stage's own breakables (roof tiles, a balustrade, a car) live in its stage module.
Everything lives in stage coordinates; effects with `zsort` are drawn in the depth pass with the fighters.
"""
import math
import numpy as np

import px
import fx
import body
import rig
from px import hexc
from proj import floor_y, proj

ICE_COAT = hexc('8cc4ec')
SCORCH = hexc('140c08')


def _rng(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ fire
def fire_puff(world, x, z, y, seed, big=1.0):
    """The teleport's fire: a column of flame swallowing (or spitting out) Scorpion, embers and smoke."""
    sy = floor_y(z) - y
    fx.burst(world, fx.FLAME, x, sy - 12, z, int(50 * big), (30, 110), angle=-math.pi / 2, spread=1.1,
             life=(0.25, 0.55), size=(1.0, 2.4), seed=seed, zs=0.5, up=40)
    fx.burst(world, fx.FLAME, x, sy - 2, z, int(24 * big), (40, 90), angle=-math.pi / 2, spread=0.25,
             life=(0.3, 0.5), size=(1.4, 2.6), seed=seed + 1, zs=0.2, up=60)
    fx.burst(world, fx.EMBER, x, sy - 14, z, int(26 * big), (30, 120), life=(0.6, 1.3), seed=seed + 2)
    fx.burst(world, fx.SMOKE, x, sy - 20, z, int(10 * big), (10, 40), life=(0.6, 1.2), size=(1.4, 2.4), seed=seed + 3)
    world.effects.append(fx.Ring(x, sy - 14, world.t, 3, 22 * big, 0.25, 'ffb050'))
    world.effects.append(FireGlow(x, z, y, world.t, world.t + 0.5, 70 * big))


class FireGlow(fx.Effect):
    def __init__(self, x, z, y, t0, t1, r):
        self.x, self.z, self.y, self.t0, self.t1, self.r = x, z, y, t0, t1, r

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        k = 1 - (t - self.t0) / (self.t1 - self.t0)
        x, y = self.x + off[0], floor_y(self.z) - self.y - 14 + off[1]
        px.glow(add, x, y, self.r * 0.4, hexc('ff7a20'), 0.6 * k)
        lights.append((x, y, self.r, hexc('ff8a3a'), 1.4 * k))


class HellColumn(fx.Effect):
    """A pillar of hellfire out of the stone: a glowing crack first, then the roaring column; a scorch stays."""
    zsort = True

    def __init__(self, x, z, t0, g=0.0, height=74.0, warn=0.22, dur=0.75, seed=0, width=7.0):
        self.x, self.z, self.g, self.t0, self.width = x, z, g, t0, width
        self.warn, self.dur, self.height, self.seed = warn, dur, height, seed
        self.t1 = t0 + warn + dur
        self.burnt = False

    def update(self, t, dt, world):
        k = t - self.t0
        sy = floor_y(self.z) - self.g
        rng = _rng(int(t * 1000) + self.seed)
        if k < self.warn:
            if rng.uniform() < 0.6:
                world.particles.spawn(fx.EMBER, [self.x + rng.uniform(-5, 5)], [self.z], [self.g + 1], [0.0], [0.0],
                                      [rng.uniform(20, 60)], [0.5], [1.0], rng=rng)
            return
        if not self.burnt:
            self.burnt = True
            world.cue('hellfire', x=self.x)
            for _ in range(10):
                world.stage.stamp(self.x + rng.uniform(-8, 8), self.z + rng.uniform(-5, 5), SCORCH,
                                  r=rng.uniform(1.5, 3.0), alpha=0.85, seed=int(rng.integers(1e6)), h=self.g)
            world.shake(t, 2.2, 0.25)
        n = 5 + int(self.height / 40)
        world.particles.spawn(fx.FLAME, self.x + rng.uniform(-5, 5, n), self.z + rng.uniform(-2, 2, n),
                              self.g + rng.uniform(0, self.height * 0.8, n), rng.uniform(-10, 10, n), np.zeros(n),
                              rng.uniform(40, 90, n), rng.uniform(0.2, 0.4, n), rng.uniform(1.2, 2.2, n), rng=rng)
        if rng.uniform() < 0.5:
            world.particles.spawn(fx.EMBER, [self.x + rng.uniform(-4, 4)], [self.z], [self.g + self.height],
                                  [rng.uniform(-30, 30)], [0.0], [rng.uniform(30, 70)], [0.9], [1.0], rng=rng)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        k = t - self.t0
        x = self.x + off[0]
        yb = floor_y(self.z) - self.g + off[1]
        if k < self.warn:
            w_ = 3 + 9 * k / self.warn
            for dx in range(-int(w_), int(w_) + 1):
                xx, yy = int(round(x + dx)), int(round(yb + (dx % 3 == 0)))
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    cv[yy, xx] = hexc('ffb040') if abs(dx) < w_ * 0.6 else hexc('ff5a10')
            px.glow(add, x, yb, 14, hexc('ff6a10'), 0.6)
            lights.append((x, yb, 40, hexc('ff7a20'), 0.8))
            return
        q = (k - self.warn) / self.dur
        grow = min(1.0, q / 0.18)
        fade = 1.0 if q < 0.7 else max(0.0, 1 - (q - 0.7) / 0.3)
        hgt = self.height * grow * (0.6 + 0.4 * fade)
        wid = self.width * (0.5 + 0.5 * fade)
        for row in range(int(hgt)):
            kk = row / max(1.0, hgt)
            half = wid * (1.0 - 0.45 * kk) * (0.85 + 0.25 * math.sin(t * 23 + row * 0.5))
            cxo = math.sin(t * 13 + row * 0.25) * 2.0 * kk
            yy = int(round(yb - row))
            if not 0 <= yy < cv.shape[0]:
                continue
            for dx in range(-int(half) - 1, int(half) + 2):
                d = abs(dx - cxo) / max(half, 0.5)
                if d > 1:
                    continue
                n = (hash((int(t * 20), row, dx, self.seed)) & 255) / 255.0
                if kk > 0.8 and n > 1.8 - kk:
                    continue
                heat = (1 - d) * (1 - kk * 0.5) * fade
                c = hexc('fffbe0') if heat > 0.6 else hexc('ffd050') if heat > 0.38 else hexc('ff8a1c') if heat > 0.18 \
                    else hexc('c8320c')
                xx = int(round(x + dx))
                if 0 <= xx < cv.shape[1]:
                    cv[yy, xx] = c
        px.glow(add, x, yb - hgt * 0.5, 26, hexc('ff7a20'), 0.7 * fade)
        lights.append((x, yb - hgt * 0.4, 90, hexc('ff8a3a'), 1.5 * fade))


def fire_trail(world, who, t0, t1, scorch=True):
    """The fire dash: flame pours off him and the stone he passes is scorched."""
    def fn(w, t, dt):
        f = w.f[who]
        if w.is_hidden(who, t):
            return
        j = w.joints[who]
        rng = _rng(int(t * 1000) + 41)
        n = 9
        pts = [j['C'], j['P'], j['FS'], j['NS'], j['Hc']]
        for _ in range(n):
            a = pts[rng.integers(len(pts))]
            w.particles.spawn(fx.FLAME, [a[0] + rng.uniform(-3, 3)], [f.z + rng.uniform(-1, 1)],
                              [floor_y(f.z) - a[1] + rng.uniform(-2, 2)], [-f.facing * rng.uniform(20, 70)], [0.0],
                              [rng.uniform(10, 40)], [rng.uniform(0.2, 0.4)], [rng.uniform(1.2, 2.2)], rng=rng)
        if scorch and rng.uniform() < 0.7:
            g = w.stage.ground(f.x, f.z, t, f.y)
            if g > -900 and f.y - g < 6:
                w.stage.stamp(f.x + rng.uniform(-3, 3), f.z + rng.uniform(-2, 2), SCORCH, r=rng.uniform(1.2, 2.2),
                              alpha=0.6, seed=int(rng.integers(1e6)), h=g)
    world.effects.append(fx.Emitter(t0, t1, fn))

    def glow(cv, add, t, off, lights, w):
        j = w.joints[who]
        lights.append((j['C'][0] + off[0], j['C'][1] + off[1], 70, hexc('ff8a3a'), 1.2))
    world.effects.append(fx.Glow(t0, t1, glow))


# ------------------------------------------------------------------ ice
def ice_coat(world, x, z, r, seed, h=0.0, k=0.55):
    world.stage.tint(x, z, r, ICE_COAT, k, seed=seed, h=h)


def slide_track(world, who, t0, t1):
    """Sub-Zero's slide leaves a frozen track behind his foot."""
    def fn(w, t, dt):
        f = w.f[who]
        g = w.stage.ground(f.x, f.z, t, f.y)
        if g > -900:
            ice_coat(w, f.x - f.facing * 4, f.z, 5.0, int(t * 997), h=g, k=0.5)
            rng = _rng(int(t * 1000) + 5)
            fx.burst(w, fx.FROST, f.x + f.facing * 10, floor_y(f.z) - g - 2, f.z, 3, (10, 40), life=(0.2, 0.5),
                     seed=int(rng.integers(1e6)))
            fx.burst(w, fx.ICE, f.x + f.facing * 12, floor_y(f.z) - g - 2, f.z, 1, (30, 70), angle=-math.pi / 2,
                     spread=1.0, life=(0.4, 0.8), seed=int(rng.integers(1e6)))
    world.effects.append(fx.Emitter(t0, t1, fn))


class Icicle(fx.Effect):
    """One shard of the ice shower: falls from the sky and bursts on what it hits (a coat of ice stays)."""

    def __init__(self, x, z, t0, t_hit, g, seed, h0=130.0):
        self.x, self.z, self.t0, self.t_hit, self.g, self.seed = x, z, t0, t_hit, g, seed
        self.t1 = t_hit
        self.h0 = h0
        self.done = False

    def h(self, t):
        k = (t - self.t0) / (self.t_hit - self.t0)
        return self.h0 + (self.g - self.h0) * k * k

    def update(self, t, dt, world):
        if t >= self.t_hit - 1e-6 and not self.done:
            self.done = True

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        h = self.h(t)
        x = self.x + off[0]
        y = floor_y(self.z) - h + off[1]
        for k in range(10):
            w_ = 1 if k < 7 else 0
            for dx in range(-w_, w_ + 1):
                xx, yy = int(round(x + dx)), int(round(y - k))
                if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                    cv[yy, xx] = fx.ICE_C[0] if dx < 0 else fx.ICE_C[2] if k < 7 else fx.ICE_C[1]
        for k in range(1, 6):
            yy = int(round(y - 10 - k * 2))
            xx = int(round(x))
            if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                add[yy, xx] += hexc('5ab4ff') * (0.5 - k * 0.08)
        lights.append((x, y, 18, hexc('7ac8ff'), 0.4))


def ice_shower(world, cx, cz, t0, n, spread, seed, hits=(), dur=1.2, g_fn=None):
    """The burst over the foe, then n icicles falling round (cx, cz) over `dur`; `hits` are (t, x, z) that must land
    exactly there (on the foe)."""
    rng = _rng(seed)

    def burst_sky(w):
        x, y = cx, floor_y(cz) - 128
        fx.burst(w, fx.ICE, x, y, cz, 40, (30, 120), life=(0.4, 0.9), size=(1, 2), seed=seed)
        fx.burst(w, fx.FROST, x, y, cz, 60, (20, 90), life=(0.4, 1.0), seed=seed + 1)
        w.effects.append(fx.Ring(x, y, w.t, 4, 30, 0.3, 'bfe8ff'))
        w.cue('ice_burst', x=x)
    world.at(t0, burst_sky)
    shards = []
    for i in range(n):
        tl = t0 + 0.1 + rng.uniform(0, dur)
        x = cx + rng.uniform(-spread, spread)
        z = cz + rng.uniform(-spread * 0.5, spread * 0.5)
        shards.append((tl, x, z))
    shards += list(hits)
    for (tl, x, z) in shards:
        g = g_fn(x, z, tl) if g_fn else world.stage.ground(x, z, tl)
        if g < -900:
            g = -200.0
        fall = 0.38
        ic = Icicle(x, z, tl - fall, tl, g, seed + int(x * 7 + z))
        world.effects.append(ic)

        def land(w, x=x, z=z, g=g, sd=int(x * 13 + z * 7)):
            if g < -150:
                return
            sy = floor_y(z) - g
            fx.burst(w, fx.ICE, x, sy - 1, z, 9, (30, 90), angle=-math.pi / 2, spread=1.3, life=(0.4, 1.0),
                     size=(1, 1.8), seed=sd)
            fx.burst(w, fx.FROST, x, sy - 2, z, 8, (10, 40), life=(0.3, 0.6), seed=sd + 1)
            ice_coat(w, x, z, 4.0, sd, h=g, k=0.6)
            w.cue('icicle', x=x)
        world.at(tl, land)
    return shards


class WaveSpike(fx.Effect):
    zsort = True

    def __init__(self, x, z, g, t0, life, size, seed):
        self.x, self.z, self.g, self.t0, self.t1, self.size, self.seed = x, z, g, t0, t0 + life, size, seed
        rng = _rng(seed)
        self.blades = [(rng.uniform(-3, 3), rng.uniform(-0.5, 0.5), rng.uniform(0.6, 1.0)) for _ in range(3)]

    def update(self, t, dt, world):
        if t + dt >= self.t1 and not getattr(self, 'gone', False):
            self.gone = True
            fx.burst(world, fx.ICE, self.x, floor_y(self.z) - self.g - 3, self.z, 4, (20, 60), angle=-math.pi / 2,
                     spread=1.2, life=(0.4, 0.8), seed=self.seed)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        k = min(1.0, (t - self.t0) / 0.08)
        x0 = self.x + off[0]
        yb = floor_y(self.z) - self.g + off[1]
        for (dx, lean, hk) in self.blades:
            hh = self.size * hk * k
            for r in range(int(hh)):
                q = r / max(1.0, hh)
                half = 1.6 * (1 - q)
                cx = x0 + dx + lean * r
                for o in range(-int(half + 0.5), int(half + 0.5) + 1):
                    xx, yy = int(round(cx + o)), int(round(yb - r))
                    if 0 <= xx < cv.shape[1] and 0 <= yy < cv.shape[0]:
                        cv[yy, xx] = fx.ICE_C[0] if o < 0 else fx.ICE_C[2] if q < 0.7 else fx.ICE_C[1]


def ground_wave(world, x0, z0, x1, z1, t0, dur, seed, g=0.0, spike=11.0):
    """The freezing wave: a front of ice spikes runs along the stone from (x0, z0) to (x1, z1); ice stays behind."""
    rng = _rng(seed)
    n = int(math.hypot(x1 - x0, z1 - z0) / 5)
    for i in range(n):
        k = i / max(1, n - 1)
        x = x0 + (x1 - x0) * k + rng.uniform(-2, 2)
        z = z0 + (z1 - z0) * k + rng.uniform(-3, 3)
        tt = t0 + dur * k
        world.effects.append(WaveSpike(x, z, g, tt, 0.9 + rng.uniform(0, 0.4), spike * rng.uniform(0.7, 1.2),
                                       seed + i))

        def coat(w, x=x, z=z, sd=seed + i * 3):
            ice_coat(w, x, z, 6.0, sd, h=g, k=0.6)
            fx.burst(w, fx.FROST, x, floor_y(z) - g - 2, z, 4, (10, 40), life=(0.3, 0.7), seed=sd)
        world.at(tt, coat)
    world.cue('ground_wave', t=t0, dur=dur, x=x0, to_x=x1)


class IceWall(fx.Effect):
    """A wall of ice raised across the fire's path: a slab (x0..x1 thick, z0..z1 across) growing out of the floor; it
    takes the fire, cracks, and shatters."""
    zsort = True

    def __init__(self, x0, x1, z0, z1, t0, t_crack, t_break, g=0.0, height=42.0, seed=0):
        self.x0, self.x1, self.z0, self.z1, self.g = x0, x1, z0, z1, g
        self.z = z1
        self.t0, self.tc, self.t1, self.height, self.seed = t0, t_crack, t_break, height, seed
        rng = _rng(seed)
        w_ = int(x1 - x0) + 1
        self.top = [height * rng.uniform(0.85, 1.05) for _ in range(w_)]
        self.cracks = []
        for i in range(5):
            pts = [(rng.uniform(x0 + 2, x1 - 2), rng.uniform(height * 0.3, height * 0.85))]
            ang = rng.uniform(0, 2 * math.pi)
            for s_ in range(18):
                ang += rng.uniform(-0.6, 0.6)
                pts.append((pts[-1][0] + math.cos(ang) * 1.2, pts[-1][1] + math.sin(ang) * 1.6))
            self.cracks.append(pts)

    def update(self, t, dt, world):
        if t + dt >= self.t1 and not getattr(self, 'gone', False):
            self.gone = True
            rng = _rng(self.seed + 9)
            n = 110
            world.particles.spawn(fx.ICE, rng.uniform(self.x0, self.x1, n), rng.uniform(self.z0, self.z1, n),
                                  self.g + rng.uniform(2, self.height, n), rng.uniform(-110, 110, n),
                                  rng.uniform(-50, 50, n), rng.uniform(20, 130, n), rng.uniform(0.8, 1.6, n),
                                  rng.uniform(1.0, 2.8, n), rng=rng)
            cx = (self.x0 + self.x1) / 2
            fx.burst(world, fx.FROST, cx, floor_y(self.z1) - self.g - self.height / 2, self.z1, 60, (30, 130),
                     life=(0.4, 0.9), seed=self.seed + 3)
            for k in range(10):
                ice_coat(world, rng.uniform(self.x0 - 10, self.x1 + 10), rng.uniform(self.z0, self.z1), 5.0,
                         self.seed + k, h=self.g)
            world.cue('ice_shatter', x=cx)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        grow = min(1.0, (t - self.t0) / 0.3)
        grow = 1 - (1 - grow) ** 3
        tint = np.array([0.62, 0.86, 1.0], np.float32)
        top_c = np.array([0.82, 0.95, 1.0], np.float32)
        span = (self.z1 - self.z0) * 0.6
        for i, xx in enumerate(range(int(self.x0), int(self.x1) + 1)):
            hh = self.top[i] * grow
            X = int(round(xx + off[0]))
            if not 0 <= X < cv.shape[1]:
                continue
            yf = floor_y(self.z1) - self.g + off[1]           # the front face's foot
            y_top_front = yf - hh
            y_top_back = y_top_front - span
            ya, yb_ = int(round(y_top_back)), int(round(yf))
            ya_c, yb_c = max(0, ya), min(cv.shape[0], yb_ + 1)
            if yb_c <= ya_c:
                continue
            col = cv[ya_c:yb_c, X]
            rows = np.arange(ya_c, yb_c)
            front = rows >= y_top_front
            facet = (((rows * 2 + xx * 3) // 9) % 3) == 0
            c = col * 0.42 + tint * 0.46
            c[~front] = col[~front] * 0.3 + top_c * 0.62
            c[facet & front] = c[facet & front] * 0.85 + 0.15
            cv[ya_c:yb_c, X] = c
            for yy in (int(round(y_top_front)), ya):
                if 0 <= yy < cv.shape[0]:
                    cv[yy, X] = hexc('f4fdff')
            if xx in (int(self.x0), int(self.x1)):
                cv[ya_c:yb_c, X] = hexc('d8f4ff')
        if t > self.tc:
            k = min(1.0, (t - self.tc) / max(0.05, self.t1 - self.tc))
            yf = floor_y(self.z1) - self.g + off[1]
            for pts in self.cracks:
                for (cx_, ch) in pts[:int(len(pts) * k)]:
                    xi = int(round(cx_ + off[0]))
                    yi = int(round(yf - ch * grow))
                    if 0 <= xi < cv.shape[1] and 0 <= yi < cv.shape[0]:
                        cv[yi, xi] = [1.0, 1.0, 1.0]
        cx = (self.x0 + self.x1) / 2 + off[0]
        lights.append((cx, floor_y(self.z1) - self.g - self.height / 2 + off[1], 70, hexc('8ad0ff'), 0.7))


class Clone(fx.Effect):
    """An ice clone: the caster's pose of that instant in solid ice; whoever strikes it freezes, then it shatters."""
    zsort = True

    def __init__(self, world, who, t0, t1, seed=0):
        self.t0, self.t1, self.seed, self.who = t0, t1, seed, who
        self.joints = None
        self.look = world.f[who].look
        self.z = world.f[who].z_at(t0)
        self.st = body.State()
        self.st.frozen = 1.0
        self.st.alpha = 0.88

    def update(self, t, dt, world):
        if self.joints is None:
            j = world.joints[self.who]
            self.joints = dict(j)
            self.facing = j['facing']
            self.scale = j['scale']
            self.z = world.f[self.who].z
            self.st.weps = dict(world.f[self.who].st.weps)
            fx.burst(world, fx.FROST, j['C'][0], j['C'][1], self.z, 40, (20, 70), life=(0.3, 0.8), seed=self.seed)
        rng = _rng(int(t * 1000) + self.seed)
        if rng.uniform() < 0.5:
            j = self.joints
            fx.burst(world, fx.FROST, j['C'][0] + rng.uniform(-6, 6), j['C'][1] + rng.uniform(-10, 10), self.z, 2,
                     (5, 20), life=(0.3, 0.6), seed=int(rng.integers(1e6)))
        if t + dt >= self.t1 and not getattr(self, 'gone', False):
            self.gone = True
            j = self.joints
            cx, cy = j['C']
            fx.burst(world, fx.ICE, cx, cy + 4, self.z, 70, (40, 170), life=(0.8, 1.6), size=(1, 2.6), seed=self.seed + 1,
                     up=30, zs=0.7)
            fx.burst(world, fx.FROST, cx, cy, self.z, 50, (20, 100), life=(0.3, 0.8), seed=self.seed + 2)
            world.effects.append(fx.Ring(cx, cy, t, 4, 30, 0.28, 'd8f4ff'))
            world.cue('ice_shatter', x=cx)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t) or self.joints is None:
            return
        jc = {k: ((v[0] + off[0], v[1] + off[1]) if isinstance(v, tuple) and len(v) == 2 and k != 'up' else v)
              for k, v in self.joints.items()}
        body.SC[0] = self.scale
        rgba, bx, by = body.draw(jc, self.look, self.st, self.facing, pad=10)
        sheen = ((np.arange(rgba.shape[0])[:, None] + np.arange(rgba.shape[1])[None, :] + int(t * 30)) % 11) == 0
        rgba[sheen & (rgba[..., 3] > 0), :3] = np.minimum(1.0, rgba[sheen & (rgba[..., 3] > 0), :3] + 0.35)
        px.blit(cv, rgba, bx, by)
        lights.append((jc['C'][0], jc['C'][1], 40, hexc('8ad0ff'), 0.5))


# ------------------------------------------------------------------ weapons loose
_PLAIN = body.Look('plain', {'suit': body.STEEL}, {}, {}, {}, {})


def draw_weapon(cv, kind, hand, ang, scale, grow=1.0, alpha=1.0):
    """A weapon on its own (flying, stuck in the floor): hand point and its pointing angle (screen radians)."""
    off = body.WEAPONS[kind]['off']
    a = ang - off
    E = (hand[0] - math.cos(a) * 8.0, hand[1] - math.sin(a) * 8.0)
    W_ = (hand[0] - math.cos(a) * 1.2 * scale, hand[1] - math.sin(a) * 1.2 * scale)
    j = {'NW': W_, 'NE': E, 'nwa': 0.0}
    reach = 30 * scale
    x0, y0 = int(min(hand[0], hand[0] + math.cos(ang) * reach) - 10), int(min(hand[1], hand[1] + math.sin(ang) * reach) - 10)
    x1, y1 = int(max(hand[0], hand[0] + math.cos(ang) * reach) + 10), int(max(hand[1], hand[1] + math.sin(ang) * reach) + 10)
    c = body.Canvas(x0, y0, x1 - x0, y1 - y0)
    body.SC[0] = scale
    st = body.State()
    items = body.weapon(c, _PLAIN, st, j, 'n', kind, grow, 1)
    body.group(c, _PLAIN, items)
    rgba = c.rgba()
    rgba[..., 3] *= alpha
    px.blit(cv, rgba, x0, y0)


class LooseWeapon(fx.Effect):
    """A weapon out of the hand: flies spinning from a to b (an arc of height h) and then sticks in the floor at a
    slant, until picked up (t_gone)."""
    zsort = True

    def __init__(self, kind, a, b, t0, t_land, t_gone, h=30.0, spins=2.0, stick_ang=None, za=0.0, zb=0.0, seed=0,
                 a0=0.0):
        self.kind, self.a, self.b, self.t0, self.tl, self.t1 = kind, a, b, t0, t_land, t_gone
        self.h, self.spins, self.za, self.zb, self.seed, self.a0 = h, spins, za, zb, seed, a0
        self.stick = stick_ang if stick_ang is not None else math.pi / 2 + 0.35
        self.z = zb
        self.landed = False

    def pos(self, t):
        if t >= self.tl:
            return self.b, self.stick
        k = (t - self.t0) / (self.tl - self.t0)
        x = self.a[0] + (self.b[0] - self.a[0]) * k
        y = self.a[1] + (self.b[1] - self.a[1]) * k - 4 * self.h * k * (1 - k)
        ang = self.a0 + self.spins * 2 * math.pi * k
        if k > 0.8:                                   # it turns to come down point first
            ang = ang + (self.stick - ang) * ((k - 0.8) / 0.2)
        self.z = self.za + (self.zb - self.za) * k
        return (x, y), ang

    def update(self, t, dt, world):
        if t >= self.tl and not self.landed:
            self.landed = True
            (x, y), ang = self.pos(t)
            fx.burst(world, fx.SPARK, x, y, self.zb, 8, (30, 90), angle=-math.pi / 2, spread=1.2, life=(0.15, 0.35),
                     seed=self.seed)
            fx.burst(world, fx.CHIP, x, y, self.zb, 5, (20, 60), angle=-math.pi / 2, spread=1.0, life=(0.6, 1.2),
                     seed=self.seed + 1)
            world.cue('blade_stick', x=x)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        (x, y), ang = self.pos(t)
        L = 24 * 0.75
        if t >= self.tl:
            hand = (x - math.cos(ang) * L * 0.55 + off[0], y - math.sin(ang) * L * 0.55 + off[1])
        else:
            hand = (x - math.cos(ang) * L * 0.5 + off[0], y - math.sin(ang) * L * 0.5 + off[1])
        draw_weapon(cv, self.kind, hand, ang, 0.75)


class DeflectSpear(fx.Spear):
    """A thrown spear on its rope batted away: it flies at him, is struck aside just short of him, and is reeled in."""

    def __init__(self, thrower, target, t0, t_hit, t1):
        super().__init__(thrower, target, t0, t_hit, t1, t1)

    def stop(self, world):
        h = self.hand(world)
        tp = self.target_point(world)
        return (tp[0] - (tp[0] - h[0]) * 0.2, tp[1] - 5)

    def tip(self, t, world):
        h = self.hand(world)
        st = self.stop(world)
        if t < self.t_hit:
            k = ((t - self.t0) / (self.t_hit - self.t0)) ** 0.8
            return (h[0] + (st[0] - h[0]) * k, h[1] + (st[1] - h[1]) * k - math.sin(k * math.pi) * 3), False
        k = min(1.0, (t - self.t_hit) / max(0.05, self.t1 - self.t_hit))
        return (st[0] + (h[0] - st[0]) * k, st[1] + (h[1] - st[1]) * k - 34 * math.sin(math.pi * k)), False


def weapon_shatter(world, x, y, z, seed):
    fx.burst(world, fx.ICE, x, y, z, 40, (40, 160), life=(0.6, 1.3), size=(1, 2.2), seed=seed, zs=0.6)
    fx.burst(world, fx.FROST, x, y, z, 30, (20, 80), life=(0.3, 0.7), seed=seed + 1)
    world.effects.append(fx.Ring(x, y, world.t, 3, 20, 0.22, 'd8f4ff'))
    world.flash(world.t, 0.18, 0.08, 'd8f4ff')


def clash_fx(world, x, y, z, seed, big=False):
    """Steel on steel (or ice): a white-gold star, sparks spraying, a shake."""
    world.effects.append(fx.HitSpark(x, y, world.t, 9.0 if big else 6.5, 'clash', seed))
    fx.burst(world, fx.SPARK, x, y, z, 22 if big else 14, (60, 200), life=(0.15, 0.45), size=(1, 1.6), seed=seed + 1,
             zs=0.5)
    world.shake(world.t, 2.6 if big else 1.6, 0.18)
    world.count['clashes'] += 1


# ================================================================== FLOW-1: bigger magic for stronger fighters
class GiantSpike(fx.Effect):
    """An ice spike as tall as a building, bursting out of the floor: a faceted shard (a lit face, a shadow face, a
    bright ridge, inner cracks) that grows in a blink, stands, and shatters."""
    zsort = True

    def __init__(self, x, z, g, t0, life, height, width, lean, seed):
        self.x, self.z, self.g, self.t0, self.t1 = x, z, g, t0, t0 + life
        self.height, self.width, self.lean, self.seed = height, width, lean, seed
        self.gone = False

    def update(self, t, dt, world):
        if t + dt >= self.t1 and not self.gone:
            self.gone = True
            rng = _rng(self.seed)
            n = int(10 + self.height / 3)
            hs = rng.uniform(0, self.height, n)
            world.particles.spawn(fx.ICE, self.x + self.lean * hs + rng.uniform(-self.width, self.width, n),
                                  self.z + rng.uniform(-3, 3, n), self.g + hs, rng.uniform(-80, 80, n),
                                  rng.uniform(-30, 30, n), rng.uniform(10, 90, n), rng.uniform(0.8, 1.6, n),
                                  rng.uniform(1.0, 2.6, n), rng=rng)
            world.cue('ice_shatter', x=self.x, small=self.height < 40)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        k = min(1.0, (t - self.t0) / 0.1)
        k = 1 - (1 - k) ** 3
        hh = self.height * k
        x0 = self.x + off[0]
        yb = floor_y(self.z) - self.g + off[1]
        H_, W_ = cv.shape[:2]
        for r in range(int(hh) + 1):
            q = r / max(1.0, hh)
            half = self.width * (1 - q) ** 0.9
            cx = x0 + self.lean * r
            yy = int(round(yb - r))
            if not 0 <= yy < H_:
                continue
            for o in range(-int(half + 0.5), int(half + 0.5) + 1):
                xx = int(round(cx + o))
                if not 0 <= xx < W_:
                    continue
                s = o / max(half, 0.5)
                if abs(s) > 0.92:
                    c = fx.ICE_C[4]
                elif s < -0.05:
                    c = fx.ICE_C[1] if ((r + o * 2) // 5) % 3 else fx.ICE_C[0]
                elif s < 0.12:
                    c = fx.ICE_C[0]
                else:
                    c = fx.ICE_C[2] if ((r * 2 - o) // 6) % 3 else fx.ICE_C[3]
                cv[yy, xx] = c
        lights.append((x0, yb - hh * 0.5, 30 + hh * 0.6, hexc('8ad0ff'), 0.6))


def ice_spikes(world, cx, cz, g, t0, n, height, spread, seed, lifes=(1.2, 2.0)):
    """A forest of giant spikes bursting from the floor round (cx, cz) one after another (the tallest in the middle)."""
    rng = _rng(seed)
    order = sorted(range(n), key=lambda i: abs(i - n // 2))
    for rank, i in enumerate(order):
        x = cx + (i - (n - 1) / 2) * spread / max(1, n - 1) * 2 + rng.uniform(-3, 3)
        z = cz + rng.uniform(-8, 8)
        hh = height * (1.0 - 0.45 * abs(i - (n - 1) / 2) / max(1.0, (n - 1) / 2)) * rng.uniform(0.85, 1.05)
        tt = t0 + rank * 0.05
        world.effects.append(GiantSpike(x, z, g, tt, rng.uniform(*lifes), hh, 2.5 + hh * 0.11,
                                        rng.uniform(-0.12, 0.12), seed + i))

        def burst(w, x=x, z=z, hh=hh, sd=seed + 31 * i):
            fx.burst(w, fx.ICE, x, floor_y(z) - g - 2, z, 10, (40, 140), angle=-math.pi / 2, spread=1.2,
                     life=(0.5, 1.1), size=(1, 2.2), seed=sd)
            fx.burst(w, fx.FROST, x, floor_y(z) - g - hh * 0.4, z, 14, (20, 80), life=(0.3, 0.8), seed=sd + 1)
            ice_coat(w, x, z, 8.0, sd, h=g, k=0.55)
        world.at(tt, burst)
    world.at(t0, lambda w: (w.cue('ice_wall', x=cx, big=True), w.shake(t0, 4.0, 0.45),
                            w.effects.append(fx.Ring(cx, floor_y(cz) - g, t0, 6, 70, 0.4, 'd8f4ff', squash=0.55,
                                                     thick=2))))


def floor_freeze(world, cx, cz, g, t0, radius, seed, dur=0.6):
    """The whole floor freezes: a ring of frost runs out from (cx, cz) and leaves the stone glazed with ice."""
    rng = _rng(seed)
    steps = 10
    for s in range(1, steps + 1):
        r = radius * s / steps
        tt = t0 + dur * s / steps
        m = int(6 + r / 6)

        def ring(w, r=r, m=m, sd=seed + s * 97):
            rr = _rng(sd)
            for i in range(m):
                a = rr.uniform(0, 2 * math.pi)
                x = cx + math.cos(a) * r
                z = cz + math.sin(a) * r * 0.6
                ice_coat(w, x, z, rr.uniform(7, 12), sd + i, h=g, k=0.5)
                if i % 3 == 0:
                    fx.burst(w, fx.FROST, x, floor_y(z) - g - 1, z, 3, (10, 40), life=(0.3, 0.7), seed=sd + 50 + i)
        world.at(tt, ring)
    world.at(t0, lambda w: (w.cue('freeze', x=cx, big=True), w.cue('ground_wave', dur=dur, x=cx),
                            w.effects.append(fx.Ring(cx, floor_y(cz) - g, t0, 4, radius, dur + 0.1, 'bfe8ff',
                                                     squash=0.55, thick=2))))


class FireTornado(fx.Effect):
    """A spinning column of hellfire round a point (a fighter caught in it), tall as the hall, roaring for a while."""
    zsort = True

    def __init__(self, x, z, g, t0, t1, height, radius, seed, follow=None):
        self.x, self.z, self.g, self.t0, self.t1 = x, z, g, t0, t1
        self.height, self.radius, self.seed, self.follow = height, radius, seed, follow

    def centre(self, world):
        if self.follow:
            f = world.f[self.follow]
            return f.x, f.z
        return self.x, self.z

    def update(self, t, dt, world):
        x, z = self.centre(world)
        self.z = z + 0.5
        rng = _rng(int(t * 1000) + self.seed)
        grow = min(1.0, (t - self.t0) / 0.3) * min(1.0, (self.t1 - t) / 0.3)
        n = int(8 * grow) + 1
        hs = rng.uniform(0, self.height * grow, n)
        a = rng.uniform(0, 2 * math.pi, n) + t * 9
        r = self.radius * (0.4 + 0.6 * hs / self.height)
        world.particles.spawn(fx.FLAME, x + np.cos(a) * r, z + np.sin(a) * r * 0.5, self.g + hs,
                              -np.sin(a) * 60, np.cos(a) * 30, rng.uniform(30, 80, n), rng.uniform(0.2, 0.4, n),
                              rng.uniform(1.4, 2.6, n), rng=rng)
        if rng.uniform() < 0.6:
            world.particles.spawn(fx.EMBER, [x + rng.uniform(-6, 6)], [z], [self.g + self.height * grow],
                                  [rng.uniform(-40, 40)], [0.0], [rng.uniform(40, 90)], [1.0], [1.0], rng=rng)
        if rng.uniform() < 0.5:
            world.stage.stamp(x + rng.uniform(-self.radius, self.radius), z + rng.uniform(-4, 4), SCORCH,
                              r=rng.uniform(1.5, 3.0), alpha=0.6, seed=int(rng.integers(1e6)), h=self.g)

    def draw(self, cv, add, t, off, lights, world):
        if not self.alive(t):
            return
        x, z = self.centre(world)
        grow = min(1.0, (t - self.t0) / 0.3) * min(1.0, (self.t1 - t) / 0.3)
        X = x + off[0]
        yb = floor_y(z) - self.g + off[1]
        hgt = self.height * grow
        H_, W_ = cv.shape[:2]
        for row in range(int(hgt)):
            q = row / max(1.0, hgt)
            r = self.radius * (0.35 + 0.75 * q)
            ph = t * 14 - row * 0.18
            yy = int(round(yb - row))
            if not 0 <= yy < H_:
                continue
            for k in range(3):
                a = ph + k * 2.1
                cxp = X + math.cos(a) * r
                w_ = 2.5 + 2.0 * (1 - q)
                front = math.sin(a) > -0.2
                for dx in range(-int(w_), int(w_) + 1):
                    xx = int(round(cxp + dx))
                    if 0 <= xx < W_:
                        heat = (1 - abs(dx) / (w_ + 0.5)) * (1 - q * 0.5) * (1.0 if front else 0.6)
                        cv[yy, xx] = hexc('fffbe0') if heat > 0.62 else hexc('ffd050') if heat > 0.4 else \
                            hexc('ff8a1c') if heat > 0.2 else hexc('c8320c')
        px.glow(add, X, yb - hgt * 0.5, 30 + self.radius, hexc('ff7a20'), 0.6 * grow)
        lights.append((X, yb - hgt * 0.45, 120, hexc('ff8a3a'), 1.6 * grow))


def hellfire_wall(world, x0, z0, x1, z1, t0, n, height, seed, g=0.0, gap=0.06, dur=0.9):
    """A wall of hellfire across the floor: n columns erupting one after another along the line."""
    for i in range(n):
        k = i / max(1, n - 1)
        x = x0 + (x1 - x0) * k
        z = z0 + (z1 - z0) * k
        world.effects.append(HellColumn(x, z, t0 + gap * i, g=g, height=height, dur=dur, seed=seed + i,
                                        width=7.0 + height * 0.04))
    return t0 + gap * (n - 1) + 0.22
