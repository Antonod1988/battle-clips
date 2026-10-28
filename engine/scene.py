"""The engine: two fighters move over a stage bigger than the screen (x, depth z) and stand on its surfaces at height y.

Each fighter has a timeline of actions (a move from moves.py over a duration, travelling dx/dz or to a point, through
the air on an arc or keeping to the surface under him). Cartoon motion rides on top: squash and stretch on a spring,
smears behind fast blows and blades, dust on take-offs and landings, afterimages. Weapons in the hands come from a
timed schedule. The camera travels after the pair and never zooms. Every event can leave a sound cue (World.cue).

A fighter's look, size, stance (his guard, stalk and run) and power tier come from his data file; the stage is a module
from stages/ (surfaces, sprites, breakables, areas and routes). Nothing here knows a fighter or a stage by name.
"""
import math
import numpy as np
from PIL import Image, ImageDraw

import px
import rig
import body
import moves
import fx
from proj import floor_y
from px import hexc, W, H

M = 8                                     # canvas margin (shake)
VOID = -999.0


class Action:
    def __init__(self, t0, move, dur, x0, z0, dx, dz, ease, air, facing, blend, params, surf, attach=None):
        self.t0, self.move, self.dur, self.x0, self.z0 = t0, move, dur, x0, z0
        self.dx, self.dz, self.ease, self.air, self.facing = dx, dz, ease, air, facing
        self.blend, self.params, self.surf = blend, params, surf
        self.attach = attach        # (holder, his joint, my joint, orient): hanging from the other's hand

    def u(self, t):
        return min(1.0, max(0.0, (t - self.t0) / self.dur)) if self.dur > 0 else 1.0


def _smooth(u, u0, u1):
    k = min(1.0, max(0.0, (u - u0) / (u1 - u0)))
    return k * k * (3 - 2 * k)


def plant(p, a, u):
    """Feet stay where they stood while the body travels forward or back, then each takes its own quick step."""
    D = a.dx * a.ease(u)
    first, second = ('ff', 'nf') if a.dx > 0 else ('nf', 'ff')
    for key, (u0, u1) in ((first, (0.04, 0.34)), (second, (0.28, 0.68))):
        s = _smooth(u, u0, u1)
        x, y = p[key]
        if y < -3.5:
            continue
        lift = 3.0 * math.sin(math.pi * s) if 0 < s < 1 else 0.0
        p[key] = (x - D * (1 - s), min(y, -2.0 - lift))
    return p


def arc(h, y0=0.0, y1=0.0):
    """Height over an action: from y0 to y1 with a hump of h."""
    return lambda u: y0 + (y1 - y0) * u + 4 * h * u * (1 - u)


def hold_y(y):
    return lambda u: y


class Fighter:
    """One fighter on the stage: his data (look, size K, stance, tier), his timeline and his state."""

    def __init__(self, world, spec, x, z, facing):
        self.world, self.spec = world, spec
        self.name = spec['id']
        self.look = spec['look_obj']
        self.K = spec.get('K', 0.75)
        self.tier = spec.get('tier', 3)
        self.stance = moves.STANCES[spec.get('stance', 'ninja')]
        self.st = body.State()
        self.actions = []
        self.x_init, self.z_init, self.face_init = x, z, facing
        self.flap = self.flap_v = 0.0
        self.sq, self.sv = 1.0, 0.0
        self.view, self.view_v = 0.0, 0.0
        self.last = None
        self.flash_t = -9.0
        self.x, self.z, self.y = x, z, 0.0
        self.weps = []           # (t, side, kind or None, grow seconds)
        self.backs = []          # (t, swords still on the back)
        self.shields = []        # (t, shield on the arm True/False)
        self.hide = []           # (t0, t1): not drawn (teleports)
        self.facing = facing
        self.heavy = self.tier >= 4 or spec.get('look', {}).get('build') in ('brute', 'heavy')
        self.grow = []           # (t, K): he swells (a titan's rage)
        self._att = (None, None)

    def other(self):
        return [f for n, f in self.world.f.items() if f is not self][0]

    def mv(self, move):
        """A generic move name through his stance (guard -> his own guard)."""
        if isinstance(move, str):
            return moves.MOVES[self.stance.get(move, move)]
        return move

    def K_at(self, t):
        k = self.K
        for (t0, kk) in self.grow:
            if t >= t0:
                k = k + (kk - k) * min(1.0, (t - t0) / 0.6)
        return k

    def add(self, t0, move, dur=1.0, dx=0.0, dz=0.0, ease=None, air=None, blend=0.06, face=None, to=None,
            attach=None, **params):
        """A move from t0: travel dx along the facing (forward +) and dz in depth, or `to`=(x, z) the end point.
        Without `air` the fighter keeps to the surface he stands on at t0 (its slope or sag included).
        attach=(holder, his joint, my joint, orient) hangs him from the other's hand (by the ankle: orient 'arm'
        turns him along the holder's arm; by the throat: None)."""
        move = self.mv(move)
        x0, z0 = self.x_at(t0), self.z_at(t0)
        y0 = self.y_at(t0)
        self.actions = [a for a in self.actions if a.t0 < t0]
        if face is None:
            ox = self.other().x_at(t0)
            face = 1 if ox > x0 + 0.5 else -1 if ox < x0 - 0.5 else self.facing_at(t0)
        if to is not None:
            dx = (to[0] - x0) * face
            dz = to[1] - z0
        if move in moves.TRAVEL and 'speed' not in params and dur > 0:
            params['speed'] = max(8.0, math.hypot(dx, dz) / dur)
            if dx < 0:
                params['back'] = True
        surf = None
        if air is None and attach is None:
            surf = self.world.stage.surface_fn(x0, z0, y0, t0)
        self.actions.append(Action(t0, move, dur, x0, z0, dx, dz, ease or moves.ease, air, face, blend, params, surf,
                                   attach))
        rate = moves.STEP_RATE.get(move)
        if rate and dur > 0.15 and math.hypot(dx, dz) > 2:
            self.world.cue('steps', t=t0, dur=dur, who=self.name, rate=rate, soft=move is moves.sidestep,
                           heavy=self.heavy)
        return t0 + dur

    def find(self, t):
        cur = prev = None
        for a in self.actions:
            if a.t0 <= t:
                prev, cur = cur, a
            else:
                break
        return cur, prev

    def attached_root(self, a, t):
        """Where a held fighter is: his joint (an ankle, the neck) sits in the holder's hand."""
        if self._att[0] == (t, id(a)):
            return self._att[1]
        holder = self.world.f[a.attach[0]]
        hj, mj = a.attach[1], a.attach[2]
        hx, hy = joint_world(holder, t, hj)
        p = self.pose_of(a, t)
        j = rig.solve(p, self.look.dims)
        fc = a.facing
        K = self.K_at(t)
        z = holder.z_at(t) + 2.0
        rx = hx - fc * K * j[mj][0]
        ry = hy - K * j[mj][1]
        out = (rx, z, floor_y(z) - ry)
        self._att = ((t, id(a)), out)
        return out

    def held_at(self, t):
        a, _ = self.find(t)
        return a is not None and a.attach is not None

    def x_at(self, t):
        a, _ = self.find(t)
        if a is not None and a.attach:
            return self.attached_root(a, t)[0]
        return self.x_init if a is None else a.x0 + a.facing * a.dx * a.ease(a.u(t))

    def z_at(self, t):
        a, _ = self.find(t)
        if a is not None and a.attach:
            return self.attached_root(a, t)[1]
        return self.z_init if a is None else a.z0 + a.dz * a.ease(a.u(t))

    def y_at(self, t):
        a, _ = self.find(t)
        if a is None:
            return self.world.stage.ground(self.x_init, self.z_init, 0.0, 999.0) if hasattr(self.world, 'stage') else 0.0
        if a.attach:
            return self.attached_root(a, t)[2]
        if a.air is not None:
            return a.air(a.u(t))
        return a.surf(a.x0 + a.facing * a.dx * a.ease(a.u(t)), a.z0 + a.dz * a.ease(a.u(t)))

    def facing_at(self, t):
        a, _ = self.find(t)
        return self.face_init if a is None else a.facing

    def move_at(self, t):
        a, _ = self.find(t)
        return None if a is None else a.move

    def end_t(self):
        return max([a.t0 + a.dur for a in self.actions if a.dur < 8.0] + [a.t0 for a in self.actions] + [0.0])

    def hidden_at(self, t):
        return any(a <= t < b for a, b in self.hide)

    def wep_at(self, t, side):
        kind, grow = None, 1.0
        for (t0, s, k, g) in self.weps:
            if s == side and t0 <= t:
                kind = k
                grow = 1.0 if not g else min(1.0, max(0.0, (t - t0) / g))
        return kind, grow

    def back_at(self, t):
        n = self.look.gear.get('scabbards', 0)
        for (t0, k) in self.backs:
            if t0 <= t:
                n = k
        return n

    def shield_at(self, t):
        on = True
        for (t0, k) in self.shields:
            if t0 <= t:
                on = k
        return on

    def pose_of(self, a, t):
        p = a.move(a.u(t), t=t - a.t0, **a.params)
        if a.dx and a.move in moves.PLANTED:
            p = plant(dict(p), a, a.u(t))
        if a.attach and a.attach[3] == 'arm':
            holder = self.world.f[a.attach[0]]
            hj = a.attach[1]
            sx, sy = joint_world(holder, t, hj[0] + 'S')
            hx, hy = joint_world(holder, t, hj)
            ux, uy = hx - sx, hy - sy
            L = math.hypot(ux, uy) + 1e-6
            p = dict(p)
            p['spin'] = math.atan2(ux / L * a.facing, -uy / L)
        return p

    def pose_at(self, t):
        a, prev = self.find(t)
        if a is None:
            return self.mv('guard')(0, t=t)
        p = self.pose_of(a, t)
        if prev is not None and (a.attach or prev.attach):
            return p
        if prev is not None and a.blend > 0 and t - a.t0 < a.blend:
            p = rig.lerp_pose(self.pose_of(prev, a.t0), p, moves.ease((t - a.t0) / a.blend))
        return p

    def view_target(self, t):
        """Three-quarter facing: travelling or striking towards the camera turns the body front, away turns it back."""
        a, _ = self.find(t)
        if a is None or a.dur <= 0 or a.attach:
            return 0.0
        if a.move in (moves.flip, moves.back_flip, moves.roll, moves.tumble, moves.lie, moves.bounce, moves.spin_slash):
            return 0.0
        u = a.u(t)
        if u >= 1.0:
            return 0.0
        ad = abs(a.dx) + abs(a.dz)
        if ad < 6.0:
            return 0.0
        k = a.dz / ad
        if abs(k) < 0.3:
            return 0.0
        return max(-1.0, min(1.0, (k - 0.3 * (1 if k > 0 else -1)) / 0.5))


def joint_world(f, t, name):
    """A joint of a fighter in stage coordinates at scene time t (from his timeline, not the last solved frame)."""
    p = f.pose_at(t)
    j = rig.solve(p, f.look.dims)
    K = f.K_at(t)
    fc = f.facing_at(t)
    return (f.x_at(t) + fc * K * j[name][0], floor_y(f.z_at(t)) - f.y_at(t) + K * j[name][1])


class World:
    def __init__(self, stage, specs, starts):
        """stage: a stage object; specs: two fighter specs (fighters/*.py data plus 'look_obj'); starts: [(x, z, facing)]."""
        self.stage = stage
        self.particles = fx.Particles()
        self.effects, self.events, self.words = [], [], []
        self.shakes, self.flashes, self.slow, self.trails, self.tint = [], [], [], [], []
        self.f = {}
        for spec, (x, z, fc) in zip(specs, starts):
            self.f[spec['id']] = Fighter(self, spec, x, z, fc)
        self.names = [s['id'] for s in specs]
        self.cam = [stage.CAM_X[0], stage.CAM_Y[0]]
        self.cam_keys = []          # (t0, t1, fn(t) -> (x, y), hard): scripted camera targets
        self.cam_bias = []          # (t0, t1, dx, dy): lead the frame a little
        self.cam_fast = []          # (t0, t1, speed)
        self.t = 0.0
        self.joints, self.prev_joints = {}, {}
        self.wtips, self.prev_wtips = {}, {}
        self.frame_no = 0
        self.hidden = {}
        self.hist = {n: [] for n in self.f}
        self.end = 30.0
        self.marks = {}
        self.key_moments = []
        self.cues = []              # the sound timeline: (scene time, kind, params), mapped to video time on export
        self.frame_t = []           # the scene time of every video frame, and the camera's x then
        self.frame_cam = []
        self.count = {'hits': 0, 'blocks': 0, 'clashes': 0, 'cuts': 0}
        self.land_hooks = []
        self.darkness = []          # (t0, t1, amount): the end card dims the scene
        self.log = []

    # ------------------------------------------------------------ scheduling
    def at(self, t, fn):
        self.events.append([t, len(self.events), fn, False])

    def shake(self, t, amp, dur=0.3):
        self.shakes.append((t, amp, dur))

    def flash(self, t, strength=0.8, dur=0.12, color='ffffff'):
        self.flashes.append((t, strength, dur, hexc(color)))

    def slowmo(self, t0, t1, rate, ramp=0.08):
        self.slow.append((t0, t1, rate, ramp))

    def trail(self, who, t0, t1):
        self.trails.append((who, t0, t1))

    def mark(self, name, t, key=True):
        """A named moment: a still of it goes into moments/, and onto the moments sheet if key."""
        base, k = name, 2
        while name in self.marks:
            name = f'{base}{k}'
            k += 1
        self.marks[name] = t
        if key:
            self.key_moments.append(name)
        return name

    def cue(self, kind, t=None, dur=None, who=None, x=None, **params):
        """A sound cue. `who` (a fighter) or `x` (stage x) places it in the stereo field."""
        self.cues.append(dict(params, kind=kind, ts=self.t if t is None else t, dur_s=dur, who=who, x=x))

    def export_cues(self, path, fps):
        """Writes the cue sheet in video seconds (slow motion and hit-stop already applied)."""
        import json
        ft = np.asarray(self.frame_t, np.float64)
        fi = np.arange(len(ft), dtype=np.float64)
        cams = np.asarray(self.frame_cam, np.float64)

        def vt(ts):
            return float(np.interp(ts, ft, fi)) / fps
        out = []
        for c in sorted(self.cues, key=lambda c: c['ts']):
            if c['ts'] > self.end + 0.5:
                continue
            d = {k: v for k, v in c.items() if k not in ('ts', 'dur_s', 'who', 'x')}
            d['t'] = round(vt(c['ts']), 4)
            dur = c['dur_s']
            if c['kind'] == 'steps' and c['who'] is not None:
                later = [a.t0 for a in self.f[c['who']].actions if a.t0 > c['ts'] + 1e-6]
                if later:
                    dur = min(dur, min(later) - c['ts'])
            if dur is not None:
                d['dur'] = round(min(vt(c['ts'] + dur), len(ft) / fps) - d['t'], 4)
            x = c['x']
            if c['who'] is not None:
                x = self.f[c['who']].x_at(c['ts'])
            if x is not None:
                cam = float(np.interp(c['ts'], ft, cams))
                d['pan'] = round(max(-0.85, min(0.85, ((x - cam) / W) * 2 - 1)), 3)
            if 'to_x' in d:
                cam = float(np.interp(c['ts'], ft, cams))
                d['pan_end'] = round(max(-0.85, min(0.85, ((d.pop('to_x') - cam) / W) * 2 - 1)), 3)
            out.append(d)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'fps': fps, 'duration': len(ft) / fps, 'cues': out}, f, indent=0)
        return out

    def rate(self, t):
        r = 1.0
        for (a, b, k, rp) in self.slow:
            if a - rp <= t <= b + rp:
                w = 1.0 if a <= t <= b else ((t - (a - rp)) / rp if t < a else 1 - (t - b) / rp)
                r = min(r, 1 + (k - 1) * w)
        return r

    def is_hidden(self, name, t=None):
        t = self.t if t is None else t
        return self.hidden.get(name) or self.f[name].hidden_at(t)

    # ------------------------------------------------------------- stepping
    def solve(self, t):
        order = sorted(self.f.items(), key=lambda kv: 1 if kv[1].held_at(t) else 0)
        for name, f in order:
            f._att = (None, None)
            p = f.pose_at(t)
            f.x, f.z, f.y = f.x_at(t), f.z_at(t), f.y_at(t)
            fc = f.facing_at(t)
            f.facing = fc
            j = rig.solve(p, f.look.dims)
            sy = f.sq
            sx = 1.0 / math.sqrt(sy)
            K = f.K_at(t)
            fore = 1.0 - 0.22 * abs(f.view)          # a three-quarter turn shortens the side view a little
            rx, ry = f.x, floor_y(f.z) - f.y
            out = {}
            for k_, v in j.items():
                if isinstance(v, tuple) and len(v) == 2 and k_ != 'up':
                    out[k_] = (rx + v[0] * fc * K * sx * fore, ry + v[1] * K * sy)
                else:
                    out[k_] = v
            out['up'] = (j['up'][0] * fc, j['up'][1])
            out['facing'] = fc
            out['scale'] = K * sx
            self.joints[name] = out
            f.st.weps = {s: f.wep_at(t, s) for s in ('f', 'n')}
            f.st.weps = {s: v for s, v in f.st.weps.items() if v[0]}
            f.st.back = f.back_at(t)
            f.st.shield = f.shield_at(t)
            f.st.view = f.view
            tips = {}
            body.SC[0] = out['scale']
            for s, (kind, grow) in f.st.weps.items():
                hand, u, tip, _ = body.weapon_axis(out, s, kind, fc, grow)
                tips[s] = (hand, tip, kind)
            self.wtips[name] = tips

    def advance(self, dt):
        t = self.t + dt
        self.t = t
        self.prev_joints = {k: dict(v) for k, v in self.joints.items()}
        self.prev_wtips = {k: dict(v) for k, v in self.wtips.items()}
        self.solve(t)
        for ev in sorted(self.events, key=lambda e: (e[0], e[1])):
            if not ev[3] and ev[0] <= t:
                ev[3] = True
                ev[2](self)
        for e in self.effects:
            if e.t0 <= t <= e.t1:
                e.update(t, dt, self)
        self.particles.update(dt, self)
        self.stage.update(self, t, dt)
        for n, f in self.f.items():
            self.cartoon(f, t, dt)
        self.solve(t)
        self.step_camera(t, dt)
        self.frame_t.append(t)
        self.frame_cam.append(self.cam[0])
        self.frame_no += 1

    def cam_target(self, t):
        for (a, b, fn, hard) in self.cam_keys:
            if a <= t <= b:
                return fn(t), hard
        vis = [(n, f) for n, f in self.f.items() if not self.is_hidden(n, t)]
        if not vis:
            vis = list(self.f.items())
        xs = [f.x for n, f in vis]
        ys = [floor_y(f.z) - f.y - 14 * f.K / 0.75 for n, f in vis]
        tx = (min(xs) + max(xs)) / 2 - W / 2
        ty = (min(ys) + max(ys)) / 2 - H * 0.58
        for (a, b, bx, by) in self.cam_bias:
            if a <= t <= b:
                tx += bx
                ty += by
        return (tx, ty), False

    def step_camera(self, t, dt):
        (tx, ty), hard = self.cam_target(t)
        cx, cy = self.stage.cam_limits(self, t)
        tx = max(cx[0], min(cx[1], tx))
        ty = max(cy[0], min(cy[1], ty))
        if hard or self.frame_no == 0:
            self.cam[0], self.cam[1] = tx, ty
            return
        spd = 2.2
        for (t0, t1, s) in self.cam_fast:
            if t0 <= t <= t1:
                spd = s
        far = math.hypot(tx - self.cam[0], ty - self.cam[1])
        spd = spd * (1.0 + min(1.2, max(0.0, far - 60) / 140.0))
        k = min(1.0, dt * spd)
        self.cam[0] += (tx - self.cam[0]) * k
        self.cam[1] += (ty - self.cam[1]) * k

    def cartoon(self, f, t, dt):
        """Squash and stretch on a spring, dust on take-offs, landings and dashes, the tabard's lag, the turn."""
        if dt <= 0:
            return
        g = self.stage.ground(f.x, f.z, t, f.y)
        ag = f.y - g if g > VOID + 1 else 99.0
        if f.last is None:
            f.last = (f.x, f.z, f.y, None, ag)
        lx, lz, ly, lmove, lag = f.last
        vx = (f.x - lx) / dt
        vy = (f.y - ly) / dt
        mv = f.move_at(t)
        target = 1.0
        if ag > 1.0:
            target = 1.0 + min(0.16, abs(vy) / 420.0)
        elif mv is moves.dash or mv is moves.fire_dash:
            target = 0.88
        hid = self.is_hidden(f.name, t) or f.held_at(t)
        heavy = 1.6 if f.heavy else 1.0
        if f.held_at(t):
            target = 1.0
        if lag > 2.0 and ag <= 0.5 and vy < -30 and not hid:
            self.cue('land', who=f.name, power=round(min(2.0, abs(vy) / 150.0) * heavy, 2))
            f.sv -= min(7.0, 2.5 + abs(vy) / 60.0)
            n = int((6 + min(14, abs(vy) / 12)) * heavy)
            fx.burst(self, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, n, (15, 50), angle=-math.pi / 2, spread=1.4,
                     life=(0.35, 0.7), size=(1.2, 2.2), seed=self.frame_no * 7 + 1, zs=0.6)
            if abs(vy) > 140:
                self.shake(t, (1.5 + abs(vy) / 200.0) * heavy, 0.2)
            for hook in self.land_hooks:
                hook(self, f, g)
        if lag <= 0.5 and ag > 1.5 and vy > 40 and not hid:
            self.cue('jump', who=f.name)
        if (mv is moves.dash) and lmove is not moves.dash and not hid:
            self.cue('dash', who=f.name)
            fx.burst(self, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, 8, (20, 60), angle=-math.pi / 2 - 0.6 * f.facing,
                     spread=0.6, life=(0.3, 0.6), size=(1.2, 2.0), seed=self.frame_no * 7 + 2, zs=0.5)
        if lmove is moves.dash and mv is not moves.dash and not hid:
            fx.burst(self, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, 6, (15, 45), angle=-math.pi / 2, spread=1.2,
                     life=(0.3, 0.6), size=(1.0, 1.8), seed=self.frame_no * 7 + 3, zs=0.5)
        if f.heavy and mv in (moves.hulk_run, moves.charge) and ag < 1 and (self.frame_no % 3 == 0):
            fx.burst(self, fx.DUST, f.x, floor_y(f.z) - g - 1, f.z, 3, (10, 30), angle=-math.pi / 2 - 0.8 * f.facing,
                     spread=0.6, life=(0.3, 0.6), size=(1.2, 2.2), seed=self.frame_no * 7 + 4, zs=0.4)
        f.sv += ((target - f.sq) * 260.0 - f.sv * 16.0) * dt
        f.sq = max(0.72, min(1.28, f.sq + f.sv * dt))
        f.last = (f.x, f.z, f.y, mv, ag)
        tgt = max(-1.0, min(1.0, -vx * f.facing * 0.01))
        f.flap_v += ((tgt - f.flap) * 60.0 - f.flap_v * 8.0) * dt
        f.flap += f.flap_v * dt
        f.st.flap = f.flap + math.sin(t * 2.3 + (0 if f.facing > 0 else 1.7)) * 0.05
        f.st.flash = 0.3 * max(0.0, 1 - (t - f.flash_t) / 0.025) if t >= f.flash_t else 0.0
        vt = f.view_target(t)
        f.view += (vt - f.view) * min(1.0, dt * 9.0)
        if abs(f.view) < 0.02:
            f.view = 0.0

    def jolt(self, who, amount=4.0):
        """A hit squashes the body for a moment (the spring brings it back with a wobble)."""
        self.f[who].sv += amount

    # ------------------------------------------------------------- the frame
    def shake_offset(self):
        amp = 0.0
        for (t0, a, d) in self.shakes:
            if t0 <= self.t <= t0 + d:
                amp += a * (1 - (self.t - t0) / d) ** 2
        amp = min(amp, 7.0)
        if amp < 0.3:
            return 0, 0
        n = self.frame_no
        return int(round((((n * 7919) % 13) / 6.0 - 1.0) * amp)), int(round((((n * 104729) % 11) / 5.0 - 1.0) * amp * 0.7))

    def render(self):
        t = self.t
        st = self.stage
        cv, off = st.view(self.cam)
        add = np.zeros_like(cv)
        lights = []
        ox, oy = off
        st.floor_lights(cv, off, t, self)
        # shadows on whatever each fighter stands over
        for name, f in self.f.items():
            if self.is_hidden(name, t):
                continue
            g = st.ground(f.x, f.z, t, f.y)
            if g <= VOID + 1:
                continue
            j = self.joints[name]
            xs = [v[0] for k_, v in j.items() if isinstance(v, tuple) and len(v) == 2 and k_ != 'up']
            k_ = max(0.3, 1 - (f.y - g) / 70.0)
            rx = max(8.0, (max(xs) - min(xs)) * 0.55) * k_
            shadow(cv, f.x + ox, floor_y(f.z) - g + oy + 0.5, rx, rx * 0.32, 0.55 * k_)
        # depth-sorted: the stage's sprites, standing effects and the fighters
        items = [(z, 0, o) for (z, o) in st.sprites(t, self)]
        items += [(f.z + (0.5 if f.held_at(t) else 0.0), 2, n) for n, f in self.f.items() if not self.is_hidden(n, t)]
        items += [(e.z + 0.2, 3, e) for e in self.effects if getattr(e, 'zsort', False) and e.alive(t)]
        for z, kind, o in sorted(items, key=lambda i: (i[0], i[1])):
            if kind == 0:
                lt = o.draw(cv, ox, oy, t)
                if lt is not None:
                    lights.append(lt)
            elif kind == 3:
                o.draw(cv, add, t, off, lights, self)
            else:
                self.draw_fighter(cv, o, off, t)
        for name in self.f:
            if self.is_hidden(name, t):
                continue
            for s, (hand, tip, kind) in self.wtips.get(name, {}).items():
                if body.WEAPONS[kind].get('ice'):
                    mx, my = (hand[0] + tip[0]) / 2 + ox, (hand[1] + tip[1]) / 2 + oy
                    px.glow(add, mx, my, 12, hexc('5ab4ff'), 0.25)
                    lights.append((mx, my, 30, hexc('7ac8ff'), 0.45))
        for e in self.effects:
            if not getattr(e, 'zsort', False):
                e.draw(cv, add, t, off, lights, self)
        self.particles.draw(cv, off, lights, add)
        for (x, y, r, c, s) in lights:
            px.boost(cv, x, y, r, c, s * 0.55, squash=0.8)
        st.overlay(cv, off, t, self)
        cv += add
        for (t0, s, d, c) in self.flashes:
            if t0 <= t <= t0 + d:
                k_ = s * (1 - (t - t0) / d)
                cv = cv * (1 - k_) + c * k_
        for (t0, t1, c, s) in self.tint:
            if t0 <= t <= t1:
                cv = cv * (1 - s) + cv * np.asarray(c, np.float32) * s
        sx, sy = self.shake_offset()
        frame = cv[M + sy:M + sy + H, M + sx:M + sx + W].copy()
        vignette(frame)
        for (t0, t1, amt) in self.darkness:
            if t >= t0:
                frame *= 1 - amt * min(1.0, (t - t0) / max(0.01, t1 - t0))
        for w in self.words:
            w.draw(frame, t)
        return frame

    def draw_fighter(self, cv, name, off, t):
        f = self.f[name]
        j = self.joints[name]
        ox, oy = off
        jc = {k_: ((v[0] + ox, v[1] + oy) if isinstance(v, tuple) and len(v) == 2 and k_ != 'up' else v)
              for k_, v in j.items()}
        mv = f.move_at(t)
        # smears behind fast blows (a pale streak from where the fist or foot was a frame ago)
        if mv in moves.SMEARS and name in self.prev_joints:
            pj = self.prev_joints[name]
            for a_, b_ in (('FE', 'FW'), ('NE', 'NW'), ('FK', 'FT'), ('NK', 'NT')):
                p0, p1 = pj[b_], j[b_]
                sp = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                if sp > 4.0:
                    smear(cv, [(pj[a_][0] + ox, pj[a_][1] + oy), (p0[0] + ox, p0[1] + oy), (p1[0] + ox, p1[1] + oy),
                               (j[a_][0] + ox, j[a_][1] + oy)], f.look.smear, min(0.75, sp / 14.0), lim=100)
        # blade arcs: the swept area between the weapon a frame ago and now
        if name in self.prev_wtips:
            for s, (hand, tip, kind) in self.wtips.get(name, {}).items():
                prev = self.prev_wtips[name].get(s)
                if prev is None or prev[2] != kind:
                    continue
                ph, pt, _ = prev
                sp = math.hypot(tip[0] - pt[0], tip[1] - pt[1])
                if sp > 5.0:
                    col = hexc('bfeaff') if body.WEAPONS[kind].get('ice') else hexc('ffffff')
                    smear(cv, [(ph[0] + ox, ph[1] + oy), (pt[0] + ox, pt[1] + oy), (tip[0] + ox, tip[1] + oy),
                               (hand[0] + ox, hand[1] + oy)], col, min(0.8, sp / 16.0), lim=120)
        body.SC[0] = j['scale']
        st = f.st
        st.xray = st.elec > 0.3 and (self.frame_no // 2) % 2 == 0
        rgba, bx, by = body.draw(jc, f.look, st, f.facing, pad=12)
        rgba[..., :3] *= np.array([0.9, 0.9, 0.97], np.float32)
        hist = self.hist[name]
        if any(a <= t <= b for (who, a, b) in self.trails if who == name):
            tint = hexc(f.spec.get('trail', 'ffffff'))
            for back, al in ((4, 0.22), (2, 0.4)):
                if len(hist) >= back:
                    g, gx, gy, gcam = hist[-back]
                    ghost = g.copy()
                    ghost[..., :3] = ghost[..., :3] * 0.3 + tint * 0.7
                    px.blit(cv, ghost, gx + int(round(gcam[0] - self.cam[0])), gy + int(round(gcam[1] - self.cam[1])), al)
        hist.append((rgba, bx, by, tuple(self.cam)))
        del hist[:-6]
        px.blit(cv, rgba, bx, by)


def smear(cv, pts, color, alpha, lim=80):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = int(math.floor(min(xs))) - 1, int(math.floor(min(ys))) - 1
    x1, y1 = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
    if x1 - x0 > lim or y1 - y0 > lim:
        return
    im = Image.new('L', (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in pts], fill=255)
    m = np.asarray(im) > 0
    ys_, xs_ = np.nonzero(m)
    ys_, xs_ = ys_ + y0, xs_ + x0
    ok = (ys_ >= 0) & (ys_ < cv.shape[0]) & (xs_ >= 0) & (xs_ < cv.shape[1])
    ys_, xs_ = ys_[ok], xs_[ok]
    c = np.asarray(color, np.float32) * 0.6 + 0.4
    cv[ys_, xs_] = cv[ys_, xs_] * (1 - alpha) + c * alpha


def shadow(cv, cx, cy, rx, ry, alpha):
    x0, x1 = int(cx - rx - 1), int(cx + rx + 2)
    y0, y1 = int(cy - ry - 1), int(cy + ry + 2)
    x0c, x1c = max(0, x0), min(cv.shape[1], x1)
    y0c, y1c = max(0, y0), min(cv.shape[0], y1)
    if x1c <= x0c or y1c <= y0c:
        return
    ys, xs = np.mgrid[y0c:y1c, x0c:x1c]
    d = ((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / max(ry, 0.8)) ** 2
    k = np.where(d <= 1, alpha * np.where(d < 0.55, 1.0, 0.6), 0.0)
    cv[y0c:y1c, x0c:x1c] *= (1 - k)[..., None]


_VIG = None


def vignette(frame):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _VIG = (1 - np.clip(d - 0.7, 0, 1) * 0.5)[..., None].astype(np.float32)
    frame *= _VIG
