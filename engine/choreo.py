"""The choreographer: builds a whole fight from a recipe, the two fighters' kits and the stage.

The fight runs through acts: intro, opening (feeling each other out, the first specials), a trip to another area,
weapons, magic, another trip, escalation, the turning point, the climax (the winner's signature), the finish. Each act
gets a share of the recipe's length (in video seconds, slow motion counted). Pinned beats from the recipe happen in
their act, in order; the rest of each act is filled with beats chosen from what both fighters can do, by weight, never
the same beat twice in a row, with short circling between some of them so nobody stands still.

Same recipe and seed, same fight; another seed, another fight.
"""
import math
import numpy as np

import acts
import beats as B
import moves
from scene import World

ACTS = ['opening', 'travel1', 'weapons', 'magic', 'travel2', 'escalation', 'turning', 'climax']
SHARE = {'opening': 0.20, 'travel1': 0.07, 'weapons': 0.30, 'magic': 0.17, 'travel2': 0.04, 'escalation': 0.12,
         'turning': 0.05, 'climax': 0.05}
FINISH_LEN = {'fire_breath': 15.0, 'ice_shatter': 11.0, 'knockout': 10.5}
INTRO_LEN = 5.0


class Choreo(acts.Ctx):
    def __init__(self, world, recipe, winner, start_area):
        super().__init__(world, np.random.default_rng(int(recipe.get('seed', 1)) * 7919 + 17), seed0=400)
        self.recipe = recipe
        self.winner = winner
        self.area = start_area
        self.visited = [start_area]
        self.recent = {}
        self.history = []
        self.stuck = None
        self.stuck_fx = None

    def target_for(self, D, t, kind=None, kinds=None, dirn=None, maxd=170, any_dir=True, area=None):
        ts = self.w.stage.targets(area or self.area, t)
        if kind:
            ts = [x for x in ts if x.kind == kind]
        if kinds:
            ts = [x for x in ts if x.kind in kinds]
        x0 = D.x_at(t)
        ts = [x for x in ts if 10 < abs(x.x - x0) <= maxd]
        if dirn:
            ts = [x for x in ts if (x.x - x0) * dirn > 0]
        if not ts:
            return None
        ts.sort(key=lambda x: (x.used, abs(x.x - x0) + abs(x.z - D.z_at(t)) * 0.5))
        return ts[0]

    def area_at(self, f, t):
        return self.w.stage.area_of(f.x_at(t), f.z_at(t), f.y_at(t), t)


def video_time(world, t, fps=30):
    """Video seconds up to scene time t, stepping exactly as the renderer does (a frame advances the scene by the
    slow-motion rate at its start, so the shortest hit-stops can fall between frames)."""
    if t <= 0:
        return 0.0
    slow = sorted(world.slow)
    tt, n = 0.0, 0
    while tt < t:
        r = 1.0
        for (a, b, k, rp) in slow:
            if a - rp > tt:
                break
            if tt <= b + rp:
                w = 1.0 if a <= tt <= b else ((tt - (a - rp)) / rp if tt < a else 1 - (tt - b) / rp)
                r = min(r, 1 + (k - 1) * w)
        tt += r / fps
        n += 1
    return n / fps


def scene_for_video(world, t_from, v_target):
    """The scene time at which video time reaches v_target (searching forward from t_from)."""
    v0 = video_time(world, t_from)
    if v_target <= v0:
        return t_from
    return t_from + (v_target - v0) * 0.98


MELEE = {'exchange', 'blade_exchange', 'blade_cut', 'counter', 'juggle', 'sweep_jump', 'air_clash', 'knock_into'}


def candidates(c, act, A, D, t, last):
    """The beats that can happen now, with weights. Rhythm: after a special (a power, a set piece, a trip) comes
    close fighting, so something lands or clashes every couple of seconds."""
    out = []
    melee_only = last is not None and last not in MELEE
    for name, b in B.BEATS.items():
        if act not in b['acts'] or not b['pool'] or b['weight'] <= 0:
            continue
        if name == last:
            continue
        try:
            ok = b['needs'] is None or b['needs'](c, A, D, t)
        except Exception:
            ok = False
        if not ok:
            continue
        if melee_only and name not in MELEE:
            continue
        wgt = b['weight']
        used = sum(1 for h in c.history if h[0] == name)
        if name in MELEE:
            wgt *= 1.5 / (1.0 + 0.6 * used)
        else:
            wgt *= 0.6 * (0.05 if used else 1.0)        # a special shows once (a second time only if nothing else)
        if name in c.pinned_later:
            wgt *= 0.15
        out.append((name, wgt))
    return out


def leader(c, A, D, beat_name=None, act=None):
    """Who leads the next beat: the winner a little more often, but never one fighter for long (the last four
    beats balance it); in the turning point, the loser."""
    if act == 'turning':
        return (D, A) if A.name == c.winner else (A, D)
    recent = [h[1] for h in c.history[-4:]]
    na = recent.count(A.name)
    if len(recent) >= 3 and na >= 3:
        return (D, A)
    if len(recent) >= 3 and na <= len(recent) - 3:
        return (A, D)
    p = 0.56 if A.name == c.winner else 0.44
    return (A, D) if c.rng.uniform() < p else (D, A)


def run_beat(c, name, A, D, t, params=None):
    params = dict(params or {})
    fn = B.BEATS[name]['fn'] if name in B.BEATS else getattr(B, name)
    t2 = fn(c, A, D, t, **params)
    c.history.append((name, A.name, t, t2))
    c.log.append(f'BEAT {name:18s} {A.name:>16s} -> {D.name:<16s} {t:7.2f} .. {t2:7.2f}  area {c.area}')
    return t2


def pick_route(c, t, prefer=None):
    st = c.w.stage
    routes = [r for r in st.ROUTES if r['a'] == c.area and st.route_open(r, t)]
    if prefer:
        routes = [r for r in routes if r['b'] == prefer or r['how'] == prefer] or routes
    if not routes:
        return None
    fresh = [r for r in routes if r['b'] not in c.visited]
    pool = fresh or routes
    return pool[int(c.rng.integers(len(pool)))]


def travel(c, F1, F2, t, prefer=None, route=None):
    route = route or pick_route(c, t, prefer)
    if route is None:
        return t
    how = route['how']
    if how in ('knocked_off', 'thrown_through', 'smash_through'):
        A, D = (F1, F2) if F1.name == c.winner else (F2, F1)
        if c.rng.uniform() < 0.35:
            A, D = D, A
    else:
        A, D = (F2, F1) if F1.name == c.winner else (F1, F2)
    t2 = B.travel(c, A, D, t, route)
    c.visited.append(route['b'])
    c.history.append(('travel', A.name, t, t2))
    c.log.append(f'TRAVEL {route["a"]} -> {route["b"]} ({how}) {t:7.2f} .. {t2:7.2f}')
    return t2


_STAGES = {}


def build(recipe, stage_mod, specs, log=print):
    """Builds the world for a recipe: the stage, the two fighters, the whole fight. The pinned beats take what they
    take, so the fight is planned again (same seed) with the generated part shortened or lengthened until the video
    length is within a couple of seconds of the recipe's. Returns the world."""
    import copy
    key = stage_mod.__name__
    if key not in _STAGES:
        _STAGES[key] = stage_mod.Stage()
    target = float(recipe.get('length', 120.0))
    finisher = recipe.get('finisher')
    body = None
    best = None
    for attempt in range(5):
        stage = copy.deepcopy(_STAGES[key])
        world = build_once(recipe, stage_mod, stage, specs, body)
        v = video_time(world, world.end)
        log(f'PLAN pass {attempt + 1}: video {v:.1f}s (target {target:.0f}s)')
        if best is None or abs(v - target) < abs(best[0] - target):
            best = (v, world)
        if abs(v - target) <= 2.0:
            break
        body = world.choreo.body - (v - target) * 1.05
        if body < 10:
            break
    return best[1]


def build_once(recipe, stage_mod, stage, specs, body=None):
    """One planning pass with `body` seconds of video for the acts between the intro and the finish."""
    start_area = recipe.get('start_area') or list(stage.AREAS)[0]
    starts = stage.START.get(start_area) or [stage.AREAS[start_area]['center']] * 2
    (xa, za), (xd, zd) = starts
    world = World(stage, specs, [(xa, za, 1), (xd, zd, -1)])
    if hasattr(stage_mod, 'setup'):
        stage_mod.setup(world)
    A, D = world.f[specs[0]['id']], world.f[specs[1]['id']]
    winner = recipe.get('winner', specs[0]['id'])
    c = Choreo(world, recipe, winner, start_area)
    W_, L_ = (A, D) if winner == A.name else (D, A)
    pins = [dict(p) for p in recipe.get('pins', [])]
    c.pinned_later = set(p['beat'] for p in pins)
    finisher = recipe.get('finisher') or W_.spec.get('finisher', 'knockout')
    length = float(recipe.get('length', 120.0))
    fin_len = FINISH_LEN.get(finisher, 10.0)
    if body is None:
        body = max(20.0, length - INTRO_LEN - fin_len)
    c.body = body
    has_weapons = bool(A.spec.get('weapons') or D.spec.get('weapons'))
    share = dict(SHARE)
    if not has_weapons:
        share['opening'] += share['weapons'] * 0.5
        share['magic'] += share['weapons'] * 0.5
        share['weapons'] = 0.0
    nroutes = len([r for r in stage.ROUTES])
    if nroutes == 0:
        share['opening'] += share['travel1']
        share['escalation'] += share['travel2']
        share['travel1'] = share['travel2'] = 0.0
    t = B.intro(c, A, D, 0.0)
    v_end = {}
    acc = INTRO_LEN
    for act in ACTS:
        acc += share[act] * body
        v_end[act] = acc
    last = None
    circled = False
    for act in ACTS:
        act_pins = [p for p in pins if p.get('act') == act]
        target_v = v_end[act]
        c.log.append(f'ACT {act} until {target_v:.1f}s of video ({len(act_pins)} pinned)')
        if act.startswith('travel'):
            for p in act_pins:
                t = run_pin(c, p, A, D, t, W_, L_)
            if not act_pins:
                t = travel(c, A, D, t)
            continue
        if act == 'turning':
            if act_pins:
                for p in act_pins:
                    t = run_pin(c, p, A, D, t, W_, L_)
            elif any(pw.get('trap') for pw in L_.spec.get('powers', [])) and W_.spec.get('breakout'):
                trap = [pw['beat'] for pw in L_.spec['powers'] if pw.get('trap')][0]
                t = run_beat(c, trap, L_, W_, t)
            else:
                t = run_beat(c, 'comeback', L_, W_, t)
            continue
        if act == 'climax':
            for p in act_pins:
                t = run_pin(c, p, A, D, t, W_, L_)
            if not any(p['beat'] in B.CLIMAXES for p in act_pins):
                sig = (W_.spec.get('signature') or {}).get('beat', 'big_combo')
                fn = sig if sig in B.CLIMAXES else 'big_combo'
                params = {'line': W_.spec['signature']['line']} if fn == 'spear' and W_.spec['signature'].get('line') else {}
                t = B.engage(c, W_, L_, t) if fn == 'big_combo' else t
                t = run_beat(c, fn, W_, L_, t, params)
            continue
        # the open acts: pins in order, generated beats between them until the act's share is used
        queue = list(act_pins)
        n_gen = 0
        while True:
            vt = video_time(world, t)
            if queue:
                remaining_pins = len(queue)
                # leave room for the pins still to come (about 4 s of video each)
                room = target_v - vt - remaining_pins * 4.0
                if room <= 0 or c.rng.uniform() < 0.45 or n_gen >= 3:
                    p = queue.pop(0)
                    t = run_pin(c, p, A, D, t, W_, L_)
                    c.pinned_later.discard(p['beat']) if not any(q['beat'] == p['beat'] for q in queue) else None
                    last = p['beat']
                    n_gen = 0
                    continue
            elif vt >= target_v - 1.0:
                break
            F1, F2 = leader(c, A, D, act=act)
            cands = candidates(c, act, F1, F2, t, last)
            if not cands:
                F1, F2 = F2, F1
                cands = candidates(c, act, F1, F2, t, last)
            if not cands:
                break
            if not circled and c.rng.uniform() < 0.3:
                t = B.circle(c, F1, F2, t)
                circled = True
                continue
            names = [n for n, _ in cands]
            wts = np.array([w_ for _, w_ in cands], np.float64)
            name = names[int(c.rng.choice(len(names), p=wts / wts.sum()))]
            t = run_beat(c, name, F1, F2, t)
            last = name
            circled = False
            n_gen += 1
    # the finish
    t = B.engage(c, W_, L_, t) if finisher == 'knockout' else t
    t_end = B.finish(c, W_, L_, t, finisher)
    c.log.append(f'FINISH {finisher} {t:.2f} .. {t_end:.2f}; video {video_time(world, t_end):.1f}s')
    world.end = t_end
    fill_idle(world, t_end)
    world.choreo = c
    return world


def run_pin(c, p, A, D, t, W_, L_):
    """A pinned beat: 'by' names who leads it (a fighter id, 'winner' or 'loser'); default the one who has it."""
    name = p['beat']
    by = p.get('by')
    if by == 'winner':
        F1, F2 = W_, L_
    elif by == 'loser':
        F1, F2 = L_, W_
    elif by in (A.name, D.name):
        F1, F2 = (A, D) if by == A.name else (D, A)
    else:
        F1, F2 = (A, D) if B.has(A, name) or not B.has(D, name) else (D, A)
    params = {k: v for k, v in p.items() if k not in ('act', 'beat', 'by')}
    if name == 'travel':
        route = None
        if 'to' in params:
            rs = [r for r in c.w.stage.ROUTES if r['a'] == c.area and r['b'] == params['to'] and
                  c.w.stage.route_open(r, t)]
            route = rs[0] if rs else None
        return travel(c, F1, F2, t, route=route)
    if name in ('stage_piece',):
        return c.w.stage.set_piece(c, F1, F2, t, params['piece'])
    return run_beat(c, name, F1, F2, t, params)


def fill_idle(world, t_end, limit=0.45):
    """No standing about: wherever a fighter would stand in his guard without travelling for longer than `limit`, he
    drifts a few pixels round and back instead (a stalk whose small steps follow the real speed)."""
    for f in world.f.values():
        acts_ = sorted(f.actions, key=lambda a: a.t0)
        new = []
        for i, a in enumerate(acts_):
            nxt = acts_[i + 1].t0 if i + 1 < len(acts_) else min(t_end, a.t0 + a.dur)
            span = min(nxt, a.t0 + a.dur) - a.t0
            if a.move in moves.STATIC and a.air is None and span > limit and abs(a.dx) < 1 and abs(a.dz) < 1 \
                    and a.move is not moves.hold:
                x0, z0 = a.x0, a.z0
                n = max(1, int(span / 0.9))
                seg = span / (2 * n)
                k = 0
                for j in range(n):
                    ang = (hash((f.name, int(a.t0 * 10), j)) % 628) / 100.0
                    ddx, ddz = math.cos(ang) * 7.0, math.sin(ang) * 6.0
                    for (sx, sz, ex, ez) in ((x0, z0, x0 + ddx, z0 + ddz), (x0 + ddx, z0 + ddz, x0, z0)):
                        dx = (ex - sx) * a.facing
                        stalk = f.mv('stalk')
                        sp = max(8.0, math.hypot(ex - sx, ez - sz) / seg)
                        b = type(a)(a.t0 + k * seg, stalk, seg, sx, sz, dx, ez - sz, moves.ease, None, a.facing,
                                    0.12, {'speed': sp, 'back': dx < 0}, a.surf)
                        new.append(b)
                        k += 1
                f.world.cue('steps', t=a.t0, dur=span, who=f.name, rate=moves.STEP_RATE.get(f.mv('stalk'), 4.6) * 0.6,
                            soft=True, heavy=f.heavy)
            else:
                new.append(a)
        f.actions = sorted(new, key=lambda a: a.t0)
