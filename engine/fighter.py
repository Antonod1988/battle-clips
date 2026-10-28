"""Loading fighters from their data files (fighters/<name>.py or fighters/fan/<name>.py, a dict named FIGHTER).

    load('scorpion')            -> the spec dict the engine uses ('look_obj', 'K' added)
    find('scorpion')            -> the file path

Missing keys get defaults so a short file works; check() lists what is wrong in a new fighter file.
"""
import importlib.util
import os

import body
import moves

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fighters')
BASE_K = 0.75                     # a normal man is about 46 px tall at this size

DEFAULTS = {
    'tier': 3, 'size': 1.0, 'stance': 'ninja', 'title_colors': 'gold', 'trail': 'ffffff', 'element': 'body',
    'weapons': [], 'powers': [], 'signature': None, 'breakout': None, 'finisher': 'knockout', 'entrance': 'walk_in',
    'voice': 'announcer',
}
STRIKE_KINDS = ('light', 'heavy', 'launcher', 'low', 'air', 'leap')


def find(name):
    """A fighter id or a path to its file."""
    if name.endswith('.py') and os.path.exists(name):
        return name
    for sub in ('', 'fan'):
        p = os.path.join(ROOT, sub, name + '.py')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no fighter "{name}" in fighters/ or fighters/fan/')


def load(name):
    path = find(name)
    spec = importlib.util.spec_from_file_location('fighter_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = dict(DEFAULTS)
    d.update(mod.FIGHTER)
    d['path'] = path
    d.setdefault('short', d['name'])
    st = {k: list(d.get('strikes', {}).get(k, [])) for k in STRIKE_KINDS}
    if not st['light']:
        st['light'] = ['jab', 'cross']
    if not st['heavy']:
        st['heavy'] = ['hook', 'roundhouse']
    d['strikes'] = st
    d['look_obj'] = body.Look.from_data(d['id'], d['look'])
    d['K'] = BASE_K * float(d.get('size', 1.0))
    problems = check(d)
    if problems:
        raise ValueError(f'fighter {path}: ' + '; '.join(problems))
    return d


def check(d):
    out = []
    for k in ('id', 'name', 'look'):
        if k not in d:
            out.append(f'missing "{k}"')
    if d.get('stance') not in moves.STANCES:
        out.append(f'stance must be one of {sorted(moves.STANCES)}')
    for kind, names in d.get('strikes', {}).items():
        for n in names:
            if n not in moves.MOVES:
                out.append(f'strike "{n}" is not a move in engine/moves.py')
    for wpn in d.get('weapons', []):
        if wpn['kind'] not in body.WEAPONS:
            out.append(f'weapon "{wpn["kind"]}" is not in body.WEAPONS')
    if not 1 <= int(d.get('tier', 3)) <= 5:
        out.append('tier must be 1..5')
    return out


def has_power(d, beat):
    return any(p['beat'] == beat for p in d.get('powers', []))


def power_spec(d, beat):
    for p in d.get('powers', []):
        if p['beat'] == beat:
            return p
    return None
