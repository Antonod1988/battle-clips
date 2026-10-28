"""What there is to build clips from, printed from the code itself (so it never goes out of date).

    python tools/kit.py                    stances, strikes (moves), weapons, beats with what they do, finishers,
                                           climaxes, head kinds, emblems, builds, fighters, stages
    python tools/kit.py fighter <id>       checks a fighter file and draws him in three views (out/kit/<id>.png)
    python tools/kit.py stage <name>       a stage's areas, routes and what can be smashed in each area
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path[:0] = [os.path.join(ROOT, 'engine'), os.path.join(ROOT, 'stages'), ROOT]

import body  # noqa: E402
import moves  # noqa: E402
import beats  # noqa: E402
import fighter  # noqa: E402


def first_line(fn):
    d = (fn.__doc__ or '').strip().replace('\n', ' ')
    return d.split('. ')[0][:150]


def everything():
    print('STANCES:', ', '.join(moves.STANCES))
    print('\nMOVES (strikes and poses):', ', '.join(sorted(moves.MOVES)))
    print('\nWEAPONS:', ', '.join(f"{k} ({'ice, formed' if v.get('ice') else 'steel'})" for k, v in body.WEAPONS.items()))
    print('\nBEATS (powers and phrases a fighter or a recipe can name):')
    for name, b in sorted(beats.BEATS.items()):
        print(f'  {name:18s} acts: {",".join(b["acts"]):34s} {first_line(b["fn"])}')
    print('\nCLIMAXES (signature):', ', '.join(beats.CLIMAXES))
    print('FINISHERS:', ', '.join(beats.FINISHERS))
    print('HEAD KINDS: ninja, helmet, mask, bare')
    print('EMBLEMS: star, spider, ring, bolt, diamond')
    print('BUILDS:', ', '.join(body.BUILDS))
    fs = []
    for sub in ('', 'fan'):
        d = os.path.join(ROOT, 'fighters', sub)
        fs += [f[:-3] for f in os.listdir(d) if f.endswith('.py') and not f.startswith('_')]
    print('\nFIGHTERS:', ', '.join(sorted(fs)))
    st = [f[:-3] for f in os.listdir(os.path.join(ROOT, 'stages')) if f.endswith('.py') and not f.startswith('_')
          and f != 'city_props.py']
    print('STAGES:', ', '.join(sorted(st)))


def show_fighter(fid):
    import numpy as np
    from PIL import Image
    import rig
    import px
    d = fighter.load(fid)
    print(f'{d["id"]}: {d["name"]}, tier {d["tier"]}, size {d["size"]}, stance {d["stance"]}')
    print('  powers:', ', '.join(p['beat'] for p in d['powers']) or '-')
    missing = [p['beat'] for p in d['powers'] if p['beat'] not in beats.BEATS]
    if missing:
        print('  NOT IN THE BEAT LIBRARY (write them in engine/beats.py):', ', '.join(missing))
    out = np.zeros((120, 260, 3), np.float32) + 0.22
    look = d['look_obj']
    guard = moves.MOVES[moves.STANCES[d['stance']]['guard']](0, t=0)
    for i, (view, pose) in enumerate(((0.0, guard), (1.0, moves.cross(0.42)), (-1.0, moves.run(0, t=0.1)))):
        j = rig.solve(pose, look.dims)
        K = d['K']
        root = (50 + i * 80, 105)
        jw = {k: ((root[0] + v[0] * K * (1 - 0.22 * abs(view)), root[1] + v[1] * K)
                  if isinstance(v, tuple) and len(v) == 2 and k != 'up' else v) for k, v in j.items()}
        st = body.State()
        st.view = view
        st.back = look.gear.get('scabbards', 0)
        body.SC[0] = K
        rgba, bx, by = body.draw(jw, look, st, 1, pad=12)
        px.blit(out, rgba, bx, by)
    os.makedirs(os.path.join(ROOT, 'out', 'kit'), exist_ok=True)
    p = os.path.join(ROOT, 'out', 'kit', f"{d['id']}.png")
    Image.fromarray(px.to_u8(out)).resize((260 * 4, 120 * 4), Image.NEAREST).save(p)
    print(f'  drawn: profile, three-quarter front, three-quarter back -> {p}')


def show_stage(name):
    import importlib
    mod = importlib.import_module(name)
    st = mod.Stage()
    print(f'{name}: areas')
    for a, v in st.AREAS.items():
        kinds = sorted(set(x.kind for x in st.targets(a, 0.0)))
        print(f'  {a:10s} x {v["x"]} z {v["z"]} ({v.get("label", "")}); can smash: {", ".join(kinds) or "-"}')
    print('routes:')
    for r in st.ROUTES:
        print(f'  {r["a"]} -> {r["b"]}: {r["how"]}')
    print('start positions:', st.START)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'fighter':
        show_fighter(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == 'stage':
        show_stage(sys.argv[2])
    else:
        everything()
