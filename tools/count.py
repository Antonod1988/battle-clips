"""Choreography check without drawing: builds a recipe's fight, steps it frame by frame (nothing rendered) and prints
the video length and the blow counts, optionally for several seeds and per beat.

    python tools/count.py recipes/scorpion-vs-subzero.py              one run
    python tools/count.py recipes/scorpion-vs-subzero.py 5 6 7 8      these seeds
    python tools/count.py recipes/scorpion-vs-subzero.py --beats      hits and clashes per beat
    python tools/count.py recipes/<clip>.py --idle                    stretches where someone stands still
"""
import bisect
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path[:0] = [os.path.join(ROOT, 'engine'), os.path.join(ROOT, 'stages'), ROOT]
import make  # noqa: E402
import moves  # noqa: E402
import render  # noqa: E402


def run(recipe, per_beat=False, idle=False):
    w = make.build_world(recipe, ) if False else make.build_world(recipe)
    snaps = []
    still = {n: [None, 0.0] for n in w.f}
    stood = []
    for i, vt, img in render.frames(w, want=lambda i, vt: False):
        snaps.append((w.t, vt, w.count['hits'], w.count['clashes']))
        if idle:
            for n, f in w.f.items():
                mv = f.move_at(w.t)
                last = f.last
                moving = last is not None and mv not in moves.STATIC
                if mv in moves.STATIC and not w.is_hidden(n):
                    if still[n][0] is None:
                        still[n] = [w.t, vt]
                else:
                    if still[n][0] is not None and w.t - still[n][0] >= 0.45:
                        stood.append((n, still[n][1], vt))
                    still[n] = [None, 0.0]
    n = len(snaps)
    print(f'SEED {recipe.get("seed")}: video {n / 30:.1f}s, {w.count}, {len(w.choreo.history)} beats')
    if per_beat:
        ts = [s[0] for s in snaps]

        def at(t):
            return snaps[min(len(snaps) - 1, bisect.bisect_left(ts, t))]
        for (name, who, t0, t1) in w.choreo.history:
            a, b = at(t0), at(t1)
            print(f'  {name:16s} {who:16s} {a[1]:6.1f}-{b[1]:6.1f}s  hits {b[2] - a[2]:2d}  clashes {b[3] - a[3]:2d}')
    if idle:
        for (who, a, b) in stood:
            print(f'  STILL {who} {a:.2f}-{b:.2f}s')
        print(f'  {len(stood)} stretches of standing still (>= 0.45 s)')
    return w


if __name__ == '__main__':
    args = [a for a in sys.argv[2:] if not a.startswith('--')]
    r = make.load_recipe(sys.argv[1])
    for sd in (args or [r.get('seed', 1)]):
        r2 = dict(r)
        r2['seed'] = int(sd)
        run(r2, '--beats' in sys.argv, '--idle' in sys.argv)
