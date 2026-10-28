"""One command from a recipe to the finished clip.

    python make.py recipes/scorpion-vs-subzero.py            the 1080p mp4 with sound, the copy under 30 MB, the
                                                              moments and contact sheets, into out/<clip>/ and the
                                                              lab-review folder the recipe names
    python make.py recipes/<clip>.py --preview               fast: 15 fps, half size, no sound (out/<clip>/)
    python make.py recipes/<clip>.py --from 40 --to 60       only that stretch of video seconds (works with --preview)
    python make.py recipes/<clip>.py --stills 12.5 30 hellfire+0.2
                                                              1080p stills at video seconds or named moments
    python make.py recipes/<clip>.py --plan                  only the choreography: the beats, the time, the counts

Nothing is published: the files stay on this machine (fan fighters are private).
"""
import argparse
import importlib.util
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
sys.path.insert(0, os.path.join(ROOT, 'stages'))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import fighter  # noqa: E402
import choreo  # noqa: E402
import render  # noqa: E402


def load_recipe(path):
    if not path.endswith('.py'):
        cand = os.path.join(ROOT, 'recipes', os.path.basename(path) + '.py')
        path = cand if os.path.exists(cand) else path + '.py'
    spec = importlib.util.spec_from_file_location('recipe', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = dict(mod.RECIPE)
    r.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    r['path'] = path
    return r


def build_world(recipe):
    specs = [fighter.load(n) for n in recipe['fighters']]
    stage_mod = importlib.import_module(recipe['stage'])
    return choreo.build(recipe, stage_mod, specs)


def main():
    ap = argparse.ArgumentParser(description='Render a battle clip from its recipe.')
    ap.add_argument('recipe')
    ap.add_argument('--preview', action='store_true', help='fast, low quality, no sound')
    ap.add_argument('--from', dest='t_from', type=float, default=None, help='start (video seconds)')
    ap.add_argument('--to', dest='t_to', type=float, default=None, help='end (video seconds)')
    ap.add_argument('--stills', nargs='+', default=None, help='video seconds or mark+offset')
    ap.add_argument('--plan', action='store_true', help='print the choreography only')
    ap.add_argument('--no-review', action='store_true', help='do not copy into the lab-review folder')
    a = ap.parse_args()
    t0 = time.time()
    recipe = load_recipe(a.recipe)
    name = recipe.get('output', recipe['name'])
    out_dir = os.path.join(config.OUT, recipe['name'])
    world = build_world(recipe)
    c = world.choreo
    vlen = choreo.video_time(world, world.end)
    print(f'PLAN {recipe["name"]}: {len(c.history)} beats, scene {world.end:.1f}s, video {vlen:.1f}s '
          f'(built in {time.time() - t0:.1f}s)')
    if a.plan:
        for line in c.log:
            print(line)
        return
    if a.stills:
        render.stills(world, os.path.join(out_dir, 'stills'), a.stills)
        print(f'STILLS -> {os.path.join(out_dir, "stills")}')
        return
    if a.preview or a.t_from is not None or a.t_to is not None:
        path, secs = render.video(world, out_dir, name, a.t_from, a.t_to, preview=a.preview)
        print(f'PREVIEW {path} ({time.time() - t0:.0f}s); blows: {world.count}')
        return
    silent, secs = render.video(world, out_dir, name)
    import sound
    m4a, loud = sound.make_track(os.path.join(out_dir, f'{name}-cues.json'), out_dir, name, recipe.get('music', 'ninja'))
    master = os.path.join(out_dir, f'{name}-1080p.mp4')
    sound.mux(silent, m4a, master)
    small_silent = os.path.join(out_dir, f'{name}-small-silent.mp4')
    render.small_copy(silent, small_silent, secs)
    small = os.path.join(out_dir, f'{name}-under-30mb.mp4')
    sound.mux(small_silent, m4a, small)
    os.remove(small_silent)
    size = os.path.getsize(small)
    if size >= config.COPY_MB * 1_000_000:
        raise SystemExit(f'the small copy is too big: {size} bytes')
    with open(os.path.join(out_dir, f'{name}-log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(c.log + [f'blows {world.count}', f'video {secs:.2f}s', f'audio {loud}']))
    print(f'DONE {master} ({os.path.getsize(master)} bytes), {small} ({size} bytes), {secs:.1f}s, '
          f'blows {world.count}, {time.time() - t0:.0f}s')
    review = recipe.get('review_dir')
    if review and not a.no_review:
        dst = os.path.join(config.LAB_REVIEW, review)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(out_dir):
            p = os.path.join(out_dir, f)
            if os.path.isfile(p) and f.startswith(name) and not f.endswith('-silent.mp4') and '-preview' not in f:
                shutil.copy2(p, os.path.join(dst, f))
        if os.path.isdir(os.path.join(out_dir, 'moments')):
            dm = os.path.join(dst, 'moments')
            os.makedirs(dm, exist_ok=True)
            for f in os.listdir(dm):                    # the review folder gets exactly this render's moments
                if f.endswith('.png'):
                    os.remove(os.path.join(dm, f))
            shutil.copytree(os.path.join(out_dir, 'moments'), dm, dirs_exist_ok=True)
        print(f'REVIEW {dst}')


if __name__ == '__main__':
    main()
