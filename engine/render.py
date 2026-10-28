"""Frames to files: the 1080p master, the copy under 30 MB, the moments sheet, the contact sheet, the cue sheet; the
fast preview; stills.

Everything is drawn at 480x270 and scaled 4:1 nearest by ffmpeg (no zoom, one pixel grid). No game window.
"""
import math
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

import px
import config


def frames(world, want=None, fps=30):
    """Steps the world one video frame at a time (slow motion: less scene time per frame); yields (i, video_t, img)."""
    i = 0
    while world.t < world.end:
        dt = world.rate(world.t) / fps
        world.advance(dt)
        vt = i / fps
        if want is None or want(i, vt):
            yield i, vt, world.render()
        else:
            yield i, vt, None
        i += 1


def big(img):
    return Image.fromarray(px.to_u8(img)).resize((px.W * px.SCALE, px.H * px.SCALE), Image.NEAREST)


def sheet(items, path, cols=4, tile=(480, 270)):
    if not items:
        return
    rows = int(math.ceil(len(items) / cols))
    out = Image.new('RGB', (cols * tile[0], rows * tile[1]), (10, 10, 12))
    d = ImageDraw.Draw(out)
    for k, (label, img) in enumerate(items):
        im = Image.fromarray(px.to_u8(img)).resize(tile, Image.NEAREST)
        x, y = (k % cols) * tile[0], (k // cols) * tile[1]
        out.paste(im, (x, y))
        d.rectangle([x + 4, y + 4, x + 8 + 7 * len(label), y + 18], fill=(0, 0, 0))
        d.text((x + 7, y + 6), label, fill=(255, 255, 255))
    out.save(path)


def encoder(path, preview=False, fps=30):
    ff = config.FFMPEG
    if preview:
        vf = 'scale=960:540:flags=neighbor,format=yuv420p'
        codec = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26']
    else:
        vf = 'scale=1920:1080:flags=neighbor:out_color_matrix=bt709:out_range=tv,format=yuv420p'
        codec = ['-c:v', 'libx264', '-preset', 'slow', '-tune', 'animation', '-crf', '17', '-colorspace', 'bt709',
                 '-color_primaries', 'bt709', '-color_trc', 'bt709']
    cmd = [ff, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
           f'{px.W}x{px.H}', '-r', str(fps), '-i', '-', '-vf', vf] + codec + \
          ['-threads', str(config.THREADS), '-movflags', '+faststart', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def video(world, out_dir, name, t_from=None, t_to=None, preview=False, log=print):
    """Renders the clip. Full: the silent master (sound is muxed later), sheets and the cue sheet. Preview: every
    second frame (15 fps) at half size, fast. t_from/t_to: only that window of video seconds is encoded."""
    os.makedirs(out_dir, exist_ok=True)
    fps = config.FPS
    step = 2 if preview else 1
    suffix = '-preview' if preview else ''
    if t_from is not None or t_to is not None:
        suffix += f'-{t_from or 0:g}-{t_to or 999:g}s'
    master = os.path.join(out_dir, f'{name}{suffix}-silent.mp4' if not preview else f'{name}{suffix}.mp4')
    enc = encoder(master, preview, fps // step)
    keep = {}
    marks = sorted(world.marks.items(), key=lambda kv: kv[1])
    t0 = time.time()
    n = 0
    lo = t_from or 0.0
    hi = t_to if t_to is not None else 1e9

    def want(i, vt):
        if not (lo <= vt <= hi):
            return False
        return i % step == 0

    last_report = time.time()
    for i, vt, img in frames(world, want, fps):
        if img is None:
            if vt > hi:
                break
            continue
        enc.stdin.write(px.to_u8(img).tobytes())
        if i % 15 == 0 or (preview and i % 16 == 0):
            keep[('t', i)] = (f'{vt:5.2f}s', img)
        for mname, mt in marks:
            if ('m', mname) not in keep and world.t >= mt + 0.12:
                keep[('m', mname)] = (f'{mname} {vt:5.2f}s', img)
        n += 1
        if time.time() - last_report > 20:
            log(f'  {i} frames, {vt:.1f}s of video, {time.time() - t0:.0f}s')
            last_report = time.time()
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit('ffmpeg failed on the master')
    total_frames = len(world.frame_t)
    seconds = total_frames / fps
    log(f'RENDER {name}{suffix}: {n} frames encoded, clip {seconds:.2f}s, {time.time() - t0:.0f}s')
    world.export_cues(os.path.join(out_dir, f'{name}-cues.json'), fps)
    tl = [v for k, v in sorted(keep.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][0] == 't' else 0)) if k[0] == 't']
    if tl:
        pick = np.linspace(0, len(tl) - 1, min(len(tl), 32)).round().astype(int)
        sheet([tl[i] for i in pick], os.path.join(out_dir, f'{name}{suffix}-contact.png'), cols=4)
    key = set(world.key_moments)
    moms = [keep[('m', m)] for m, _ in marks if ('m', m) in keep and m in key]
    sheet(moms, os.path.join(out_dir, f'{name}{suffix}-moments.png'), cols=4)
    if not preview:
        os.makedirs(os.path.join(out_dir, 'moments'), exist_ok=True)
        for k, (mname, _) in enumerate(marks):
            if ('m', mname) in keep:
                big(keep[('m', mname)][1]).save(os.path.join(out_dir, 'moments', f'{k + 1:02d}-{mname}.png'))
    return master, seconds


def small_copy(master, out, seconds, audio_kbps=192, log=print):
    """The copy under 30 MB: two-pass libx264 at a bitrate from the length, leaving room for the sound."""
    ff = config.FFMPEG
    budget = config.COPY_MB * 0.93 * 8 * 1000 / seconds - audio_kbps
    master_kbps = os.path.getsize(master) * 8 / 1000 / seconds
    kbps = int(min(budget, master_kbps * 0.85))
    passlog = os.path.join(os.path.dirname(out), 'x264-pass')
    for p in (1, 2):
        dst = out if p == 2 else 'NUL'
        fmt = [] if p == 2 else ['-f', 'mp4']
        subprocess.check_call([ff, '-hide_banner', '-loglevel', 'error', '-y', '-i', master, '-an', '-c:v', 'libx264',
                               '-preset', 'slow', '-tune', 'animation', '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.6)}k',
                               '-bufsize', f'{kbps * 2}k', '-pass', str(p), '-passlogfile', passlog, '-threads',
                               str(config.THREADS), '-pix_fmt', 'yuv420p', '-movflags', '+faststart'] + fmt + [dst])
    for f in os.listdir(os.path.dirname(out)):
        if f.startswith('x264-pass'):
            os.remove(os.path.join(os.path.dirname(out), f))
    log(f'COPY {out}: {os.path.getsize(out)} bytes at {kbps} kbps')
    return kbps


def stills(world, out_dir, times_, fps=30):
    """Stills (1920x1080) at video seconds, or 'mark+offset' scene moments, and a sheet of them."""
    os.makedirs(out_dir, exist_ok=True)
    want_frames = sorted(int(round(float(t) * fps)) for t in times_ if not t[0].isalpha())
    scene_t = []
    for a in times_:
        if a[0].isalpha():
            nm, _, off = a.partition('+')
            if nm not in world.marks:
                print(f'STILLS: no moment "{nm}" (moments: {", ".join(sorted(world.marks))})')
                continue
            scene_t.append(world.marks[nm] + float(off or 0))
    items = []
    pending = sorted(scene_t)
    for i, vt, img in frames(world, want=lambda i, vt: i in want_frames, fps=fps):
        hit = False
        while pending and world.t >= pending[0]:
            pending.pop(0)
            hit = True
        if i in want_frames or hit:
            if img is None:
                img = world.render()
            big(img).save(os.path.join(out_dir, f'p-{vt:06.2f}.png'))
            items.append((f'{vt:.2f}s', img))
        if not pending and (not want_frames or i > want_frames[-1]):
            break
    for k in range(0, len(items), 16):
        sheet(items[k:k + 16], os.path.join(out_dir, f'stills{k // 16 + 1}.png'), cols=4)
    return items
