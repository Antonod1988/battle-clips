"""The clip's words: a heavy system face thresholded to whole art pixels, then a colour ramp, a bevel, an outline
and a drop shadow, so every word sits on the same 4:1 pixel grid as the art (from VERSUS-1 v3)."""
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import binary_dilation

from px import ramp

FONTS = {'big': ('C:/Windows/Fonts/impact.ttf', 38), 'mid': ('C:/Windows/Fonts/impact.ttf', 30),
         'name': ('C:/Windows/Fonts/ariblk.ttf', 19), 'huge': ('C:/Windows/Fonts/impact.ttf', 50),
         'small': ('C:/Windows/Fonts/ariblk.ttf', 12)}
FIRE = ramp('fff7c0', 'ffd84a', 'ffa21c', 'e0600e', 'a8320a')
ICE = ramp('ffffff', 'c8f0ff', '86ccf4', '3f8ee0', '1f4fa8')
BLOOD = ramp('ff9070', 'ff3a24', 'd8140c', 'a00a08', '5a0404')
GOLD = ramp('fffbe0', 'ffe680', 'f4b830', 'c07a10', '6a3a06')
BONE = ramp('ffffff', 'f0ece2', 'd2cbbd', 'a69e8c', '6c6556')
HULK = ramp('eaffb8', 'aef064', '5cc432', '2e7c18', '144a0a')
CAP = ramp('ffffff', 'dce6ff', '7c9eff', '2c4cd2', '142c8c')
RED = ramp('ffd8c8', 'ff7a5a', 'e02a1e', 'a81410', '5a0806')
_cache = {}


def word(text, font='big', colors=FIRE, spacing=1):
    key = (text, font, id(colors))
    if key in _cache:
        return _cache[key]
    path, size = FONTS[font]
    f = ImageFont.truetype(path, size)
    im = Image.new('L', (len(text) * size + 20, size * 2), 0)
    d = ImageDraw.Draw(im)
    x = 4
    for ch in text:
        d.text((x, 2), ch, font=f, fill=255)
        x += int(round(d.textlength(ch, font=f))) + spacing
    m = np.asarray(im) > 120
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    pad = 4
    big = np.zeros((h + pad * 2, w + pad * 2), bool)
    big[pad:pad + h, pad:pad + w] = m
    rgba = np.zeros(big.shape + (4,), np.float32)
    # shadow, outline, fill
    sh = np.zeros_like(big)
    sh[2:, 2:] = big[:-2, :-2]
    sh = binary_dilation(sh, iterations=1)
    rgba[sh] = [0.0, 0.0, 0.0, 0.6]
    ol = binary_dilation(big, structure=np.ones((3, 3), bool))
    rgba[ol] = [0.05, 0.02, 0.03, 1.0]
    yy = np.arange(big.shape[0])[:, None].repeat(big.shape[1], 1)
    k = np.clip((yy - pad) / max(1, h - 1), 0, 1)
    idx = np.clip((k * (len(colors) - 0.01)).astype(int), 0, len(colors) - 1)
    fill = colors[idx]
    rgba[big, :3] = fill[big]
    rgba[big, 3] = 1.0
    # bevel: a bright top edge, a dark bottom edge inside each stroke
    top = big & ~np.roll(big, 1, axis=0)
    bot = big & ~np.roll(big, -1, axis=0)
    rgba[top, :3] = np.minimum(1.0, rgba[top, :3] * 0.5 + 0.55)
    rgba[bot, :3] = rgba[bot, :3] * 0.55
    _cache[key] = rgba
    return rgba


class Word:
    def __init__(self, text, t0, t1, cx, cy, font='big', colors=FIRE, drop=8, flash=0.08, fade=0.25):
        self.img = word(text, font, colors)
        self.t0, self.t1 = t0, t1
        self.cx, self.cy = cx, cy
        self.drop, self.flash, self.fade = drop, flash, fade

    def draw(self, frame, t):
        if t < self.t0 or t > self.t1 + self.fade:
            return
        a = 1.0
        if t > self.t1:
            a = max(0.0, 1 - (t - self.t1) / self.fade)
        k = min(1.0, (t - self.t0) / 0.09)
        dy = int(round(self.drop * (1 - k) ** 2))
        img = self.img
        if t - self.t0 < self.flash:
            img = img.copy()
            m = img[..., 3] > 0.99
            img[m, :3] = img[m, :3] * 0.3 + 0.7
        h, w = img.shape[:2]
        x = int(round(self.cx - w / 2))
        y = int(round(self.cy - h / 2)) - dy
        from px import blit
        blit(frame, img, x, y, a)
