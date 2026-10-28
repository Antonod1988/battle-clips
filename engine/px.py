"""Pixel helpers for the VERSUS-1 v3 clip: colours, a float RGB frame, blits, glows, noise.

The art is drawn at 480x270 and shown 4:1 (nearest) at 1920x1080. Colours are float32 0..1.
"""
import numpy as np

W, H = 480, 270
SCALE = 4


def hexc(s):
    s = s.lstrip('#')
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def ramp(*hexes):
    return np.stack([hexc(h) for h in hexes]).astype(np.float32)


def mix(a, b, t):
    return a + (b - a) * t


def hash_noise(xs, ys, seed=0):
    """Deterministic white noise in [0,1) for integer grids (numpy arrays)."""
    x = np.asarray(xs).astype(np.int64)
    y = np.asarray(ys).astype(np.int64)
    h = (x * 374761393 + y * 668265263 + seed * 982451653) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float32) / 65536.0


def value_noise(w, h, cell, seed=0):
    """Smooth value noise (w x h), lattice spacing `cell` pixels."""
    gx = np.arange(w) / cell
    gy = np.arange(h) / cell
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    X0, Y0 = np.meshgrid(x0, y0)
    FX, FY = np.meshgrid(fx, fy)
    a = hash_noise(X0, Y0, seed)
    b = hash_noise(X0 + 1, Y0, seed)
    c = hash_noise(X0, Y0 + 1, seed)
    d = hash_noise(X0 + 1, Y0 + 1, seed)
    return (a * (1 - FX) + b * FX) * (1 - FY) + (c * (1 - FX) + d * FX) * FY


def fbm(w, h, cell, seed=0, octaves=3):
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        out += value_noise(w, h, max(1.0, cell / (2 ** o)), seed + o * 17) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def blit(dst, rgba, x, y, alpha=1.0):
    """Alpha-composite an RGBA float array onto an RGB float array at integer (x, y)."""
    h, w = rgba.shape[:2]
    x = int(x)
    y = int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] * alpha
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + src[..., :3] * a


def add_rgba(dst, rgba, x, y, strength=1.0):
    """Additive blit (emissive)."""
    h, w = rgba.shape[:2]
    x = int(x)
    y = int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    dst[y0:y1, x0:x1] += src[..., :3] * src[..., 3:4] * strength


def glow(dst, cx, cy, radius, color, strength=1.0, power=2.0, squash=1.0):
    """Additive radial glow centred at (cx, cy); squash < 1 flattens it vertically."""
    r = radius
    x0 = int(max(0, np.floor(cx - r)))
    x1 = int(min(dst.shape[1], np.ceil(cx + r) + 1))
    y0 = int(max(0, np.floor(cy - r * squash)))
    y1 = int(min(dst.shape[0], np.ceil(cy + r * squash) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xs + 0.5 - cx) / r) ** 2 + ((ys + 0.5 - cy) / (r * squash)) ** 2)
    f = np.clip(1 - d, 0, 1) ** power * strength
    dst[y0:y1, x0:x1] += f[..., None] * np.asarray(color, np.float32)


def boost(dst, cx, cy, radius, color, strength=1.0, squash=1.0):
    """Multiplicative light: brightens what is there (a lamp on a wall) instead of adding colour."""
    r = radius
    x0 = int(max(0, np.floor(cx - r)))
    x1 = int(min(dst.shape[1], np.ceil(cx + r) + 1))
    y0 = int(max(0, np.floor(cy - r * squash)))
    y1 = int(min(dst.shape[0], np.ceil(cy + r * squash) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xs + 0.5 - cx) / r) ** 2 + ((ys + 0.5 - cy) / (r * squash)) ** 2)
    f = np.clip(1 - d, 0, 1) ** 1.6 * strength
    dst[y0:y1, x0:x1] *= 1 + f[..., None] * np.asarray(color, np.float32)


def put(dst, x, y, color, alpha=1.0):
    x = int(round(x))
    y = int(round(y))
    if 0 <= x < dst.shape[1] and 0 <= y < dst.shape[0]:
        dst[y, x] = dst[y, x] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def line_points(x0, y0, x1, y1):
    """Bresenham points between two integer points (inclusive)."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    pts = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
