"""Procedural drawing engine: primitives, silhouettes, particles and the film finish."""
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 24
BAR = 132  # cinemascope letterbox height
HERE = os.path.dirname(os.path.abspath(__file__))

FONT_FILES = {
    "title": f"{HERE}/assets/fonts/Cinzel-Bold.ttf",
    "deco": f"{HERE}/assets/fonts/CinzelDeco-Black.ttf",
    "italic": f"{HERE}/assets/fonts/Cormorant-Italic.ttf",
    "serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "serifb": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "sansb": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "emoji": "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
}


@lru_cache(maxsize=None)
def font(name, size):
    return ImageFont.truetype(FONT_FILES[name], size)


# ---------------------------------------------------------------- math helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, x):
    return a + (b - a) * x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, t0, d=0.8):
    """0 before t0, eases to 1 over d seconds. t0=None -> 0."""
    if t0 is None:
        return 0.0
    return smooth((t - t0) / d)


def mix(c1, c2, x):
    return tuple(int(lerp(a, b, x)) for a, b in zip(c1, c2))


def rgba(c, a):
    return (int(c[0]), int(c[1]), int(c[2]), int(clamp(a) * 255))


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- noise
def noise1d(n, seed=0, octaves=5, base=8):
    r = rng(seed)
    out = np.zeros(n)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = base * 2 ** o
        pts = r.uniform(-1, 1, k + 3)
        x = np.linspace(0, k, n)
        i = x.astype(int)
        f = x - i
        f = f * f * (3 - 2 * f)
        out += amp * (pts[i] * (1 - f) + pts[i + 1] * f)
        tot += amp
        amp *= 0.5
    return out / tot


def noise2d(w, h, seed=0, octaves=5, base=4):
    """Smooth fractal noise in [0,1], built from upscaled random grids."""
    r = rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = base * 2 ** o
        g = r.random((max(2, int(k * h / w) + 2), k + 2)).astype(np.float32)
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += amp * np.asarray(img, np.float32) / 255
        tot += amp
        amp *= 0.55
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    return acc


# ---------------------------------------------------------------- gradients
def vgrad(stops, w=W, h=H):
    """stops: list of (pos 0..1, (r,g,b))."""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, 3), np.float32)
    ps = [s[0] for s in stops]
    for ch in range(3):
        out[:, ch] = np.interp(ys, ps, [s[1][ch] for s in stops])
    arr = np.repeat(out[:, None, :], w, axis=1)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def radial(w, h, cx, cy, r, c_in, c_out, power=1.6):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    k = np.clip(d, 0, 1) ** (1 / power)
    arr = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        arr[..., ch] = c_in[ch] * (1 - k) + c_out[ch] * k
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def glow_layer(size, draw_fn, blur, scale=4):
    """Draw at 1/scale resolution, blur, upscale. Returns RGB for screen blending."""
    w, h = size
    small = Image.new("RGB", (w // scale, h // scale), (0, 0, 0))
    d = ImageDraw.Draw(small)
    draw_fn(d, 1.0 / scale)
    small = small.filter(ImageFilter.GaussianBlur(blur / scale))
    return small.resize((w, h), Image.BILINEAR)


def screen(base, light):
    return ImageChops.screen(base, light)


def add(base, light):
    return ImageChops.add(base, light)


def paste_alpha(base, layer, alpha, pos=(0, 0)):
    """Composite RGBA layer onto base with extra global alpha."""
    if alpha <= 0:
        return
    if alpha < 1:
        a = layer.getchannel("A").point([int(v * alpha) for v in range(256)])
        base.paste(layer, pos, a)
        return
    base.paste(layer, pos, layer)


# ---------------------------------------------------------------- landscape
def ridge_points(x0, x1, base, amp, seed, step=8, octaves=6, nbase=3, bottom=H):
    xs = np.arange(x0, x1 + step, step)
    n = noise1d(len(xs), seed, octaves, nbase)
    pts = [(float(x), float(base - amp * (v * 0.5 + 0.5))) for x, v in zip(xs, n)]
    return [(x0, bottom)] + pts + [(x1, bottom)]


def pine(d, x, y, h, color):
    w = h * 0.36
    tiers = 4
    for i in range(tiers):
        ty = y - h + i * h * 0.22
        tw = w * (0.35 + 0.65 * (i + 1) / tiers)
        d.polygon([(x, ty), (x - tw, ty + h * 0.34), (x + tw, ty + h * 0.34)], fill=color)
    d.rectangle([x - h * 0.03, y - h * 0.12, x + h * 0.03, y], fill=color)


def palm(d, x, y, h, color, lean=0.15, t=0.0, seed=0):
    # curved trunk
    pts = []
    for i in range(12):
        f = i / 11
        pts.append((x + lean * h * f * f + math.sin(t * 0.8 + seed) * 4 * f, y - h * f))
    for i in range(len(pts) - 1):
        wdt = int(lerp(h * 0.05, h * 0.025, i / 11))
        d.line([pts[i], pts[i + 1]], fill=color, width=max(2, wdt))
    tx, ty = pts[-1]
    for k in range(7):
        ang = -math.pi / 2 + (k - 3) * 0.55 + math.sin(t * 1.3 + k + seed) * 0.05
        L = h * 0.45
        fr = []
        for j in range(9):
            f = j / 8
            a = ang + f * (0.5 if math.cos(ang) > 0 else -0.5)
            fr.append((tx + math.cos(a) * L * f, ty + math.sin(a) * L * f + (L * 0.18) * f * f))
        d.line(fr, fill=color, width=max(2, int(h * 0.03)))
        for j in range(2, 9):
            px, py = fr[j]
            d.line([(px, py), (px + (fr[j][0] - fr[j - 1][0]) * 0.2, py + h * 0.06)], fill=color, width=max(1, int(h * 0.012)))


def stars_layer(seed=3, n=500, horizon=H):
    r = rng(seed)
    return np.stack([r.uniform(0, W, n), r.uniform(BAR, horizon, n) ** 1.0,
                     r.uniform(0.3, 1.0, n), r.uniform(0, 6.28, n), r.uniform(0.5, 2.2, n)], 1)


def draw_stars(d, st, t, alpha=1.0):
    for x, y, b, ph, s in st:
        v = b * (0.65 + 0.35 * math.sin(t * 2.2 + ph)) * alpha
        c = int(255 * v)
        if s > 1.8:
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(c, c, min(255, c + 10)))
        else:
            d.point((x, y), fill=(c, c, min(255, c + 10)))


# ---------------------------------------------------------------- particles
class Particles:
    """Deterministic particle field; positions are closed-form functions of time."""

    def __init__(self, n, seed, vx=(0, 0), vy=(-30, -60), size=(1, 3), region=(0, BAR, W, H - BAR),
                 wobble=10, life=None):
        r = rng(seed)
        vx, vy, size = sorted(vx), sorted(vy), sorted(size)
        self.n = n
        x0, y0, x1, y1 = region
        self.region = region
        self.x = r.uniform(x0, x1, n)
        self.y = r.uniform(y0, y1, n)
        self.vx = r.uniform(*vx, n)
        self.vy = r.uniform(*vy, n)
        self.s = r.uniform(*size, n)
        self.ph = r.uniform(0, 6.28, n)
        self.b = r.uniform(0.4, 1.0, n)
        self.wob = wobble

    def pos(self, t):
        x0, y0, x1, y1 = self.region
        w, h = x1 - x0, y1 - y0
        x = x0 + (self.x - x0 + self.vx * t + np.sin(t * 0.7 + self.ph) * self.wob) % w
        y = y0 + (self.y - y0 + self.vy * t) % h
        return x, y

    def draw(self, d, t, color, alpha=1.0, flicker=0.0, streak=0):
        x, y = self.pos(t)
        for i in range(self.n):
            a = self.b[i] * alpha * (1 - flicker + flicker * (0.5 + 0.5 * math.sin(t * 7 + self.ph[i] * 3)))
            if a <= 0.02:
                continue
            c = (int(color[0] * a), int(color[1] * a), int(color[2] * a))
            s = self.s[i]
            if streak:
                d.line([(x[i], y[i]), (x[i] - self.vx[i] * streak, y[i] - self.vy[i] * streak)], fill=c, width=max(1, int(s)))
            else:
                d.ellipse([x[i] - s, y[i] - s, x[i] + s, y[i] + s], fill=c)


# ---------------------------------------------------------------- ships
def longship(scale=1.0, sail=(170, 30, 30), sail2=(225, 210, 180), hull=(18, 14, 12), lit=(60, 45, 35), oars=True):
    """Returns RGBA image of a longship, origin at waterline center."""
    s = scale
    w, h = int(700 * s), int(560 * s)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, wl = w / 2, h * 0.82
    # hull
    top = []
    for i in range(41):
        f = i / 40
        x = cx + (f - 0.5) * 520 * s
        y = wl - 40 * s - 30 * s * (abs(f - 0.5) * 2) ** 3
        top.append((x, y))
    bot = []
    for i in range(41):
        f = 1 - i / 40
        x = cx + (f - 0.5) * 470 * s
        y = wl + 6 * s - 18 * s * (1 - (abs(f - 0.5) * 2) ** 2)
        bot.append((x, y + 18 * s))
    d.polygon(top + bot, fill=hull)
    # prow and stern curls
    for side in (-1, 1):
        base = (cx + side * 255 * s, wl - 60 * s)
        pts = []
        for i in range(16):
            f = i / 15
            pts.append((base[0] + side * (40 * s * math.sin(f * 1.9)), base[1] - 120 * s * f))
        d.line(pts, fill=hull, width=int(20 * s))
        if side == 1:  # dragon head
            hx, hy = pts[-1]
            d.polygon([(hx - 8 * s, hy + 10 * s), (hx + 34 * s, hy - 6 * s), (hx + 40 * s, hy + 6 * s),
                       (hx + 10 * s, hy + 18 * s)], fill=hull)
            d.polygon([(hx - 4 * s, hy - 2 * s), (hx - 16 * s, hy - 26 * s), (hx + 4 * s, hy - 8 * s)], fill=hull)
        else:
            tx, ty = pts[-1]
            d.ellipse([tx - 18 * s, ty - 10 * s, tx + 4 * s, ty + 12 * s], outline=hull, width=int(8 * s))
    # shields
    for i in range(11):
        x = cx + (i - 5) * 40 * s
        y = wl - 42 * s
        col = (150, 30, 25) if i % 2 else (200, 170, 90)
        col = mix(col, (0, 0, 0), 0.55)
        d.ellipse([x - 15 * s, y - 15 * s, x + 15 * s, y + 15 * s], fill=col, outline=hull, width=max(1, int(3 * s)))
        d.ellipse([x - 4 * s, y - 4 * s, x + 4 * s, y + 4 * s], fill=hull)
    # oars
    if oars:
        for i in range(10):
            x = cx + (i - 4.5) * 40 * s
            d.line([(x, wl - 30 * s), (x - 45 * s, wl + 45 * s)], fill=hull, width=max(1, int(4 * s)))
    # mast + sail
    mx = cx - 10 * s
    d.line([(mx, wl - 40 * s), (mx, wl - 440 * s)], fill=hull, width=int(10 * s))
    d.line([(mx - 190 * s, wl - 400 * s), (mx + 190 * s, wl - 400 * s)], fill=hull, width=int(8 * s))
    sail_pts = [(mx - 185 * s, wl - 398 * s), (mx + 185 * s, wl - 398 * s),
                (mx + 200 * s, wl - 150 * s), (mx - 200 * s, wl - 150 * s)]
    sail_im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sail_im)
    sd.polygon(sail_pts, fill=sail)
    for k in range(-4, 5, 2):
        x0 = mx + k * 42 * s
        sd.polygon([(x0, wl - 398 * s), (x0 + 42 * s, wl - 398 * s), (x0 + 46 * s, wl - 150 * s), (x0 + 4 * s, wl - 150 * s)], fill=sail2)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(sail_pts, fill=255)
    im.paste(sail_im, (0, 0), mask)
    # rigging
    d.line([(mx, wl - 440 * s), (cx + 250 * s, wl - 70 * s)], fill=hull, width=max(1, int(3 * s)))
    d.line([(mx, wl - 440 * s), (cx - 250 * s, wl - 70 * s)], fill=hull, width=max(1, int(3 * s)))
    return im


def caravel(scale=1.0, sail=(215, 200, 170), cross=(160, 25, 25), hull=(20, 14, 10), broken=False):
    s = scale
    w, h = int(760 * s), int(760 * s)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, wl = w / 2, h * 0.86
    # hull with high stern castle (left) and forecastle (right)
    pts = [(cx - 300 * s, wl - 150 * s), (cx - 180 * s, wl - 150 * s), (cx - 170 * s, wl - 95 * s),
           (cx + 150 * s, wl - 85 * s), (cx + 170 * s, wl - 120 * s), (cx + 270 * s, wl - 125 * s),
           (cx + 320 * s, wl - 140 * s), (cx + 250 * s, wl - 40 * s), (cx + 190 * s, wl + 20 * s),
           (cx - 210 * s, wl + 25 * s), (cx - 280 * s, wl - 30 * s)]
    if broken:
        pts = [(x, y + (40 * s if x > cx else 0)) for x, y in pts]
    d.polygon(pts, fill=hull)
    # hull planking highlight
    for k in range(3):
        y = wl - 70 * s + k * 26 * s
        d.line([(cx - 250 * s, y - 10 * s), (cx + 230 * s, y + 5 * s)], fill=mix(hull, (90, 60, 35), 0.35), width=max(1, int(3 * s)))
    # bowsprit
    d.line([(cx + 290 * s, wl - 130 * s), (cx + 420 * s, wl - 210 * s)], fill=hull, width=int(8 * s))
    masts = [(cx - 190 * s, 420 * s, 0.55, "lateen"), (cx - 20 * s, 620 * s, 1.0, "square"), (cx + 170 * s, 470 * s, 0.7, "square")]
    if broken:
        masts = [(cx - 20 * s, 300 * s, 0.9, "torn")]
    for mx, mh, k, kind in masts:
        top = wl - 90 * s - mh
        d.line([(mx, wl - 90 * s), (mx, top)], fill=hull, width=int(11 * s * k + 2))
        if kind == "square":
            for lvl, (y0, y1, sw) in enumerate([(0.12, 0.52, 150), (0.56, 0.86, 170)]):
                a, b = top + mh * y0, top + mh * y1
                ww = sw * s * k
                sp = [(mx - ww * 0.8, a), (mx + ww * 0.8, a), (mx + ww, b), (mx - ww, b)]
                bulge = [(mx - ww * 0.8, a), (mx + ww * 0.8, a), (mx + ww * 1.02, (a + b) / 2), (mx + ww, b),
                         (mx, b + 18 * s * k), (mx - ww, b), (mx - ww * 1.02, (a + b) / 2)]
                d.polygon(bulge, fill=sail)
                d.line([(mx - ww * 0.9, a), (mx + ww * 0.9, a)], fill=hull, width=int(6 * s * k + 1))
                if cross and lvl == 1 and k > 0.9:
                    cy = (a + b) / 2 + 5 * s
                    cw = ww * 0.28
                    d.rectangle([mx - cw * 0.22, cy - cw, mx + cw * 0.22, cy + cw], fill=cross)
                    d.rectangle([mx - cw, cy - cw * 0.22, mx + cw, cy + cw * 0.22], fill=cross)
        elif kind == "lateen":
            d.line([(mx - 120 * s, top + mh * 0.75), (mx + 90 * s, top - 10 * s)], fill=hull, width=int(6 * s))
            d.polygon([(mx + 88 * s, top - 6 * s), (mx - 116 * s, top + mh * 0.74), (mx + 20 * s, top + mh * 0.82)], fill=sail)
        elif kind == "torn":
            d.polygon([(mx - 120 * s, top + 40 * s), (mx + 110 * s, top + 50 * s), (mx + 60 * s, top + 120 * s),
                       (mx + 10 * s, top + 90 * s), (mx - 40 * s, top + 150 * s), (mx - 90 * s, top + 100 * s)], fill=sail)
        # flag
        if kind != "torn":
            d.polygon([(mx, top), (mx + 40 * s * k, top + 10 * s), (mx, top + 20 * s)], fill=cross or sail)
    # rigging lines
    for mx, mh, k, kind in masts:
        top = wl - 90 * s - mh
        d.line([(mx, top), (cx - 290 * s, wl - 150 * s)], fill=hull, width=max(1, int(2 * s)))
        d.line([(mx, top), (cx + 300 * s, wl - 135 * s)], fill=hull, width=max(1, int(2 * s)))
    return im


_ROT = {}


def rotated(im, angle):
    """Rotated sprite, cached with the angle quantized to 0.25 degrees."""
    q = round(angle * 4)
    key = (id(im), q)
    r = _ROT.get(key)
    if r is None:
        if len(_ROT) > 3000:
            _ROT.clear()
        r = _ROT[key] = im.rotate(q / 4, resample=Image.BICUBIC, expand=True)
    return r


def rotate_paste(base, im, x, y, angle, alpha=1.0):
    """Paste ship image so its waterline center lands at (x, y)."""
    r = rotated(im, angle)
    ox = int(x - r.width / 2)
    oy = int(y - r.height / 2 - (im.height * 0.34) * math.cos(math.radians(angle)))
    if alpha < 1:
        base.paste(r, (ox, oy), r.getchannel("A").point([int(v * alpha) for v in range(256)]))
        return
    base.paste(r, (ox, oy), r)


# ---------------------------------------------------------------- figures / busts
def bust(size, look, body=(8, 7, 8)):
    """Head-and-shoulders silhouette, RGBA, size = height in px."""
    s = size / 1000
    w, h = int(900 * s), int(1000 * s)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w / 2
    c = body + (255,)
    # shoulders
    sh = [(cx - 430 * s, h), (cx - 400 * s, 760 * s), (cx - 300 * s, 660 * s), (cx - 120 * s, 610 * s),
          (cx + 120 * s, 610 * s), (cx + 300 * s, 660 * s), (cx + 400 * s, 760 * s), (cx + 430 * s, h)]
    d.polygon(sh, fill=c)
    d.rectangle([cx - 70 * s, 470 * s, cx + 70 * s, 650 * s], fill=c)  # neck
    d.ellipse([cx - 150 * s, 200 * s, cx + 150 * s, 560 * s], fill=c)  # head
    d.polygon([(cx + 140 * s, 360 * s), (cx + 175 * s, 420 * s), (cx + 140 * s, 440 * s)], fill=c)  # nose (profile hint)
    if look.startswith("viking") or look == "king":
        # beard
        if look != "viking_young":
            d.polygon([(cx - 130 * s, 430 * s), (cx + 140 * s, 430 * s), (cx + 110 * s, 640 * s), (cx + 20 * s, 740 * s),
                       (cx - 80 * s, 680 * s), (cx - 140 * s, 560 * s)], fill=c)
        else:
            d.polygon([(cx - 120 * s, 470 * s), (cx + 130 * s, 470 * s), (cx + 60 * s, 600 * s), (cx - 60 * s, 600 * s)], fill=c)
            # long hair
            d.polygon([(cx - 150 * s, 330 * s), (cx - 200 * s, 650 * s), (cx - 90 * s, 640 * s), (cx - 110 * s, 360 * s)], fill=c)
        # fur cloak
        fur = []
        for i in range(19):
            f = i / 18
            x = cx - 420 * s + 840 * s * f
            y = 700 * s - 90 * s * math.sin(f * math.pi) + (18 * s if i % 2 else -10 * s)
            fur.append((x, y))
        d.polygon(fur + [(cx + 430 * s, h), (cx - 430 * s, h)], fill=c)
    if look in ("viking", "viking_red"):
        d.chord([cx - 162 * s, 170 * s, cx + 162 * s, 470 * s], 180, 360, fill=c)  # helmet dome
        d.rectangle([cx - 165 * s, 310 * s, cx + 165 * s, 335 * s], fill=c)
        d.rectangle([cx + 105 * s, 320 * s, cx + 135 * s, 430 * s], fill=c)  # nasal
        d.line([(cx, 170 * s), (cx, 120 * s)], fill=c, width=int(10 * s))
    if look == "king":
        pts = [(cx - 150 * s, 250 * s)]
        for i in range(5):
            x = cx - 150 * s + i * 75 * s
            pts += [(x + 18 * s, 130 * s), (x + 37 * s, 220 * s)]
        pts += [(cx + 150 * s, 250 * s)]
        d.polygon(pts, fill=c)
    if look in ("woman", "woman_dark"):
        d.polygon([(cx - 160 * s, 300 * s), (cx - 230 * s, 760 * s), (cx - 60 * s, 700 * s), (cx - 110 * s, 420 * s)], fill=c)
        d.polygon([(cx + 150 * s, 300 * s), (cx + 190 * s, 720 * s), (cx + 80 * s, 690 * s), (cx + 120 * s, 440 * s)], fill=c)
        if look == "woman_dark":  # hood
            d.chord([cx - 210 * s, 150 * s, cx + 210 * s, 620 * s], 170, 370, fill=c)
            d.rectangle([cx - 200 * s, 380 * s, cx + 200 * s, 620 * s], fill=c)
            d.polygon([(cx - 210 * s, 390 * s), (cx - 330 * s, 760 * s), (cx + 330 * s, 760 * s), (cx + 210 * s, 390 * s),
                       (cx + 150 * s, 560 * s), (cx - 150 * s, 560 * s)], fill=c)
        else:  # braid bun
            d.ellipse([cx - 90 * s, 150 * s, cx + 40 * s, 250 * s], fill=c)
    if look in ("columbus", "vespucci"):
        # bob hair to the jaw
        d.polygon([(cx - 160 * s, 280 * s), (cx - 175 * s, 560 * s), (cx - 60 * s, 540 * s), (cx - 60 * s, 300 * s)], fill=c)
        d.polygon([(cx + 150 * s, 290 * s), (cx + 160 * s, 520 * s), (cx + 100 * s, 500 * s)], fill=c)
        # flat cap
        d.ellipse([cx - 210 * s, 190 * s, cx + 210 * s, 290 * s], fill=c)
        d.chord([cx - 160 * s, 130 * s, cx + 160 * s, 300 * s], 180, 360, fill=c)
        # collar
        d.polygon([(cx - 160 * s, 610 * s), (cx, 700 * s), (cx + 160 * s, 610 * s), (cx + 200 * s, 660 * s),
                   (cx, 760 * s), (cx - 200 * s, 660 * s)], fill=c)
    if look == "vespucci":
        for i in range(9):
            a = math.pi * (0.1 + 0.8 * i / 8)
            d.ellipse([cx + math.cos(a) * 170 * s - 30 * s, 620 * s - math.sin(a) * 30 * s - 30 * s,
                       cx + math.cos(a) * 170 * s + 30 * s, 620 * s - math.sin(a) * 30 * s + 30 * s], fill=c)
    if look == "priest":
        d.chord([cx - 220 * s, 140 * s, cx + 220 * s, 640 * s], 160, 380, fill=c)
        d.rectangle([cx - 210 * s, 380 * s, cx + 210 * s, 640 * s], fill=c)
        d.polygon([(cx - 220 * s, 400 * s), (cx - 420 * s, h), (cx + 420 * s, h), (cx + 220 * s, 400 * s),
                   (cx + 160 * s, 600 * s), (cx - 160 * s, 600 * s)], fill=c)
    return im


def rim_light(sil, color, strength=1.0, radius=10):
    """Colored glow behind a silhouette (returns RGBA same size)."""
    a = sil.getchannel("A")
    g = a.filter(ImageFilter.GaussianBlur(radius))
    g = g.point(lambda v: int(min(255, v * 1.6 * strength)))
    layer = Image.new("RGBA", sil.size, tuple(color) + (0,))
    layer.putalpha(g)
    return layer


# ---------------------------------------------------------------- text
def text_size(txt, f, spacing=0):
    if spacing:
        return sum(f.getlength(ch) + spacing for ch in txt) - spacing, f.getbbox("Ag")[3]
    b = f.getbbox(txt)
    return f.getlength(txt), b[3]


@lru_cache(maxsize=1024)
def _text_layer(txt, f, fill, spacing, shadow, glow):
    """Render (and cache) a text sprite with its shadow/glow; the blur is the expensive part."""
    tw, th = text_size(txt, f, spacing)
    pad = 60
    lw, lh = int(tw + pad * 2), int(f.size * 1.6 + pad * 2)
    lay = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    y0 = pad + f.size * 0.8

    def put(dd, col):
        if spacing:
            x = pad
            for ch in txt:
                dd.text((x, y0), ch, font=f, fill=col, anchor="lm")
                x += f.getlength(ch) + spacing
        else:
            dd.text((pad, y0), txt, font=f, fill=col, anchor="lm")

    if shadow or glow:
        sh = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        put(ImageDraw.Draw(sh), rgba(glow or (0, 0, 0), 0.9 if glow else 0.85))
        sh = sh.filter(ImageFilter.GaussianBlur(f.size * (0.25 if glow else 0.12)))
        lay = Image.alpha_composite(lay, sh)
    put(ImageDraw.Draw(lay), rgba(fill, 1))
    return lay, pad, y0


def draw_text(img, xy, txt, f, fill, alpha=1.0, anchor="mm", spacing=0, shadow=True, glow=None):
    """Draw (optionally letter-spaced) text with soft shadow. img is RGB; sprites are cached."""
    if alpha <= 0.01 or not txt:
        return
    lay, pad, y0 = _text_layer(txt, f, tuple(int(c) for c in fill), int(spacing), bool(shadow),
                               tuple(int(c) for c in glow) if glow else None)
    lw = lay.width
    x, y = xy
    ax, ay = anchor
    ox = {"l": x - pad, "m": x - lw / 2, "r": x - lw + pad}[ax]
    oy = y - y0 if ay == "m" else y - y0 + f.size * 0.5 if ay == "t" else y - y0 - f.size * 0.4
    paste_alpha(img, lay, alpha, (int(ox), int(oy)))


def wrap(txt, f, maxw):
    words, lines, cur = txt.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if f.getlength(test) > maxw and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


@lru_cache(maxsize=256)
def emoji(ch, size):
    f = font("emoji", 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor="mm")
    bb = im.getbbox()
    if bb:
        im = im.crop(bb)
    k = size / max(im.size)
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)


# ---------------------------------------------------------------- film finish
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_vig = 1 - 0.55 * np.clip(((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.78)) ** 2, 0, 1) ** 1.3
VIGNETTE = Image.fromarray((_vig * 255).astype(np.uint8)).convert("RGB")
del _yy, _xx, _vig
_GRAIN_CACHE = {}
WHITE = None


def grain_frames(amp, n=6):
    """Full-res grain split into (+) and (-) parts so it can be applied with saturating C ops."""
    if amp not in _GRAIN_CACHE:
        g = rng(99)
        frames = []
        for _ in range(n):
            a = g.normal(0, 1, (H // 2, W // 2)).astype(np.float32) * amp
            a = np.repeat(np.repeat(a, 2, 0), 2, 1)
            pos = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
            neg = Image.fromarray(np.clip(-a, 0, 255).astype(np.uint8)).convert("RGB")
            frames.append((pos, neg))
        _GRAIN_CACHE[amp] = frames
    return _GRAIN_CACHE[amp]


try:
    import cv2
    cv2.setNumThreads(1)  # we already run one process per core
except ImportError:  # pure-PIL fallback below
    cv2 = None
_CV = {}


def _cv_film(img, t, fade, grain, flash):
    """Vignette + grain + flash + fade on uint8 arrays with OpenCV's SIMD kernels."""
    if "vig" not in _CV:
        _CV["vig"] = np.asarray(VIGNETTE, np.uint8).copy()
    if grain and grain not in _CV:
        _CV[grain] = [(np.asarray(p).copy(), np.asarray(n).copy()) for p, n in grain_frames(float(grain))]
    a = np.asarray(img)
    out = cv2.multiply(a, _CV["vig"], scale=1 / 255)
    if grain:
        pos, neg = _CV[grain][int(t * FPS) % 6]
        cv2.add(out, pos, dst=out)
        cv2.subtract(out, neg, dst=out)
    if flash > 0.003:
        out = cv2.addWeighted(out, 1 - min(1.0, flash), out, 0, 255 * min(1.0, flash))
    if fade < 0.999:
        out = cv2.convertScaleAbs(out, alpha=fade)
    return Image.fromarray(out)


def zoom_crop(img, box):
    """Camera move: scale the box region to full frame (OpenCV when available)."""
    if cv2 is None:
        return img.resize((W, H), Image.BILINEAR, box=box)
    x0, y0, x1, y1 = box
    sx, sy = W / (x1 - x0), H / (y1 - y0)
    M = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], np.float32)
    return Image.fromarray(cv2.warpAffine(np.asarray(img), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def finish(img, t, fade=1.0, grain=6.0, tint=None, caption=None, stamp=None, stamp_alpha=0.0, flash=0.0):
    global WHITE
    if cv2 is not None and tint is None:
        out = _cv_film(img, t, fade, grain, flash)
        return _overlays(out, t, fade, caption, stamp, stamp_alpha)
    out = img
    if tint is not None:
        out = out.point([int(v * tint[c]) for c in range(3) for v in range(256)])
    out = ImageChops.multiply(out, VIGNETTE)
    if grain:
        pos, neg = grain_frames(float(grain))[int(t * FPS) % 6]
        out = ImageChops.subtract(ImageChops.add(out, pos), neg)
    if flash > 0.003:
        if WHITE is None:
            WHITE = Image.new("RGB", (W, H), (255, 255, 255))
        out = Image.blend(out, WHITE, min(1.0, flash))
    if fade < 0.999:
        out = out.point([int(v * fade) for v in range(256)] * 3)
    return _overlays(out, t, fade, caption, stamp, stamp_alpha)


def _overlays(out, t, fade, caption, stamp, stamp_alpha):
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, BAR], fill=(0, 0, 0))
    d.rectangle([0, H - BAR, W, H], fill=(0, 0, 0))
    if stamp and stamp_alpha > 0:
        f = font("title", 30)
        a = stamp_alpha * fade
        x, y = 90, BAR + 70
        d2 = ImageDraw.Draw(out)
        tw = sum(f.getlength(ch) + 5 for ch in stamp)
        draw_text(out, (x, y), stamp, f, (235, 225, 200), alpha=a, anchor="lm", spacing=5)
        d2.line([(x, y + 30), (x + tw * min(1, stamp_alpha * 1.4), y + 30)], fill=mix((0, 0, 0), (200, 160, 80), a), width=2)
    if caption:
        txt, a = caption
        if a > 0.01:
            f = font("serif", 42)
            lines = wrap(txt, f, W - 360)
            n = len(lines)
            for i, ln in enumerate(lines):
                y = H - BAR / 2 + (i - (n - 1) / 2) * 48
                draw_text(out, (W / 2, y), ln, f, (238, 234, 222), alpha=a * fade, anchor="mm", shadow=False)
    return out
