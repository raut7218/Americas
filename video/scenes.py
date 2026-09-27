"""Scene renderers. Each scene draws one RGB frame at time t (seconds from scene start)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import maps
from engine import (BAR, FPS, H, W, Particles, bust, caravel, clamp, draw_stars, draw_text, ease_in_out, ease_out,
                    emoji, finish, font, glow_layer, lerp, longship, mix, noise1d, noise2d, palm, paste_alpha, pine,
                    radial, ramp, rgba, ridge_points, rim_light, rng, rotate_paste, screen, smooth, stars_layer, vgrad,
                    wrap)

REG = {}


def scene(kind):
    def deco(cls):
        REG[kind] = cls
        return cls
    return deco


class Scene:
    zoom = (1.0, 1.06)
    fade_in = 0.45
    fade_out = 0.45
    grain = 5.0

    def __init__(self, spec, tl):
        self.p = spec
        self.dur = tl["dur"]
        self.chunks = tl["chunks"]
        self.setup()

    def setup(self):
        pass

    # cue helpers ---------------------------------------------------
    def cue(self, k):
        if isinstance(k, int) and 0 <= k < len(self.chunks):
            return self.chunks[k]["start"]
        return None

    def cue_named(self, name):
        for k, v in self.p.get("cues", {}).items():
            if v == name:
                return self.cue(k)
        return None

    def chunk_end(self, k):
        return self.chunks[k]["end"] if 0 <= k < len(self.chunks) else self.dur

    def caption(self, t):
        for c in self.chunks:
            if c["start"] - 0.15 <= t <= c["end"] + 0.3:
                a = min(ramp(t, c["start"] - 0.15, 0.2), 1 - ramp(t, c["end"] + 0.1, 0.2))
                return c["text"], a
        return None

    # frame ---------------------------------------------------------
    def draw(self, t):
        return Image.new("RGB", (W, H), (0, 0, 0))

    def render(self, t):
        img = self.draw(t)
        z0, z1 = self.zoom
        z = lerp(z0, z1, ease_in_out(t / max(self.dur, 0.01)))
        if abs(z - 1) > 1e-3:
            cw, ch = W / z, H / z
            x0, y0 = (W - cw) / 2, (H - ch) / 2
            img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
        fade = min(ramp(t, 0, self.fade_in) if self.fade_in else 1, 1 - ramp(t, self.dur - self.fade_out, self.fade_out) if self.fade_out else 1)
        stamp = self.p.get("stamp")
        sa = ramp(t, 0.6, 1.0) * (1 - ramp(t, 6.5, 1.0)) if stamp else 0
        return finish(img, t, fade=fade, grain=self.grain, caption=self.caption(t), stamp=stamp, stamp_alpha=sa,
                      flash=self.flash(t))

    def flash(self, t):
        return 0.0


# ======================================================================= helpers
def person(d, x, y, h, color, kind="plain", step=0.0):
    """Standing figure silhouette with feet at (x, y)."""
    hw = h * 0.13
    d.ellipse([x - h * 0.075, y - h, x + h * 0.075, y - h + h * 0.16], fill=color)
    body = [(x - hw, y - h * 0.8), (x + hw, y - h * 0.8), (x + hw * 0.8, y - h * 0.42), (x - hw * 0.8, y - h * 0.42)]
    d.polygon(body, fill=color)
    lg = math.sin(step) * h * 0.08
    d.line([(x - hw * 0.4, y - h * 0.45), (x - hw * 0.5 - lg, y)], fill=color, width=max(2, int(h * 0.07)))
    d.line([(x + hw * 0.4, y - h * 0.45), (x + hw * 0.5 + lg, y)], fill=color, width=max(2, int(h * 0.07)))
    d.line([(x - hw, y - h * 0.78), (x - hw * 1.4, y - h * 0.45)], fill=color, width=max(2, int(h * 0.055)))
    d.line([(x + hw, y - h * 0.78), (x + hw * 1.4, y - h * 0.45)], fill=color, width=max(2, int(h * 0.055)))
    if kind == "spear":
        d.line([(x + hw * 1.4, y - h * 1.25), (x + hw * 1.4, y - h * 0.1)], fill=color, width=max(2, int(h * 0.03)))
        d.polygon([(x + hw * 1.4, y - h * 1.35), (x + hw * 1.25, y - h * 1.22), (x + hw * 1.55, y - h * 1.22)], fill=color)
    if kind == "helmet":
        d.chord([x - h * 0.09, y - h * 1.03, x + h * 0.09, y - h * 0.87], 180, 360, fill=color)
        d.line([(x + hw * 1.4, y - h * 1.1), (x + hw * 1.4, y - h * 0.1)], fill=color, width=max(2, int(h * 0.03)))
    if kind == "banner":
        d.line([(x + hw * 1.4, y - h * 1.5), (x + hw * 1.4, y - h * 0.1)], fill=color, width=max(2, int(h * 0.03)))
        d.polygon([(x + hw * 1.4, y - h * 1.5), (x + hw * 1.4 + h * 0.35, y - h * 1.4), (x + hw * 1.4, y - h * 1.2)], fill=color)
    if kind == "hat":
        d.ellipse([x - h * 0.12, y - h * 0.98, x + h * 0.12, y - h * 0.92], fill=color)
    if kind == "bent":
        pass


SKIES = {
    "night": dict(sky=[(0, (4, 6, 16)), (0.45, (10, 16, 32)), (0.6, (24, 34, 54))], sea=(8, 14, 26), near=(2, 4, 8),
                  crest=(60, 80, 110), sun=None, moon=(1350, 300)),
    "dawn": dict(sky=[(0, (22, 28, 60)), (0.35, (110, 80, 110)), (0.52, (235, 150, 100)), (0.6, (255, 205, 140))],
                 sea=(90, 70, 80), near=(12, 10, 18), crest=(250, 190, 140), sun=(1120, 610, (255, 220, 160))),
    "day": dict(sky=[(0, (40, 80, 130)), (0.45, (120, 160, 190)), (0.6, (200, 210, 210))], sea=(50, 90, 110),
                near=(8, 22, 34), crest=(200, 220, 230), sun=(1500, 260, (255, 250, 230))),
    "dusk": dict(sky=[(0, (16, 12, 36)), (0.3, (70, 30, 60)), (0.5, (190, 70, 50)), (0.6, (250, 150, 60))],
                 sea=(70, 35, 40), near=(10, 5, 10), crest=(250, 140, 70), sun=(700, 600, (255, 170, 80))),
    "fog": dict(sky=[(0, (70, 80, 88)), (0.6, (150, 158, 160))], sea=(90, 100, 105), near=(18, 24, 28),
                crest=(170, 180, 180), sun=None),
}
HORIZON = int(H * 0.58)


class Sea:
    """Layered animated ocean with optional ships."""

    def __init__(self, tod="day", weather="calm", seed=1, horizon=HORIZON, dark=False):
        pal = SKIES["fog" if weather == "fog" else tod]
        self.pal = pal
        self.h0 = horizon
        self.storm = 2.2 if weather == "storm" else 1.0
        self.weather = weather
        stops = [(p, c) for p, c in pal["sky"]]
        sky = vgrad([(p * H / horizon * 0.6 if p else 0, c) for p, c in stops] + [(1, stops[-1][1])], W, horizon)
        sky = sky.resize((W, horizon))
        if pal.get("sun"):
            sx, sy, sc = pal["sun"]
            g = radial(W, horizon, sx, min(sy, horizon - 10), 700, sc, (0, 0, 0), power=0.6)
            sky = screen(sky, g)
            dd = ImageDraw.Draw(sky)
            if weather != "fog":
                dd.ellipse([sx - 60, sy - 60, sx + 60, sy + 60], fill=mix(sc, (255, 255, 255), 0.4))
        if pal.get("moon"):
            mx, my = pal["moon"]
            g = radial(W, horizon, mx, my, 420, (70, 90, 120), (0, 0, 0), power=0.5)
            sky = screen(sky, g)
            dd = ImageDraw.Draw(sky)
            dd.ellipse([mx - 46, my - 46, mx + 46, my + 46], fill=(225, 230, 235))
            dd.ellipse([mx - 30, my - 50, mx + 62, my + 42], fill=mix(stops[1][1], (225, 230, 235), 0.1))
        if dark:
            a = np.asarray(sky, np.float32)
            g = a.mean(2, keepdims=True)
            a = (a * 0.35 + g * 0.65) * np.array([0.9, 0.55, 0.5])
            sky = Image.fromarray(a.clip(0, 255).astype(np.uint8))
            self.pal = dict(pal, sea=mix(pal["sea"], (30, 10, 10), 0.6), crest=mix(pal["crest"], (120, 60, 50), 0.6))
        self.sky = sky
        # clouds: a wide strip we scroll
        nz = noise2d(W * 2 // 4, horizon // 4, seed=seed + 10, octaves=5, base=6)
        dens = 0.35 if weather == "calm" else 0.55 if weather == "fog" else 0.75
        a = np.clip((nz - (1 - dens)) / dens, 0, 1) ** 1.3
        top_c = np.array(stops[0][1], np.float32) * (0.6 if tod != "night" else 1.8) + (6 if tod == "night" else 0)
        bot_c = np.array(stops[-1][1], np.float32) * 0.9
        yy = np.linspace(0, 1, a.shape[0])[:, None, None]
        col = top_c * (1 - yy) + bot_c * yy
        if weather == "storm":
            col = col * 0.55
        cl = np.dstack([np.broadcast_to(col, a.shape + (3,)), (a * 235)[..., None]]).astype(np.uint8)
        self.clouds = Image.fromarray(cl, "RGBA").resize((W * 2, horizon), Image.BICUBIC)
        self.stars = stars_layer(4, 450, horizon) if tod == "night" else None
        self.L = 11
        self.xs = np.arange(-40, W + 60, 16, dtype=np.float32)
        r = rng(seed)
        self.ph = r.uniform(0, 6.28, (self.L, 4))
        self.rain = Particles(420, seed + 3, vx=(-500, -350), vy=(1400, 1900), size=(1, 1.6), region=(0, 0, W + 300, H)) \
            if weather == "storm" else None
        if weather == "fog":
            f = noise2d(W // 4, H // 4, seed=seed + 20, octaves=4, base=3)
            f = np.dstack([np.full(f.shape + (3,), 170, np.float32), (np.clip(f * 1.4 - 0.2, 0, 1) * 200)[..., None]])
            self.fog = Image.fromarray(f.astype(np.uint8), "RGBA").resize((W * 2, H), Image.BICUBIC)
            fa = np.asarray(self.fog)[:, :W]
            self.fog = Image.fromarray(np.concatenate([fa, fa[:, ::-1]], 1), "RGBA")
        else:
            self.fog = None

    def layer_y(self, i, x, t):
        f = i / (self.L - 1)
        base = self.h0 + (H - self.h0) * f ** 1.7 + 2
        amp = (1.5 + 26 * f ** 2) * self.storm
        k = 0.004 + 0.008 * (1 - f)
        sp = (0.9 + 0.6 * f) * (1.4 if self.storm > 1 else 1)
        p = self.ph[i]
        return base + amp * (np.sin(x * k + t * sp + p[0]) * 0.6 + np.sin(x * k * 2.3 - t * sp * 1.3 + p[1]) * 0.3
                             + np.sin(x * k * 0.45 + t * sp * 0.6 + p[2]) * 0.5)

    def draw_sky(self, img, t, flash=0.0):
        img.paste(self.sky, (0, 0))
        off = int((t * (40 if self.storm > 1 else 12)) % W)
        cl = self.clouds.crop((off, 0, off + W, self.h0))
        img.paste(cl, (0, 0), cl)
        if self.stars is not None:
            draw_stars(ImageDraw.Draw(img), self.stars, t, 0.9)
        if flash > 0:
            a = np.asarray(img.crop((0, 0, W, self.h0)), np.float32)
            a = a + (np.array([200, 210, 255]) - a) * flash * 0.6
            img.paste(Image.fromarray(a.clip(0, 255).astype(np.uint8)), (0, 0))

    def draw_layers(self, img, t, lo, hi, flash=0.0):
        d = ImageDraw.Draw(img)
        pal = self.pal
        for i in range(lo, hi):
            f = i / (self.L - 1)
            ys = self.layer_y(i, self.xs, t)
            col = mix(pal["sea"], pal["near"], f ** 0.8)
            if flash:
                col = mix(col, (150, 160, 190), flash * 0.5 * (1 - f))
            pts = list(zip(self.xs.tolist(), ys.tolist())) + [(W + 60, H), (-40, H)]
            d.polygon(pts, fill=col)
            crest = mix(col, pal["crest"], 0.35 * (1 - f) + 0.12)
            d.line(list(zip(self.xs.tolist(), ys.tolist())), fill=crest, width=1 + int(2 * f))
            if self.storm > 1 and f > 0.3:
                # foam flecks on crests
                idx = np.where(np.diff(np.sign(np.diff(ys))) > 0)[0]
                for j in idx[::2]:
                    x, y = float(self.xs[j + 1]), float(ys[j + 1])
                    d.ellipse([x - 16 * f, y - 3, x + 16 * f, y + 3], fill=mix(crest, (230, 235, 240), 0.5))
        # glints of sun/moon on water
        src = pal.get("sun") or (pal.get("moon") and pal["moon"] + ((200, 210, 230),))
        if src and self.weather != "fog":
            sx = src[0]
            r = rng(int(t * 8))
            for k in range(40):
                y = self.h0 + 4 + (H - self.h0) * r.random() ** 1.8
                if y > H - BAR:
                    continue
                ww = 6 + 50 * (y - self.h0) / (H - self.h0)
                x = sx + r.normal(0, 18 + 90 * (y - self.h0) / (H - self.h0))
                d.line([(x - ww, y), (x + ww, y)], fill=mix(src[2], (255, 255, 255), 0.3), width=2)

    def ship_pose(self, i, x, t):
        y = float(self.layer_y(i, np.array([x], np.float32), t)[0])
        y2 = float(self.layer_y(i, np.array([x + 30.0], np.float32), t)[0])
        ang = -math.degrees(math.atan2(y2 - y, 30.0)) * 0.8
        return y, ang

    def draw_overlays(self, img, t):
        if self.rain is not None:
            self.rain.draw(ImageDraw.Draw(img), t, (150, 160, 180), alpha=0.5, streak=0.018)
        if self.fog is not None:
            off = int((t * 25) % W)
            fg = self.fog.crop((off, 0, off + W, H))
            img.paste(fg, (0, 0), fg)


def coast_silhouette(kind, horizon, seed=5):
    im = Image.new("RGBA", (W, horizon + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if kind == "forest":
        pts = ridge_points(700, W + 20, horizon - 20, 90, seed, bottom=horizon + 4)
        d.polygon(pts, fill=(30, 40, 42, 255))
        r = rng(seed)
        for x in range(720, W, 13):
            yb = min(y for xx, y in pts if abs(xx - x) < 9) if any(abs(xx - x) < 9 for xx, _ in pts) else horizon
            pine(d, x + r.uniform(-4, 4), yb + 6, r.uniform(40, 80), (26, 34, 36, 255))
    elif kind == "palm":
        d.polygon([(900, horizon + 4), (1050, horizon - 18), (1400, horizon - 26), (1700, horizon - 12), (1850, horizon + 4)], fill=(14, 10, 14, 255))
        for i, x in enumerate([1080, 1160, 1300, 1420, 1560, 1650]):
            palm(d, x, horizon - 16, 90 + (i % 3) * 25, (14, 10, 14, 255), lean=0.2 * (-1) ** i, seed=i)
    elif kind in ("light", "island"):
        d.polygon([(1150, horizon + 4), (1260, horizon - 12), (1500, horizon - 16), (1620, horizon + 4)], fill=(6, 8, 14, 255))
    return im


# ======================================================================= SEA
@scene("sea")
class SeaScene(Scene):
    def setup(self):
        p = self.p
        self.sea = Sea(p.get("tod", "day"), p.get("weather", "calm"), seed=hash(p["id"]) % 1000, dark=p.get("dark", False))
        self.coast = coast_silhouette(p["coast"], HORIZON) if p.get("coast") else None
        n = p.get("n", 1)
        kind = p.get("ship", "none")
        self.ships = []
        r = rng(7)
        if kind == "longship":
            self.ships.append(dict(img=longship(0.62 if n == 1 else 0.3), layer=7, x=820, vx=14))
        elif kind == "caravel":
            if n == 1:
                self.ships.append(dict(img=caravel(0.62), layer=7, x=900, vx=12))
            elif n == 3:
                for j, (sc, lay, x) in enumerate([(0.3, 3, 520), (0.4, 5, 1450), (0.58, 8, 960)]):
                    self.ships.append(dict(img=caravel(sc), layer=lay, x=x, vx=10 + j * 2, name=j))
            else:
                for j in range(n):
                    f = j / max(1, n - 1)
                    lay = 1 + int(f * 7)
                    sc = 0.08 + 0.4 * (lay / 10) ** 1.6
                    self.ships.append(dict(img=caravel(sc), layer=lay, x=float(r.uniform(80, W - 80)), vx=8 + lay))
                self.ships.sort(key=lambda s: s["layer"])
        elif kind == "wreck":
            self.ships.append(dict(img=caravel(0.5, broken=True), layer=6, x=700, vx=0, tilt=-12))
        self.lightning = []
        if p.get("lightning"):
            r = rng(3)
            tt = 1.5
            while tt < self.dur:
                self.lightning.append(tt)
                tt += r.uniform(2.5, 5.5)

    def flash_amt(self, t):
        a = 0
        for lt in self.lightning:
            if 0 <= t - lt < 0.5:
                a = max(a, (1 - (t - lt) / 0.5) ** 2 * (0.6 + 0.4 * math.sin((t - lt) * 60)))
        return a

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        fl = self.flash_amt(t)
        self.sea.draw_sky(img, t, fl)
        if fl > 0.3:
            d = ImageDraw.Draw(img)
            r = rng(int(t * 3))
            x, y = r.uniform(300, W - 300), BAR
            pts = [(x, y)]
            while y < HORIZON - 20:
                x += r.uniform(-60, 60)
                y += r.uniform(30, 70)
                pts.append((x, y))
            d.line(pts, fill=(230, 235, 255), width=4)
        if self.coast is not None:
            ca = 1.0
            cc = self.p.get("cues", {})
            if "coast" in cc.values():
                ca = ramp(t, self.cue_named("coast"), 3.0)
            paste_alpha(img, self.coast, ca * (0.7 if self.p.get("weather") == "fog" else 1), (0, 0))
            if self.p.get("coast") == "light":
                c0 = self.cue_named("coast") or 0
                if t > 1:
                    fl2 = 0.6 + 0.4 * math.sin(t * 9) * math.sin(t * 3.1)
                    gl = glow_layer((W, H), lambda d, k: d.ellipse([(1380 - 30) * k, (HORIZON - 40) * k, (1380 + 30) * k, (HORIZON - 5) * k],
                                                                   fill=(255, 190, 90)), 30)
                    img = Image.blend(img, screen(img, gl), clamp(fl2 * (0.4 + 0.6 * ramp(t, c0, 1.0))))
        # waves + ships interleaved by layer
        lo = 0
        for s in self.ships + [dict(layer=self.sea.L)]:
            self.sea.draw_layers(img, t, lo, s["layer"], fl)
            lo = s["layer"]
            if "img" in s:
                x = s["x"] + s["vx"] * t
                if self.p.get("n", 1) > 3:
                    x = (x + 200) % (W + 400) - 200
                y, ang = self.sea.ship_pose(s["layer"], x, t)
                rotate_paste(img, s["img"], x, y + 6, ang + s.get("tilt", 0))
                if "name" in s and self.p.get("names"):
                    nm = self.p["names"][s["name"]]
                    a = ramp(t, 1.0 + s["name"] * 1.1, 0.8) * (1 - ramp(t, self.dur - 2, 1))
                    hh = s["img"].height * 0.95
                    draw_text(img, (x, y - hh), nm, font("title", 28), (240, 230, 200), alpha=a, spacing=4)
        self.sea.draw_overlays(img, t)
        if self.p.get("cues", {}).get(0) == "count":
            a = ramp(t, 1.2, 0.8)
            draw_text(img, (W / 2, 330), "17 SHIPS", font("title", 110), (245, 230, 190), alpha=a, spacing=8)
            a2 = ramp(t, self.cue(1) or 5, 0.8)
            draw_text(img, (W / 2, 450), "1,200 MEN", font("title", 64), (230, 200, 150), alpha=a2 * (1 - ramp(t, self.cue(2) or 99, 1)), spacing=6)
            a = a * (1 - ramp(t, self.cue(2) or 99, 1))
        return img

    def flash(self, t):
        return self.flash_amt(t) * 0.15


# ======================================================================= TITLE / CHAPTER / END
def ember_bg(seed, col=(255, 150, 60)):
    return Particles(90, seed, vx=(-10, 10), vy=(-25, -70), size=(1, 3), region=(0, 0, W, H), wobble=20)


@scene("title")
class TitleScene(Scene):
    zoom = (1.0, 1.05)
    fade_in = 1.0

    def setup(self):
        img, self.proj = maps.render_map((-100, 30, -10, 76), (W, H), ocean=(6, 8, 12), land=(22, 20, 18),
                                         coast=(120, 95, 50), grid=(18, 18, 20), glow=(60, 40, 10), coast_w=1)
        self.map = img
        self.emb = ember_bg(1)

    def draw(self, t):
        img = Image.blend(Image.new("RGB", (W, H)), self.map, 0.55 * ramp(t, 0, 2))
        d = ImageDraw.Draw(img)
        self.emb.draw(d, t, (255, 160, 70), alpha=0.8, flicker=0.4)
        # light sweep
        sx = lerp(-400, W + 400, clamp((t - 0.8) / 3.5))
        gl = glow_layer((W, H), lambda dd, k: dd.polygon([((sx - 120) * k, 0), ((sx + 40) * k, 0), ((sx - 200) * k, H * k), ((sx - 360) * k, H * k)],
                                                         fill=(90, 70, 40)), 80)
        img = screen(img, gl)
        f = font("deco", 118)
        title = "UNDERSTANDING"
        a = ramp(t, 0.6, 1.4)
        draw_text(img, (W / 2, 440), title, f, (240, 220, 170), alpha=a, spacing=int(lerp(40, 14, ease_out((t - 0.6) / 3))), glow=(160, 100, 30))
        draw_text(img, (W / 2, 580), "AMERICA", font("deco", 150), (250, 235, 190), alpha=ramp(t, 1.3, 1.4),
                  spacing=int(lerp(60, 26, ease_out((t - 1.3) / 3))), glow=(180, 110, 30))
        w_ = 500 * ramp(t, 2.2, 1.2)
        d = ImageDraw.Draw(img)
        d.line([(W / 2 - w_, 670), (W / 2 + w_, 670)], fill=(170, 130, 60), width=2)
        draw_text(img, (W / 2, 725), "EPISODE ONE", font("title", 34), (220, 200, 160), alpha=ramp(t, 2.8, 1), spacing=10)
        draw_text(img, (W / 2, 790), "From Vikings to Columbus", font("italic", 60), (235, 225, 205), alpha=ramp(t, 3.3, 1.2))
        return img

    def flash(self, t):
        return 0.0


@scene("chapter")
class ChapterScene(Scene):
    zoom = (1.08, 1.0)
    fade_in = 0.15

    def setup(self):
        hue = {"I": (40, 70, 110), "II": (120, 30, 20), "III": (60, 90, 50), "IV": (100, 60, 30), "V": (110, 20, 30),
               "VI": (50, 60, 80), "VII": (120, 90, 40), "VIII": (130, 100, 50), "IX": (130, 90, 20), "X": (70, 40, 40),
               "XI": (60, 90, 80), "XII": (90, 70, 40), "XIII": (110, 15, 15), "XIV": (80, 70, 50)}.get(self.p["num"], (80, 60, 40))
        self.bg = radial(W, H, W / 2, H / 2, 1100, hue, (0, 0, 0), power=1.0)
        smoke = noise2d(W // 4, H // 4, seed=len(self.p["title"]), octaves=5, base=3)
        self.smoke = Image.fromarray((np.clip(smoke * 1.5 - 0.5, 0, 1) * 70).astype(np.uint8)).resize((W * 2, H), Image.BICUBIC)
        self.emb = ember_bg(len(self.p["title"]))

    def draw(self, t):
        img = self.bg.copy()
        off = int(t * 30)
        sm = self.smoke.crop((off, 0, off + W, H))
        img = Image.composite(Image.new("RGB", (W, H), (120, 110, 100)), img, sm)
        self.emb.draw(ImageDraw.Draw(img), t, (255, 170, 80), alpha=0.9, flicker=0.4)
        draw_text(img, (W / 2, 390), f"CHAPTER {self.p['num']}", font("title", 36), (200, 180, 140), alpha=ramp(t, 0.2, 0.6), spacing=14)
        tt = self.p["title"]
        size = 104 if len(tt) < 16 else 82 if len(tt) < 24 else 64
        draw_text(img, (W / 2, 520), tt, font("deco", size), (245, 232, 200), alpha=ramp(t, 0.4, 0.8),
                  spacing=int(lerp(30, 8, ease_out(t / 3))), glow=(150, 90, 30))
        w_ = 380 * ramp(t, 0.7, 1.0)
        d = ImageDraw.Draw(img)
        d.line([(W / 2 - w_, 610), (W / 2 + w_, 610)], fill=(180, 140, 70), width=2)
        d.polygon([(W / 2, 600), (W / 2 + 10, 610), (W / 2, 620), (W / 2 - 10, 610)], fill=(200, 160, 80))
        draw_text(img, (W / 2, 670), self.p["sub"], font("italic", 50), (225, 210, 180), alpha=ramp(t, 1.0, 0.8))
        return img

    def flash(self, t):
        return 0.55 * (1 - ramp(t, 0, 0.5))


@scene("end")
class EndScene(TitleScene):
    fade_out = 2.0

    def draw(self, t):
        img = Image.blend(Image.new("RGB", (W, H)), self.map, 0.4)
        self.emb.draw(ImageDraw.Draw(img), t, (255, 160, 70), alpha=0.7, flicker=0.4)
        draw_text(img, (W / 2, 380), "UNDERSTANDING AMERICA", font("deco", 84), (240, 225, 185), alpha=ramp(t, 0.3, 1), spacing=10, glow=(150, 90, 30))
        draw_text(img, (W / 2, 470), "Episode One · From Vikings to Columbus", font("italic", 50), (225, 215, 195), alpha=ramp(t, 0.9, 1))
        draw_text(img, (W / 2, 600), "Based on  A VOYAGE LONG AND STRANGE  by Tony Horwitz", font("title", 28), (200, 185, 150), alpha=ramp(t, 1.8, 1), spacing=3)
        draw_text(img, (W / 2, 700), "NEXT:  CONQUEST  ·  SETTLEMENT", font("title", 40), (230, 200, 130), alpha=ramp(t, 2.8, 1), spacing=8)
        draw_text(img, (W / 2, 800), "Thank you for watching", font("italic", 44), (210, 200, 185), alpha=ramp(t, 3.8, 1))
        return img


# ======================================================================= ROME / BOOKS
def temple(d, cx, base, w, h, color):
    d.rectangle([cx - w / 2 - 20, base - 20, cx + w / 2 + 20, base], fill=color)
    d.rectangle([cx - w / 2 - 10, base - 40, cx + w / 2 + 10, base - 20], fill=color)
    n = 8
    for i in range(n):
        x = cx - w / 2 + 30 + i * (w - 60) / (n - 1)
        d.rectangle([x - 18, base - h, x + 18, base - 40], fill=color)
        d.rectangle([x - 26, base - h - 12, x + 26, base - h], fill=color)
    d.rectangle([cx - w / 2, base - h - 60, cx + w / 2, base - h - 12], fill=color)
    d.polygon([(cx - w / 2 - 20, base - h - 60), (cx, base - h - 200), (cx + w / 2 + 20, base - h - 60)], fill=color)


def capitol(d, cx, base, s, color):
    d.rectangle([cx - 700 * s, base - 120 * s, cx + 700 * s, base], fill=color)
    d.rectangle([cx - 200 * s, base - 200 * s, cx + 200 * s, base - 110 * s], fill=color)
    for i in range(12):
        x = cx - 180 * s + i * 33 * s
        d.rectangle([x - 6 * s, base - 330 * s, x + 6 * s, base - 200 * s], fill=color)
    d.rectangle([cx - 190 * s, base - 350 * s, cx + 190 * s, base - 325 * s], fill=color)
    d.chord([cx - 170 * s, base - 560 * s, cx + 170 * s, base - 200 * s], 180, 360, fill=color)
    d.rectangle([cx - 35 * s, base - 620 * s, cx + 35 * s, base - 540 * s], fill=color)
    d.polygon([(cx - 12 * s, base - 620 * s), (cx, base - 680 * s), (cx + 12 * s, base - 620 * s)], fill=color)
    for side in (-1, 1):
        for i in range(10):
            x = cx + side * (260 * s + i * 42 * s)
            d.rectangle([x - 8 * s, base - 200 * s, x + 8 * s, base - 120 * s], fill=color)
        d.rectangle([cx + side * 240 * s - (0 if side > 0 else 440 * s), base - 215 * s, cx + side * 240 * s + (440 * s if side > 0 else 0), base - 195 * s], fill=color)


@scene("rome")
class RomeScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (20, 16, 30)), (0.45, (120, 60, 50)), (0.7, (240, 150, 70)), (1, (60, 30, 20))])
        g = radial(W, H, W / 2, 720, 800, (255, 190, 110), (0, 0, 0), 0.6)
        self.bg = screen(self.bg, g)
        self.cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        capitol(ImageDraw.Draw(self.cap), W / 2, H - BAR + 5, 1.0, (14, 10, 12, 255))
        self.tem = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        temple(ImageDraw.Draw(self.tem), W / 2, H - BAR + 5, 1100, 420, (16, 10, 10, 255))
        self.dust = Particles(120, 11, vx=(-8, 8), vy=(-10, -25), size=(1, 2.5))
        self.words = ["IANVARIVS", "FEBRVARIVS", "AVGVSTVS", "Homo sapiens", "Quercus robur", "Canis lupus",
                      "Panthera leo", "MARTIVS", "IVLIVS", "Rosa canina"]

    def draw(self, t):
        img = self.bg.copy()
        c2, c5 = self.cue(2), self.cue(5)
        rome = ramp(t, c2, 1.5) * (1 - ramp(t, c5, 1.5))
        paste_alpha(img, self.cap, 1 - rome)
        paste_alpha(img, self.tem, rome)
        d = ImageDraw.Draw(img)
        self.dust.draw(d, t, (255, 210, 150), alpha=0.7)
        if c2 is not None and rome > 0:
            for i, w_ in enumerate(self.words):
                t0 = c2 + 0.6 + i * 1.3
                if t < t0:
                    continue
                k = t - t0
                x = W * (0.15 + 0.7 * ((i * 0.37) % 1))
                y = 700 - k * 45 - (i % 3) * 60
                a = min(ramp(t, t0, 1.0), 1 - ramp(k, 5, 2)) * rome
                f = font("italic" if w_[0].islower() or " " in w_ else "title", 46 if " " in w_ else 54)
                draw_text(img, (x, y), w_, f, (255, 235, 200), alpha=a, spacing=0 if " " in w_ else 6, glow=(180, 100, 40))
        if c5 is not None:
            draw_text(img, (W / 2, 300), "THE MOST MISUNDERSTOOD NATION", font("title", 58), (250, 235, 205),
                      alpha=ramp(t, self.cue(6), 1.0), spacing=6, glow=(120, 60, 20))
        return img


@scene("books")
class BooksScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2 + 100, 1200, (60, 45, 30), (6, 5, 5), 0.9)
        r = rng(5)
        self.books = []
        x = 170
        while x < W - 170:
            bw = r.uniform(28, 60)
            bh = r.uniform(220, 360)
            col = tuple(int(v) for v in r.choice([(90, 30, 25), (30, 50, 70), (40, 60, 35), (100, 80, 40), (60, 30, 60), (25, 25, 30)]) * r.uniform(0.7, 1.1))
            self.books.append((x, bw, bh, col))
            x += bw + 2
        self.dust = Particles(100, 9, vx=(-6, 6), vy=(-8, -20), size=(1, 2.2))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        base = 820
        cb, cp, cd = self.cue_named("book"), self.cue_named("parts"), self.cue_named("discovery")
        dim = 1 - 0.7 * ramp(t, cb, 1.0)
        n = len(self.books)
        for i, (x, bw, bh, col) in enumerate(self.books):
            k = ramp(t, 0.3 + i * 0.05, 0.6)
            if k <= 0:
                continue
            c = mix((0, 0, 0), col, k * dim)
            y0 = base - bh * ease_out(k)
            d.rectangle([x, y0, x + bw, base], fill=c)
            d.line([(x + 4, y0 + 20), (x + bw - 4, y0 + 20)], fill=mix(c, (200, 170, 90), 0.5), width=2)
            d.line([(x + 4, base - 30), (x + bw - 4, base - 30)], fill=mix(c, (200, 170, 90), 0.5), width=2)
        d.rectangle([100, base, W - 100, base + 18], fill=(40, 26, 16))
        cnt = int(100 * ease_out(clamp((t - 0.5) / 4)))
        draw_text(img, (W / 2, 290), f"{cnt}", font("deco", 150), (245, 225, 180), alpha=ramp(t, 0.3, 0.6) * dim, glow=(150, 90, 30))
        draw_text(img, (W / 2, 400), "BOOKS · ONE JOURNEY", font("title", 40), (220, 200, 160), alpha=ramp(t, 1.5, 0.8) * dim, spacing=10)
        if cb is not None:
            a = ramp(t, cb, 1.0) * (1 - ramp(t, cp, 1.0) if cp else 1)
            # featured book cover
            bx, by = W / 2, 520
            cov = [bx - 190, by - 250, bx + 190, by + 250]
            k = ease_out(ramp(t, cb, 1.2))
            if a > 0:
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ld = ImageDraw.Draw(lay)
                ld.rectangle(cov, fill=(120, 40, 25, 255), outline=(210, 170, 90, 255), width=4)
                ld.rectangle([cov[0] + 20, cov[1] + 20, cov[2] - 20, cov[3] - 20], outline=(210, 170, 90, 255), width=2)
                paste_alpha(img, lay, a)
                for j, ln in enumerate(["A VOYAGE", "LONG AND", "STRANGE"]):
                    draw_text(img, (bx, by - 120 + j * 70), ln, font("title", 50), (240, 215, 150), alpha=a)
                draw_text(img, (bx, by + 150), "Tony Horwitz", font("italic", 44), (240, 215, 150), alpha=a)
        if cp is not None:
            for j, w_ in enumerate(["DISCOVERY", "CONQUEST", "SETTLEMENT"]):
                a = ramp(t, cp + j * 0.6, 0.6)
                hi = ramp(t, cd, 0.8)
                c = (255, 220, 140) if j == 0 else mix((230, 220, 200), (90, 85, 80), hi)
                draw_text(img, (W / 2 + (j - 1) * 560, 540), w_, font("deco", 70 if j == 0 else 60), c, alpha=a,
                          spacing=6, glow=(200, 120, 30) if j == 0 and hi > 0.5 else None)
        self.dust.draw(ImageDraw.Draw(img), t, (255, 210, 150), alpha=0.6)
        return img


# ======================================================================= LANDSCAPES
def layered_mountains(stops, layers, seed, horizon):
    """Pre-render a sky with a stack of ridge layers. layers: list of (base_y, amp, color)."""
    img = vgrad(stops)
    d = ImageDraw.Draw(img)
    for i, (by, amp, col) in enumerate(layers):
        d.polygon(ridge_points(-20, W + 20, by, amp, seed + i, octaves=7, nbase=2 + i), fill=col)
    return img


@scene("raid")
class RaidScene(Scene):
    def setup(self):
        self.sea = Sea("night", "calm", seed=21, horizon=640)
        self.sea.pal = dict(self.sea.pal, sea=(40, 20, 20), near=(6, 3, 4), crest=(200, 90, 40))
        self.sky = vgrad([(0, (8, 4, 8)), (0.35, (50, 16, 14)), (0.6, (160, 60, 20))], W, 640)
        cliff = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(cliff)
        d.polygon([(1000, H), (1080, 620), (1250, 560), (1500, 540), (W + 10, 530), (W + 10, H)], fill=(10, 6, 6, 255))
        c = (6, 3, 4, 255)
        # monastery
        d.rectangle([1320, 380, 1640, 545], fill=c)
        d.polygon([(1300, 380), (1480, 290), (1660, 380)], fill=c)
        d.rectangle([1640, 300, 1720, 545], fill=c)
        d.polygon([(1630, 300), (1680, 220), (1730, 300)], fill=c)
        d.rectangle([1676, 170, 1684, 225], fill=c)
        d.rectangle([1664, 186, 1696, 194], fill=c)
        for i in range(4):
            d.chord([1350 + i * 70, 420, 1380 + i * 70, 470], 180, 360, fill=(255, 150, 50, 255))
            d.rectangle([1350 + i * 70, 445, 1380 + i * 70, 480], fill=(255, 140, 40, 255))
        self.cliff = cliff
        self.fire = Particles(260, 5, vx=(-30, 30), vy=(-120, -260), size=(2, 6), region=(1280, 150, 1760, 560), wobble=25)
        self.emb = Particles(160, 6, vx=(-60, 10), vy=(-60, -140), size=(1, 2.5), region=(900, 0, W, 640), wobble=30)
        self.ships = [(longship(0.34), 420, 6), (longship(0.26), 150, 4), (longship(0.2), 760, 3)]
        sm = noise2d(W // 4, H // 4, seed=33, octaves=5, base=4)
        self.smoke = Image.fromarray(np.dstack([np.full(sm.shape + (3,), 30, np.float32),
                                                (np.clip(sm * 1.6 - 0.5, 0, 1) * 200)[..., None]]).astype(np.uint8), "RGBA").resize((W, H * 2), Image.BICUBIC)

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        img.paste(self.sky, (0, 0))
        off = int((t * 50) % H)
        sm = self.smoke.crop((0, H - off, W, 2 * H - off)) if off else self.smoke.crop((0, 0, W, H))
        img.paste(sm, (0, -200), sm)
        lo = 0
        for sh, x, lay in sorted(self.ships, key=lambda s: s[2]):
            self.sea.draw_layers(img, t, lo, lay)
            lo = lay
            y, ang = self.sea.ship_pose(lay, x, t)
            rotate_paste(img, sh, x, y + 4, ang * 0.5)
        self.sea.draw_layers(img, t, lo, self.sea.L)
        img.paste(self.cliff, (0, 0), self.cliff)
        flick = 0.8 + 0.2 * math.sin(t * 13) * math.sin(t * 7.3)
        gl = glow_layer((W, H), lambda d, k: d.ellipse([1200 * k, 180 * k, 1800 * k, 620 * k], fill=(255, 110, 30)), 120)
        img = Image.blend(img, screen(img, gl), flick)
        d = ImageDraw.Draw(img)
        self.fire.draw(d, t, (255, 150, 40), flicker=0.6)
        self.emb.draw(d, t, (255, 190, 90), flicker=0.5)
        return img


@scene("fjord")
class FjordScene(Scene):
    def setup(self):
        stops = [(0, (40, 60, 90)), (0.5, (170, 180, 190)), (0.62, (230, 200, 170))]
        self.bg = layered_mountains(stops, [(430, 260, (110, 130, 150)), (500, 220, (80, 98, 118)),
                                            (580, 160, (50, 64, 80)), (640, 110, (30, 40, 52))], 40, 640)
        d = ImageDraw.Draw(self.bg)
        d.rectangle([0, 660, W, H], fill=(60, 80, 100))
        self.refl = self.bg.crop((0, 200, W, 660)).transpose(Image.FLIP_TOP_BOTTOM).resize((W, 420))
        self.farm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fd = ImageDraw.Draw(self.farm)
        c = (20, 22, 26, 255)
        fd.polygon([(-10, 760), (300, 700), (700, 720), (900, 800), (900, H), (-10, H)], fill=c)
        for x, s in [(200, 1.0), (420, 0.8), (560, 0.7)]:
            fd.polygon([(x - 90 * s, 715), (x, 650 * s + 715 * (1 - s) - 20), (x + 90 * s, 715)], fill=c)
            fd.rectangle([x - 80 * s, 710, x + 80 * s, 760], fill=c)
            fd.rectangle([x - 12 * s, 725, x + 12 * s, 760], fill=(200, 120, 50, 255))
        for x in range(640, 900, 22):
            pine(fd, x, 800, 70 + (x % 40), c)
        self.ship = longship(0.3)
        self.snow = Particles(220, 8, vx=(-20, 20), vy=(30, 70), size=(1, 2.4), region=(0, 0, W, H))
        self.smoke = Particles(50, 3, vx=(5, 20), vy=(-30, -50), size=(4, 12), region=(180, 420, 260, 660), wobble=15)

    def draw(self, t):
        img = self.bg.copy()
        rf = self.refl
        rarr = np.asarray(rf, np.float32) * 0.55 + np.array([10, 20, 30]) * 0.45
        img.paste(Image.fromarray(rarr.clip(0, 255).astype(np.uint8)), (0, 660))
        d = ImageDraw.Draw(img)
        for i in range(30):
            y = 670 + i * 13
            d.line([(0, y), (W, y + math.sin(t + i) * 2)], fill=(90, 110, 130), width=1)
        x = 1900 - t * 22
        rotate_paste(img, self.ship, x, 780 + math.sin(t * 1.3) * 3, math.sin(t) * 1.5)
        img.paste(self.farm, (0, 0), self.farm)
        d = ImageDraw.Draw(img)
        self.smoke.draw(d, t, (110, 115, 120), alpha=0.35)
        self.snow.draw(d, t, (230, 235, 240), alpha=0.8)
        return img


@scene("iceland")
class IcelandScene(Scene):
    def setup(self):
        stops = [(0, (20, 16, 30)), (0.45, (80, 40, 50)), (0.62, (200, 90, 50))]
        self.bg = vgrad(stops)
        d = ImageDraw.Draw(self.bg)
        d.polygon([(500, 720), (880, 330), (980, 320), (1400, 720)], fill=(26, 18, 20))
        d.polygon(ridge_points(-20, W + 20, 700, 120, 51), fill=(18, 12, 14))
        d.polygon(ridge_points(-20, W + 20, 820, 80, 52), fill=(10, 7, 8))
        self.lava = Particles(160, 4, vx=(-60, 60), vy=(-120, -300), size=(2, 5), region=(840, 150, 1020, 330), wobble=10)
        self.gey = Particles(200, 7, vx=(-15, 15), vy=(-150, -260), size=(3, 9), region=(300, 420, 340, 790), wobble=6)
        self.stars = stars_layer(6, 200, 500)

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.5)
        gl = glow_layer((W, H), lambda dd, k: dd.ellipse([780 * k, 220 * k, 1080 * k, 420 * k], fill=(255, 120, 30)), 90)
        img = screen(img, gl)
        d = ImageDraw.Draw(img)
        self.lava.draw(d, t, (255, 140, 40), flicker=0.5)
        self.gey.draw(d, t, (200, 205, 215), alpha=0.55)
        return img


@scene("greenland")
class GreenlandScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (4, 8, 20)), (0.5, (10, 30, 50)), (0.62, (30, 60, 80))])
        self.stars = stars_layer(8, 400, 600)
        ice = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ice)
        pts = ridge_points(-20, W + 20, 560, 50, 71, octaves=4, nbase=2, bottom=720)
        d.polygon(pts, fill=(150, 185, 205, 255))
        d.polygon(ridge_points(-20, W + 20, 640, 30, 72, octaves=6, nbase=6, bottom=720), fill=(110, 145, 170, 255))
        d.rectangle([0, 715, W, H], fill=(8, 20, 34, 255))
        for x, w_, h_ in [(300, 160, 70), (900, 90, 40), (1500, 220, 90), (1200, 60, 25)]:
            d.polygon([(x - w_ / 2, 760), (x - w_ / 3, 760 - h_), (x + w_ / 5, 760 - h_ * 1.1), (x + w_ / 2, 760)], fill=(170, 200, 220, 255))
            d.polygon([(x - w_ / 2, 760), (x + w_ / 2, 760), (x + w_ / 3, 760 + h_ * 0.6), (x - w_ / 3, 760 + h_ * 0.5)], fill=(40, 70, 90, 255))
        self.ice = ice
        self.snow = Particles(260, 12, vx=(-30, 10), vy=(30, 80), size=(1, 2.5), region=(0, 0, W, H))
        self.aur = rng(9).uniform(0, 6.28, 6)

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.8)

        def aurora(dd, k):
            for j in range(5):
                xs = np.linspace(-100, W + 100, 60)
                ys = 250 + 60 * np.sin(xs * 0.003 + t * 0.25 + self.aur[j]) + 30 * np.sin(xs * 0.008 - t * 0.4 + j) + j * 22
                col = mix((30, 255, 140), (140, 60, 220), j / 5)
                for x, y in zip(xs, ys):
                    hgt = 170 + 70 * math.sin(x * 0.01 + t + j)
                    dd.line([(x * k, (y - hgt) * k), (x * k, y * k)], fill=mix((0, 0, 0), col, 0.35), width=max(1, int(22 * k)))
        img = screen(img, glow_layer((W, H), aurora, 30))
        img.paste(self.ice, (0, 0), self.ice)
        d = ImageDraw.Draw(img)
        # reflections of aurora on the sea
        for i in range(12):
            y = 725 + i * 22
            d.line([(0, y), (W, y)], fill=(20, 60 + int(20 * math.sin(t + i)), 60), width=1)
        c1, c2 = self.cue(0), self.cue(2)
        a = ramp(t, c1, 0.8) * (1 - ramp(t, self.cue(1), 0.8))
        draw_text(img, (W / 2, 330), "≈ 80%", font("deco", 140), (230, 245, 255), alpha=a, glow=(60, 140, 200))
        draw_text(img, (W / 2, 450), "BENEATH THE ICE", font("title", 44), (210, 230, 240), alpha=a, spacing=10)
        if c2 is not None:
            for i, x in enumerate([260, 330, 420, 1580, 1650]):
                aa = ramp(t, c2 + i * 0.4, 0.6)
                if aa > 0:
                    gl = glow_layer((W, H), lambda dd, k, x=x: dd.ellipse([(x - 12) * k, 690 * k, (x + 12) * k, 712 * k], fill=(255, 170, 70)), 16)
                    img = Image.blend(img, screen(img, gl), aa)
        self.snow.draw(ImageDraw.Draw(img), t, (230, 240, 250), alpha=0.8)
        return img


@scene("vines")
class VinesScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (40, 30, 15)), (0.5, (150, 110, 40)), (1, (30, 25, 10))])
        self.bg = screen(self.bg, radial(W, H, W * 0.6, 200, 900, (255, 210, 120), (0, 0, 0), 0.7))
        d = ImageDraw.Draw(self.bg)
        for i, x in enumerate(range(-40, W + 60, 150)):
            d.rectangle([x, 300 - (i % 3) * 40, x + 26 + (i % 2) * 18, H], fill=(38 + i % 3 * 8, 30, 14))
        d.polygon(ridge_points(-20, W + 20, 880, 60, 81), fill=(22, 18, 8))
        self.vines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        vd = ImageDraw.Draw(self.vines)
        r = rng(3)
        for k in range(9):
            x0 = r.uniform(0, W)
            pts = [(x0 + 60 * math.sin(j * 0.5 + k), BAR + j * 28) for j in range(int(r.uniform(6, 14)))]
            vd.line(pts, fill=(20, 26, 10, 255), width=6)
            for (px, py) in pts[2::2]:
                for m in range(3):
                    lx, ly = px + r.uniform(-40, 40), py + r.uniform(-10, 20)
                    vd.polygon([(lx, ly - 26), (lx + 26, ly), (lx, ly + 22), (lx - 26, ly)], fill=(26, 40, 14, 255))
            gx, gy = pts[-1]
            for m in range(22):
                row = int(math.sqrt(m * 2))
                bx = gx + r.uniform(-1, 1) * (26 - row * 3)
                by = gy + 10 + row * 16
                vd.ellipse([bx - 11, by - 11, bx + 11, by + 11], fill=(70, 20, 60, 255), outline=(120, 60, 110, 255))
        self.pollen = Particles(160, 21, vx=(-10, 10), vy=(-6, -20), size=(1, 2.6))

    def draw(self, t):
        img = self.bg.copy()
        rays = glow_layer((W, H), lambda d, k: [d.polygon([((1300 + i * 120) * k, 0), ((1340 + i * 120) * k, 0),
                                                          ((700 + i * 180) * k, H * k), ((600 + i * 180) * k, H * k)],
                                                         fill=(90, 70, 30)) for i in range(4)], 40)
        img = Image.blend(img, screen(img, rays), 0.7 + 0.3 * math.sin(t * 0.5))
        sway = math.sin(t * 0.7) * 6
        img.paste(self.vines, (int(sway), 0), self.vines)
        self.pollen.draw(ImageDraw.Draw(img), t, (255, 230, 160), alpha=0.8)
        c = self.cue_named("title")
        a = ramp(t, c, 1.2)
        draw_text(img, (W / 2, 480), "VINLAND", font("deco", 150), (255, 235, 190), alpha=a, spacing=int(lerp(40, 16, ramp(t, c, 3))), glow=(200, 110, 30))
        draw_text(img, (W / 2, 600), "The Land of Wine", font("italic", 60), (250, 230, 200), alpha=ramp(t, (c or 0) + 0.8, 1) if c else 0)
        draw_text(img, (W / 2, 690), "LEIF “THE LUCKY”", font("title", 40), (240, 210, 150), alpha=ramp(t, self.cue(4), 1), spacing=8)
        return img


@scene("beach")
class BeachScene(Scene):
    def setup(self):
        self.sea = Sea("dusk", "calm", seed=31, horizon=560)
        self.var = self.p["variant"]
        self.shore = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.shore)
        d.polygon([(-10, 800), (500, 760), (1100, 790), (W + 10, 740), (W + 10, H), (-10, H)], fill=(34, 20, 18, 255))
        for x in range(1500, W + 40, 24):
            pine(d, x, 760 - (x - 1500) * 0.1, 120 + (x % 60), (12, 8, 10, 255))
        self.arrows = Particles(160, 5, vx=(700, 900), vy=(300, 420), size=(1, 2), region=(-400, -200, W, H - 300))
        self.canoes = [(rng(i).uniform(100, 1700), 1 + int(rng(i + 50).uniform(0, 8))) for i in range(22)]

    def canoe(self, d, x, y, s, col):
        d.chord([x - 70 * s, y - 20 * s, x + 70 * s, y + 18 * s], 0, 180, fill=col)
        person(d, x - 25 * s, y + 2 * s, 42 * s, col)
        person(d, x + 20 * s, y + 2 * s, 42 * s, col)

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        self.sea.draw_sky(img, t)
        self.sea.draw_layers(img, t, 0, self.sea.L)
        d = ImageDraw.Draw(img)
        if self.var == "arrows":
            for i, (x, lay) in enumerate(self.canoes):
                a = ramp(t, i * 0.25, 0.8)
                xx = x + math.sin(t * 0.3 + i) * 20
                y, _ = self.sea.ship_pose(lay, xx, t)
                s = 0.35 + lay * 0.12
                self.canoe(d, xx, y + 4, s, mix((200, 120, 80), (8, 5, 6), a))
        img.paste(self.shore, (0, 0), self.shore)
        d = ImageDraw.Draw(img)
        if self.var == "canoes":
            cr, ca, ce = self.cue_named("reveal"), self.cue_named("attack"), self.cue_named("escape")
            for i, x in enumerate([700, 900, 1100]):
                y = 790
                rev = ramp(t, cr, 1.0)
                d.chord([x - 90, y - 45, x + 90, y + 30], 180, 360, fill=(14, 9, 9))
                if rev > 0 and not ramp(t, ca, 0.3):
                    gl = glow_layer((W, H), lambda dd, k, x=x: dd.arc([(x - 95) * k, (y - 50) * k, (x + 95) * k, (y + 34) * k], 180, 360, fill=(255, 160, 80), width=3), 6, 2)
                    img = Image.blend(img, screen(img, gl), rev * (0.6 + 0.4 * math.sin(t * 4)))
                    d = ImageDraw.Draw(img)
            # norse advancing from left
            adv = ramp(t, self.cue(1), 6)
            for j in range(6):
                person(d, lerp(-60, 380, adv) + j * 55 - (j % 2) * 20, 815 + (j % 2) * 20, 150, (8, 6, 8), "helmet", step=t * 4 + j)
            if ce is not None:
                k = clamp((t - ce) / 6)
                self.canoe(d, 1150 + k * 700, 640 + math.sin(t * 2) * 3, 0.55, (10, 6, 8))
                draw_text(img, (W / 2, 330), "EIGHT KILLED  ·  ONE ESCAPED", font("title", 56), (240, 200, 180), alpha=ramp(t, ce, 0.8), spacing=6)
        else:
            cv = self.cue_named("volley")
            if cv is not None and t > cv:
                a = ramp(t, cv, 0.5) * (1 - ramp(t, self.dur - 2, 1.5))
                self.arrows.draw(d, t, (20, 12, 10), alpha=a, streak=0.05)
                self.arrows.draw(d, t + 0.37, (30, 16, 12), alpha=a, streak=0.05)
            person(d, 420, 850, 190, (8, 5, 6), "helmet")
            person(d, 520, 860, 170, (8, 5, 6), "spear")
        return img

    def flash(self, t):
        ca = self.cue_named("attack")
        if ca is not None and t >= ca:
            return 0.5 * (1 - ramp(t, ca, 0.6))
        return 0.0


@scene("word")
class WordScene(Scene):
    zoom = (1.0, 1.08)

    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (50, 38, 28), (5, 4, 4), 0.8)
        tex = noise2d(W // 2, H // 2, seed=5, octaves=6, base=8)
        arr = np.asarray(self.bg, np.float32) * (0.8 + 0.4 * np.asarray(Image.fromarray((tex * 255).astype(np.uint8)).resize((W, H)), np.float32)[..., None] / 255)
        self.bg = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        self.dust = Particles(120, 15, vx=(-8, 8), vy=(-8, -18), size=(1, 2))

    def draw(self, t):
        img = self.bg.copy()
        self.dust.draw(ImageDraw.Draw(img), t, (230, 200, 150), alpha=0.6)
        word = self.p["word"]
        f = font("deco", 170)
        n = len(word)
        tw = sum(f.getlength(c) + 20 for c in word)
        x = W / 2 - tw / 2
        for i, ch in enumerate(word):
            a = ramp(t, 0.5 + i * 0.15, 0.6)
            draw_text(img, (x + f.getlength(ch) / 2, 470 - 20 * (1 - a)), ch, f, (235, 220, 190), alpha=a, glow=(150, 80, 30))
            x += f.getlength(ch) + 20
        draw_text(img, (W / 2, 640), self.p["meaning"], font("italic", 64), (220, 200, 170), alpha=ramp(t, 2.0, 1))
        return img


@scene("icons")
class IconsScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1100, (60, 50, 35), (6, 5, 5), 0.9)
        self.emb = Particles(80, 3, vx=(-8, 8), vy=(-10, -25), size=(1, 2.5))

    def draw(self, t):
        img = self.bg.copy()
        self.emb.draw(ImageDraw.Draw(img), t, (255, 210, 150), alpha=0.6)
        cs = self.cue_named("snorri")
        a0 = 1 - ramp(t, cs, 1.0)
        for side, spec, cx in [(0, self.p["left"], W * 0.28), (1, self.p["right"], W * 0.72)]:
            draw_text(img, (cx, 290), spec["title"], font("title", 48), (235, 215, 170), alpha=ramp(t, 0.4 + side * 0.4, 0.8) * a0, spacing=8)
            for j, (e, nm) in enumerate(zip(spec["items"], spec["names"])):
                a = ramp(t, 1.0 + side * 0.8 + j * 0.4, 0.7) * a0
                if a <= 0:
                    continue
                em = emoji(e, 150)
                x = int(cx + (j - 0.5) * 240 - em.width / 2)
                y = int(470 - em.height / 2 + math.sin(t * 1.5 + j + side) * 8)
                paste_alpha(img, em, a, (x, y))
                draw_text(img, (x + em.width / 2, 600), nm, font("italic", 44), (230, 215, 190), alpha=a)
        # exchange arrows
        k = (t * 0.6) % 1
        a = ramp(t, 2.5, 0.8) * a0
        if a > 0:
            d = ImageDraw.Draw(img)
            for dirn, y in [(1, 430), (-1, 520)]:
                x0, x1 = (W * 0.43, W * 0.57) if dirn > 0 else (W * 0.57, W * 0.43)
                xm = lerp(x0, x1, k)
                d.line([(x0, y), (x1, y)], fill=mix((0, 0, 0), (180, 150, 90), a), width=3)
                d.polygon([(x1, y), (x1 - 20 * dirn, y - 12), (x1 - 20 * dirn, y + 12)], fill=mix((0, 0, 0), (220, 180, 100), a))
                d.ellipse([xm - 6, y - 6, xm + 6, y + 6], fill=mix((0, 0, 0), (255, 220, 150), a))
        if cs is not None:
            a = ramp(t, cs, 1.2)
            gl = glow_layer((W, H), lambda d, k: d.ellipse([(W / 2 - 160) * k, 250 * k, (W / 2 + 160) * k, 570 * k], fill=(160, 130, 80)), 80)
            img = Image.blend(img, screen(img, gl), a)
            em = emoji("⭐", 110)
            paste_alpha(img, em, a, (int(W / 2 - em.width / 2), 350))
            draw_text(img, (W / 2, 620), "SNORRI THORFINNSSON", font("deco", 74), (245, 230, 195), alpha=a, spacing=6, glow=(150, 100, 40))
            draw_text(img, (W / 2, 720), "first European child known to be born in the Americas", font("italic", 48), (225, 210, 185), alpha=ramp(t, self.cue(2), 1))
        return img


@scene("portraits")
class PortraitScene(Scene):
    zoom = (1.0, 1.05)

    def setup(self):
        self.people = self.p["people"]
        self.busts = [bust(760, pp["look"]) for pp in self.people]
        self.rims = [rim_light(b, pp["c"], 1.0, 14) for b, pp in zip(self.busts, self.people)]
        self.bgs = [radial(W, H, W * 0.33, H * 0.45, 900, tuple(int(v * 0.8) for v in pp["c"]), (4, 4, 6), 0.8) for pp in self.people]
        sm = noise2d(W // 4, H // 4, seed=len(self.p["id"]), octaves=5, base=3)
        self.smoke = Image.fromarray((np.clip(sm * 1.6 - 0.55, 0, 1) * 90).astype(np.uint8)).resize((W * 2, H), Image.BICUBIC)
        self.emb = Particles(90, 4, vx=(-8, 12), vy=(-20, -50), size=(1, 2.6), region=(0, 0, W, H))
        cues = self.p.get("cues", {})
        self.starts = [0.0] * len(self.people)
        for k, v in cues.items():
            if isinstance(v, int) and v < len(self.people):
                self.starts[v] = self.cue(k) or 0.0
        self.starts[0] = 0.0

    def draw(self, t):
        idx = 0
        for i, s in enumerate(self.starts):
            if t >= s - 0.4:
                idx = i
        k = ramp(t, self.starts[idx] - 0.4, 0.8) if idx > 0 else 1
        img = self.layer(idx, t)
        if k < 1 and idx > 0:
            img = Image.blend(self.layer(idx - 1, t), img, k)
        return img

    def layer(self, i, t):
        pp = self.people[i]
        img = self.bgs[i].copy()
        off = int(t * 25) % W
        sm = self.smoke.crop((off, 0, off + W, H))
        img = Image.composite(Image.new("RGB", (W, H), mix(pp["c"], (120, 120, 120), 0.5)), img, sm)
        tl = t - self.starts[i]
        x = int(W * 0.33 - self.busts[i].width / 2 + lerp(-30, 0, ease_out(tl / 3)))
        y = int(H - BAR - self.busts[i].height + 40 + math.sin(t * 0.9) * 4)
        img.paste(self.rims[i], (x - 8, y - 6), self.rims[i])
        img.paste(self.busts[i], (x, y), self.busts[i])
        d = ImageDraw.Draw(img)
        self.emb.draw(d, t, mix(pp["c"], (255, 220, 160), 0.6), alpha=0.8, flicker=0.4)
        nx = int(W * 0.6)
        name = pp["name"]
        size = 84
        while size > 30 and sum(font("deco", size).getlength(ch) + 4 for ch in name) > W - nx - 70:
            size -= 2
        a = ramp(tl, 0.3, 0.8)
        draw_text(img, (nx, 470), name, font("deco", size), (245, 235, 210), alpha=a, anchor="lm", spacing=4, glow=tuple(int(v * 0.6) for v in pp["c"]))
        d.line([(nx, 540), (nx + 420 * ramp(tl, 0.6, 1.0), 540)], fill=mix((0, 0, 0), pp["c"], 0.9), width=3)
        draw_text(img, (nx, 600), pp["epi"], font("italic", 52), (225, 215, 200), alpha=ramp(tl, 0.9, 0.8), anchor="lm")
        return img


# ======================================================================= MAP
@scene("map")
class MapScene(Scene):
    zoom = (1.0, 1.0)
    MARGIN = 1.12

    def setup(self):
        view = self.p["view"]
        mw, mh = int(W * self.MARGIN), int(H * self.MARGIN)
        hl = []
        if self.p.get("spanish"):
            hl = [(("484", "320", "340", "222", "558", "188", "591", "170", "862", "218", "604", "068", "152", "032",
                    "600", "858", "192", "214", "630"), (120, 60, 30))]
        self.map, self.proj = maps.render_map(view, (mw, mh), highlight=hl)
        if self.p.get("spanish"):
            self.us = maps.country_mask(view, (mw, mh), ("840",))
        self.mw, self.mh = mw, mh
        self.routes = []
        for r in self.p["routes"]:
            pts = np.array(r["pts"], np.float64)
            # densify along great-ish segments
            dense = []
            for a, b in zip(pts[:-1], pts[1:]):
                for s in np.linspace(0, 1, 30, endpoint=False):
                    dense.append(a + (b - a) * s)
            dense.append(pts[-1])
            dense = np.array(dense)
            x, y = self.proj(dense[:, 0], dense[:, 1])
            xy = np.stack([x, y], 1)
            # smooth path
            if len(xy) > 5:
                k = np.ones(9) / 9
                xs = np.convolve(np.pad(xy[:, 0], 4, mode="edge"), k, "valid")
                ys = np.convolve(np.pad(xy[:, 1], 4, mode="edge"), k, "valid")
                xy = np.stack([xs, ys], 1)
            seg = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
            cum = np.concatenate([[0], np.cumsum(seg)])
            self.routes.append(dict(xy=xy, cum=cum, cue=self.cue(r["cue"]), color=r.get("color")))
        self.pulse = 0

    def cam(self, t):
        z = lerp(1.0, 1.07, ease_in_out(t / self.dur))
        cw, ch = W / z, H / z
        x0 = (self.mw - cw) / 2 + math.sin(t * 0.1) * 20
        y0 = (self.mh - ch) / 2
        return x0, y0, z

    def draw(self, t):
        x0, y0, z = self.cam(t)
        cw, ch = W / z, H / z
        img = self.map.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)

        def P(x, y):
            return (x - x0) * z, (y - y0) * z

        if self.p.get("spanish"):
            a = ramp(t, self.cue(1), 1.0)
            if a > 0:
                m = self.us.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
                m = m.point(lambda v: int(v * a * (0.45 + 0.1 * math.sin(t * 3))))
                img = Image.composite(Image.new("RGB", (W, H), (200, 150, 60)), img, m)
        light = Image.new("RGB", (W, H))
        ld = ImageDraw.Draw(light)
        d = ImageDraw.Draw(img)
        for r in self.routes:
            if r["cue"] is None or t < r["cue"]:
                continue
            k = ease_in_out((t - r["cue"]) / 3.2)
            L = r["cum"][-1] * k
            n = int(np.searchsorted(r["cum"], L)) + 1
            pts = [P(*p) for p in r["xy"][:n]]
            if len(pts) < 2:
                continue
            col = (230, 80, 60) if r["color"] == "red" else (255, 200, 90)
            ld.line(pts, fill=mix((0, 0, 0), col, 0.8), width=12)
            # dashed core
            for j in range(0, len(pts) - 1, 2):
                d.line([pts[j], pts[j + 1]], fill=col, width=4)
            hx, hy = pts[-1]
            if k < 1:
                d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=(255, 245, 220))
            else:
                d.polygon(self.arrow(pts), fill=col)
        img = screen(img, light.filter(ImageFilter.GaussianBlur(8)))
        for lb in self.p["labels"]:
            c = self.cue(lb["cue"])
            a = ramp(t, c, 0.8)
            if a <= 0:
                continue
            x, y = P(*self.proj(*lb["at"]))
            dot = lb.get("dot")
            if dot:
                dx, dy = P(*self.proj(*(lb["at"] if dot is True else dot)))
                pr = 10 + 18 * ((t - c) * 0.8 % 1)
                d = ImageDraw.Draw(img)
                d.ellipse([dx - pr, dy - pr, dx + pr, dy + pr], outline=mix((0, 0, 0), (255, 210, 120), a * (1 - (t - c) * 0.8 % 1)), width=2)
                d.ellipse([dx - 7, dy - 7, dx + 7, dy + 7], fill=mix((0, 0, 0), (255, 220, 140), a))
            f = font("title", 30 if lb.get("small") else 44)
            draw_text(img, (x, y), lb["t"], f, (245, 235, 205), alpha=a, spacing=6)
            if lb.get("sub"):
                draw_text(img, (x, y + 44), lb["sub"], font("italic", 36), (225, 205, 160), alpha=a)
        return img

    @staticmethod
    def arrow(pts):
        (x1, y1), (x2, y2) = pts[-3] if len(pts) > 2 else pts[0], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        s = 20
        return [(x2 + math.cos(a) * s, y2 + math.sin(a) * s),
                (x2 + math.cos(a + 2.5) * s, y2 + math.sin(a + 2.5) * s),
                (x2 + math.cos(a - 2.5) * s, y2 + math.sin(a - 2.5) * s)]
