"""Second half of the scene library (Freydis through the finale)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import maps
from engine import (BAR, H, W, Particles, bust, caravel, clamp, draw_stars, draw_text, ease_in_out, ease_out, emoji,
                    font, glow_layer, lerp, longship, mix, noise2d, palm, paste_alpha, pine, radial, ramp,
                    ridge_points, rim_light, rng, rotate_paste, screen, smooth, stars_layer, vgrad, wrap)
from scenes import HORIZON, Scene, Sea, person, scene


def smoke_strip(seed, gray=40, dens=1.6, cut=0.5, alpha=180, w=W, h=H):
    sm = noise2d(w // 4, h // 4, seed=seed, octaves=5, base=3)
    a = np.clip(sm * dens - cut, 0, 1) * alpha
    arr = np.dstack([np.full(sm.shape + (3,), gray, np.float32), a[..., None]]).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA").resize((w, h), Image.BICUBIC)
    a = np.asarray(img)
    return Image.fromarray(np.concatenate([a, a[:, ::-1]], 1), "RGBA")


def scroll(base, strip, t, speed, dy=0):
    off = int((t * speed) % W)
    s = strip.crop((off, 0, off + W, strip.height))
    base.paste(s, (0, dy), s)


def longhouse(d, x, y, w, h, c, window=None):
    d.chord([x - w / 2, y - h * 1.6, x + w / 2, y + h * 0.4], 180, 360, fill=c)
    d.rectangle([x - w / 2, y - h * 0.6, x + w / 2, y], fill=c)
    if window:
        d.rectangle([x - 14, y - 50, x + 14, y - 5], fill=window)


# ======================================================================= NORSE FINALE
@scene("night")
class NightScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (4, 6, 14)), (0.55, (20, 26, 44)), (0.62, (60, 70, 90)), (1, (40, 44, 56))])
        d = ImageDraw.Draw(self.bg)
        d.ellipse([1500, 220, 1580, 300], fill=(220, 225, 235))
        self.bg = screen(self.bg, radial(W, H, 1540, 260, 400, (60, 70, 90), (0, 0, 0), 0.5))
        d = ImageDraw.Draw(self.bg)
        d.polygon(ridge_points(-20, W + 20, 620, 70, 91), fill=(14, 16, 24))
        d.polygon([(-10, 700), (W + 10, 680), (W + 10, H), (-10, H)], fill=(70, 78, 96))
        longhouse(d, 520, 700, 560, 120, (10, 10, 14), window=(255, 170, 70))
        longhouse(d, 1320, 700, 420, 100, (10, 10, 14), window=(90, 70, 50))
        self.stars = stars_layer(12, 300, 600)
        self.snow = Particles(240, 3, vx=(-15, 15), vy=(25, 60), size=(1, 2.4), region=(0, 0, W, H))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.7)
        # footprints appearing from the far house toward the lit one
        steps = int(t * 1.6)
        for i in range(min(steps, 26)):
            f = i / 26
            x = lerp(1250, 600, f) + (12 if i % 2 else -12)
            y = lerp(730, 760, f) + 60 * math.sin(f * 3.1)
            d.ellipse([x - 7, y - 4, x + 7, y + 4], fill=(40, 46, 62))
        # the walking figure
        f = clamp(t / 17)
        x = lerp(1250, 600, f)
        y = lerp(730, 760, f) + 60 * math.sin(f * 3.1)
        if f < 1:
            person(d, x, y, 110, (6, 6, 10), step=t * 5)
        gl = glow_layer((W, H), lambda dd, k: dd.ellipse([460 * k, 580 * k, 580 * k, 720 * k], fill=(160, 90, 30)), 50)
        img = screen(img, gl)
        self.snow.draw(ImageDraw.Draw(img), t, (220, 225, 240), alpha=0.8)
        return img


def axe(d, cx, cy, s, ang, col):
    ca, sa = math.cos(ang), math.sin(ang)

    def R(x, y):
        return cx + x * ca - y * sa, cy + x * sa + y * ca
    d.line([R(0, 0), R(0, -520 * s)], fill=col, width=int(26 * s))
    head = [(0, -500 * s), (170 * s, -600 * s), (210 * s, -450 * s), (170 * s, -320 * s), (0, -400 * s), (-40 * s, -440 * s)]
    d.polygon([R(*p) for p in head], fill=col)


@scene("axe")
class AxeScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (90, 10, 10), (4, 0, 0), 0.8)
        self.smoke = smoke_strip(44, gray=60, alpha=150)
        self.sea = Sea("dusk", "calm", seed=51, dark=True)
        self.ship = longship(0.36)
        self.crowd = [(rng(i).uniform(0, W), rng(i + 99).uniform(0, 1)) for i in range(70)]

    def draw(self, t):
        cs, c2, cl = self.cue_named("strike"), self.cue(2), self.cue_named("leave")
        if c2 is None or t < c2:
            img = self.bg.copy()
            scroll(img, self.smoke, t, 30)
            d = ImageDraw.Draw(img)
            ang = -0.2
            if cs is not None and t > cs:
                k = (t - cs) % 3.0
                ang = -0.2 - 0.9 * smooth(k / 1.2) + 1.6 * smooth((k - 1.3) / 0.25)
            axe(d, W * 0.52, H * 0.9, 1.0, ang, (6, 2, 2))
            return img
        img = Image.new("RGB", (W, H))
        self.sea.draw_sky(img, t)
        self.sea.draw_layers(img, t, 0, 6)
        leave = ramp(t, cl, 1)
        if cl is not None:
            k = clamp((t - cl) / 10)
            y, a = self.sea.ship_pose(4, 900 + k * 500, t)
            s = self.ship.resize((int(self.ship.width * (1 - 0.6 * k)), int(self.ship.height * (1 - 0.6 * k))))
            rotate_paste(img, s, 900 + k * 500, y - 30 * k, a)
        self.sea.draw_layers(img, t, 6, self.sea.L)
        d = ImageDraw.Draw(img)
        # growing crowd on the shore ridge (the natives were many)
        d.polygon([(-10, 860), (600, 800), (1200, 830), (W + 10, 790), (W + 10, H), (-10, H)], fill=(10, 4, 5))
        n = int(clamp((t - c2) / 6) * len(self.crowd)) if c2 is not None else 0
        for i, (x, r) in enumerate(self.crowd[:n]):
            if abs(x - 1100) < 250 and leave > 0:
                continue
            person(d, x, 830 + r * 60 - (x / W) * 20, 90 + r * 60, (8, 3, 4), "spear" if i % 3 == 0 else "plain")
        return img

    def flash(self, t):
        cs = self.cue_named("strike")
        c2 = self.cue(2)
        if cs is not None and cs < t < (c2 or 1e9):
            k = (t - cs) % 3.0
            if 1.45 < k < 1.8:
                return 0.5 * (1 - (k - 1.45) / 0.35)
        return 0.0


@scene("sod")
class SodScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (60, 70, 80)), (0.55, (150, 160, 165)), (0.6, (120, 130, 135)), (1, (60, 70, 50))])
        d = ImageDraw.Draw(self.bg)
        d.rectangle([0, 600, W, 640], fill=(80, 95, 105))
        d.polygon(ridge_points(-20, W + 20, 700, 40, 61, bottom=H), fill=(70, 90, 50))
        for x, w_, h_ in [(500, 420, 110), (1050, 300, 90), (1450, 380, 100)]:
            d.chord([x - w_ / 2, 760 - h_, x + w_ / 2, 760 + h_], 180, 360, fill=(58, 76, 38))
            d.chord([x - w_ / 2 + 20, 760 - h_ + 20, x + w_ / 2 - 20, 760 + h_], 180, 360, fill=(66, 86, 44))
            d.rectangle([x - 22, 700, x + 22, 760], fill=(20, 20, 16))
        self.grass = [(rng(i).uniform(0, W), rng(i + 7).uniform(780, H - BAR)) for i in range(260)]
        self.clouds = smoke_strip(7, gray=200, alpha=120, h=500)

    def draw(self, t):
        img = self.bg.copy()
        scroll(img, self.clouds, t, 18, dy=BAR - 40)
        d = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(self.grass):
            b = math.sin(t * 2 + x * 0.01) * 10
            d.line([(x, y), (x + b, y - 30)], fill=(50, 70, 34), width=2)
        cd, cu = self.cue_named("dig"), self.cue_named("unesco")
        a = ramp(t, cd, 1.2) * (1 - ramp(t, cu, 1))
        if a > 0:
            for x in range(200, W - 150, 120):
                d.line([(x, 640), (x - (x - W / 2) * 0.3, H - BAR)], fill=mix((66, 86, 44), (240, 220, 160), a * 0.7), width=2)
            for y in range(660, H - BAR, 50):
                d.line([(0, y), (W, y)], fill=mix((66, 86, 44), (240, 220, 160), a * 0.5), width=1)
            draw_text(img, (W / 2, 330), "REDISCOVERED · 1960s", font("title", 58), (250, 245, 230), alpha=a, spacing=8)
        a = ramp(t, cu, 1.0)
        draw_text(img, (W / 2, 300), "UNESCO WORLD HERITAGE SITE", font("title", 56), (255, 250, 240), alpha=a, spacing=6)
        draw_text(img, (W / 2, 380), "one of the first ever inscribed, 1978", font("italic", 48), (240, 240, 230), alpha=ramp(t, (cu or 0) + 0.6, 1) if cu else 0)
        return img


@scene("fires")
class FiresScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (4, 5, 12)), (0.5, (12, 16, 28)), (1, (4, 5, 8))])
        d = ImageDraw.Draw(self.bg)
        d.polygon(ridge_points(-20, W + 20, 520, 120, 17), fill=(8, 10, 16))
        d.polygon(ridge_points(-20, W + 20, 700, 90, 18), fill=(6, 7, 10))
        for x in range(0, W, 30):
            pine(d, x + (x * 7) % 13, 560 + (x * 13) % 60, 60 + (x * 7) % 40, (6, 7, 10))
        self.stars = stars_layer(2, 350, 520)
        r = rng(4)
        self.fires = [(r.uniform(150, W - 150), r.uniform(560, 900)) for _ in range(40)]
        order = r.permutation(len(self.fires))
        self.death = np.zeros(len(self.fires))
        cf, c3, cl = self.cue_named("fade"), self.cue_named("three"), self.cue_named("last")
        cf = cf or self.dur * 0.4
        c3 = c3 or self.dur * 0.6
        cl = cl or self.dur * 0.9
        for rank, i in enumerate(order):
            if rank < 37:
                self.death[i] = lerp(cf, c3, rank / 37)
            elif rank < 39:
                self.death[i] = lerp(c3 + 2, cl - 1, (rank - 37) / 2)
            else:
                self.death[i] = cl + 3.5
                self.last = i

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.8)
        live = []
        for i, (x, y) in enumerate(self.fires):
            a = 1 - ramp(t, self.death[i], 1.2)
            if a > 0:
                live.append((x, y, a, i))

        def glows(dd, k):
            for x, y, a, i in live:
                fl = 0.75 + 0.25 * math.sin(t * 11 + i) * math.sin(t * 5 + i * 2)
                c = mix((0, 0, 0), (255, 130, 40), a * fl)
                dd.ellipse([(x - 40) * k, (y - 34) * k, (x + 40) * k, (y + 14) * k], fill=c)
        img = screen(img, glow_layer((W, H), glows, 26))
        d = ImageDraw.Draw(img)
        for x, y, a, i in live:
            fl = 0.7 + 0.3 * math.sin(t * 13 + i)
            d.polygon([(x - 6, y), (x, y - 16 * fl), (x + 6, y)], fill=mix((0, 0, 0), (255, 220, 140), a))
        cl = self.cue_named("last")
        if cl is not None:
            a = ramp(t, cl, 1.0)
            draw_text(img, (W / 2, 330), "SHANAWDITHIT", font("deco", 80), (230, 220, 200), alpha=a, spacing=8, glow=(120, 60, 20))
            draw_text(img, (W / 2, 420), "c. 1801 – 1829", font("italic", 50), (210, 200, 185), alpha=a)
        return img


# ======================================================================= COLUMBUS
@scene("globe")
class GlobeScene(Scene):
    zoom = (1.0, 1.0)

    def setup(self):
        self.stars = stars_layer(9, 600, H)
        self.bg = radial(W, H, W / 2, H / 2, 1100, (14, 18, 36), (2, 2, 6), 0.8)
        self.pear = self.p.get("variant") == "pear"

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.9)
        R = 300
        lon = -40 + t * 6 if not self.pear else -65 + t * 1.5
        cx, cy = W / 2, H / 2
        if not self.pear:
            cr, cs = self.cue_named("round"), self.cue_named("size")
            rnd = ramp(t, cr, 1.5)
            if rnd < 1:
                # flat disc with an edge the ships fall off
                a = 1 - rnd
                d.ellipse([cx - 420, cy - 60, cx + 420, cy + 60], fill=mix((0, 0, 0), (40, 80, 120), a), outline=mix((0, 0, 0), (180, 160, 110), a), width=3)
                d.ellipse([cx - 300, cy - 40, cx - 60, cy + 20], fill=mix((0, 0, 0), (70, 110, 60), a))
                d.ellipse([cx + 40, cy - 30, cx + 280, cy + 30], fill=mix((0, 0, 0), (70, 110, 60), a))
                for i in range(12):
                    x = cx + 420 + (i % 3) * 5
                    d.line([(x - 10, cy + 10 + i * 18), (x - 10, cy + 20 + i * 18)], fill=mix((0, 0, 0), (60, 110, 160), a), width=6)
                draw_text(img, (W / 2, 330), "THE FLAT-EARTH MYTH", font("title", 56), (230, 220, 200), alpha=a * ramp(t, 0.5, 0.8), spacing=6)
                if cr is not None and t > cr - 0.8:
                    k = ramp(t, cr - 0.8, 0.4) * a
                    d = ImageDraw.Draw(img)
                    d.line([(cx - 450, cy - 200), (cx + 450, cy + 200)], fill=mix((0, 0, 0), (200, 40, 30), k), width=14)
                    d.line([(cx + 450, cy - 200), (cx - 450, cy + 200)], fill=mix((0, 0, 0), (200, 40, 30), k), width=14)
            if rnd > 0:
                if cs is not None and t > cs:
                    k = ease_in_out((t - cs) / 1.5)
                    for j, (rr, dx) in enumerate([(120, -560), (200, 0), (290, 600)]):
                        rrr = int(lerp(R, rr, k))
                        x = cx + lerp(0, dx, k)
                        g = maps.globe(rrr, lon + j * 40, 20)
                        paste_alpha(img, g, rnd, (int(x - rrr), int(cy - rrr)))
                        draw_text(img, (x, cy + rrr + 60), "?", font("deco", 80), (240, 220, 170), alpha=k)
                else:
                    g = maps.globe(R, lon, 20)
                    paste_alpha(img, g, rnd, (int(cx - R), int(cy - R)))
            return img
        # pear variant
        cp = self.cue_named("pear")
        pr = ramp(t, cp, 3.0)
        g = maps.globe(R, lon, 8, pear=pr)
        gl = glow_layer((W, H), lambda dd, k: dd.ellipse([(cx - R - 30) * k, (cy - R - 30) * k, (cx + R + 30) * k, (cy + R + 30) * k], fill=(30, 60, 120)), 60)
        img = screen(img, gl)
        img.paste(g, (int(cx - R), int(cy - R)), g)
        a = ramp(t, self.cue(1), 1.0) * (1 - pr)
        draw_text(img, (W / 2, 190), "VENEZUELA · 1498", font("title", 46), (240, 220, 170), alpha=a, spacing=6)
        if pr > 0:
            tip = glow_layer((W, H), lambda dd, k: dd.ellipse([(cx - 60) * k, (cy - R - 80) * k, (cx + 60) * k, (cy - R + 20) * k], fill=(255, 230, 150)), 40)
            img = Image.blend(img, screen(img, tip), pr * (0.7 + 0.3 * math.sin(t * 3)))
            draw_text(img, (W / 2, cy - R - 110), "EDEN?", font("deco", 60), (255, 240, 200), alpha=pr, spacing=8, glow=(200, 150, 50))
        return img


def skyline_constantinople(d, base, c):
    d.rectangle([0, base - 60, W, base], fill=c)
    for i in range(0, W, 60):
        d.rectangle([i, base - 80, i + 34, base - 60], fill=c)
    domes = [(500, 240, 150), (820, 180, 110), (1250, 260, 170), (1580, 160, 100)]
    for x, w_, h_ in domes:
        d.rectangle([x - w_ * 0.8, base - h_ * 0.9, x + w_ * 0.8, base - 60], fill=c)
        d.chord([x - w_ / 2, base - h_ * 0.9 - w_ / 2, x + w_ / 2, base - h_ * 0.9 + w_ / 2], 180, 360, fill=c)
        d.rectangle([x - 4, base - h_ * 0.9 - w_ / 2 - 50, x + 4, base - h_ * 0.9 - w_ / 2], fill=c)
        for side in (-1, 1):
            mx = x + side * w_ * 0.95
            d.rectangle([mx - 9, base - h_ * 1.9, mx + 9, base - 60], fill=c)
            d.polygon([(mx - 11, base - h_ * 1.9), (mx, base - h_ * 1.9 - 60), (mx + 11, base - h_ * 1.9)], fill=c)


@scene("city")
class CityScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (20, 10, 26)), (0.45, (90, 40, 50)), (0.62, (230, 120, 60)), (1, (40, 20, 20))])
        self.city = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        skyline_constantinople(ImageDraw.Draw(self.city), 860, (10, 6, 10, 255))
        self.fire = Particles(200, 5, vx=(-20, 20), vy=(-80, -200), size=(2, 5), region=(200, 500, 1800, 860), wobble=20)
        self.smoke = smoke_strip(71, gray=35, alpha=170)

    def draw(self, t):
        img = self.bg.copy()
        cf = self.cue_named("fall") or 0
        burn = ramp(t, cf + 1.5, 3)
        if burn > 0:
            gl = glow_layer((W, H), lambda d, k: d.rectangle([100 * k, 650 * k, 1800 * k, 900 * k], fill=(255, 90, 20)), 100)
            img = Image.blend(img, screen(img, gl), burn * (0.8 + 0.2 * math.sin(t * 9)))
            scroll(img, self.smoke, t, 40, dy=-150)
        d = ImageDraw.Draw(img)
        # crescent moon rising
        my = lerp(520, 260, ramp(t, cf, 5))
        d.ellipse([1500, my, 1600, my + 100], fill=(240, 230, 200))
        d.ellipse([1528, my - 10, 1628, my + 90], fill=mix((90, 40, 50), (20, 10, 26), 0.3))
        img.paste(self.city, (0, 0), self.city)
        d = ImageDraw.Draw(img)
        if burn > 0:
            self.fire.draw(d, t, (255, 160, 60), alpha=burn, flicker=0.5)
        c1 = self.cue(1)
        if c1 is not None:
            a = ramp(t, c1, 1)
            y = 300
            x1 = lerp(150, 1770, clamp((t - c1) / 3))
            for x in range(150, int(x1), 30):
                if abs(x - W / 2) > 40:
                    d.line([(x, y), (x + 16, y)], fill=mix((0, 0, 0), (255, 210, 110), a), width=4)
            draw_text(img, (170, y - 50), "EUROPE", font("title", 34), (240, 220, 180), alpha=a, anchor="lm", spacing=4)
            draw_text(img, (1760, y - 50), "THE INDIES", font("title", 34), (240, 220, 180), alpha=a, anchor="rm", spacing=4)
            k = ramp(t, c1 + 3.2, 0.4)
            d.line([(W / 2 - 40, y - 40), (W / 2 + 40, y + 40)], fill=mix((0, 0, 0), (220, 40, 30), k), width=10)
            d.line([(W / 2 + 40, y - 40), (W / 2 - 40, y + 40)], fill=mix((0, 0, 0), (220, 40, 30), k), width=10)
        return img


@scene("compare")
class CompareScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1100, (20, 34, 54), (3, 5, 9), 0.9)
        d = ImageDraw.Draw(self.bg)
        for x in range(0, W, 80):
            d.line([(x, 0), (x, H)], fill=(18, 28, 42))
        for y in range(0, H, 80):
            d.line([(0, y), (W, y)], fill=(18, 28, 42))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        a, b = self.p["a"], self.p["b"]
        x0, maxw = 200, 1500
        scale = maxw / b["value"]
        cb, cg = self.cue_named("b"), self.cue_named("gap")
        for i, (spec, c0, col) in enumerate([(a, 0.4, (230, 180, 80)), (b, cb, (80, 170, 230))]):
            if c0 is None:
                continue
            k = ease_out((t - c0) / 2.5) if t > c0 else 0
            y = 430 + i * 230
            draw_text(img, (x0, y - 70), spec["label"], font("title", 38), (230, 225, 210), alpha=ramp(t, c0, 0.6), anchor="lm", spacing=6)
            w_ = spec["value"] * scale * k
            d.rectangle([x0, y - 30, x0 + w_, y + 30], fill=col)
            val = int(spec["value"] * k)
            draw_text(img, (x0 + w_ + 30, y), f"{val:,} {spec['unit']}", font("title", 50), (250, 245, 235), alpha=ramp(t, c0, 0.6), anchor="lm")
        if cg is not None:
            k = ramp(t, cg, 1.0)
            xa = x0 + a["value"] * scale
            xb = x0 + b["value"] * scale
            d.rectangle([xa, 380, xb, 720], outline=mix((0, 0, 0), (230, 70, 60), k), width=4)
            draw_text(img, ((xa + xb) / 2, 800), "13,000 KM OF UNKNOWN", font("title", 52), (255, 200, 180), alpha=k, spacing=6)
        return img


@scene("court")
class CourtScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (20, 10, 8)), (0.6, (60, 30, 18)), (1, (25, 12, 8))])
        d = ImageDraw.Draw(self.bg)
        c = (12, 6, 5)
        for i in range(7):
            x = 140 + i * 280
            d.rectangle([x - 40, BAR, x + 40, H], fill=c)
        for i in range(6):
            x = 280 + i * 280
            d.rectangle([x - 100, BAR, x + 100, 300], fill=c)
            d.chord([x - 100, 200, x + 100, 400], 0, 180, fill=c)
        for x in (500, 1420):
            d.rectangle([x - 70, 330, x + 70, 620], fill=(130, 20, 20))
            d.polygon([(x - 70, 620), (x, 670), (x + 70, 620)], fill=(130, 20, 20))
            d.rectangle([x - 30, 420, x + 30, 490], fill=(200, 160, 60))
        d.rectangle([0, 860, W, H], fill=(18, 9, 7))
        d.polygon([(700, 860), (1220, 860), (1100, 780), (820, 780)], fill=(26, 12, 9))
        self.royals = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(self.royals)
        for x in (820, 1100):
            rd.rectangle([x - 110, 560, x + 110, 830], fill=(8, 4, 4, 255))
            rd.polygon([(x - 110, 560), (x - 110, 470), (x - 60, 500), (x, 440), (x + 60, 500), (x + 110, 470), (x + 110, 560)], fill=(8, 4, 4, 255))
            rd.rectangle([x - 150, 700, x + 150, 740], fill=(8, 4, 4, 255))
            b = bust(360, "king")
            self.royals.paste(b, (int(x - b.width / 2), 440), b)
        self.coins = Particles(140, 8, vx=(-20, 20), vy=(160, 300), size=(4, 8), region=(0, -50, W, H), wobble=10)
        self.dust = Particles(120, 4, vx=(-5, 5), vy=(-5, -15), size=(1, 2))

    def draw(self, t):
        img = self.bg.copy()
        rays = glow_layer((W, H), lambda d, k: [d.polygon([((380 + i * 280) * k, 250 * k), ((460 + i * 280) * k, 250 * k),
                                                          ((700 + i * 320) * k, H * k), ((560 + i * 320) * k, H * k)],
                                                         fill=(110, 70, 30)) for i in range(5)], 50)
        img = screen(img, rays)
        cr = self.cue_named("royals")
        paste_alpha(img, self.royals, ramp(t, cr if cr is not None else 0, 1.2))
        d = ImageDraw.Draw(img)
        self.dust.draw(d, t, (255, 220, 160), alpha=0.6)
        cg = self.cue_named("gold")
        if cg is not None and t > cg:
            k = ramp(t, cg, 0.6)
            gl = glow_layer((W, H), lambda dd, kk: dd.ellipse([600 * kk, 300 * kk, 1320 * kk, 800 * kk], fill=(200, 150, 40)), 120)
            img = Image.blend(img, screen(img, gl), k)
            self.coins.draw(ImageDraw.Draw(img), t - cg, (255, 210, 90), alpha=k)
            if self.p.get("variant") != "triumph":
                draw_text(img, (W / 2, 330), "GOLD", font("deco", 190), (255, 225, 130), alpha=k, spacing=24, glow=(220, 140, 20))
        ct = self.cue_named("title")
        if ct is not None:
            a = ramp(t, ct, 1)
            draw_text(img, (W / 2, 270), "ADMIRAL OF THE OCEAN SEA", font("deco", 70), (255, 235, 180), alpha=a, spacing=6, glow=(180, 110, 20))
        return img


@scene("logs")
class LogsScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, 500, 900, (80, 55, 30), (6, 4, 3), 0.8)

    def page(self, img, cx, title, frac, t, t0, col):
        a = ramp(t, t0, 0.8)
        if a <= 0:
            return
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rectangle([cx - 330, 250, cx + 330, 850], fill=(215, 195, 150, 255))
        for i in range(12):
            y = 380 + i * 36
            L = 520 * clamp((t - t0) * 0.7 - i * 0.3)
            d.line([(cx - 270, y), (cx - 270 + L, y)], fill=(90, 70, 50, 255), width=3)
        paste_alpha(img, lay, a)
        draw_text(img, (cx, 310), title, font("title", 40), (70, 40, 20), alpha=a, spacing=4, shadow=False)
        # distance bar
        k = clamp((t - t0) / 8)
        dd = ImageDraw.Draw(img)
        dd.rectangle([cx - 270, 800, cx - 270 + 540 * frac * k, 820], fill=mix((0, 0, 0), col, a))
        draw_text(img, (cx, 770), "leagues sailed", font("italic", 34), (70, 40, 20), alpha=a, shadow=False)

    def draw(self, t):
        img = self.bg.copy()
        c = self.cue_named("logs")
        if c is None:
            return img
        self.page(img, W * 0.3, "THE TRUE LOG", 1.0, t, c, (150, 30, 20))
        self.page(img, W * 0.7, "THE LOG FOR THE CREW", 0.72, t, c + 0.8, (40, 80, 120))
        flick = 0.85 + 0.15 * math.sin(t * 9) * math.sin(t * 4.3)
        gl = glow_layer((W, H), lambda d, k: d.ellipse([(W / 2 - 200) * k, 100 * k, (W / 2 + 200) * k, 500 * k], fill=(120, 80, 30)), 100)
        return Image.blend(img, screen(img, gl), flick)


@scene("landing")
class LandingScene(Scene):
    def setup(self):
        self.sea = Sea("day", "calm", seed=61, horizon=560)
        self.sea.pal = dict(self.sea.pal, sea=(40, 150, 170), near=(20, 110, 130), crest=(200, 240, 240))
        self.ships = [(caravel(0.28), 400), (caravel(0.22), 700), (caravel(0.18), 1000)]
        self.beach = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.beach)
        d.polygon([(-10, 780), (700, 740), (1300, 760), (W + 10, 720), (W + 10, H), (-10, H)], fill=(225, 205, 160, 255))
        d.polygon([(-10, 820), (900, 800), (W + 10, 790), (W + 10, H), (-10, H)], fill=(210, 185, 140, 255))
        for i, x in enumerate([1480, 1600, 1740, 1850]):
            palm(d, x, 760, 360 + (i % 2) * 80, (30, 40, 25, 255), lean=-0.25, seed=i)

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        self.sea.draw_sky(img, t)
        self.sea.draw_layers(img, t, 0, 5)
        for s, x in self.ships:
            y, a = self.sea.ship_pose(3, x, t)
            rotate_paste(img, s, x, y, a)
        self.sea.draw_layers(img, t, 5, self.sea.L)
        img.paste(self.beach, (0, 0), self.beach)
        d = ImageDraw.Draw(img)
        for j in range(5):
            person(d, 400 + j * 70, 830 + (j % 2) * 10, 170, (30, 22, 20), "banner" if j == 0 else "helmet")
        ct = self.cue_named("taino")
        n = int(clamp((t - (ct or 1e9)) / 3) * 7)
        for j in range(n):
            person(d, 1150 + j * 60, 820 + (j % 2) * 12, 190, (110, 60, 35), "plain")
        ctr = self.cue_named("trade")
        if ctr is not None and t > ctr:
            for j, e in enumerate(["\U0001f99c", "\U0001f4ff", "\U0001f514"]):
                a = ramp(t, ctr + j * 0.6, 0.6) * (1 - ramp(t, self.cue(3) or 1e9, 0.8))
                em = emoji(e, 90)
                x = int(W / 2 - 200 + j * 200 - em.width / 2)
                paste_alpha(img, em, a, (x, int(300 + math.sin(t * 2 + j) * 10)))
        c4, c5 = self.cue(4), self.cue(5)
        if c4 is not None:
            draw_text(img, (W / 2, 330), "NO METAL", font("deco", 90), (255, 250, 240), alpha=ramp(t, c4, 0.6) * (1 - ramp(t, c5, 0.6)), spacing=10)
        if c5 is not None and t > c5:
            em = emoji("\U0001f5e1️", 180)
            paste_alpha(img, em, ramp(t, c5, 0.5), (int(W / 2 - em.width / 2), 380))
        return img


@scene("quote")
class QuoteScene(Scene):
    zoom = (1.0, 1.07)

    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (60, 40, 22), (4, 3, 2), 0.9)

    def draw(self, t):
        img = self.bg.copy()
        c = self.cue(1) or 1.5
        q = self.p["quote"]
        n = int(clamp((t - c + 0.3) / 1.8) * len(q))
        draw_text(img, (W / 2, 470), q[:n], font("italic", 110), (240, 225, 195), alpha=1 if n else 0, glow=(120, 70, 20))
        draw_text(img, (W / 2, 620), self.p["who"], font("title", 34), (200, 180, 140), alpha=ramp(t, c + 2, 1), spacing=6)
        flick = 0.8 + 0.2 * math.sin(t * 8) * math.sin(t * 3.7)
        gl = glow_layer((W, H), lambda d, k: d.ellipse([(W / 2 - 400) * k, 250 * k, (W / 2 + 400) * k, 700 * k], fill=(90, 55, 20)), 120)
        return Image.blend(img, screen(img, gl), flick)


@scene("wreck")
class WreckScene(Scene):
    def setup(self):
        self.sea = Sea("night", "calm", seed=71)
        self.ship = caravel(0.55)
        self.shore = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.shore)
        d.polygon([(900, H), (1100, 760), (1500, 720), (W + 10, 700), (W + 10, H)], fill=(10, 12, 14, 255))
        for i, x in enumerate([1700, 1820]):
            palm(d, x, 720, 260, (6, 8, 10, 255), lean=-0.3, seed=i)
        self.fort = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fd = ImageDraw.Draw(self.fort)
        for i, x in enumerate(range(1250, 1650, 22)):
            fd.polygon([(x, 740), (x, 620 - (i % 2) * 10), (x + 10, 600 - (i % 2) * 10), (x + 20, 620 - (i % 2) * 10), (x + 20, 740)], fill=(14, 10, 10, 255))
        fd.line([(1450, 600), (1450, 480)], fill=(14, 10, 10, 255), width=5)
        fd.polygon([(1450, 480), (1520, 500), (1450, 520)], fill=(150, 30, 20, 255))

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        self.sea.draw_sky(img, t)
        self.sea.draw_layers(img, t, 0, 7)
        cf, cs = self.cue_named("fort"), self.cue_named("sail")
        wa = 1 - ramp(t, cf, 2)
        if wa > 0:
            y, _ = self.sea.ship_pose(7, 700, t)
            rotate_paste(img, self.ship, 700, y + 30 * (1 - wa) + 20, -14 + math.sin(t) * 1, alpha=wa)
        if cs is not None and t > cs:
            k = clamp((t - cs) / 8)
            s = self.ship.resize((int(self.ship.width * 0.35), int(self.ship.height * 0.35)))
            y, a = self.sea.ship_pose(2, 700 - k * 600, t)
            rotate_paste(img, s, 700 - k * 600, y, a, alpha=ramp(t, cs, 1))
        self.sea.draw_layers(img, t, 7, self.sea.L)
        img.paste(self.shore, (0, 0), self.shore)
        fa = ramp(t, cf, 2.5)
        if fa > 0:
            paste_alpha(img, self.fort, fa)
            gl = glow_layer((W, H), lambda d, k: [d.ellipse([(x - 15) * k, 650 * k, (x + 15) * k, 680 * k], fill=(255, 170, 70)) for x in (1300, 1420, 1560)], 20)
            img = Image.blend(img, screen(img, gl), fa)
            draw_text(img, (1450, 380), "LA NAVIDAD", font("title", 50), (240, 225, 190), alpha=fa, spacing=8)
            draw_text(img, (1450, 440), "39 men", font("italic", 42), (220, 205, 180), alpha=ramp(t, self.cue(2), 1))
        return img


@scene("ruins")
class RuinsScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (40, 40, 45)), (0.6, (120, 110, 100)), (1, (30, 26, 24))])
        d = ImageDraw.Draw(self.bg)
        d.polygon([(-10, 760), (W + 10, 720), (W + 10, H), (-10, H)], fill=(34, 28, 24))
        r = rng(5)
        for i, x in enumerate(range(500, 1500, 26)):
            h_ = r.uniform(10, 90)
            d.polygon([(x, 760), (x + 4, 760 - h_), (x + 16, 760 - h_ * 0.8), (x + 20, 760)], fill=(14, 10, 8))
        self.smoke = Particles(60, 5, vx=(5, 25), vy=(-20, -50), size=(20, 45), region=(500, 250, 1500, 760), wobble=30)
        self.birds = [(rng(i).uniform(0, W), rng(i + 3).uniform(200, 400), rng(i + 9).uniform(40, 80)) for i in range(7)]

    def draw(self, t):
        img = self.bg.copy()
        sm = Image.new("RGB", (W, H))
        self.smoke.draw(ImageDraw.Draw(sm), t, (255, 255, 255), alpha=1.0)
        sm = sm.convert("L").filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * 0.5))
        img = Image.composite(Image.new("RGB", (W, H), (60, 56, 54)), img, sm)
        d = ImageDraw.Draw(img)
        for x, y, v in self.birds:
            bx = (x + v * t) % (W + 200) - 100
            by = y + math.sin(t + x) * 10
            wv = math.sin(t * 6 + x) * 8
            d.line([(bx - 14, by - wv), (bx, by), (bx + 14, by - wv)], fill=(10, 10, 12), width=3)
        return img


@scene("quota")
class QuotaScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (40, 28, 12), (3, 2, 1), 0.8)
        self.gold = Particles(180, 3, vx=(-6, 6), vy=(-6, -20), size=(1, 3))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        cx, cy = W / 2, 540
        k = ((t - (self.cue_named("quota") or 0)) / 6) % 1 if t > (self.cue_named("quota") or 1e9) else 0.2
        gcol = (220, 170, 60)
        d.polygon([(cx - 150, cy - 230), (cx + 150, cy - 230), (cx + 10, cy), (cx - 10, cy)], outline=(200, 180, 150), width=4)
        d.polygon([(cx - 10, cy), (cx + 10, cy), (cx + 150, cy + 230), (cx - 150, cy + 230)], outline=(200, 180, 150), width=4)
        top = lerp(0.1, 1, k)
        tw = 130 * (1 - top)
        d.polygon([(cx - tw, cy - 230 * (1 - top) + 0), (cx + tw, cy - 230 * (1 - top)), (cx + 8, cy - 4), (cx - 8, cy - 4)], fill=gcol)
        bh = 200 * k
        d.polygon([(cx - 140, cy + 225), (cx + 140, cy + 225), (cx + 140 - bh * 0.4, cy + 225 - bh * 0.5), (cx, cy + 225 - bh * 0.9), (cx - 140 + bh * 0.4, cy + 225 - bh * 0.5)], fill=gcol)
        d.line([(cx, cy), (cx, cy + 225 - bh * 0.9)], fill=gcol, width=3)
        for yy in (cy - 245, cy + 245):
            d.rectangle([cx - 190, yy - 12, cx + 190, yy + 12], fill=(60, 36, 20))
        self.gold.draw(d, t, (255, 200, 90), alpha=0.7)
        cq = self.cue_named("quota")
        draw_text(img, (W * 0.22, 470), "A QUOTA", font("deco", 70), (250, 220, 150), alpha=ramp(t, cq, 0.8), spacing=6)
        draw_text(img, (W * 0.22, 560), "OF GOLD", font("deco", 70), (250, 220, 150), alpha=ramp(t, cq, 0.8), spacing=6)
        draw_text(img, (W * 0.78, 515), "EVERY 3 MONTHS", font("title", 50), (240, 225, 200), alpha=ramp(t, (cq or 0) + 1, 0.8) if cq else 0, spacing=6)
        cb = self.cue_named("blood")
        if cb is not None:
            a = ramp(t, cb, 2)
            arr = np.asarray(img, np.float32)
            arr = arr * (1 - 0.5 * a) + np.array([120, 0, 0]) * 0.5 * a
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return img


@scene("split")
class SplitScene(Scene):
    def setup(self):
        left = vgrad([(0, (10, 8, 8)), (1, (40, 26, 18))], W // 2, H)
        d = ImageDraw.Draw(left)
        d.polygon(ridge_points(-10, W // 2 + 10, 300, 120, 3, bottom=0), fill=(4, 3, 3))
        right = vgrad([(0, (230, 180, 90)), (0.5, (240, 210, 140)), (0.62, (120, 110, 60)), (1, (80, 70, 40))], W // 2, H)
        rd = ImageDraw.Draw(right)
        for i in range(12):
            y = 680 + i * 30
            rd.line([(0, y), (W // 2, y - 40)], fill=(60, 70, 30), width=6)
        self.left, self.right = left, right
        self.dust = Particles(80, 3, vx=(-5, 5), vy=(-8, -16), size=(1, 2), region=(0, 0, W // 2, H))

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        img.paste(self.left, (0, 0))
        img.paste(self.right, (W // 2, 0))
        d = ImageDraw.Draw(img)
        for j in range(5):
            x = 150 + j * 150
            dig = math.sin(t * 3 + j) * 0.5
            person(d, x, 860, 170, (6, 4, 4))
            d.line([(x + 30, 740), (x + 90, 740 + 80 * dig)], fill=(6, 4, 4), width=6)
        for j in range(5):
            x = W // 2 + 150 + j * 160
            person(d, x, 880 - j * 15, 150, (60, 40, 20))
        self.dust.draw(d, t, (200, 150, 80), alpha=0.6)
        d.line([(W // 2, BAR), (W // 2, H - BAR)], fill=(0, 0, 0), width=8)
        draw_text(img, (W * 0.25, 260), "THE MINES", font("title", 56), (230, 210, 170), alpha=ramp(t, 0.5, 0.8), spacing=8)
        draw_text(img, (W * 0.75, 260), "THE FIELDS", font("title", 56), (60, 40, 20), alpha=ramp(t, 0.9, 0.8), spacing=8, shadow=False)
        cc, c3 = self.cue_named("children"), self.cue_named("third")
        a = ramp(t, cc, 1) * (1 - ramp(t, c3, 1))
        if a > 0:
            arr = np.asarray(img, np.float32) * (1 - 0.6 * a)
            img = Image.fromarray(arr.astype(np.uint8))
            em = emoji("\U0001f56f️", 130)
            paste_alpha(img, em, a, (int(W / 2 - em.width / 2), 420))
        if c3 is not None and t > c3:
            k = ramp(t, c3, 0.6)
            arr = np.asarray(img, np.float32) * (1 - 0.8 * k)
            img = Image.fromarray(arr.astype(np.uint8))
            d = ImageDraw.Draw(img)
            for i in range(60):
                r_, c_ = divmod(i, 20)
                x = W / 2 - 20 * 45 / 2 + c_ * 45
                y = 480 + r_ * 110
                dead = r_ == 2 and ramp(t, c3 + 1 + c_ * 0.1, 0.5) > 0.5
                person(d, x, y + 80, 80, (90, 80, 70) if dead else (220, 200, 170))
            draw_text(img, (W / 2, 330), "1 IN 3", font("deco", 100), (240, 200, 180), alpha=k, spacing=10)
        return img


@scene("chains")
class ChainsScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (50, 20, 12), (3, 1, 1), 0.8)
        self.smoke = smoke_strip(81, gray=50, alpha=150)
        self.fire = Particles(160, 3, vx=(-30, 30), vy=(-80, -200), size=(2, 4), region=(0, 600, W, H), wobble=20)
        self.col = bust(700, "columbus")
        self.rim = rim_light(self.col, (220, 120, 50), 1.0, 12)

    def draw(self, t):
        img = self.bg.copy()
        scroll(img, self.smoke, t, 40)
        d = ImageDraw.Draw(img)
        self.fire.draw(d, t, (255, 140, 40), alpha=1 - ramp(t, self.cue(1), 2), flicker=0.5)
        a = ramp(t, self.cue(1), 1.5)
        if a > 0:
            x = int(W / 2 - self.col.width / 2)
            y = int(H - BAR - self.col.height + 60 + 20 * ramp(t, self.cue_named("chains"), 2))
            paste_alpha(img, self.rim, a, (x - 6, y - 6))
            paste_alpha(img, self.col, a, (x, y))
        cc = self.cue_named("chains")
        if cc is not None and t > cc:
            k = ease_out((t - cc) / 0.5)
            d = ImageDraw.Draw(img)
            for side in (-1, 1):
                for i in range(14):
                    f = i / 13
                    sw = math.sin(t * 1.5 + side) * 12 * f
                    x = W / 2 + side * lerp(900, 120, f * k) + sw
                    y = lerp(150, 830, f) * k + (1 - k) * 100
                    if i % 2:
                        d.ellipse([x - 22, y - 34, x + 22, y + 34], outline=(150, 140, 130), width=9)
                    else:
                        d.ellipse([x - 34, y - 20, x + 34, y + 20], outline=(120, 112, 105), width=9)
        return img

    def flash(self, t):
        cc = self.cue_named("chains")
        if cc is not None and cc < t < cc + 0.6:
            return 0.4 * (1 - (t - cc) / 0.6)
        return 0.0


@scene("naming")
class NamingScene(Scene):
    zoom = (1.0, 1.0)

    def setup(self):
        view = (-130, -10, -50, 62)
        img, self.proj = maps.render_map(view, (W, H), ocean=(212, 190, 145), land=(180, 150, 100), coast=(90, 60, 35),
                                         grid=(195, 172, 128), glow=(160, 120, 70))
        tex = noise2d(W // 2, H // 2, seed=3, octaves=6, base=6)
        arr = np.asarray(img, np.float32) * (0.82 + 0.3 * np.asarray(Image.fromarray((tex * 255).astype(np.uint8)).resize((W, H)), np.float32)[..., None] / 255)
        self.map = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        self.namerio = self.proj(-58, -12)
        self.na = self.proj(-100, 45)

    def draw(self, t):
        img = self.map.copy()
        ca = self.cue_named("america")
        if ca is not None and t > ca - 0.3:
            word = "AMERICA"
            n = clamp((t - ca + 0.3) / 2.0) * len(word)
            f = font("deco", 110)
            x, y = self.namerio
            draw_text(img, (x, y), word[:int(math.ceil(n))], f, (110, 30, 20), alpha=1, spacing=10, shadow=False)
        draw_text(img, (W / 2, 210), "UNIVERSALIS COSMOGRAPHIA · 1507", font("title", 40), (80, 50, 30), alpha=ramp(t, 0.8, 1), spacing=6, shadow=False)
        cc = self.cue_named("columbia")
        if cc is not None:
            a = ramp(t, cc, 1)
            draw_text(img, (W * 0.2, 760), "COLUMBUS →", font("title", 40), (80, 40, 25), alpha=a, spacing=4, shadow=False)
            draw_text(img, (W * 0.2, 820), "District of Columbia · Colombia", font("italic", 42), (80, 40, 25), alpha=a, shadow=False)
        cn = self.cue_named("continents")
        if cn is not None:
            a = ramp(t, cn, 1)
            x, y = self.na
            draw_text(img, (x, y), "AMERICA", font("deco", 90), (110, 30, 20), alpha=a, spacing=10, shadow=False)
            draw_text(img, (W * 0.8, 820), "VESPUCCI → two continents", font("italic", 44), (110, 30, 20), alpha=a, shadow=False)
        return img


@scene("crypt")
class CryptScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, 600, 900, (40, 30, 22), (2, 2, 2), 0.8)
        d = ImageDraw.Draw(self.bg)
        for i in range(5):
            x = 200 + i * 380
            d.rectangle([x - 50, BAR, x + 50, H], fill=(8, 6, 5))
        self.cands = [(460, 760), (580, 720), (1340, 720), (1460, 760)]

    def draw(self, t):
        img = self.bg.copy()
        co, ct = self.cue_named("open"), self.cue_named("text")

        def glows(d, k):
            for i, (x, y) in enumerate(self.cands):
                f = 0.8 + 0.2 * math.sin(t * 10 + i * 2) * math.sin(t * 4.1 + i)
                d.ellipse([(x - 90 * f) * k, (y - 140 * f) * k, (x + 90 * f) * k, (y + 40) * k], fill=(200, 120, 40))
            if co is not None and t > co:
                a = ramp(t, co, 1.5)
                d.rectangle([(W / 2 - 250) * k, (620 - 100 * a) * k, (W / 2 + 250) * k, 660 * k], fill=mix((0, 0, 0), (200, 180, 140), a))
        img = screen(img, glow_layer((W, H), glows, 60))
        d = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(self.cands):
            d.rectangle([x - 10, y, x + 10, y + 120], fill=(200, 190, 170))
            f = 0.8 + 0.2 * math.sin(t * 10 + i * 2)
            d.polygon([(x - 7, y), (x, y - 28 * f), (x + 7, y)], fill=(255, 220, 140))
        # lead box
        d.rectangle([W / 2 - 280, 640, W / 2 + 280, 860], fill=(70, 72, 76), outline=(110, 112, 118), width=4)
        lid = ramp(t, co, 2) if co is not None else 0
        ly = 620 - 120 * lid
        lx = W / 2 + 200 * lid
        d.polygon([(lx - 300, ly), (lx + 300, ly), (lx + 280, ly + 30), (lx - 280, ly + 30)], fill=(85, 88, 92), outline=(120, 122, 128))
        if ct is not None:
            a = ramp(t, ct, 1.5)
            draw_text(img, (W / 2, 330), "“Illustrious and distinguished man,", font("italic", 62), (245, 230, 200), alpha=a, glow=(140, 90, 30))
            draw_text(img, (W / 2, 410), "Don Cristóbal Colón”", font("italic", 62), (245, 230, 200), alpha=ramp(t, ct + 0.8, 1.5), glow=(140, 90, 30))
        return img


@scene("lighthouse")
class LighthouseScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (4, 5, 12)), (0.7, (16, 20, 34)), (1, (8, 8, 10))])
        self.stars = stars_layer(22, 400, 700)
        self.clouds = smoke_strip(91, gray=70, alpha=150, h=500)
        self.mon = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.mon)
        c = (10, 10, 12, 255)
        for i in range(8):
            w_ = 1500 - i * 150
            d.rectangle([W / 2 - w_ / 2, 860 - (i + 1) * 36, W / 2 + w_ / 2, 860 - i * 36], fill=c)
        d.rectangle([0, 860, W, H], fill=(6, 6, 8, 255))
        d.polygon([(0, 880), (300, 800), (360, 880)], fill=(6, 6, 8, 255))
        for x in range(1500, W, 60):
            d.rectangle([x, 780 - (x % 7) * 20, x + 40, 880], fill=(6, 6, 8, 255))
            for k in range(3):
                d.rectangle([x + 8, 800 + k * 25 - (x % 7) * 10, x + 16, 810 + k * 25 - (x % 7) * 10], fill=(90, 70, 30, 255))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        draw_stars(d, self.stars, t, 0.8)
        con, coff = self.cue_named("on"), self.cue_named("off")
        on = ramp(t, con, 1.5)
        if coff is not None and t > coff:
            k = t - coff
            on *= 0 if k > 2.2 else (0.5 + 0.5 * math.sin(k * 25)) * (1 - k / 2.2)
        scroll(img, self.clouds, t, 10, dy=BAR)
        if on > 0:
            def beams(dd, kk):
                for i in range(-6, 7):
                    x0 = W / 2 + i * 40
                    dd.polygon([(x0 * kk, 580 * kk), ((x0 + 10) * kk, 580 * kk), ((W / 2 + 10) * kk, 330 * kk), ((W / 2 - 10) * kk, 330 * kk)], fill=(120, 130, 150))
                dd.rectangle([(W / 2 - 22) * kk, 150 * kk, (W / 2 + 22) * kk, 480 * kk], fill=(200, 210, 230))
                dd.rectangle([(W / 2 - 170) * kk, 250 * kk, (W / 2 + 170) * kk, 290 * kk], fill=(200, 210, 230))
            img = Image.blend(img, screen(img, glow_layer((W, H), beams, 20)), on)
        img.paste(self.mon, (0, 0), self.mon)
        c3 = self.cue(3)
        if c3 is not None:
            draw_text(img, (W / 2, 400), "FUKÚ", font("deco", 150), (190, 40, 30), alpha=ramp(t, c3, 1.5) * 0.9, spacing=20, glow=(120, 20, 10))
        return img


@scene("church")
class ChurchScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, 300, 900, (40, 34, 40), (4, 3, 5), 0.8)
        d = ImageDraw.Draw(self.bg)
        d.rectangle([W / 2 - 120, 160, W / 2 + 120, 520], fill=(120, 90, 60))
        d.chord([W / 2 - 120, 60, W / 2 + 120, 280], 180, 360, fill=(120, 90, 60))
        for i, col in enumerate([(150, 40, 40), (40, 70, 150), (180, 150, 60), (40, 120, 70)]):
            d.rectangle([W / 2 - 110 + i * 55, 180, W / 2 - 60 + i * 55, 500], fill=col)
        d.rectangle([W / 2 - 6, 160, W / 2 + 6, 520], fill=(20, 16, 18))
        d.rectangle([W / 2 - 120, 330, W / 2 + 120, 342], fill=(20, 16, 18))
        d.polygon([(700, H), (760, 720), (1160, 720), (1220, H)], fill=(12, 9, 10))
        d.rectangle([780, 650, 1140, 720], fill=(16, 12, 12))
        self.fri = bust(360, "priest")
        self.dust = Particles(120, 7, vx=(-4, 4), vy=(4, 12), size=(1, 2), region=(700, 150, 1200, 900))

    def draw(self, t):
        img = self.bg.copy()
        rays = glow_layer((W, H), lambda d, k: d.polygon([((W / 2 - 110) * k, 200 * k), ((W / 2 + 110) * k, 200 * k),
                                                          ((W / 2 + 300) * k, H * k), ((W / 2 - 300) * k, H * k)], fill=(90, 70, 60)), 40)
        img = screen(img, rays)
        img.paste(self.fri, (int(W / 2 - self.fri.width / 2), 400), self.fri)
        d = ImageDraw.Draw(img)
        self.dust.draw(d, t, (230, 210, 180), alpha=0.7)
        cw = self.cue_named("words")
        if cw is not None:
            a = ramp(t, cw, 0.8) * (1 - ramp(t, self.cue(2), 1))
            draw_text(img, (W * 0.2, 400), "“Are these", font("italic", 80), (250, 240, 220), alpha=a)
            draw_text(img, (W * 0.2, 490), "not men?”", font("italic", 80), (250, 240, 220), alpha=a)
        return img


@scene("book")
class OldBookScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1000, (40, 12, 8), (3, 1, 1), 0.8)
        self.emb = Particles(160, 9, vx=(-20, 20), vy=(-30, -80), size=(1, 3), region=(0, 0, W, H))

    def draw(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        d.rectangle([W / 2 - 330, 200, W / 2 + 330, 890], fill=(205, 185, 140))
        d.rectangle([W / 2 - 300, 230, W / 2 + 300, 860], outline=(120, 80, 50), width=3)
        lines = wrap(self.p["title"], font("title", 50), 520)
        for i, ln in enumerate(lines):
            draw_text(img, (W / 2, 330 + i * 70), ln, font("title", 50), (80, 30, 20), alpha=ramp(t, 0.3 + i * 0.3, 0.8), shadow=False)
        draw_text(img, (W / 2, 700), "Bartolomé de las Casas", font("italic", 50), (80, 30, 20), alpha=ramp(t, 1.5, 1), shadow=False)
        draw_text(img, (W / 2, 780), self.p["year"], font("title", 44), (80, 30, 20), alpha=ramp(t, 1.8, 1), shadow=False)
        dark = ramp(t, self.cue(1), 3)
        if dark > 0:
            arr = np.asarray(img, np.float32) * (1 - 0.45 * dark) + np.array([60, 0, 0]) * 0.3 * dark
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        self.emb.draw(ImageDraw.Draw(img), t, (255, 140, 50), alpha=0.4 + 0.6 * dark, flicker=0.5)
        return img


@scene("census")
class CensusScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1100, (30, 22, 18), (3, 2, 2), 0.9)
        icon = Image.new("RGBA", (28, 52), (0, 0, 0, 0))
        person(ImageDraw.Draw(icon), 14, 51, 50, (230, 210, 180, 255))
        self.icon = icon
        r = rng(3)
        self.cells = [(160 + (i % 50) * 32, 330 + (i // 50) * 56) for i in range(500)]
        self.survivors = set(r.choice(500, 23, replace=False).tolist())

    def draw(self, t):
        img = self.bg.copy()
        cd, cg, cw = self.cue_named("drop"), self.cue_named("gone"), self.cue_named("word")
        drop = ramp(t, cd, 3)
        gone = ramp(t, cg, 2)
        for i, (x, y) in enumerate(self.cells):
            a = ramp(t, 0.2 + i * 0.004, 0.4)
            if i not in self.survivors:
                a *= 1 - smooth(drop * 1.5 - (i % 37) / 37 * 0.5)
            else:
                a *= 1 - gone
            if a > 0.02:
                paste_alpha(img, self.icon, a, (x, y))
        a0 = 1 - ramp(t, cw, 0.8)
        val = int(lerp(500000, 22726, smooth(drop))) if cd is not None and t > cd else 500000
        label = "1492" if not (cd is not None and t > cd) else "1514"
        txt = f"{val:,}+" if label == "1492" else "< 23,000"
        draw_text(img, (W / 2, 220), f"{label}:  {txt}", font("deco", 70), (240, 220, 190), alpha=ramp(t, 0.5, 0.8) * a0 * (1 - gone), spacing=4)
        draw_text(img, (W / 2, 900), "each figure = 1,000 people", font("italic", 36), (190, 175, 150), alpha=ramp(t, 1.5, 1) * a0 * (1 - gone))
        if cw is not None:
            a = ramp(t, cw + 2.2, 1.5)
            draw_text(img, (W / 2, 540), "GENOCIDE", font("deco", 170), (200, 30, 25), alpha=a, spacing=24, glow=(90, 0, 0))
        return img


@scene("words")
class WordsScene(Scene):
    def setup(self):
        self.bg = vgrad([(0, (20, 40, 50)), (0.6, (60, 110, 110)), (1, (20, 30, 30))])
        d = ImageDraw.Draw(self.bg)
        for i, x in enumerate([80, 220, 1700, 1840]):
            palm(d, x, H - BAR, 520 + (i % 2) * 80, (10, 20, 22), lean=0.25 if x < W / 2 else -0.25, seed=i)
        self.bg = self.bg.filter(ImageFilter.GaussianBlur(2))
        pairs = self.p["pairs"]
        c1, c2 = self.cue(1), self.cue(2)
        self.slots = []
        for j in range(3):
            self.slots.append((c1 + j * (self.chunk_end(1) - c1) / 3, pairs[j]))
        for j in range(3, 7):
            self.slots.append((c2 + (j - 3) * (self.chunk_end(2) - c2) / 4, pairs[j]))

    def draw(self, t):
        img = self.bg.copy()
        c0, cc = self.cue(0), self.cue_named("crops")
        draw_text(img, (W / 2, 300), "THE TAÍNO LIVE ON IN WORDS", font("title", 50), (230, 240, 230), alpha=ramp(t, 0.5, 1) * (1 - ramp(t, cc, 1)), spacing=6)
        for i, (s, (a, b)) in enumerate(self.slots):
            e = self.slots[i + 1][0] if i + 1 < len(self.slots) else (cc or self.dur)
            if not (s - 0.2 <= t < e + 0.3):
                continue
            vis = ramp(t, s - 0.2, 0.3) * (1 - ramp(t, e, 0.3))
            k = ramp(t, s + (e - s) * 0.45, 0.4)
            draw_text(img, (W / 2, 520 - 40 * k), a, font("italic", 120), (255, 230, 170), alpha=vis * (1 - 0.6 * k))
            draw_text(img, (W / 2, 640), "↓  " + b.upper(), font("deco", 90), (240, 250, 245), alpha=vis * k, spacing=6, glow=(40, 120, 110))
        if cc is not None:
            for j, e in enumerate(self.p["crops"]):
                a = ramp(t, cc + j * 0.35, 0.6)
                em = emoji(e, 150)
                x = int(W / 2 + (j - (len(self.p["crops"]) - 1) / 2) * 230 - em.width / 2)
                paste_alpha(img, em, a, (x, int(480 + math.sin(t * 2 + j) * 12)))
            draw_text(img, (W / 2, 330), "THE CROPS OF THE AMERICAS", font("title", 50), (240, 245, 235), alpha=ramp(t, cc, 1), spacing=6)
        return img


@scene("versus")
class VersusScene(Scene):
    def setup(self):
        self.bg = radial(W, H, W / 2, H / 2, 1100, (40, 30, 26), (4, 3, 3), 0.9)
        d = ImageDraw.Draw(self.bg)
        d.line([(W / 2 + 160, 240), (W / 2 + 160, 900)], fill=(80, 60, 40), width=2)
        d.line([(W / 2 + 480, 240), (W / 2 + 480, 900)], fill=(80, 60, 40), width=2)

    def mark(self, d, x, y, ok, a):
        if ok:
            d.line([(x - 22, y), (x - 6, y + 18), (x + 26, y - 20)], fill=mix((0, 0, 0), (120, 210, 120), a), width=9)
        else:
            d.line([(x - 18, y - 18), (x + 18, y + 18)], fill=mix((0, 0, 0), (220, 70, 60), a), width=9)
            d.line([(x + 18, y - 18), (x - 18, y + 18)], fill=mix((0, 0, 0), (220, 70, 60), a), width=9)

    def draw(self, t):
        img = self.bg.copy()
        draw_text(img, (W / 2 + 0, 250), "THE NORSE", font("title", 44), (180, 200, 230), alpha=ramp(t, 0.4, 0.8), anchor="mm", spacing=6)
        draw_text(img, (W / 2 + 480 + 160, 250), "SPAIN", font("title", 44), (240, 200, 120), alpha=ramp(t, 0.8, 0.8), spacing=6)
        cues = {v: k for k, v in self.p["cues"].items()}
        d = ImageDraw.Draw(img)
        for i, (name, norse, spain) in enumerate(self.p["rows"]):
            c = self.cue(cues.get(i, -1))
            a = ramp(t, c, 0.6)
            if a <= 0:
                continue
            y = 370 + i * 110
            draw_text(img, (W / 2 - 180, y), name, font("title", 42), (235, 225, 205), alpha=a, anchor="rm", spacing=4)
            d = ImageDraw.Draw(img)
            self.mark(d, W / 2 + 0, y, norse, a)
            self.mark(d, W / 2 + 640, y, spain, ramp(t, (c or 0) + 0.3, 0.5))
        return img


@scene("next")
class NextScene(Scene):
    fade_out = 1.0

    def setup(self):
        self.sea = Sea("dusk", "calm", seed=91)
        self.emb = Particles(80, 5, vx=(-8, 8), vy=(-15, -35), size=(1, 2.5))

    def draw(self, t):
        img = Image.new("RGB", (W, H))
        self.sea.draw_sky(img, t)
        self.sea.draw_layers(img, t, 0, self.sea.L)
        arr = np.asarray(img, np.float32) * 0.55
        img = Image.fromarray(arr.astype(np.uint8))
        self.emb.draw(ImageDraw.Draw(img), t, (255, 200, 120), alpha=0.8)
        cn = self.cue_named("next")
        for j, w_ in enumerate(["DISCOVERY", "CONQUEST", "SETTLEMENT"]):
            x = W / 2 + (j - 1) * 580
            if j == 0:
                a = ramp(t, 0.5, 0.8)
                draw_text(img, (x, 500), w_, font("deco", 66), (170, 160, 140), alpha=a, spacing=6)
                d = ImageDraw.Draw(img)
                d.line([(x - 220, 500), (x - 220 + 440 * ramp(t, 1.5, 0.8), 500)], fill=(200, 150, 80), width=4)
            else:
                a = ramp(t, (cn or 99) + (j - 1) * 0.8, 1)
                draw_text(img, (x, 500), w_, font("deco", 74), (255, 230, 170), alpha=a, spacing=6, glow=(200, 120, 30))
        draw_text(img, (W / 2, 330), "NEXT TIME", font("title", 44), (240, 220, 180), alpha=ramp(t, cn, 1), spacing=14)
        return img
