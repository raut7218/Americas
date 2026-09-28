"""High-contrast YouTube thumbnail (16:9) and Shorts/Reels cover (9:16).

  python3 thumbs.py   -> output/E1_thumbnail_youtube.jpg, output/E1_cover_vertical.jpg
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import maps
import scenes  # noqa: F401
import scenes2  # noqa: F401
from engine import HERE, H, W, caravel, glow_layer, longship, radial, screen
from scenes import REG
from script import SCENES

BUILD = os.environ.get("BUILD", "build")
OUT = os.environ.get("OUT", "output")
HEAVY = f"{HERE}/assets/fonts/Anton-Regular.ttf"
if not os.path.exists(HEAVY):
    HEAVY = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
YELLOW = (255, 210, 40)


def hf(size):
    return ImageFont.truetype(HEAVY, size)


def scene_frame(sid, t):
    tl = {s["id"]: s for s in json.load(open(f"{BUILD}/timeline.json"))["scenes"]}
    spec = next(s for s in SCENES if s["id"] == sid)
    return REG[spec["kind"]](spec, tl[sid]).draw(t)


def grade(img, tint, contrast=1.35, sat=1.3):
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(sat)
    a = np.asarray(img, np.float32) * np.array(tint, np.float32)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fit(img, size):
    """Cover-fit img into size (center crop)."""
    w, h = size
    k = max(w / img.width, h / img.height)
    im = img.resize((int(img.width * k + 1), int(img.height * k + 1)), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def stroked(d, xy, txt, f, fill, stroke=10, anchor="mm", shadow=True):
    x, y = xy
    if shadow:
        d.text((x + 8, y + 10), txt, font=f, fill=(0, 0, 0), anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))
    d.text((x, y), txt, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def text_glow(img, xy, txt, f, color, radius=30, anchor="mm"):
    lay = Image.new("RGB", img.size, (0, 0, 0))
    ImageDraw.Draw(lay).text(xy, txt, font=f, fill=color, anchor=anchor, stroke_width=14, stroke_fill=color)
    return screen(img, lay.filter(ImageFilter.GaussianBlur(radius)))


def vs_badge(img, cx, cy, r):
    d = ImageDraw.Draw(img)
    g = glow_layer(img.size, lambda dd, k: dd.ellipse([(cx - r * 1.5) * k, (cy - r * 1.5) * k, (cx + r * 1.5) * k, (cy + r * 1.5) * k],
                                                       fill=(255, 120, 30)), r * 0.8)
    img = screen(img, g)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10], fill=(0, 0, 0))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(205, 25, 25), outline=(255, 230, 120), width=max(4, r // 14))
    stroked(d, (cx, cy + r * 0.04), "VS", hf(int(r * 1.05)), (255, 255, 255), stroke=max(4, r // 18), shadow=False)
    return img


def backgrounds(size):
    storm = scene_frame("open", 1.62)   # a lightning frame
    gold = scene_frame("fleet", 3.0)
    storm = grade(storm, (0.8, 0.95, 1.25))
    gold = grade(gold, (1.2, 1.0, 0.75))
    return fit(storm, size), fit(gold, size)


def compose(size, vertical):
    w, h = size
    storm, gold = backgrounds(size)
    mask = Image.new("L", size, 0)
    md = ImageDraw.Draw(mask)
    if vertical:
        md.polygon([(0, 0), (w, 0), (w, h * 0.44), (0, h * 0.56)], fill=255)
    else:
        md.polygon([(0, 0), (w * 0.58, 0), (w * 0.42, h), (0, h)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(3))
    img = Image.composite(storm, gold, mask)
    s = h / 1080 if not vertical else w / 1080
    # hero ships, big and cropped by the frame edge for scale
    ls = longship(1.05 * s)
    cv = caravel(1.0 * s)
    if vertical:
        img.paste(ls, (int(-ls.width * 0.12), int(h * 0.47 - ls.height)), ls)
        img.paste(cv, (int(w - cv.width * 0.9), int(h * 1.02 - cv.height)), cv)
    else:
        img.paste(ls, (int(-ls.width * 0.18), int(h * 1.05 - ls.height)), ls)
        img.paste(cv, (int(w - cv.width * 0.85), int(h * 1.07 - cv.height)), cv)
    # glowing seam
    seam = Image.new("RGB", size, (0, 0, 0))
    sd = ImageDraw.Draw(seam)
    line = [(0, h * 0.56), (w, h * 0.44)] if vertical else [(w * 0.58, 0), (w * 0.42, h)]
    sd.line(line, fill=(255, 220, 140), width=int(10 * s))
    img = screen(img, seam.filter(ImageFilter.GaussianBlur(18 * s)))
    img = screen(img, seam.filter(ImageFilter.GaussianBlur(3 * s)))
    # globe behind the VS badge
    R = int(250 * s)
    cx, cy = (w / 2, h * 0.5) if vertical else (w / 2, h * 0.5)
    g = maps.globe(R, -50, 25, ocean=(20, 60, 110), land=(90, 150, 80))
    halo = glow_layer(size, lambda dd, k: dd.ellipse([(cx - R * 1.25) * k, (cy - R * 1.25) * k, (cx + R * 1.25) * k, (cy + R * 1.25) * k],
                                                     fill=(90, 150, 255)), R * 0.5)
    img = screen(img, halo)
    img.paste(g, (int(cx - R), int(cy - R)), g)
    img = vs_badge(img, int(cx), int(cy), int(95 * s))
    # vignette
    v = radial(w, h, w / 2, h / 2, max(w, h) * 0.75, (255, 255, 255), (70, 70, 70), 1.4)
    img = Image.fromarray((np.asarray(img, np.float32) * np.asarray(v, np.float32) / 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if vertical:
        img = text_glow(img, (w / 2, h * 0.14), "VIKINGS", hf(int(250 * s)), (60, 140, 255))
        d = ImageDraw.Draw(img)
        stroked(d, (w / 2, h * 0.14), "VIKINGS", hf(int(250 * s)), (240, 248, 255), stroke=int(12 * s))
        img = text_glow(img, (w / 2, h * 0.86), "COLUMBUS", hf(int(210 * s)), (255, 140, 20))
        d = ImageDraw.Draw(img)
        stroked(d, (w / 2, h * 0.86), "COLUMBUS", hf(int(210 * s)), YELLOW, stroke=int(12 * s))
        stroked(d, (w / 2, h * 0.255), "WHO REALLY FOUND AMERICA?", hf(int(78 * s)), (255, 255, 255), stroke=int(7 * s))
        d.rounded_rectangle([w * 0.2, h * 0.935, w * 0.8, h * 0.975], radius=int(20 * s), fill=(200, 25, 25))
        d.text((w / 2, h * 0.955), "UNDERSTANDING AMERICA · EP 1", font=hf(int(46 * s)), fill=(255, 255, 255), anchor="mm")
    else:
        img = text_glow(img, (w * 0.25, h * 0.2), "VIKINGS", hf(int(230 * s)), (60, 140, 255))
        img = text_glow(img, (w * 0.75, h * 0.2), "COLUMBUS", hf(int(200 * s)), (255, 140, 20))
        d = ImageDraw.Draw(img)
        stroked(d, (w * 0.25, h * 0.2), "VIKINGS", hf(int(230 * s)), (240, 248, 255), stroke=int(12 * s))
        stroked(d, (w * 0.75, h * 0.2), "COLUMBUS", hf(int(200 * s)), YELLOW, stroke=int(12 * s))
        stroked(d, (w * 0.5, h * 0.86), "WHO WAS FIRST?", hf(int(150 * s)), (255, 255, 255), stroke=int(12 * s))
        d.rounded_rectangle([w * 0.015, h * 0.02, w * 0.27, h * 0.085], radius=int(14 * s), fill=(200, 25, 25))
        d.text((w * 0.1425, h * 0.0525), "EP 1 · UNDERSTANDING AMERICA", font=hf(int(34 * s)), fill=(255, 255, 255), anchor="mm")
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    yt = compose((1920, 1080), vertical=False)
    yt.resize((1280, 720), Image.LANCZOS).save(f"{OUT}/E1_thumbnail_youtube.jpg", quality=92)
    yt.save(f"{BUILD}/prev/thumb_yt_full.jpg", quality=90)
    compose((1080, 1920), vertical=True).save(f"{OUT}/E1_cover_vertical.jpg", quality=92)
    print("thumbnails written")


if __name__ == "__main__":
    main()
