"""Vertical (9:16) trailer for Shorts / Reels / TikTok, cut from the film's scene renderers.

  python3 trailer.py          # -> output/E1_trailer_vertical.mp4 (+ small copy for chat/WhatsApp)

Reuses the episode's scenes (drawn fresh at 1920x1080 without letterbox), reframes each shot for
9:16 (full-bleed crop or framed over a blurred fill), adds a trailer voiceover, word-by-word
captions and a trailer score built from music.py instruments.
"""
import json
import math
import os
import re
import subprocess
from functools import lru_cache
from multiprocessing import Pool

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly

import music as M
import scenes  # noqa: F401  (registers kinds)
import scenes2  # noqa: F401
from engine import FPS, H, W, HERE, Particles, draw_text, font, mix, ramp
from scenes import REG
from script import SAY, SCENES

BUILD = os.environ.get("BUILD", "build")
OUT = os.environ.get("OUT", "output")
MODELS = os.environ.get("MODELS", "models")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VW, VH = 1080, 1920
SR = 44100
FONT_HEAVY = f"{HERE}/assets/fonts/Anton-Regular.ttf"
if not os.path.exists(FONT_HEAVY):
    FONT_HEAVY = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
YELLOW = (255, 214, 64)

# t: seconds into the scene, or "cue:<name>[+/-offset]" relative to a named visual cue.
# mode: "crop" = full-bleed 9:16 crop centred at cx; "frame" = 16:9 picture over a blurred fill.
SHOTS = [
    dict(scene="open", t=1.0, mode="crop", cx=0.43, hit=True,
         vo="Five hundred years before Columbus, the Vikings found America."),
    dict(scene="erik", t=0.8, mode="frame", vo="It started with a killer. Erik the Red. Banished. Twice."),
    dict(scene="greenland", t=0.5, mode="crop", cx=0.5, vo="He sailed west, and sold the world an island of ice, as Greenland."),
    dict(scene="grapes", t="cue:title-0.3", mode="frame", vo="His son Leif found a land of wild grapes. Vinland."),
    dict(scene="arrows", t="cue:volley-0.2", mode="crop", cx=0.34, hit=True,
         vo="Then they met the people who already lived there."),
    dict(scene="axe", t="cue:strike-0.3", mode="crop", cx=0.52, hit=True, vo="And Erik's daughter took up an axe."),
    dict(scene="bridge", t=1.0, mode="crop", cx=0.58, vo="Then the world forgot, for five hundred years."),
    dict(scene="fleet", t=0.8, mode="crop", cx=0.5, hit=True, vo="Until fourteen ninety-two."),
    dict(scene="landing", t="cue:trade-0.5", mode="frame", vo="Columbus promised gold. He found paradise."),
    dict(scene="census", t="cue:word+2.0", mode="frame", hit=True, vo="And within a generation, its people were gone."),
    dict(scene="chains", t="cue:chains+0.2", mode="crop", cx=0.5, vo="He died disgraced, still sure he'd reached India."),
    dict(scene="naming", t="cue:america-0.2", mode="frame", vo="And the land he found? Named after someone else."),
    dict(scene="END", t=0, mode="end", hit=True, vo="Understanding America. Episode one. The full story is on the channel."),
]
PAD = 0.45        # silence after each line before the cut
LEAD = 0.25       # silence before each line


# ------------------------------------------------------------------ voice
def say(text):
    for k in sorted(SAY, key=len, reverse=True):
        text = re.sub(re.escape(k), SAY[k], text)
    return text


def synth_vo():
    os.makedirs(f"{BUILD}/trailer", exist_ok=True)
    kok = None
    lines = []
    for i, sh in enumerate(SHOTS):
        path = f"{BUILD}/trailer/vo_{i}.wav"
        if not os.path.exists(path):
            if kok is None:
                from kokoro_onnx import Kokoro
                kok = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
            s, sr = kok.create(say(sh["vo"]), voice="bm_george", speed=1.0, lang="en-gb")
            s = resample_poly(s, SR // 300, sr // 300).astype(np.float32)
            idx = np.where(np.abs(s) > 0.01)[0]
            s = s[max(0, idx[0] - 600): idx[-1] + 2000]
            sf.write(path, s, SR)
        s, _ = sf.read(path, dtype="float32")
        lines.append(s)
    return lines


# ------------------------------------------------------------------ captions
@lru_cache(maxsize=None)
def heavy(size):
    return ImageFont.truetype(FONT_HEAVY, size)


def word_times(text, start, dur):
    """Approximate per-word timing from syllable-ish weights (no forced aligner offline)."""
    words = text.split()
    w = np.array([max(1.0, len(re.sub(r"[^A-Za-z]", "", x)) ** 0.8 + (1.2 if x[-1] in ",.?!" else 0)) for x in words])
    edges = start + np.concatenate([[0], np.cumsum(w)]) / w.sum() * dur
    return list(zip(words, edges[:-1], edges[1:]))


def groups(words, n=3):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == n or w[0][-1] in ",.?!":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


@lru_cache(maxsize=512)
def caption_sprite(words, active):
    f = heavy(96)
    txt = [w.upper() for w in words]
    widths = [f.getlength(w) for w in txt]
    gap = 26
    tw = sum(widths) + gap * (len(txt) - 1)
    lines = [(txt, widths)] if tw < VW - 120 else None
    if lines is None:  # wrap to two lines
        k = max(1, len(txt) // 2)
        lines = [(txt[:k], widths[:k]), (txt[k:], widths[k:])]
    im = Image.new("RGBA", (VW, 140 * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    idx = 0
    for li, (ws, wd) in enumerate(lines):
        x = (VW - (sum(wd) + gap * (len(ws) - 1))) / 2
        for w, wdt in zip(ws, wd):
            col = YELLOW if idx == active else (255, 255, 255)
            d.text((x, 20 + li * 130), w, font=f, fill=col, stroke_width=9, stroke_fill=(0, 0, 0))
            x += wdt + gap
            idx += 1
    return im


# ------------------------------------------------------------------ framing
_VIG = None
_GRAIN = None


def vfinish(arr, t, flash=0.0, fade=1.0):
    global _VIG, _GRAIN
    if _VIG is None:
        yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
        v = 1 - 0.5 * np.clip(((xx - VW / 2) / (VW * 0.75)) ** 2 + ((yy - VH / 2) / (VH * 0.62)) ** 2, 0, 1) ** 1.3
        _VIG = np.dstack([(v * 255).astype(np.uint8)] * 3)
        g = np.random.default_rng(7)
        _GRAIN = []
        for _ in range(6):
            a = np.repeat(np.repeat(g.normal(0, 5, (VH // 2, VW // 2)), 2, 0), 2, 1)
            _GRAIN.append((np.dstack([np.clip(a, 0, 255).astype(np.uint8)] * 3), np.dstack([np.clip(-a, 0, 255).astype(np.uint8)] * 3)))
    out = cv2.multiply(arr, _VIG, scale=1 / 255)
    p, n = _GRAIN[int(t * FPS) % 6]
    cv2.add(out, p, dst=out)
    cv2.subtract(out, n, dst=out)
    if flash > 0.01:
        out = cv2.addWeighted(out, 1 - flash, out, 0, 255 * flash)
    if fade < 0.999:
        out = cv2.convertScaleAbs(out, alpha=fade)
    return out


def reframe(src, mode, cx, zoom):
    a = np.asarray(src)
    if mode == "crop":
        ch = H / zoom
        cw = ch * VW / VH
        x0 = min(max(cx * W - cw / 2, 0), W - cw)
        y0 = (H - ch) / 2
        s = VH / ch
        M_ = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        return cv2.warpAffine(a, M_, (VW, VH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # frame: blurred, darkened full-bleed fill + the whole 16:9 picture
    ch = H
    cw = ch * VW / VH
    x0 = (W - cw) / 2
    bg = cv2.resize(a[:, int(x0):int(x0 + cw)], (VW // 8, VH // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (VW, VH), interpolation=cv2.INTER_LINEAR)
    bg = cv2.convertScaleAbs(bg, alpha=0.45)
    fw = int(VW * zoom * 1.12)  # slightly wider than the screen: trims the sides a touch, feels bigger
    fh = int(fw * H / W)
    fr = cv2.resize(a, (fw, fh), interpolation=cv2.INTER_AREA)
    fx = (VW - fw) // 2
    fy = 880 - fh // 2
    x_a, x_b = max(0, -fx), min(fw, VW - fx)
    bg[fy:fy + fh, fx + x_a:fx + x_b] = fr[:, x_a:x_b]
    # thin gold rules above/below the picture
    bg[fy - 6:fy - 2, 80:VW - 80] = (200, 160, 80)
    bg[fy + fh + 2:fy + fh + 6, 80:VW - 80] = (200, 160, 80)
    return bg


class EndCard:
    def __init__(self):
        self.emb = Particles(140, 5, vx=(-10, 10), vy=(-40, -90), size=(1.5, 4), region=(0, 0, VW, VH), wobble=20)
        yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
        d = np.sqrt((xx - VW / 2) ** 2 + (yy - VH * 0.42) ** 2) / 1100
        k = np.clip(d, 0, 1) ** 0.8
        bg = np.dstack([90 * (1 - k) + 4 * k, 45 * (1 - k) + 3 * k, 12 * (1 - k) + 3 * k]).astype(np.uint8)
        self.bg = Image.fromarray(bg)

    def draw(self, t):
        img = self.bg.copy()
        self.emb.draw(ImageDraw.Draw(img), t, (255, 170, 70), alpha=0.9, flicker=0.4)
        draw_text(img, (VW / 2, 560), "UNDERSTANDING", font("deco", 96), (245, 228, 185), alpha=ramp(t, 0.1, 0.5), spacing=4, glow=(170, 100, 30))
        draw_text(img, (VW / 2, 690), "AMERICA", font("deco", 150), (255, 236, 190), alpha=ramp(t, 0.3, 0.5), spacing=10, glow=(190, 110, 30))
        draw_text(img, (VW / 2, 820), "EPISODE 1  ·  VIKINGS TO COLUMBUS", font("title", 38), (230, 205, 150), alpha=ramp(t, 0.7, 0.5), spacing=3)
        a = ramp(t, 1.2, 0.5)
        if a > 0:
            d = ImageDraw.Draw(img)
            pulse = 1 + 0.03 * math.sin(t * 6)
            bw, bh = 760 * pulse, 130 * pulse
            box = [VW / 2 - bw / 2, 1000 - bh / 2, VW / 2 + bw / 2, 1000 + bh / 2]
            d.rounded_rectangle(box, radius=26, fill=mix((0, 0, 0), (200, 30, 30), a))
            f = heavy(64)
            d.text((VW / 2, 1000), "▶  WATCH FULL EPISODE", font=f, fill=mix((0, 0, 0), (255, 255, 255), a), anchor="mm")
        draw_text(img, (VW / 2, 1130), "full 24-minute documentary on the channel", font("italic", 46), (235, 220, 200), alpha=ramp(t, 1.6, 0.5))
        return img


# ------------------------------------------------------------------ build
def resolve_t(sc, t):
    if isinstance(t, (int, float)):
        return float(t)
    m = re.match(r"cue:(\w+)([+-][\d.]+)?", t)
    c = sc.cue_named(m.group(1)) or 0.0
    return c + float(m.group(2) or 0)


def plan():
    lines = synth_vo()
    t, out = 0.0, []
    for sh, s in zip(SHOTS, lines):
        d = LEAD + len(s) / SR + PAD
        if sh["mode"] == "end":
            d = max(d, 5.5)
        out.append(dict(sh, start=t, dur=round(d * FPS) / FPS, vo_len=len(s) / SR))
        t += out[-1]["dur"]
    return out, lines, t


def render_shot(args):
    i, sh = args
    path = f"{BUILD}/trailer/shot_{i:02d}.mp4"
    tl = json.load(open(f"{BUILD}/timeline.json"))
    by = {s["id"]: s for s in tl["scenes"]}
    if sh["mode"] == "end":
        sc, t0 = EndCard(), 0.0
    else:
        spec = next(s for s in SCENES if s["id"] == sh["scene"])
        sc = REG[spec["kind"]](spec, by[sh["scene"]])
        t0 = resolve_t(sc, sh["t"])
    words = word_times(sh["vo"], LEAD, sh["vo_len"])
    grp = groups(words)
    n = int(round(sh["dur"] * FPS))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        tt = k / FPS
        if sh["mode"] == "end":
            arr = np.asarray(sc.draw(tt))
        else:
            src = sc.draw(t0 + tt)
            z = 1.0 + 0.10 * (tt / sh["dur"])  # constant push-in keeps every shot moving
            arr = reframe(src, sh["mode"], sh.get("cx", 0.5), z)
        flash = 0.8 * (1 - ramp(tt, 0, 0.18)) if sh.get("hit") else 0.25 * (1 - ramp(tt, 0, 0.12))
        arr = vfinish(np.ascontiguousarray(arr), tt, flash=flash)
        img = Image.fromarray(arr)
        # persistent hook title (top) on story shots
        if sh["mode"] != "end":
            d = ImageDraw.Draw(img)
            d.text((VW / 2, 250), "WHO REALLY FOUND", font=heavy(78), fill=(255, 255, 255), anchor="mm", stroke_width=7, stroke_fill=(0, 0, 0))
            d.text((VW / 2, 350), "AMERICA FIRST?", font=heavy(92), fill=YELLOW, anchor="mm", stroke_width=8, stroke_fill=(0, 0, 0))
        # word-by-word captions
        if sh["mode"] != "end":
            for g in grp:
                if g[0][1] - 0.05 <= tt < g[-1][2] + 0.25:
                    active = max([j for j, w in enumerate(g) if w[1] <= tt] or [0])
                    spr = caption_sprite(tuple(w[0] for w in g), active)
                    pop = 1 + 0.08 * (1 - ramp(tt, g[0][1] - 0.05, 0.12))
                    if pop > 1.001:
                        spr = spr.resize((int(spr.width * pop), int(spr.height * pop)))
                    img.paste(spr, (int((VW - spr.width) / 2), int(1330 - spr.height / 2 + 60)), spr)
                    break
        fade = 1 - ramp(tt, sh["dur"] - 0.6, 0.6) if sh["mode"] == "end" else 1.0
        if fade < 1:
            img = Image.fromarray(cv2.convertScaleAbs(np.asarray(img), alpha=fade))
        p.stdin.write(img.tobytes())
    p.stdin.close()
    p.wait()
    return i


def score(shots, total, lines):
    N = int((total + 1) * SR)
    mus = np.zeros((N, 2), np.float32)
    sfx = np.zeros((N, 2), np.float32)
    vo = np.zeros(N, np.float32)
    rng = np.random.default_rng(3)
    for sh, s in zip(shots, lines):
        a = int((sh["start"] + LEAD) * SR)
        vo[a:a + len(s)] += s[:max(0, N - a)]
    # drone + epic pads through the whole trailer
    d = M.drone(N, 0) * 0.35
    mus += np.stack([d, d], 1)
    prog = M.PROG["epic"]
    for ci in range(int(total // M.CHORD) + 2):
        for j, m in enumerate(M.CH[prog[ci % 4]]):
            M.put(mus, M.pad_voice(m + (12 if j == 3 else 0), M.CHORD + 2.2, 1), ci * M.CHORD, pan=(j - 1.5) * 0.4, gain=0.07)
    mus = M.lp(mus, 2400).astype(np.float32)
    end_start = shots[-1]["start"]
    # ostinato + taiko build, silent on the end card until the final hit
    step = M.BEAT / 4
    for i in range(int(end_start / step)):
        tt = i * step
        ch = M.CH[prog[int(tt // M.CHORD) % 4]]
        m = ch[0] - 12 + (7 if i % 4 == 2 else 12 if i % 8 == 5 else 0)
        build = 0.5 + 0.5 * tt / end_start
        M.put(mus, M.pluck_string(m, step * 1.6), tt, pan=0.1, gain=0.2 * build * (1.0 if i % 4 == 0 else 0.7))
    for bi in range(int(end_start / M.BAR) + 1):
        for j, pp in enumerate([0, 1.5, 2, 3, 3.5]):
            tt = bi * M.BAR + pp * M.BEAT
            if tt < end_start - 0.2 and (j == 0 or rng.random() < 0.8):
                M.put(mus, M.taiko(big=j == 0), tt, pan=rng.uniform(-.3, .3), gain=0.34)
    # hits, whooshes, riser into the end card, thunder on the storm opener
    for k, sh in enumerate(shots):
        if sh.get("hit"):
            M.put(sfx, M.braam(4.0) * (1.3 if sh["mode"] == "end" else 0.9), sh["start"], gain=0.45)
        if k:
            w = M.whoosh(0.9)
            M.put(sfx, np.stack([w, np.roll(w, 300)], 1), sh["start"] - 0.45, gain=0.4)
    r = M.riser(2.2)
    M.put(sfx, np.stack([r, r], 1), end_start - 2.2, gain=0.35)
    th = M.thunder(5)
    M.put(sfx, np.stack([th, np.roll(th, 400)], 1), 0.4, gain=0.6)
    oc = M.ocean(int(shots[1]["start"] * SR), storm=True)
    M.put(sfx, oc * 0.25, 0)
    music = mus + sfx
    music = music / (np.sqrt((music ** 2).mean()) + 1e-9) * 0.14
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    hop = 441
    act = (np.sqrt(np.convolve(vo ** 2, np.ones(hop) / hop, "same")[::hop]) > 0.02).astype(np.float32)
    env, g = np.zeros_like(act), 0.0
    for i, v in enumerate(act):
        g = g + (v - g) * (0.35 if v > g else 0.04)
        env[i] = g
    env = np.interp(np.arange(N) / hop, np.arange(len(env)), env)
    fade = np.ones(N)
    fade[-int(1.0 * SR):] = np.linspace(1, 0, int(1.0 * SR))
    out = (music * (1 - 0.45 * env)[:, None] + vo[:, None] * 0.95) * fade[:, None]
    out = np.tanh(out * 1.2) / np.tanh(1.2)
    out = out / np.abs(out).max() * 0.95
    sf.write(f"{BUILD}/trailer/mix.wav", out[:int(total * SR)].astype(np.float32), SR, subtype="PCM_16")


def main():
    shots, lines, total = plan()
    print(f"trailer length {total:.1f}s")
    assert total < 60, "Shorts must stay under 60 s"
    score(shots, total, lines)
    with Pool(os.cpu_count()) as pool:
        for i in pool.imap_unordered(render_shot, list(enumerate(shots))):
            print("shot", i, flush=True)
    with open(f"{BUILD}/trailer/concat.txt", "w") as f:
        for i in range(len(shots)):
            f.write(f"file 'shot_{i:02d}.mp4'\n")
    os.makedirs(OUT, exist_ok=True)
    name = f"{OUT}/E1_trailer_vertical"
    subprocess.check_call([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{BUILD}/trailer/concat.txt",
                           "-i", f"{BUILD}/trailer/mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                           "-shortest", "-movflags", "+faststart", f"{name}.mp4"])
    # small copy that fits chat / messaging upload limits
    subprocess.check_call([FFMPEG, "-y", "-loglevel", "error", "-i", f"{name}.mp4", "-c:v", "libx264", "-preset", "medium",
                           "-b:v", "3200k", "-maxrate", "3800k", "-bufsize", "6M", "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart", f"{name}_small.mp4"])
    for f_ in (f"{name}.mp4", f"{name}_small.mp4"):
        print(f_, round(os.path.getsize(f_) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
