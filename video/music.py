"""Procedural cinematic score + ambience, mixed under the narration.

Output: BUILD/mix.wav (stereo 44.1 kHz)
"""
import json
import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, oaconvolve, sosfilt

from script import SCENES

BUILD = os.environ.get("BUILD", "build")
SR = 44100
BPM = 72
BEAT = 60 / BPM
BAR = BEAT * 4
CHORD = BAR * 2

CH = {
    "Dm": [50, 53, 57, 62], "Bb": [46, 53, 58, 62], "Gm": [43, 50, 55, 58], "A": [45, 52, 57, 61],
    "F": [41, 48, 53, 57], "C": [48, 52, 55, 60], "Eb": [51, 55, 58, 63], "Cdim": [49, 52, 55, 58],
    "D": [50, 54, 57, 62],
}
PROG = {
    "mystery": ["Dm", "Bb", "Gm", "A"], "tension": ["Dm", "Eb", "Dm", "Cdim"], "epic": ["Dm", "Bb", "F", "C"],
    "wonder": ["F", "C", "Dm", "Bb"], "dark": ["Dm", "Eb", "Gm", "A"], "tragic": ["Gm", "Dm", "Bb", "A"],
    "hopeful": ["Bb", "F", "C", "Dm"], "finale": ["Bb", "C", "Dm", "Dm"], "end": ["Bb", "F", "C", "D"],
    "hit": ["Dm", "Dm", "Dm", "Dm"],
}
# layer gains per mood: drone, pad, ostinato, taiko, bells, piano, heart, tick
MIX = {
    "mystery": dict(drone=.8, pad=.55, bells=.3),
    "tension": dict(drone=.7, pad=.45, ost=.55, ost_div=4, taiko=.35, tick=.18),
    "epic": dict(drone=.6, pad=.65, ost=.6, ost_div=2, taiko=1.0, bright=1),
    "wonder": dict(drone=.35, pad=.6, bells=.55, bright=1),
    "dark": dict(drone=1.0, pad=.4, heart=.8),
    "tragic": dict(drone=.45, pad=.55, piano=.6),
    "hopeful": dict(drone=.35, pad=.6, bells=.45, ost=.25, ost_div=2, bright=1),
    "finale": dict(drone=.6, pad=.7, taiko=.7, bells=.4, bright=1),
    "end": dict(drone=.4, pad=.6, bells=.5, bright=1),
    "hit": dict(drone=.6, pad=.25),
}

TAB_N = 4096


def _saw_table(nh=40):
    ph = np.arange(TAB_N) / TAB_N * 2 * np.pi
    return sum(np.sin(ph * k) / k for k in range(1, nh + 1)).astype(np.float32) * 0.6


SAW = _saw_table()
SAW_DARK = _saw_table(10)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def osc(f, n, table=SAW, phase=0.0, vib=0.0, vib_rate=5.0, t0=0.0):
    t = np.arange(n) / SR + t0
    fr = f * (1 + vib * np.sin(2 * np.pi * vib_rate * t))
    ph = phase + np.cumsum(fr) / SR
    return table[((ph % 1) * TAB_N).astype(np.int32)]


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x, axis=0)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x, axis=0)


def sweep(x, fc_fn, kind="low", blk=1024, width=0.5):
    """Time-varying filter that carries filter state across blocks (no clicks)."""
    out = np.zeros_like(x)
    zi = None
    n = len(x)
    for i in range(0, n, blk):
        fc = fc_fn(i / n, i / SR)
        if kind == "low":
            sos = butter(2, min(fc, SR * 0.45), "low", fs=SR, output="sos")
        else:
            sos = butter(2, [fc * width, min(fc, SR * 0.45)], "band", fs=SR, output="sos")
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = np.zeros((sos.shape[0], 2) + x.shape[1:])
        out[i:i + blk], zi = sosfilt(sos, x[i:i + blk], axis=0, zi=zi)
    return out


def adsr(n, a, r, sustain=1.0):
    e = np.full(n, sustain, np.float32)
    na, nr = int(a * SR), int(r * SR)
    if na:
        e[:min(na, n)] = np.linspace(0, sustain, na)[:n]
    if nr and nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def perc(n, decay):
    return np.exp(-np.arange(n) / SR / decay).astype(np.float32)


def put(buf, x, t, pan=0.0, gain=1.0):
    a = int(t * SR)
    if a < 0:
        x = x[-a:]
        a = 0
    b = min(len(buf), a + len(x))
    if b <= a:
        return
    x = x[:b - a] * gain
    if x.ndim == 1:
        l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        buf[a:b, 0] += x * l * 1.41
        buf[a:b, 1] += x * r * 1.41
    else:
        buf[a:b] += x


# ---------------------------------------------------------------- instruments
def pad_voice(m, dur, bright):
    n = int(dur * SR)
    x = osc(hz(m), n, SAW, np.random.rand(), vib=0.002, vib_rate=4.3) + osc(hz(m) * 1.004, n, SAW, np.random.rand(), vib=0.002, vib_rate=3.7)
    return x * adsr(n, 1.8, 2.2)


def drone(n, t0, root=38):
    x = osc(hz(root), n, SAW_DARK, t0=t0) * 0.6 + osc(hz(root + 7), n, SAW_DARK, t0=t0) * 0.3
    x += np.sin(2 * np.pi * hz(root - 12) * (np.arange(n) / SR + t0)) * 0.8
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * (np.arange(n) / SR + t0))
    return lp(x * lfo, 220)


def pluck_string(m, dur):
    n = int(dur * SR)
    x = osc(hz(m), n, SAW) * perc(n, dur * 0.5) * adsr(n, 0.005, 0.03)
    return lp(x, 1400)


def taiko(big=False):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    f = 45 + 80 * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (3.5 if big else 5))
    click = lp(np.random.randn(n) * np.exp(-t * 60), 2500) * 0.5
    return (body + click) * (1.3 if big else 0.8)


def bell(m, dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 3) + 0.2 * np.sin(2 * np.pi * f * 3.02 * t) * np.exp(-t * 6))
    return x * np.exp(-t * 1.4) * adsr(n, 0.004, 0.2)


def piano(m, dur=4.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    x = sum(np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k * k)) * (0.6 ** k) * np.exp(-t * (0.9 + 0.8 * k)) for k in range(1, 6))
    return x * adsr(n, 0.003, 0.3)


def heart():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    beat = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 14)
    out = np.zeros(int(0.9 * SR))
    out[:n] += beat
    out[int(0.28 * SR):int(0.28 * SR) + n] += beat * 0.7
    return lp(out, 150) * 2.2


def braam(dur=4.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(osc(hz(m), n, SAW, np.random.rand()) * g for m, g in [(26, .8), (38, 1), (45, .7), (50, .5), (53, .35)])
    x = x + sum(osc(hz(m) * 1.006, n, SAW, np.random.rand()) * g for m, g in [(38, .6), (45, .5)])
    env = np.minimum(1, t / 0.05) * np.exp(-t * 0.7)
    # filter sweep open then close, in blocks
    out = sweep(x, lambda k, tt: 180 + 2200 * math.exp(-((tt - 0.35) ** 2) / 0.35))
    boom = np.sin(2 * np.pi * np.cumsum(38 + 30 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.2) * 1.4
    return (out * env * 1.2 + boom)


def riser(dur=2.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.random.randn(n)
    out = sweep(x, lambda k, tt: 300 + 6000 * k ** 2, kind="band")
    return out * (t / dur) ** 2.5 * 0.9


def whoosh(dur=1.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.random.randn(n)
    out = sweep(x, lambda k, tt: 400 + 2500 * math.sin(math.pi * k), kind="band", width=0.6)
    return out * np.sin(np.pi * t / dur) ** 2 * 0.5


# ---------------------------------------------------------------- ambience
def ocean(n, storm=False, seed=0):
    r = np.random.default_rng(seed)
    x = np.cumsum(r.standard_normal((n, 2)), axis=0)
    x = hp(x, 30)
    x = x / (np.abs(x).max() + 1e-9)
    t = np.arange(n) / SR
    sw = 0.55 + 0.45 * np.sin(2 * np.pi * t / (7 if not storm else 4.5))[:, None] ** 2
    y = lp(x, 900 if not storm else 1800) * sw * (2.5 if storm else 1.6)
    if storm:
        y += hp(r.standard_normal((n, 2)), 2500) * 0.05  # rain hiss
    return y


def wind(n, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, 2))
    t = np.arange(n) / SR
    out = sweep(x, lambda k, tt: 500 + 300 * math.sin(2 * math.pi * tt / 6.0), kind="band", blk=4096, width=0.7)
    return out * (0.5 + 0.5 * np.sin(2 * np.pi * t / 9)[:, None] ** 2) * 0.35


def fire(n, seed=0):
    r = np.random.default_rng(seed)
    x = lp(r.standard_normal((n, 2)), 600) * 0.25
    k = r.random(n) < 30 / SR
    cr = np.zeros(n)
    cr[k] = r.uniform(0.3, 1, k.sum()) * r.choice([-1, 1], k.sum())
    cr = hp(np.convolve(cr, np.exp(-np.arange(200) / 25), "same"), 1500)
    return x + np.stack([cr, np.roll(cr, 40)], 1) * 0.9


def thunder(seed):
    r = np.random.default_rng(seed)
    n = int(3 * SR)
    t = np.arange(n) / SR
    x = lp(np.cumsum(r.standard_normal(n)), 300)
    x = x / np.abs(x).max()
    return x * np.exp(-t * 1.2) * np.minimum(1, t / 0.03) * 1.2


AMB = {"sea": "sea", "raid": "fire", "city": "fire", "fires": "fire", "fjord": "wind", "greenland": "wind",
       "iceland": "wind", "night": "wind", "sod": "wind", "ruins": "wind", "beach": "sea", "landing": "sea",
       "wreck": "sea", "axe": "wind", "chains": "fire", "next": "sea", "lighthouse": "wind"}


def main():
    tl = json.load(open(f"{BUILD}/timeline.json"))
    total = tl["total"]
    N = int((total + 2) * SR)
    mus = np.zeros((N, 2), np.float32)
    amb = np.zeros((N, 2), np.float32)
    sfx = np.zeros((N, 2), np.float32)
    rng = np.random.default_rng(1)
    spec = {s["id"]: s for s in SCENES}
    X = 1.0  # crossfade overlap each side

    for k, sc in enumerate(tl["scenes"]):
        sp = spec[sc["id"]]
        mood = sp["mood"]
        mix = MIX[mood]
        s0, s1 = sc["start"], sc["start"] + sc["dur"]
        a, b = max(0, s0 - X), s1 + X
        n = int((b - a) * SR)
        seg = np.zeros((n, 2), np.float32)
        fade = np.ones(n, np.float32)
        nx = int(2 * X * SR)
        fade[:nx] = np.linspace(0, 1, nx) if s0 > 0 else 1
        fade[-nx:] = np.linspace(1, 0, nx)
        prog = PROG[mood]
        # drone
        if mix.get("drone"):
            d = drone(n, a) * mix["drone"] * 0.35
            seg += np.stack([d, d], 1)
        # pad chords on the global grid
        if mix.get("pad"):
            padbuf = np.zeros((n, 2), np.float32)
            c0 = int(a // CHORD)
            for ci in range(c0, int(b // CHORD) + 1):
                tc = ci * CHORD
                notes = CH[prog[ci % 4]]
                for j, m in enumerate(notes):
                    v = pad_voice(m + (12 if mix.get("bright") and j == 3 else 0), CHORD + 2.2, mix.get("bright", 0))
                    put(padbuf, v, tc - a, pan=(j - 1.5) * 0.4, gain=0.09)
            padbuf = lp(padbuf, 2200 if mix.get("bright") else 1100)
            seg += padbuf * mix["pad"]
        # ostinato
        if mix.get("ost"):
            step = BEAT / mix["ost_div"]
            i0 = int(math.ceil(a / step))
            for i in range(i0, int(b / step)):
                tt = i * step
                ch = CH[prog[int(tt // CHORD) % 4]]
                m = ch[0] - 12 + (7 if i % 4 == 2 else 12 if i % 8 == 5 else 0)
                acc = 1.0 if i % 4 == 0 else 0.7
                put(seg, pluck_string(m, step * 1.6), tt - a, pan=0.1, gain=0.22 * mix["ost"] * acc)
        # taiko
        if mix.get("taiko"):
            patt = [0, 1.5, 2, 3, 3.5]
            for bi in range(int(a // BAR), int(b // BAR) + 1):
                for j, p in enumerate(patt):
                    tt = bi * BAR + p * BEAT
                    if a <= tt < b and (j == 0 or rng.random() < 0.75):
                        put(seg, taiko(big=j == 0 and bi % 2 == 0), tt - a, pan=rng.uniform(-.3, .3), gain=0.33 * mix["taiko"])
        # tick (clock)
        if mix.get("tick"):
            for i in range(int(math.ceil(a / (BEAT / 2))), int(b / (BEAT / 2))):
                tt = i * BEAT / 2
                clk = hp(np.random.randn(int(0.03 * SR)) * perc(int(0.03 * SR), 0.005), 3000)
                put(seg, clk, tt - a, pan=0.4, gain=mix["tick"] * (1 if i % 2 == 0 else 0.6))
        # bells: arpeggio of chord tones an octave up
        if mix.get("bells"):
            step = BEAT
            for i in range(int(math.ceil(a / step)), int(b / step)):
                if rng.random() < 0.55:
                    tt = i * step
                    ch = CH[prog[int(tt // CHORD) % 4]]
                    m = ch[rng.integers(0, 4)] + 24
                    put(seg, bell(m), tt - a, pan=rng.uniform(-.6, .6), gain=0.07 * mix["bells"])
        # piano: slow melody
        if mix.get("piano"):
            step = BEAT * 2
            for i in range(int(math.ceil(a / step)), int(b / step)):
                tt = i * step
                ch = CH[prog[int(tt // CHORD) % 4]]
                m = ch[(i * 3) % 4] + 12
                put(seg, piano(m), tt - a, pan=-0.2, gain=0.12 * mix["piano"])
                if i % 2 == 0:
                    put(seg, piano(ch[0] - 12), tt - a, pan=0.1, gain=0.1 * mix["piano"])
        # heartbeat
        if mix.get("heart"):
            for i in range(int(math.ceil(a / 1.0)), int(b / 1.0)):
                put(seg, heart(), i * 1.0 - a, gain=0.35 * mix["heart"])
        seg *= fade[:, None]
        put(mus, seg, a)

        # chapter hits
        if mood == "hit":
            put(sfx, braam(4.5), s0, gain=0.55)
            r = riser(2.5)
            put(sfx, np.stack([r, r[::-1] * 0 + r], 1), s0 - 2.5, gain=0.35)
        elif mood in ("finale",):
            put(sfx, braam(5.0), s0 + (sc["chunks"][1]["start"] if len(sc["chunks"]) > 1 else 0), gain=0.35)
        elif k > 0:
            w = whoosh()
            put(sfx, np.stack([w, np.roll(w, 300)], 1), s0 - 0.6, gain=0.35)

        # ambience
        kind = sp["kind"]
        amb_kind = AMB.get(kind)
        if amb_kind:
            m = int(sc["dur"] * SR)
            if amb_kind == "sea":
                x = ocean(m, storm=sp.get("weather") == "storm", seed=k)
            elif amb_kind == "wind":
                x = wind(m, seed=k)
            else:
                x = fire(m, seed=k) + wind(m, seed=k) * 0.3
            e = adsr(m, 0.8, 0.8)[:, None]
            put(amb, (x * e).astype(np.float32), s0, gain=0.22)
        if sp.get("lightning"):
            r3 = np.random.default_rng(3)
            tt = 1.5
            while tt < sc["dur"]:
                th = thunder(int(tt * 10))
                put(sfx, np.stack([th, np.roll(th, 500)], 1), s0 + tt + 0.15, gain=0.6)
                tt += r3.uniform(2.5, 5.5)
        if kind == "chains":
            c = sc["chunks"][2]["start"] if len(sc["chunks"]) > 2 else 0
            n2 = int(1.5 * SR)
            tt_ = np.arange(n2) / SR
            clang = sum(np.sin(2 * np.pi * f * tt_) * np.exp(-tt_ * d) for f, d in [(523, 3), (1187, 5), (1760, 7), (2493, 9)])
            put(sfx, clang * 0.25, s0 + c, gain=0.8)
            put(sfx, clang * 0.2, s0 + c + 0.18, pan=0.3, gain=0.6)

    # reverb on the music bus
    ir_n = int(3.2 * SR)
    t = np.arange(ir_n) / SR
    r = np.random.default_rng(5)
    ir = r.standard_normal((ir_n, 2)) * np.exp(-t * 2.1)[:, None]
    ir = lp(ir, 5000) * 0.03
    ir[0] += 1.0
    wet = np.stack([oaconvolve(mus[:, c] + sfx[:, c] * 0.4, ir[:, c])[:N] for c in range(2)], 1)
    music = wet + sfx * 0.6

    vo, vsr = sf.read(f"{BUILD}/vo.wav", dtype="float32")
    vo = vo[:N]
    vo = np.pad(vo, (0, N - len(vo)))
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    # ducking envelope from narration activity
    hop = 441
    frames = np.sqrt(np.convolve(vo ** 2, np.ones(hop) / hop, "same")[::hop])
    act = (frames > 0.02).astype(np.float32)
    env = np.zeros_like(act)
    g = 0.0
    for i, v in enumerate(act):
        g = g + (v - g) * (0.35 if v > g else 0.03)
        env[i] = g
    env = np.interp(np.arange(N) / hop, np.arange(len(env)), env)
    duck = (1 - 0.55 * env)[:, None]

    music = music / (np.sqrt((music ** 2).mean()) + 1e-9) * 0.12
    amb = amb / (np.sqrt((amb ** 2).mean()) + 1e-9) * 0.035
    out = music * duck + amb * (1 - 0.3 * env)[:, None] + vo[:, None] * 0.95
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    out = out / np.abs(out).max() * 0.95
    sf.write(f"{BUILD}/mix.wav", out.astype(np.float32), SR, subtype="PCM_16")
    print("mix written", out.shape[0] / SR / 60, "min")


if __name__ == "__main__":
    main()
