"""Synthesize the narration with Kokoro (offline neural TTS) and lay out the timeline.

Outputs (in BUILD dir):
  vo.wav         – full-length narration track, 44.1 kHz mono
  timeline.json  – per-scene start/duration and per-caption timings
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from script import SAY, SCENES

BUILD = os.environ.get("BUILD", "build")
MODELS = os.environ.get("MODELS", "models")
VOICE = os.environ.get("VOICE", "bm_george")
SPEED = float(os.environ.get("SPEED", "0.93"))
SR = 44100

LEAD, GAP, TAIL = 0.8, 0.42, 1.2
CHAPTER_HOLD = 3.9


def say(text):
    for k in sorted(SAY, key=len, reverse=True):
        text = re.sub(re.escape(k), SAY[k], text)
    return text


def main():
    from kokoro_onnx import Kokoro

    os.makedirs(f"{BUILD}/vo", exist_ok=True)
    kokoro = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
    lang = "en-gb" if VOICE.startswith("b") else "en-us"

    timeline, track, t = [], [], 0.0
    for sc in SCENES:
        chunks = []
        if sc["vo"]:
            cursor = LEAD
            for i, line in enumerate(sc["vo"]):
                cache = f"{BUILD}/vo/{sc['id']}_{i}.wav"
                if not os.path.exists(cache):
                    samples, sr = kokoro.create(say(line), voice=VOICE, speed=SPEED, lang=lang)
                    samples = resample_poly(samples, SR // 300, sr // 300).astype(np.float32)
                    # trim silence the model leaves at the edges
                    idx = np.where(np.abs(samples) > 0.01)[0]
                    if len(idx):
                        samples = samples[max(0, idx[0] - 800): idx[-1] + 2400]
                    sf.write(cache, samples, SR)
                samples, _ = sf.read(cache, dtype="float32")
                d = len(samples) / SR
                chunks.append(dict(text=line, start=cursor, end=cursor + d, file=cache))
                cursor += d + GAP
            dur = cursor - GAP + TAIL
        else:
            dur = sc.get("hold", CHAPTER_HOLD)
        dur = round(dur * 24) / 24  # whole frames
        timeline.append(dict(id=sc["id"], start=t, dur=dur, chunks=chunks))
        t += dur
        print(f"{sc['id']:16s} {dur:6.2f}s  total {t/60:5.2f} min", flush=True)

    vo = np.zeros(int(t * SR) + SR, dtype=np.float32)
    for sc in timeline:
        for c in sc["chunks"]:
            s, _ = sf.read(c["file"], dtype="float32")
            a = int((sc["start"] + c["start"]) * SR)
            vo[a:a + len(s)] += s
    sf.write(f"{BUILD}/vo.wav", vo, SR)
    json.dump(dict(total=t, scenes=timeline), open(f"{BUILD}/timeline.json", "w"), indent=1)
    print("total", t / 60, "minutes")


if __name__ == "__main__":
    sys.exit(main())
