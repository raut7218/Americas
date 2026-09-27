"""Render all scenes to video segments in parallel, then concatenate.

  python3 render.py                 # render every scene (skips finished segments)
  python3 render.py --preview a,b   # write stills at 25/50/80% of the named scenes
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import imageio_ffmpeg

import scenes  # noqa: F401  (registers kinds)
import scenes2  # noqa: F401
from engine import FPS, H, W
from scenes import REG
from script import SCENES

BUILD = os.environ.get("BUILD", "build")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load():
    tl = json.load(open(f"{BUILD}/timeline.json"))
    by = {s["id"]: s for s in tl["scenes"]}
    return tl, by


def make(spec, tl):
    return REG[spec["kind"]](spec, tl)


def render_scene(args):
    spec, tl = args
    out = f"{BUILD}/seg/{spec['id']}.mp4"
    if os.path.exists(out):
        return spec["id"], 0.0
    t0 = time.time()
    sc = make(spec, tl)
    n = int(round(tl["dur"] * FPS))
    tmp = out + ".part.mp4"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(sc.render(i / FPS).tobytes())
    p.stdin.close()
    p.wait()
    os.rename(tmp, out)
    return spec["id"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview")
    ap.add_argument("--only")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    a = ap.parse_args()
    tl, by = load()
    if a.preview:
        os.makedirs(f"{BUILD}/prev", exist_ok=True)
        ids = a.preview.split(",")
        for spec in SCENES:
            if spec["id"] in ids or a.preview == "all":
                sc = make(spec, by[spec["id"]])
                for f in (0.25, 0.5, 0.85):
                    t = by[spec["id"]]["dur"] * f
                    sc.render(t).save(f"{BUILD}/prev/{spec['id']}_{int(f*100)}.jpg", quality=85)
                print("preview", spec["id"], flush=True)
        return
    os.makedirs(f"{BUILD}/seg", exist_ok=True)
    jobs = [(s, by[s["id"]]) for s in SCENES if not a.only or s["id"] in a.only.split(",")]
    jobs.sort(key=lambda j: -j[1]["dur"])
    with Pool(a.j) as pool:
        for sid, dt in pool.imap_unordered(render_scene, jobs):
            print(f"done {sid} {dt:.0f}s", flush=True)
    with open(f"{BUILD}/concat.txt", "w") as f:
        for s in SCENES:
            f.write(f"file 'seg/{s['id']}.mp4'\n")
    subprocess.check_call([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{BUILD}/concat.txt",
                           "-c", "copy", f"{BUILD}/video_only.mp4"])
    print("video_only.mp4 written")


if __name__ == "__main__":
    sys.exit(main())
