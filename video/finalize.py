"""Mux picture + soundtrack (loudness-normalized for YouTube), write SRT subtitles and a thumbnail."""
import json
import os
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw

from engine import draw_text, font, radial, screen, glow_layer
import maps

BUILD = os.environ.get("BUILD", "build")
OUT = os.environ.get("OUT", "output")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
NAME = "Understanding_America_E1_From_Vikings_to_Columbus"


def srt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def write_srt(path):
    tl = json.load(open(f"{BUILD}/timeline.json"))
    n = 0
    with open(path, "w") as f:
        for sc in tl["scenes"]:
            for c in sc["chunks"]:
                n += 1
                a, b = sc["start"] + c["start"], sc["start"] + c["end"] + 0.2
                f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{c['text']}\n\n")


def thumbnail(path):
    W, H = 1280, 720
    img, _ = maps.render_map((-80, 20, 10, 72), (W, H), ocean=(8, 12, 20), land=(40, 34, 26), coast=(200, 150, 70),
                             grid=(20, 22, 28), glow=(90, 60, 20))
    img = screen(img, radial(W, H, W * 0.5, H * 0.45, 700, (120, 60, 20), (0, 0, 0), 0.7))
    from engine import longship, caravel
    ls = longship(0.55)
    img.paste(ls, (-60, H - ls.height + 40), ls)
    cv = caravel(0.55)
    img.paste(cv, (W - cv.width + 90, H - cv.height + 40), cv)
    d = ImageDraw.Draw(img)
    draw_text(img, (W / 2, 150), "UNDERSTANDING AMERICA · EP 1", font("title", 40), (240, 220, 170), spacing=6)
    draw_text(img, (W / 2, 290), "VIKINGS", font("deco", 140), (250, 240, 215), spacing=8, glow=(180, 90, 20))
    draw_text(img, (W / 2, 395), "to", font("italic", 70), (240, 210, 150))
    draw_text(img, (W / 2, 500), "COLUMBUS", font("deco", 140), (255, 215, 120), spacing=8, glow=(180, 90, 20))
    img.save(path, quality=92)


def main():
    os.makedirs(OUT, exist_ok=True)
    mp4 = f"{OUT}/{NAME}.mp4"
    subprocess.check_call([FFMPEG, "-y", "-loglevel", "error", "-i", f"{BUILD}/video_only.mp4", "-i", f"{BUILD}/mix.wav",
                           "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                           "-shortest", "-movflags", "+faststart", mp4])
    write_srt(f"{OUT}/{NAME}.en.srt")
    thumbnail(f"{OUT}/{NAME}_thumbnail.jpg")
    print("wrote", mp4, os.path.getsize(mp4) / 1e6, "MB")


if __name__ == "__main__":
    main()
