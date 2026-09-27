# Understanding America — Episode 1: From Vikings to Columbus

A ~24-minute animated documentary built from the episode transcript: narrated, scored and
animated entirely in code (no stock footage or stock music).

## What's in it

- **Narration**: Kokoro neural TTS (offline), British documentary voice, with timed captions.
- **Picture**: 80 animated scenes, 1920×1080 @ 24 fps, with a cinemascope letterbox and burned-in captions.
  The film has 14 chapters: Norsemen, Erik the Red, Vinland, First Contact, Freydis, the Forgotten World,
  Columbus, 1492, Rivers of Gold, the Admiral in Chains, the naming of America, the wandering bones,
  Las Casas, and why Spain succeeded where the Vikings failed.
  The maps use real Natural Earth coastlines (world-atlas), including a rotating 3D globe.
- **Score**: a procedural cinematic soundtrack, synthesized in `music.py`. It has drones, string pads,
  ostinatos, taiko drums, braams on chapter cards, risers, bells and piano, changing with each scene's mood.
  It also includes sound effects (ocean, storm and thunder, wind, fire, chains). The music ducks under
  the narration, and the final mix is normalized to −14 LUFS for YouTube.

## Build

```bash
pip install pillow numpy scipy soundfile kokoro-onnx imageio-ffmpeg
# Kokoro model files -> models/ (kokoro-v1.0.onnx, voices-v1.0.bin from
# https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
python3 tts.py        # narration + timeline      -> build/vo.wav, build/timeline.json
python3 music.py      # score + sfx + mix         -> build/mix.wav
python3 render.py     # all scenes (parallel)     -> build/video_only.mp4
python3 finalize.py   # mux, loudnorm, SRT, thumb -> output/
```

`python3 render.py --preview <scene_id,...|all>` writes still frames for quick review.

## Files

| file | role |
|---|---|
| `script.py` | the screenplay: narration, chapter cards, per-scene visual direction and cues |
| `engine.py` | drawing primitives: ships, silhouettes, particles, text, film finish (grain, vignette, letterbox, captions) |
| `maps.py` | coastline decoding, Mercator maps, parchment maps, orthographic globe |
| `scenes.py`, `scenes2.py` | one renderer per scene type |
| `music.py` | procedural score, ambience and final mix |
| `finalize.py` | final MP4, `.srt` subtitles, YouTube thumbnail |
