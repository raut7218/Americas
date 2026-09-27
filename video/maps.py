"""Coastline data (Natural Earth via world-atlas) -> map images, projections and a globe texture."""
import json
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def _topo():
    return json.load(open(f"{HERE}/assets/countries-50m.json"))


@lru_cache(maxsize=1)
def _arcs():
    t = _topo()
    sx, sy = t["transform"]["scale"]
    tx, ty = t["transform"]["translate"]
    out = []
    for arc in t["arcs"]:
        a = np.cumsum(np.array(arc, dtype=np.float64), axis=0)
        a[:, 0] = a[:, 0] * sx + tx
        a[:, 1] = a[:, 1] * sy + ty
        out.append(a)
    return out


def _ring(idx):
    arcs = _arcs()
    pts = []
    for i in idx:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts.append(a if not pts else a[1:])
    return np.concatenate(pts)


def _polys(geom):
    if geom["type"] == "Polygon":
        return [[_ring(r) for r in geom["arcs"]]]
    if geom["type"] == "MultiPolygon":
        return [[_ring(r) for r in p] for p in geom["arcs"]]
    return []


@lru_cache(maxsize=1)
def land_polys():
    t = _topo()
    out = []
    for g in t["objects"]["land"]["geometries"]:
        out += _polys(g)
    return out


@lru_cache(maxsize=4)
def country_polys(ids):
    t = _topo()
    out = []
    for g in t["objects"]["countries"]["geometries"]:
        if g.get("id") in ids:
            out += [p for p in _polys(g) if np.ptp(p[0][:, 0]) < 180]
    return out


def merc_y(lat):
    lat = np.clip(lat, -85, 85)
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


class Proj:
    """Mercator projection fitted so lon range spans `width` px, centered on the view."""

    def __init__(self, view, width, height):
        lon0, lon1, lat0, lat1 = view
        self.k = width / math.radians(lon1 - lon0)
        self.cx = (lon0 + lon1) / 2
        self.cy = float(merc_y((lat0 + lat1) / 2))
        self.w, self.h = width, height

    def __call__(self, lon, lat):
        x = self.w / 2 + np.radians(np.asarray(lon) - self.cx) * self.k
        y = self.h / 2 - (merc_y(np.asarray(lat)) - self.cy) * self.k
        return x, y


def render_map(view, size, ocean=(8, 16, 26), land=(34, 44, 46), coast=(120, 160, 150), grid=(22, 36, 50),
               highlight=None, ss=2, coast_w=2, glow=(40, 90, 90)):
    """Return (RGB image, Proj). highlight: list of (country_id_tuple, rgb)."""
    w, h = size
    W2, H2 = w * ss, h * ss
    proj = Proj(view, W2, H2)
    img = Image.new("RGB", (W2, H2), ocean)
    d = ImageDraw.Draw(img)
    # graticule every 10 degrees
    for lon in range(-180, 181, 10):
        x, _ = proj(lon, 0)
        d.line([(float(x), 0), (float(x), H2)], fill=grid, width=ss)
    for lat in range(-80, 81, 10):
        _, y = proj(0, lat)
        d.line([(0, float(y)), (W2, float(y))], fill=grid, width=ss)
    lon0, lon1, lat0, lat1 = view
    pad = 40

    def visible(r):
        return r[:, 0].max() > lon0 - pad and r[:, 0].min() < lon1 + pad and r[:, 1].max() > lat0 - pad and r[:, 1].min() < lat1 + pad

    polys = [p for p in land_polys() if visible(p[0]) and np.ptp(p[0][:, 0]) < 180]
    mask = Image.new("L", (W2, H2), 0)
    md = ImageDraw.Draw(mask)
    for p in polys:
        x, y = proj(p[0][:, 0], p[0][:, 1])
        md.polygon(list(zip(x.tolist(), y.tolist())), fill=255)
        for hole in p[1:]:
            x, y = proj(hole[:, 0], hole[:, 1])
            md.polygon(list(zip(x.tolist(), y.tolist())), fill=0)
    # coastal glow into the ocean
    if glow:
        g = mask.filter(ImageFilter.GaussianBlur(14 * ss))
        glow_img = Image.new("RGB", (W2, H2), glow)
        img = Image.composite(glow_img, img, g.point(lambda v: int(v * 0.55)))
    # land with subtle relief noise
    rs = np.random.default_rng(7)
    nz = rs.random((H2 // 64 + 2, W2 // 64 + 2)).astype(np.float32)
    nz = np.asarray(Image.fromarray((nz * 255).astype(np.uint8)).resize((W2, H2), Image.BICUBIC), np.float32) / 255
    land_arr = np.empty((H2, W2, 3), np.float32)
    for c in range(3):
        land_arr[..., c] = land[c] * (0.85 + 0.3 * nz)
    land_img = Image.fromarray(land_arr.clip(0, 255).astype(np.uint8))
    img = Image.composite(land_img, img, mask)
    d = ImageDraw.Draw(img)
    if highlight:
        for ids, col in highlight:
            for p in country_polys(tuple(ids)):
                x, y = proj(p[0][:, 0], p[0][:, 1])
                d.polygon(list(zip(x.tolist(), y.tolist())), fill=col)
    for p in polys:
        for r in p:
            x, y = proj(r[:, 0], r[:, 1])
            d.line(list(zip(x.tolist(), y.tolist())), fill=coast, width=coast_w * ss, joint="curve")
    img = img.resize((w, h), Image.LANCZOS)

    class P2:
        def __call__(self, lon, lat):
            x, y = proj(lon, lat)
            return x / ss, y / ss

    return img, P2()


def country_mask(view, size, ids, ss=1):
    w, h = size
    proj = Proj(view, w * ss, h * ss)
    m = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(m)
    for p in country_polys(tuple(ids)):
        x, y = proj(p[0][:, 0], p[0][:, 1])
        d.polygon(list(zip(x.tolist(), y.tolist())), fill=255)
    return m.resize((w, h), Image.LANCZOS) if ss > 1 else m


@lru_cache(maxsize=1)
def equirect_land(w=2048, h=1024):
    """Land mask in equirectangular projection, float32 0..1."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for p in land_polys():
        for k, r in enumerate(p):
            if r[:, 0].max() - r[:, 0].min() > 180:
                continue
            x = (r[:, 0] + 180) / 360 * w
            y = (90 - r[:, 1]) / 180 * h
            d.polygon(list(zip(x.tolist(), y.tolist())), fill=255 if k == 0 else 0)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(1)), np.float32) / 255


def globe(radius, lon0, lat0, pear=0.0, ocean=(20, 50, 90), land=(70, 110, 70), light=(-0.5, -0.5, 0.7), night=0.15):
    """Orthographic globe as RGBA (2r x 2r*1.25 when pear>0). Pixel-sampled from the land texture."""
    tex = equirect_land()
    th, tw = tex.shape
    R = radius
    hh = 2 * R
    ww = 2 * R
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    top = 0
    u = (xs - R) / R
    v = (ys - top - R) / R  # -1 top .. 1 bottom of sphere part
    # pear: narrower on top, with a nipple rising above the sphere
    widen = 1 + pear * 0.3 * v
    vv = v
    us = u / np.maximum(widen, 0.2)
    inside = us ** 2 + vv ** 2 <= 1
    z = np.sqrt(np.clip(1 - us ** 2 - vv ** 2, 0, 1))
    # rotate: view direction -> lat/lon
    la0, lo0 = math.radians(lat0), math.radians(lon0)
    x3, y3, z3 = us, -vv, z
    # rotate around x axis by lat0
    y4 = y3 * math.cos(la0) + z3 * math.sin(la0)
    z4 = -y3 * math.sin(la0) + z3 * math.cos(la0)
    lat = np.arcsin(np.clip(y4, -1, 1))
    lon = np.arctan2(x3, z4) + lo0
    tx = ((np.degrees(lon) + 180) % 360 / 360 * (tw - 1)).astype(np.int32)
    ty = ((90 - np.degrees(lat)) / 180 * (th - 1)).astype(np.int32)
    L = tex[ty, tx][..., None]
    col = np.array(ocean, np.float32) * (1 - L) + np.array(land, np.float32) * L
    lx, ly, lz = light
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    shade = np.clip((us * lx + vv * ly + z * lz) / n, 0, 1)
    shade = night + (1 - night) * shade
    rim = np.clip(1 - z, 0, 1) ** 3
    col = col * shade[..., None] + np.array([90, 150, 230], np.float32) * rim[..., None] * 0.6
    alpha = inside.astype(np.float32)
    # soft edge
    edge = np.clip((1 - np.sqrt(us ** 2 + vv ** 2)) * R, 0, 1)
    alpha = alpha * edge
    out = np.dstack([col.clip(0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")
