# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The ground heights the OpenStreetMap example generators share: the GSI
elevation tiles of the Geospatial Information Authority of Japan, sampled
bilinearly and kept on disk, the height a structure tag adds, and the cubic
Hermite segment of an ``<elevation>`` record.

Elevation tiles (c) Geospatial Information Authority of Japan, under the
GSI terms of use (compatible with CC BY 4.0).
"""

import math
import os
import pathlib
import urllib.error
import urllib.request

import numpy as np

from _osmroad import tagged

TILE_URL = 'https://cyberjapandata.gsi.go.jp/xyz/{src}/{z}/{x}/{y}.txt'
SOURCES = ('dem5a', 'dem5b', 'dem10b')
ZOOM = 15
CACHE_DIR = pathlib.Path(os.environ.get(
    'GSI_TILE_CACHE', pathlib.Path.home() / '.cache' / 'solvcon-gsi'))
VIADUCT = 7.0
EMBANKMENT = 4.0
CUTTING = -4.0
TUNNEL = -10.0
ATTRIBUTION = ('Elevation (c) Geospatial Information Authority of Japan, '
               'elevation tiles DEM5A, DEM5B, DEM10B')


class Tiles:
    """GSI elevation tiles, fetched once and sampled bilinearly. A tile is
    saved under ``cache_dir`` and read from there on a rerun; an empty file
    stands for a tile the server does not have."""

    def __init__(self, cache_dir: pathlib.Path | None = CACHE_DIR) -> None:
        self.cache: dict[tuple, np.ndarray | None] = {}
        self.cache_dir = cache_dir
        self.fetched = 0

    def _text(self, src: str, x: int, y: int) -> str:
        path = None
        if self.cache_dir is not None:
            path = pathlib.Path(self.cache_dir) / src / str(ZOOM) / str(x) \
                / f'{y}.txt'
            if path.exists():
                return path.read_text('ascii')
        url = TILE_URL.format(src=src, z=ZOOM, x=x, y=y)
        req = urllib.request.Request(url, headers={'User-Agent': 'solvcon'})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode('ascii')
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            text = ''
        self.fetched += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, 'ascii')
        return text

    def tile(self, src: str, x: int, y: int) -> np.ndarray | None:
        key = (src, x, y)
        if key not in self.cache:
            text = self._text(src, x, y)
            if not text.strip():
                self.cache[key] = None
            else:
                rows = [[math.nan if v == 'e' else float(v)
                         for v in line.split(',')]
                        for line in text.strip().splitlines()]
                self.cache[key] = np.array(rows, dtype='float64')
        return self.cache[key]

    def height(self, lat: float, lon: float) -> tuple[float, str]:
        n = 2 ** ZOOM
        px = (lon + 180.0) / 360.0 * n * 256.0
        phi = math.radians(lat)
        py = ((1.0 - math.log(math.tan(phi) + 1.0 / math.cos(phi)) / math.pi)
              / 2.0 * n * 256.0)
        for src in SOURCES:
            value = self._bilinear(src, px, py)
            if not math.isnan(value):
                return value, src
        return math.nan, ''

    def _bilinear(self, src: str, px: float, py: float) -> float:
        corners = []
        for fy in (math.floor(py - 0.5), math.floor(py - 0.5) + 1):
            for fx in (math.floor(px - 0.5), math.floor(px - 0.5) + 1):
                tile = self.tile(src, fx // 256, fy // 256)
                corners.append(math.nan if tile is None
                               else tile[fy % 256, fx % 256])
        if any(math.isnan(v) for v in corners):
            return math.nan
        u, v = px - 0.5 - math.floor(px - 0.5), py - 0.5 - math.floor(py - 0.5)
        top = corners[0] * (1.0 - u) + corners[1] * u
        bottom = corners[2] * (1.0 - u) + corners[3] * u
        return top * (1.0 - v) + bottom * v


def structure_offset(tags: dict) -> float:
    """The height of the structure an OSM way is tagged with above the
    ground, scaled by ``layer``."""
    try:
        layer = int(tags.get('layer', '0') or 0)
    except ValueError:
        layer = 0
    if tagged(tags, 'tunnel'):
        return TUNNEL * max(-layer, 1)
    if tagged(tags, 'bridge'):
        return VIADUCT * max(layer, 1)
    if tagged(tags, 'embankment'):
        return EMBANKMENT
    if tagged(tags, 'cutting'):
        return CUTTING
    return 0.0


def height_segment(s0: float, s1: float, p0: float, p1: float, m0: float,
                   m1: float) -> tuple[float, float, float, float, float]:
    """The ``(s, a, b, c, d)`` cubic from height ``p0`` and slope ``m0`` at
    ``s0`` to ``p1`` and ``m1`` at ``s1``."""
    length = s1 - s0
    c = 3.0 * (p1 - p0) / length ** 2 - (2.0 * m0 + m1) / length
    d = 2.0 * (p0 - p1) / length ** 3 + (m0 + m1) / length ** 2
    return s0, p0, m0, c, d

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
