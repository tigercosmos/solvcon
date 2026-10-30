# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Add an elevation profile to the Yamanote Line map from GSI elevation tiles.

The map that ``yamanote.py`` writes is flat: OpenStreetMap carries no
elevation. This script reads that map, samples the ground height under each
reference line from the elevation tiles of the Geospatial Information
Authority of Japan (GSI, DEM5A first, then DEM5B and DEM10B where the laser
survey has a hole), adds the height of the structure the OSM way is tagged
with (viaduct, embankment, cutting, tunnel, scaled by ``layer``), smooths
the sum along each loop and writes it as ``<elevationProfile>`` cubics.
Nothing else in the map changes. The OSM structure tags the generator did
not keep are added to the road ``<userData>`` so the offsets can be redone
without another Overpass query.

Each loop is one closed chain of roads, so the profile is smoothed and
differentiated around the whole loop before it is cut back into roads; the
height and the slope are then continuous across every road boundary.

Usage: python3 odrcore/examples/yamanote_elevation.py IN.xodr OUT.xodr
       [OVERPASS.json]

Without OVERPASS.json the OSM ways are fetched as ``yamanote.py`` does.

Elevation tiles (c) Geospatial Information Authority of Japan, under the
GSI terms of use (compatible with CC BY 4.0).
"""

import json
import math
import pathlib
import sys
import urllib.error
import urllib.request

import numpy as np

import odrcore
from odrcore.geom import PiecewisePoly3
from odrcore.model import Road
from odrcore.osm import LocalTangentPlane, origin_of

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _osmroad import fetch, loop_chain  # noqa: E402
from yamanote import QUERY, TRACKS  # noqa: E402

TILE_URL = 'https://cyberjapandata.gsi.go.jp/xyz/{src}/{z}/{x}/{y}.txt'
SOURCES = ('dem5a', 'dem5b', 'dem10b')
ZOOM = 15
SAMPLE = 5.0
KNOT = 20.0
TERRAIN_WINDOW = 100.0
STRUCTURE_WINDOW = 60.0
VIADUCT = 7.0
EMBANKMENT = 4.0
CUTTING = -4.0
TUNNEL = -10.0
ATTRIBUTION = ('Elevation (c) Geospatial Information Authority of Japan, '
               'elevation tiles DEM5A, DEM5B, DEM10B')


class Tiles:
    """GSI elevation tiles, fetched once and sampled bilinearly."""

    def __init__(self) -> None:
        self.cache: dict[tuple, np.ndarray | None] = {}
        self.fetched = 0

    def tile(self, src: str, x: int, y: int) -> np.ndarray | None:
        key = (src, x, y)
        if key not in self.cache:
            url = TILE_URL.format(src=src, z=ZOOM, x=x, y=y)
            req = urllib.request.Request(url,
                                         headers={'User-Agent': 'solvcon'})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    text = resp.read().decode('ascii')
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                self.cache[key] = None
            else:
                rows = [[math.nan if v == 'e' else float(v)
                         for v in line.split(',')]
                        for line in text.strip().splitlines()]
                self.cache[key] = np.array(rows, dtype='float64')
            self.fetched += 1
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
    layer = int(tags.get('layer', '0') or 0)
    if tags.get('tunnel'):
        return TUNNEL * max(-layer, 1)
    if tags.get('bridge'):
        return VIADUCT * max(layer, 1)
    if tags.get('embankment'):
        return EMBANKMENT
    if tags.get('cutting'):
        return CUTTING
    return 0.0


def circular_mean(values: np.ndarray, window: float) -> np.ndarray:
    half = int(round(window / SAMPLE / 2.0))
    if half < 1:
        return values
    kernel = np.ones(2 * half + 1, dtype='float64') / (2 * half + 1)
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, kernel, mode='valid')


def fill_gaps(values: np.ndarray) -> np.ndarray:
    bad = np.isnan(values)
    if bad.any():
        idx = np.arange(len(values), dtype='int64')
        values = values.copy()
        values[bad] = np.interp(idx[bad], idx[~bad], values[~bad],
                                period=len(values))
    return values


def hermite(s0: float, s1: float, p0: float, p1: float, m0: float,
            m1: float) -> tuple[float, float, float, float, float]:
    length = s1 - s0
    c = 3.0 * (p1 - p0) / length ** 2 - (2.0 * m0 + m1) / length
    d = 2.0 * (p0 - p1) / length ** 3 + (m0 + m1) / length ** 2
    return s0, p0, m0, c, d


def profile_loop(chain: list[Road], tags: dict[str, dict],
                 plane: LocalTangentPlane, tiles: Tiles) -> dict[str, int]:
    starts = np.cumsum([0.0] + [r.length for r in chain], dtype='float64')
    total = float(starts[-1])
    s_global = np.arange(0.0, total, SAMPLE, dtype='float64')
    which = np.searchsorted(starts, s_global, side='right') - 1

    terrain, offset, used = [], [], {}
    for sg, k in zip(s_global, which):
        road = chain[k]
        x, y, _ = road.refline.point(float(sg - starts[k]), 0.0)
        lat, lon, _ = plane.to_geodetic(x, y, 0.0)
        z, src = tiles.height(lat, lon)
        used[src] = used.get(src, 0) + 1
        terrain.append(z)
        offset.append(structure_offset(tags.get(road.id, {})))

    terrain = fill_gaps(np.array(terrain, dtype='float64'))
    height = (circular_mean(terrain, TERRAIN_WINDOW)
              + circular_mean(np.array(offset, dtype='float64'),
                              STRUCTURE_WINDOW))
    slope = np.gradient(np.concatenate([height[-1:], height, height[:1]]),
                        SAMPLE)[1:-1]

    def at(sg: float) -> tuple[float, float]:
        sg = sg % total
        return (float(np.interp(sg, s_global, height, period=total)),
                float(np.interp(sg, s_global, slope, period=total)))

    for k, road in enumerate(chain):
        knots = list(np.arange(0.0, road.length, KNOT))
        if road.length - knots[-1] < SAMPLE and len(knots) > 1:
            knots.pop()
        knots.append(road.length)
        segs = []
        for s0, s1 in zip(knots[:-1], knots[1:]):
            p0, m0 = at(starts[k] + s0)
            p1, m1 = at(starts[k] + s1)
            segs.append(hermite(s0, s1, p0, p1, m0, m1))
        road.refline.elevation = PiecewisePoly3(segs)
    return used


def annotate(road: Road, tags: dict) -> None:
    present = road.attrs.get('userData', '')
    extra = ''.join(f'<userData code="osm:{key}" value="{tags[key]}"/>'
                    for key in ('layer', 'embankment', 'cutting')
                    if key in tags and f'code="osm:{key}"' not in present)
    road.attrs['userData'] = present + extra


def main(argv: list[str]) -> int:
    src, out = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    if len(argv) > 3:
        data = json.loads(pathlib.Path(argv[3]).read_text('utf-8'))
    else:
        data = fetch(QUERY)
    tags = {str(e['id']): e.get('tags', {})
            for e in data['elements'] if e['type'] == 'way'}

    m = odrcore.load(src)
    plane = LocalTangentPlane(*origin_of(m))
    tiles = Tiles()
    for _, label in TRACKS:
        chain = loop_chain(m, label)
        used = profile_loop(chain, tags, plane, tiles)
        for road in chain:
            annotate(road, tags.get(road.id, {}))
        z = [road.refline.elevation(s) for road in chain
             for s in np.linspace(0.0, road.length, 8)]
        print(f'{label}: {len(chain)} roads, elevation {min(z):.1f} to '
              f'{max(z):.1f} m, samples by source {used}')
    if 'source:elevation' not in m.header.attrs['userData']:
        m.header.attrs['userData'] += (
            f'<userData code="source:elevation" value="{ATTRIBUTION}"/>')
    odrcore.dump(m, out)
    print(f'{tiles.fetched} tiles fetched, wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
