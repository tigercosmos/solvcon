# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Build an OpenDRIVE map of the roads within 500 m of the JR East Yamanote
Line from OpenStreetMap.

The search area follows the Outer loop of the track map that
``yamanote.py`` writes: its reference line is sampled every ``STEP`` and
sent to the Overpass API as an ``around`` polyline. Every road way of a
``HIGHWAYS`` class that comes within ``RADIUS`` of it is kept whole, not
clipped at the edge of the buffer. Each way becomes one road, with the OSM
way id as the road id and no road links or junctions. Ways that meet share
a node, so their end points coincide, but each way is fitted on its own and
the heading may kink where they meet.

A road carries left-hand-traffic ``driving`` lanes: the ones running along
the way on the left (positive ids), the ones running against it on the
right. The lane counts come from the ``lanes``, ``lanes:forward``,
``lanes:backward`` and ``oneway`` tags, else from the class. A two-way way
tagged with a single lane gets one lane of direction ``both``. The lanes are
centered on the way. A road also has a bridge or tunnel ``<objects>`` when
the way is tagged so, and ``<userData>`` with the way id, class, name,
layer, the traffic signal and crossing nodes on it (``s`` along the road),
and the ``segment`` of the line it belongs to: the two Outer-loop stations
on either side of the point of the loop closest to the middle of the way,
in station number order, e.g. ``JY05-JY06``.

The geometry uses the same local tangent plane and geoReference as the track
map, so the two maps overlay, and the same ``paramPoly3`` fit. The roads are
flat (z = 0).

Map data (c) OpenStreetMap contributors, ODbL 1.0. The output is a derived
database under the same license.

Usage: python3 odrcore/examples/yamanote_roads.py [--base YAMANOTE.xodr]
       OUT.xodr [OVERPASS.json]

Without OVERPASS.json the ways are fetched from the Overpass API, or from
the instance that the environment variable ``OVERPASS_URL`` names, and the
response is saved as OUT with the suffix ``.json`` for reruns. The base map
defaults to ``yamanote.xodr`` next to this script.
"""

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
from xml.sax.saxutils import quoteattr

import numpy as np

import odrcore
from odrcore.geom import PiecewisePoly3
from odrcore.model import (Boundary, Frame, Header, Lane, LaneSection, Map,
                           Road, SpeedLimit, boundary_id)
from odrcore.osm import LocalTangentPlane, origin_of

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _osmroad import (MIN_SPACING, fetch, hermite, loop_chain,  # noqa: E402
                      open_tangents, structure, tangents, thin)

BASE_MAP = pathlib.Path(__file__).resolve().parent / 'yamanote.xodr'
LOOP = 'Outer'
RADIUS = 500.0
STEP = 100.0
CHUNK = 10
PROJECT_STEP = 5.0
MAJOR = ('motorway', 'trunk', 'primary', 'secondary', 'tertiary')
MINOR = ('unclassified', 'residential', 'living_street')
HIGHWAYS = MAJOR + MINOR + tuple(f'{c}_link' for c in MAJOR)
NODE_TAGS = {'traffic_signals': 'signal', 'crossing': 'crossing'}
# Lanes in each direction when the way has no lane tags.
DEFAULT_LANES = {'motorway': 2, 'trunk': 2, 'primary': 2}
DEFAULT_SPEED = {
    'motorway': 80.0, 'trunk': 60.0, 'primary': 50.0, 'secondary': 40.0,
    'tertiary': 40.0, 'unclassified': 30.0, 'residential': 30.0,
    'living_street': 20.0, 'motorway_link': 40.0,
}
LANE_WIDTH = 3.0
NARROW_WIDTH = 2.75
WIDTH_RANGE = (2.0, 5.0)
MPH = 1.609344
RETRY_PAUSE = 60.0
SERVER_TIMEOUT = 180.0
# Longer than the server timeout, which counts from when the query starts
# to run and not from when it is queued.
CLIENT_TIMEOUT = 300.0
ATTRIBUTION = ('(c) OpenStreetMap contributors, ODbL 1.0, highway ways '
               f'within {RADIUS:g} m of the Yamanote Line')
STOP_RE = re.compile(r'osm:stop" value="(\w+) .*? s=([-0-9.e+]+)')
HIGHWAY_RE = re.compile(r'osm:highway" value="(\w+)"')


def query(points: list[tuple[float, float]]) -> str:
    """The ways around the polyline through ``points``. An ``around`` line
    of the whole loop timed out or came back empty from Overpass, so the
    line goes in short pieces of ``CHUNK`` points, each searched on its own
    small bounding box."""
    classes, marks = '|'.join(HIGHWAYS), '|'.join(NODE_TAGS)
    pieces = []
    for i in range(0, len(points) - 1, CHUNK):
        line = ','.join(f'{lat:.6f},{lon:.6f}'
                        for lat, lon in points[i:i + CHUNK + 1])
        pieces.append(f'way["highway"~"^({classes})$"]'
                      f'(around:{RADIUS:g},{line});')
    return (f'[out:json][timeout:{SERVER_TIMEOUT:g}];'
            f'({"".join(pieces)})->.w;'
            f'.w out body geom;node(w.w)["highway"~"^({marks})$"];out;')


class Loop:
    """The Outer loop of the track map, sampled for the search polyline,
    the nearest-point projection and the stations along it."""

    def __init__(self, m: Map) -> None:
        self.plane = LocalTangentPlane(*origin_of(m))
        xy, s_all, stops, offset = [], [], [], 0.0
        for road in loop_chain(m, LOOP):
            for s in np.arange(0.0, road.length, PROJECT_STEP,
                               dtype='float64'):
                xy.append(road.refline.point(float(s), 0.0)[:2])
                s_all.append(offset + s)
            for code, s in STOP_RE.findall(road.attrs.get('userData', '')):
                stops.append((offset + float(s), code))
            offset += road.length
        self.length = offset
        self.xy = np.array(xy, dtype='float64')
        self.s = np.array(s_all, dtype='float64')
        stops.sort()
        self.stop_s = np.array([s for s, _ in stops], dtype='float64')
        self.codes = [code for _, code in stops]

    def search_line(self) -> list[tuple[float, float]]:
        """``(lat, lon)`` every ``STEP`` around the loop, closed."""
        pts = []
        for s in np.arange(0.0, self.length, STEP, dtype='float64'):
            k = min(int(np.searchsorted(self.s, s)), len(self.s) - 1)
            x, y = self.xy[k]
            pts.append(self.plane.to_geodetic(x, y, 0.0)[:2])
        return pts + pts[:1]

    def project(self, xy) -> float:
        d = np.hypot(*(self.xy - np.asarray(xy, dtype='float64')).T)
        return float(self.s[int(np.argmin(d))])

    def segment(self, s: float) -> str:
        """The two stations around ``s`` in station number order, which
        puts JY30 before JY01."""
        k = int(np.searchsorted(self.stop_s, s, side='right'))
        a, b = self.codes[k - 1], self.codes[k % len(self.codes)]
        count = len(self.codes)
        if (int(b[2:]) - int(a[2:])) % count == 1:
            return f'{a}-{b}'
        return f'{b}-{a}'


def number(text: str) -> float | None:
    hit = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)', text or '')
    return float(hit.group(1)) if hit else None


def lane_counts(tags: dict) -> tuple[int, int, bool]:
    """``(forward, backward, both)``: the lanes along and against the way,
    and whether a single lane carries both directions."""
    highway = tags['highway']
    oneway = tags.get('oneway', '')
    if oneway == '' and (highway == 'motorway' or tags.get('junction') in (
            'roundabout', 'circular')):
        oneway = 'yes'
    total = number(tags.get('lanes', ''))
    total = int(total) if total else None
    fwd = number(tags.get('lanes:forward', ''))
    bwd = number(tags.get('lanes:backward', ''))
    per = DEFAULT_LANES.get(highway, 1)
    if oneway in ('yes', 'true', '1'):
        return total or per, 0, False
    if oneway == '-1':
        return 0, total or per, False
    if fwd is not None or bwd is not None:
        if fwd is None:
            fwd = total - bwd if total and total > bwd else per
        if bwd is None:
            bwd = total - fwd if total and total > fwd else per
        if fwd + bwd > 0:
            return int(fwd), int(bwd), False
    if total == 1:
        return 1, 0, True
    if total:
        return total - total // 2, total // 2, False
    return per, per, False


def lane_width(tags: dict, count: int) -> float:
    width = number(tags.get('width', ''))
    if width:
        return min(max(width / count, WIDTH_RANGE[0]), WIDTH_RANGE[1])
    if tags['highway'] in ('residential', 'living_street'):
        return NARROW_WIDTH
    return LANE_WIDTH


def speed(tags: dict) -> float:
    text = tags.get('maxspeed', '')
    value = number(text)
    if value is None:
        highway = tags['highway']
        return DEFAULT_SPEED.get(highway, DEFAULT_SPEED.get(
            highway.removesuffix('_link'), 40.0))
    return value * MPH if 'mph' in text else value


def vertices(way: dict, ltp: LocalTangentPlane,
             pinned: set[int]) -> tuple[np.ndarray, list[int]]:
    """The thinned vertices of a way and their node ids. Consecutive nodes
    at the same place are merged, and a free vertex closer than
    ``MIN_SPACING`` to the last node is dropped so that the last segment
    is not a stub."""
    ids, xy = [], []
    for nid, g in zip(way['nodes'], way['geometry']):
        p = np.array(ltp.to_local(g['lat'], g['lon'], 0.0)[:2],
                     dtype='float64')
        if xy and np.hypot(*(p - xy[-1])) < 1e-3:
            if nid in pinned:
                ids[-1] = nid
            continue
        ids.append(nid)
        xy.append(p)
    if len(xy) < 2:
        return np.empty((0, 2), dtype='float64'), []
    xy = np.array(xy, dtype='float64')
    keep = {len(xy) - 1} | {i for i, n in enumerate(ids) if n in pinned}
    idx = thin(xy, keep)
    if (len(idx) > 2 and idx[-2] not in keep
            and np.hypot(*(xy[idx[-1]] - xy[idx[-2]])) < MIN_SPACING):
        idx.pop(-2)
    return xy[idx], [ids[i] for i in idx]


def way_tangents(xy: np.ndarray, closed: bool) -> np.ndarray:
    if closed and len(xy) > 3:
        tan = tangents(xy[:-1])
        return np.vstack([tan, tan[:1]])
    return open_tangents(xy)


def user_data(way: dict, segment: str,
              marks: list[tuple[str, float, int]]) -> str:
    tags = way['tags']
    name = tags.get('name:en') or tags.get('name', '')
    items = [('osm:way', str(way['id'])), ('osm:highway', tags['highway']),
             ('osm:name', name), ('osm:layer', tags.get('layer', '0')),
             ('segment', segment)]
    items += [(f'osm:{kind}', f's={s!r} node={nid}')
              for kind, s, nid in marks]
    return ''.join(f'<userData code="{code}" value={quoteattr(value)}/>'
                   for code, value in items)


def add_lanes(road: Road, tags: dict) -> None:
    fwd, bwd, both = lane_counts(tags)
    width = lane_width(tags, fwd + bwd)
    limit = [SpeedLimit(0.0, speed(tags), 'km/h')]
    sec = LaneSection(0.0, road)
    road.sections.append(sec)
    rid = road.id
    sec.center = Boundary(boundary_id(rid, 0, 0), sec,
                          PiecewisePoly3.constant(0.5 * (bwd - fwd) * width))
    inner = sec.center
    for k in range(1, fwd + 1):
        outer = Boundary(boundary_id(rid, 0, k), sec,
                         PiecewisePoly3.constant(inner.t(0.0) + width))
        sec.left.append(Lane(sec, k, inner, outer,
                             direction='both' if both else 'standard',
                             speed=list(limit)))
        inner = outer
    inner = sec.center
    for k in range(1, bwd + 1):
        outer = Boundary(boundary_id(rid, 0, -k), sec,
                         PiecewisePoly3.constant(inner.t(0.0) - width))
        sec.right.append(Lane(sec, -k, inner, outer, speed=list(limit)))
        inner = outer


def make_road(way: dict, xy: np.ndarray, ids: list[int],
              marked: dict[int, str], loop: Loop) -> Road:
    tags = way['tags']
    rid = str(way['id'])
    road = Road(id=rid, name=tags.get('name:en') or tags.get('name', ''),
                rule='LHT')
    closed = way['nodes'][0] == way['nodes'][-1]
    tan = way_tangents(xy, closed)
    s, marks = 0.0, []
    for k in range(len(xy) - 1):
        if ids[k] in marked:
            marks.append((marked[ids[k]], s, ids[k]))
        geom = hermite(s, xy[k], xy[k + 1], tan[k], tan[k + 1])
        road.refline.plan.append(geom)
        s += geom.length
    if ids[-1] in marked and not closed:
        marks.append((marked[ids[-1]], s, ids[-1]))
    road.length = s

    objects = structure(way, s)
    if objects:
        road.attrs['_unknown'] = objects
    middle = road.refline.point(0.5 * s, 0.0)[:2]
    road.attrs['userData'] = user_data(
        way, loop.segment(loop.project(middle)), marks)
    add_lanes(road, tags)
    return road


def build(data: dict, base: Map) -> tuple[Map, list[str]]:
    """The road map and the ids of the ways left out."""
    ways = {}
    for e in data['elements']:
        if e['type'] == 'way':
            ways.setdefault(e['id'], e)
    marked = {e['id']: NODE_TAGS[e['tags']['highway']]
              for e in data['elements']
              if e['type'] == 'node'
              and e.get('tags', {}).get('highway') in NODE_TAGS}
    uses: dict[int, int] = {}
    for way in ways.values():
        for nid in set(way['nodes']):
            uses[nid] = uses.get(nid, 0) + 1
    pinned = {n for n, c in uses.items() if c > 1} | set(marked)

    loop = Loop(base)
    ltp = loop.plane
    m, skipped, corners = Map(), [], []
    for way in ways.values():
        xy, ids = vertices(way, ltp, pinned)
        if len(xy) < 2:
            skipped.append(str(way['id']))
            continue
        road = make_road(way, xy, ids, marked, loop)
        m.roads[road.id] = road
        corners += [xy.min(axis=0), xy.max(axis=0)]

    if not m.roads:
        raise ValueError('the Overpass data has no road way')
    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    m.header = Header(name='JR East Yamanote Line roads', version='1.0',
                      date='2026-10-31', vendor='solvcon odrcore',
                      south=float(lo[1]), west=float(lo[0]),
                      north=float(hi[1]), east=float(hi[0]))
    m.header.attrs['userData'] = (
        f'<userData code="source" value="{ATTRIBUTION}"/>')
    m.frame = Frame(geo_reference=base.frame.geo_reference)
    m.reindex()
    return m, skipped


def fetch_politely(text: str) -> dict:
    """Fetch, and once more after a pause when Overpass is busy."""
    try:
        return fetch(text, CLIENT_TIMEOUT)
    except (urllib.error.HTTPError, TimeoutError) as exc:
        if getattr(exc, 'code', 504) not in (429, 502, 503, 504):
            raise
        print(f'Overpass failed ({exc}), retrying in {RETRY_PAUSE:g} s')
        time.sleep(RETRY_PAUSE)
        return fetch(text, CLIENT_TIMEOUT)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split('\n\n')[0])
    parser.add_argument('--base', type=pathlib.Path, default=BASE_MAP)
    parser.add_argument('out', type=pathlib.Path)
    parser.add_argument('overpass', type=pathlib.Path, nargs='?')
    args = parser.parse_args(argv[1:])

    base = odrcore.load(args.base)
    if args.overpass:
        data = json.loads(args.overpass.read_text('utf-8'))
    else:
        data = fetch_politely(query(Loop(base).search_line()))
        cache = args.out.with_suffix('.json')
        cache.write_text(json.dumps(data), 'utf-8')
        print(f'saved the Overpass response to {cache}')
    m, skipped = build(data, base)
    odrcore.dump(m, args.out)

    lengths: dict[str, list[float]] = {h: [] for h in HIGHWAYS}
    for road in m.roads.values():
        highway = HIGHWAY_RE.search(road.attrs['userData']).group(1)
        lengths[highway].append(road.length)
    for highway, of_class in lengths.items():
        if of_class:
            print(f'{highway}: {len(of_class)} roads, '
                  f'{sum(of_class) / 1000.0:.3f} km')
    total = sum(r.length for r in m.roads.values())
    print(f'total: {len(m.roads)} roads, {total / 1000.0:.3f} km')
    if skipped:
        print(f'skipped {len(skipped)} ways with one distinct node: '
              f'{", ".join(skipped)}')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
