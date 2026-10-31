# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Build an OpenDRIVE map of the roads within 500 m of the JR East Yamanote
Line from OpenStreetMap.

The search area follows the Outer loop of the track map that
``yamanote.py`` writes. The Overpass API is asked for the road ways of the
``HIGHWAYS`` classes in the ``CELL`` squares around the loop reference line
(its ``around`` filter returns nothing or hangs), and a way is kept when
one of its nodes is within ``RADIUS`` of the line. A way is kept whole, not
clipped at the edge of the buffer.

A way is split into roads at every interior node that another kept way
also uses or that the way visits again, and a piece that would start and
end at one node is split in the middle. The first piece takes the OSM way
id as the road id, the others ``<way id>.<k>``. Where exactly two road ends
meet they are linked as predecessor and successor, with lane links from
the lanes that arrive to the lanes that leave, innermost first. Where three
or more meet they join a direct junction ``j<node id>`` with a connection
for every move from one road to another that some lane can make. The
heading is continuous where a road continues into one other: the only other
road end at the node, or the next piece of the same way.

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
from odrcore.model import (Boundary, Connection, End, Frame, Header,
                           Junction, Lane, LaneSection, Map, Road, RoadLink,
                           SpeedLimit, boundary_id)
from odrcore.osm import LocalTangentPlane, origin_of

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _osmroad import (MIN_SPACING, fetch, hermite, loop_chain,  # noqa: E402
                      open_tangents, structure, tangents, thin)

BASE_MAP = pathlib.Path(__file__).resolve().parent / 'yamanote.xodr'
LOOP = 'Outer'
RADIUS = 500.0
PROJECT_STEP = 5.0
NEAR_STEP = 25.0
CELL = 500.0
BOXES_PER_QUERY = 25
QUERY_PAUSE = 10.0
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
# The cosine of the sharpest turn through a node that keeps the heading
# continuous across it.
MIN_JOINT_COS = 0.5
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


def query(boxes: list[tuple[float, float, float, float]]) -> str:
    """The road ways with a node in one of the ``(south, west, north,
    east)`` boxes, and their signal and crossing nodes."""
    classes, marks = '|'.join(HIGHWAYS), '|'.join(NODE_TAGS)
    pieces = ''.join(f'way["highway"~"^({classes})$"]'
                     f'({",".join(f"{v:.6f}" for v in box)});'
                     for box in boxes)
    return (f'[out:json][timeout:{SERVER_TIMEOUT:g}];({pieces})->.w;'
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

    def boxes(self) -> list[tuple[float, float, float, float]]:
        """``(south, west, north, east)`` boxes that cover the ``RADIUS``
        buffer of the loop: the ``CELL`` squares of the plane the loop
        passes through and their neighbors, merged along each row."""
        cells = {(int(i) + di, int(j) + dj)
                 for i, j in np.floor(self.xy / CELL)
                 for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        out = []
        for j in sorted({j for _, j in cells}):
            row = sorted(i for i, jj in cells if jj == j)
            start = row[0]
            for prev, i in zip(row, row[1:] + [None]):
                if i == prev + 1:
                    continue
                corners = [self.plane.to_geodetic(x * CELL, y * CELL, 0.0)
                           for x in (start, prev + 1) for y in (j, j + 1)]
                lat = [c[0] for c in corners]
                lon = [c[1] for c in corners]
                out.append((min(lat), min(lon), max(lat), max(lon)))
                start = i
        return out

    def near(self, xy: np.ndarray) -> np.ndarray:
        """Whether each point is within ``RADIUS`` of the loop."""
        step = max(int(round(NEAR_STEP / PROJECT_STEP)), 1)
        line = self.xy[::step]
        out, chunk = np.zeros(len(xy), dtype='bool'), 2048
        for k in range(0, len(xy), chunk):
            d = xy[k:k + chunk, None, :] - line[None, :, :]
            out[k:k + chunk] = (np.einsum('ijk,ijk->ij', d, d).min(axis=1)
                                <= RADIUS * RADIUS)
        return out

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


def pieces(way: dict, uses: dict[int, int]) -> list[tuple[int, int]]:
    """The ``(first, last)`` node indices of the pieces of a way: cut at
    every interior node another way uses or this way visits again, and in
    the middle of a piece that would start and end at one node."""
    nodes = way['nodes']
    visits: dict[int, int] = {}
    for nid in nodes:
        visits[nid] = visits.get(nid, 0) + 1
    cuts = [0] + [i for i in range(1, len(nodes) - 1)
                  if uses[nodes[i]] > 1 or visits[nodes[i]] > 1]
    cuts.append(len(nodes) - 1)
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if nodes[a] == nodes[b] and b - a > 1:
            out += [(a, (a + b) // 2), ((a + b) // 2, b)]
        else:
            out.append((a, b))
    return out


def joint_tangents(geo: dict[str, tuple[dict, np.ndarray, list[int]]],
                   ends: dict[int, list[tuple[str, End]]],
                   chains: list[tuple[list[str], bool]]
                   ) -> dict[str, np.ndarray]:
    """The vertex tangents of each road. At an end that continues into one
    other road end, the only other one at the node or the next piece of the
    same way, the tangent is that of the polyline through both roads."""
    partner: dict[tuple[str, End], tuple[str, End]] = {}
    for at in ends.values():
        if len(at) == 2 and at[0][0] != at[1][0]:
            partner[at[0]], partner[at[1]] = at[1], at[0]
    for chain, closed in chains:
        pairs = list(zip(chain, chain[1:]))
        if closed and len(chain) > 1:
            pairs.append((chain[-1], chain[0]))
        for a, b in pairs:
            if geo[a][2][-1] == geo[b][2][0]:
                partner.setdefault((a, 'end'), (b, 'start'))
                partner.setdefault((b, 'start'), (a, 'end'))

    tan = {rid: open_tangents(xy) for rid, (_, xy, _) in geo.items()}
    for (rid, end), (oid, oend) in partner.items():
        xy, oxy = geo[rid][1], geo[oid][1]
        beyond = oxy[1] if oend == 'start' else oxy[-2]
        three = [xy[-2], xy[-1], beyond] if end == 'end' \
            else [beyond, xy[0], xy[1]]
        three = np.array(three, dtype='float64')
        d0, d1 = three[1] - three[0], three[2] - three[1]
        # Past a sharp turn the shared tangent would loop the shorter
        # segment, so both ends keep their own chords.
        if d0 @ d1 > MIN_JOINT_COS * np.hypot(*d0) * np.hypot(*d1):
            tan[rid][-1 if end == 'end' else 0] = tangents(three)[1]
    return tan


def lanes_at(road: Road, end: End) -> tuple[list[int], list[int]]:
    """The ids of the lanes that arrive at ``end`` of the road and of those
    that leave from it, innermost first."""
    sec = road.sections[0]
    fwd = [ln.odr_id for ln in sec.left]
    bwd = [ln.odr_id for ln in sec.right]
    if sec.left and sec.left[0].direction == 'both':
        bwd = fwd
    return (fwd, bwd) if end == 'end' else (bwd, fwd)


def add_lane_link(road: Road, odr_id: int, end: End, other: int) -> None:
    link = road.sections[0].lane(odr_id).link
    ids = link.successors if end == 'end' else link.predecessors
    if other not in ids:
        ids.append(other)


def set_road_link(road: Road, end: End, link: RoadLink) -> None:
    if end == 'end':
        road.successor = link
    else:
        road.predecessor = link


def link_pair(m: Map, here: tuple[str, End], there: tuple[str, End]) -> None:
    (rid, end), (oid, oend) = here, there
    road, other = m.roads[rid], m.roads[oid]
    set_road_link(road, end, RoadLink('road', oid, oend))
    set_road_link(other, oend, RoadLink('road', rid, end))
    arrive, leave = lanes_at(road, end)
    o_arrive, o_leave = lanes_at(other, oend)
    for a, b in list(zip(arrive, o_leave)) + list(zip(leave, o_arrive)):
        add_lane_link(road, a, end, b)
        add_lane_link(other, b, oend, a)


def direct_junction(m: Map, nid: int,
                    at: list[tuple[str, End]]) -> Junction:
    """A direct junction with a connection for every move from one road end
    to the end of another road that a lane can make."""
    junction = Junction(f'j{nid}', type='direct')
    for rid, end in at:
        set_road_link(m.roads[rid], end, RoadLink('junction', junction.id))
        arrive = lanes_at(m.roads[rid], end)[0]
        for oid, oend in at:
            leave = lanes_at(m.roads[oid], oend)[1]
            if oid == rid or not arrive or not leave:
                continue
            junction.connections.append(Connection(
                str(len(junction.connections)), rid, linked_road=oid,
                contact_point=oend, lane_links=list(zip(arrive, leave))))
    return junction


def link_roads(m: Map, ends: dict[int, list[tuple[str, End]]]) -> None:
    for nid, at in ends.items():
        if len(at) == 2 and at[0][0] != at[1][0]:
            link_pair(m, *at)
        elif len(at) > 2:
            m.junctions[f'j{nid}'] = direct_junction(m, nid, at)


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


def make_road(rid: str, way: dict, xy: np.ndarray, tan: np.ndarray,
              ids: list[int], marked: dict[int, str], loop: Loop) -> Road:
    tags = way['tags']
    road = Road(id=rid, name=tags.get('name:en') or tags.get('name', ''),
                rule='LHT')
    s, marks = 0.0, []
    for k in range(len(xy) - 1):
        if ids[k] in marked:
            marks.append((marked[ids[k]], s, ids[k]))
        geom = hermite(s, xy[k], xy[k + 1], tan[k], tan[k + 1])
        road.refline.plan.append(geom)
        s += geom.length
    if ids[-1] in marked:
        marks.append((marked[ids[-1]], s, ids[-1]))
    road.length = s

    objects = structure(way, s, rid)
    if objects:
        road.attrs['_unknown'] = objects
    middle = road.refline.point(0.5 * s, 0.0)[:2]
    road.attrs['userData'] = user_data(
        way, loop.segment(loop.project(middle)), marks)
    add_lanes(road, tags)
    return road


def build(data: dict, base: Map) -> tuple[Map, list[str], int]:
    """The road map, the ids of the roads of one distinct node and the
    number of ways farther than ``RADIUS`` from the loop."""
    loop = Loop(base)
    ltp = loop.plane
    ways = {}
    for e in data['elements']:
        if e['type'] == 'way':
            ways.setdefault(e['id'], e)
    coords = {nid: g for way in ways.values()
              for nid, g in zip(way['nodes'], way['geometry'])}
    order = list(coords)
    near = loop.near(np.array(
        [ltp.to_local(coords[n]['lat'], coords[n]['lon'], 0.0)[:2]
         for n in order], dtype='float64').reshape(-1, 2))
    near_ids = {n for n, ok in zip(order, near) if ok}
    inside = {k: w for k, w in ways.items()
              if near_ids.intersection(w['nodes'])}
    outside = len(ways) - len(inside)
    ways = inside

    marked = {e['id']: NODE_TAGS[e['tags']['highway']]
              for e in data['elements']
              if e['type'] == 'node'
              and e.get('tags', {}).get('highway') in NODE_TAGS}
    uses: dict[int, int] = {}
    for way in ways.values():
        for nid in set(way['nodes']):
            uses[nid] = uses.get(nid, 0) + 1
    cut = {way['id']: pieces(way, uses) for way in ways.values()}
    pinned = {n for n, c in uses.items() if c > 1} | set(marked)
    pinned |= {way['nodes'][i] for way in ways.values()
               for span in cut[way['id']] for i in span}

    geo, chains, skipped = {}, [], []
    ends: dict[int, list[tuple[str, End]]] = {}
    for way in ways.values():
        chain = []
        for k, (a, b) in enumerate(cut[way['id']]):
            rid = f'{way["id"]}.{k}' if k else str(way['id'])
            piece = dict(way, nodes=way['nodes'][a:b + 1],
                         geometry=way['geometry'][a:b + 1])
            xy, ids = vertices(piece, ltp, pinned)
            if len(xy) < 2:
                skipped.append(rid)
                continue
            geo[rid] = (way, xy, ids)
            chain.append(rid)
            ends.setdefault(ids[0], []).append((rid, 'start'))
            ends.setdefault(ids[-1], []).append((rid, 'end'))
        chains.append((chain, way['nodes'][0] == way['nodes'][-1]))
    tan = joint_tangents(geo, ends, chains)

    m, corners = Map(), []
    for rid, (way, xy, ids) in geo.items():
        m.roads[rid] = make_road(rid, way, xy, tan[rid], ids, marked, loop)
        corners += [xy.min(axis=0), xy.max(axis=0)]
    if not m.roads:
        raise ValueError('the Overpass data has no road way')
    link_roads(m, ends)

    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    m.header = Header(name='JR East Yamanote Line roads', version='1.0',
                      date='2026-10-31', vendor='solvcon odrcore',
                      south=float(lo[1]), west=float(lo[0]),
                      north=float(hi[1]), east=float(hi[0]))
    m.header.attrs['userData'] = (
        f'<userData code="source" value="{ATTRIBUTION}"/>')
    m.frame = Frame(geo_reference=base.frame.geo_reference)
    m.reindex()
    return m, skipped, outside


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


def fetch_all(boxes: list[tuple[float, float, float, float]]) -> dict:
    """The union of the answers to the queries of ``BOXES_PER_QUERY``
    boxes each."""
    elements, seen = [], set()
    for i in range(0, len(boxes), BOXES_PER_QUERY):
        if i:
            time.sleep(QUERY_PAUSE)
        part = fetch_politely(query(boxes[i:i + BOXES_PER_QUERY]))
        for e in part['elements']:
            if (e['type'], e['id']) not in seen:
                seen.add((e['type'], e['id']))
                elements.append(e)
        print(f'boxes {i + 1} to {min(i + BOXES_PER_QUERY, len(boxes))} '
              f'of {len(boxes)}: {len(part["elements"])} elements')
    return {'elements': elements}


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
        data = fetch_all(Loop(base).boxes())
        cache = args.out.with_suffix('.json')
        cache.write_text(json.dumps(data), 'utf-8')
        print(f'saved the Overpass response to {cache}')
    m, skipped, outside = build(data, base)
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
    links = sum(link is not None and link.element_type == 'road'
                for r in m.roads.values()
                for link in (r.predecessor, r.successor)) // 2
    moves = sum(len(j.connections) for j in m.junctions.values())
    print(f'{sum("." in rid for rid in m.roads)} roads cut from longer '
          f'ways, {links} road links, {len(m.junctions)} junctions with '
          f'{moves} connections')
    print(f'left out {outside} ways farther than {RADIUS:g} m')
    if skipped:
        print(f'skipped {len(skipped)} ways with one distinct node: '
              f'{", ".join(skipped)}')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
