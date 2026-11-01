# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Build an OpenDRIVE map of the JR East Yamanote Line from OpenStreetMap.

The two OSM route relations, the outer loop (sotomawari, clockwise) and the
inner loop (uchimawari, counterclockwise), each follow their own track. Each
track way of a relation becomes one road, with the OSM way id as the road
id, and the roads are linked in member order into a closed chain. The
reference line is the track centerline in its direction of travel. A road
carries one left-hand-traffic ``rail`` lane centered on the track, an
``<objects>`` bridge or tunnel when the way is tagged so, and
``<userData>`` with the way id and the stops on it (``s`` along the road).

The geometry is projected onto the local tangent plane that ``odrcore.osm``
uses about the geoReference origin, and fitted with one normalized
``paramPoly3`` per vertex pair (cubic Hermite with the Bessel tangent), so
the heading is continuous along each loop and across roads. Interior nodes
closer than ``MIN_SPACING`` are dropped; way ends and stops are kept.

Map data (c) OpenStreetMap contributors, ODbL 1.0. The output is a derived
database under the same license.

Usage: python3 odrcore/examples/yamanote.py OUT.xodr [OVERPASS.json]

Without OVERPASS.json the relations are fetched from the Overpass API.
"""

import json
import pathlib
import sys
from xml.sax.saxutils import quoteattr

import numpy as np

import odrcore
from odrcore.geom import PiecewisePoly3
from odrcore.model import (Boundary, Frame, Header, Lane, LaneLink,
                           LaneSection, Map, Road, RoadLink, SpeedLimit,
                           boundary_id)
from odrcore.osm import LocalTangentPlane

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _osmroad import (fetch, hermite, structure, tangents,  # noqa: E402
                      thin)

OUTER, INNER = 1972920, 1972960
TRACKS = ((OUTER, 'Outer'), (INNER, 'Inner'))
QUERY = (f'[out:json][timeout:120];rel(id:{OUTER},{INNER})->.r;.r out body;'
         'way(r.r:"");out body geom;node(r.r:"stop");out;')
LANE_WIDTH = 3.0
DEFAULT_SPEED = 90.0
ATTRIBUTION = ('(c) OpenStreetMap contributors, ODbL 1.0, relations '
               f'{OUTER} and {INNER}')


def chain(rel: dict, ways: dict) -> list[tuple[dict, list]]:
    """The track ways in member order, each with its ``(node id, lon,
    lat)`` list turned to the direction of travel."""
    members = [ways[m['ref']] for m in rel['members']
               if m['type'] == 'way' and m['role'] == '']
    out = []
    for k, way in enumerate(members):
        nodes = [(n, g['lon'], g['lat'])
                 for n, g in zip(way['nodes'], way['geometry'])]
        if k == 0:
            nxt = members[1]['nodes']
            if nodes[0][0] in (nxt[0], nxt[-1]):
                nodes.reverse()
        elif nodes[0][0] != out[-1][1][-1][0]:
            nodes.reverse()
        if out and nodes[0][0] != out[-1][1][-1][0]:
            raise ValueError(f'way {way["id"]} does not continue the track')
        out.append((way, nodes))
    if out[0][1][0][0] != out[-1][1][-1][0]:
        raise ValueError(f'relation {rel["id"]} is not closed')
    return out


def user_data(way: dict, stops: list[tuple[float, dict]]) -> str:
    items = [f'<userData code="osm:way" value="{way["id"]}"/>']
    for s, stop in stops:
        tags = stop['tags']
        value = (f'{tags.get("ref", "")} {tags.get("name:en", "")} '
                 f'{tags["name"]} s={s!r} node={stop["id"]}')
        items.append(f'<userData code="osm:stop" value={quoteattr(value)}/>')
    return ''.join(items)


def make_road(way: dict, label: str, xy: np.ndarray, tan: np.ndarray,
              stop_at: dict[int, dict], prev_id: str,
              next_id: str) -> Road:
    rid = str(way['id'])
    road = Road(id=rid, name=f'{label} loop, OSM way {rid}', rule='LHT',
                predecessor=RoadLink('road', prev_id, 'end'),
                successor=RoadLink('road', next_id, 'start'))
    s, stops = 0.0, []
    for k in range(len(xy) - 1):
        if k in stop_at:
            stops.append((s, stop_at[k]))
        geom = hermite(s, xy[k], xy[k + 1], tan[k], tan[k + 1])
        road.refline.plan.append(geom)
        s += geom.length
    road.length = s
    objects = structure(way, s)
    if objects:
        road.attrs['_unknown'] = objects
    road.attrs['userData'] = user_data(way, stops)

    maxspeed = way['tags'].get('maxspeed', '')
    speed = float(maxspeed) if maxspeed.isdigit() else DEFAULT_SPEED
    sec = LaneSection(0.0, road)
    road.sections.append(sec)
    sec.center = Boundary(boundary_id(rid, 0, 0), sec,
                          PiecewisePoly3.constant(-0.5 * LANE_WIDTH))
    outer = Boundary(boundary_id(rid, 0, 1), sec,
                     PiecewisePoly3.constant(0.5 * LANE_WIDTH))
    sec.left.append(Lane(sec, 1, sec.center, outer, type='rail',
                         speed=[SpeedLimit(0.0, speed, 'km/h')],
                         link=LaneLink([1], [1])))
    return road


def track_roads(rel: dict, ways: dict, nodes: dict,
                ltp: LocalTangentPlane, label: str) -> list[Road]:
    segments = chain(rel, ways)
    stop_ids = {m['ref'] for m in rel['members'] if m['role'] == 'stop'}

    # One vertex list for the whole loop, so that the tangent at a way end
    # is shared by the two roads meeting there.
    verts, stop_of, ends = [], {}, []
    for way, pts in segments:
        xy = np.array([ltp.to_local(lat, lon, 0.0)[:2]
                       for _, lon, lat in pts])
        if not verts:
            verts.append(xy[0])
        ends.append(len(verts) - 1)
        pinned = {len(pts) - 1} | {i for i, (n, _, _) in enumerate(pts)
                                   if n in stop_ids}
        for i in thin(xy, pinned)[1:]:
            verts.append(xy[i])
            if pts[i][0] in stop_ids:
                stop_of[len(verts) - 1] = nodes[pts[i][0]]
    # The last vertex closes the loop onto the first.
    verts.pop()
    if len(verts) in stop_of:
        stop_of[0] = stop_of.pop(len(verts))
    ends.append(len(verts))
    verts = np.array(verts)
    tan = tangents(verts)
    closed = np.vstack([verts, verts[:1]])
    tan_closed = np.vstack([tan, tan[:1]])

    roads = []
    n = len(segments)
    for w, (way, _) in enumerate(segments):
        lo, hi = ends[w], ends[w + 1]
        stop_at = {k - lo: stop for k, stop in stop_of.items()
                   if lo <= k < hi}
        roads.append(make_road(
            way, label, closed[lo:hi + 1], tan_closed[lo:hi + 1], stop_at,
            str(segments[(w - 1) % n][0]['id']),
            str(segments[(w + 1) % n][0]['id'])))
    return roads


def build(data: dict) -> Map:
    rels = {e['id']: e for e in data['elements'] if e['type'] == 'relation'}
    ways = {e['id']: e for e in data['elements'] if e['type'] == 'way'}
    nodes = {e['id']: e for e in data['elements'] if e['type'] == 'node'}
    lat = [g['lat'] for w in ways.values() for g in w['geometry']]
    lon = [g['lon'] for w in ways.values() for g in w['geometry']]
    lat0, lon0 = round(float(np.mean(lat)), 4), round(float(np.mean(lon)), 4)
    ltp = LocalTangentPlane(lat0, lon0)

    m = Map()
    for rel_id, label in TRACKS:
        for road in track_roads(rels[rel_id], ways, nodes, ltp, label):
            m.roads[road.id] = road

    corners = [ltp.to_local(a, b, 0.0) for a, b in
               ((min(lat), min(lon)), (max(lat), max(lon)))]
    m.header = Header(name='JR East Yamanote Line', version='1.0',
                      date='2026-10-29', vendor='solvcon odrcore',
                      south=corners[0][1], west=corners[0][0],
                      north=corners[1][1], east=corners[1][0])
    m.header.attrs['userData'] = (
        f'<userData code="source" value="{ATTRIBUTION}"/>')
    m.frame = Frame(geo_reference=(
        f'+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 '
        '+ellps=WGS84 +units=m +no_defs'))
    m.reindex()
    return m


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[1])
    if len(argv) > 2:
        data = json.loads(pathlib.Path(argv[2]).read_text('utf-8'))
    else:
        data = fetch(QUERY)
    m = build(data)
    odrcore.dump(m, out)
    for rel_id, label in TRACKS:
        roads = [r for r in m.roads.values() if r.name.startswith(label)]
        total = sum(r.length for r in roads)
        stops = sum(r.attrs['userData'].count('osm:stop') for r in roads)
        print(f'{label}: {len(roads)} roads, {total / 1000.0:.3f} km, '
              f'{stops} stops')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
