# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import pathlib
import re
import sys

import numpy as np
import pytest

import odrcore
from odrcore.model import RoadLink

EXAMPLES = pathlib.Path(odrcore.__file__).parent / 'examples'
sys.path.insert(0, str(EXAMPLES))
import _gsi  # noqa: E402
import yamanote_roads  # noqa: E402


@pytest.fixture(scope='module')
def base():
    return odrcore.load(EXAMPLES / 'yamanote.xodr')


@pytest.fixture(scope='module')
def loop(base):
    return yamanote_roads.Loop(base)


def loop_frame(loop, s):
    """A point of the loop at ``s`` and the unit normal to its left."""
    k = int(np.searchsorted(loop.s, s))
    p, q = loop.xy[k], loop.xy[k + 1]
    tan = (q - p) / np.hypot(*(q - p))
    return p, np.array([-tan[1], tan[0]], dtype='float64'), tan


def on_ellipsoid(loop, local):
    """The OSM coordinates of the plane points and the points they map
    back to."""
    geo = {nid: dict(zip(('lat', 'lon'), loop.plane.to_geodetic(*xy, 0.0)))
           for nid, xy in local.items()}
    # The generator drops the height, so the nodes land on the ellipsoid
    # slightly off the tangent plane.
    local = {nid: np.array(loop.plane.to_local(g['lat'], g['lon'], 0.0)[:2],
                           dtype='float64')
             for nid, g in geo.items()}
    return geo, local


def osm_way(geo, wid, ids, **tags):
    return {'type': 'way', 'id': wid, 'nodes': ids,
            'geometry': [geo[nid] for nid in ids], 'tags': tags}


def overpass(loop, s_mid, s_wrap):
    """Ways 100 m beside the loop at ``s_mid``: a two-way residential way,
    a straight one-way primary with three lanes and a signal in the middle,
    a secondary with lanes per direction and a width, all chained through
    shared nodes, a way of one repeated node, way 16 out of the buffer and
    way 17 reaching out of it from within. Way 15 lies beside the
    loop at ``s_wrap``, just before the end of the chain."""
    p, nrm, tan = loop_frame(loop, s_mid)
    origin = p + 100.0 * nrm
    local = {1: origin - 120.0 * tan + 8.0 * nrm,
             2: origin - 60.0 * tan + 3.0 * nrm,
             3: origin,
             4: origin + 40.0 * tan,
             5: origin + 80.0 * tan,
             6: origin + 110.0 * tan + 30.0 * nrm,
             7: origin + 200.0 * nrm,
             20: origin + 600.0 * nrm, 21: origin + 800.0 * nrm}
    p, nrm, tan = loop_frame(loop, s_wrap)
    local[8] = p + 80.0 * nrm - 20.0 * tan
    local[9] = p + 80.0 * nrm + 20.0 * tan
    geo, local = on_ellipsoid(loop, local)

    def way(wid, ids, **tags):
        return osm_way(geo, wid, ids, **tags)

    elements = [
        way(11, [1, 2, 3], highway='residential', name='Sakura-dori'),
        way(12, [3, 4, 5], highway='primary', oneway='yes', lanes='3',
            maxspeed='50', bridge='yes', layer='1'),
        way(13, [5, 6], highway='secondary', **{
            'lanes:forward': '2', 'lanes:backward': '1', 'width': '9',
            'name': 'Ginza-dori', 'name:en': 'Ginza Avenue'}),
        way(14, [7, 7], highway='tertiary'),
        way(15, [8, 9], highway='unclassified'),
        way(16, [20, 21], highway='residential'),
        way(17, [7, 20], highway='residential'),
        {'type': 'node', 'id': 4, **geo[4],
         'tags': {'highway': 'traffic_signals'}},
    ]
    return {'elements': elements}, local


@pytest.fixture(scope='module')
def built(base, loop):
    s_mid = 0.5 * (loop.stop_s[0] + loop.stop_s[1])
    data, local = overpass(loop, s_mid, loop.length - 50.0)
    m, skipped, outside = yamanote_roads.build(data, base)
    assert outside == 1
    return m, skipped, local


def user_data(road):
    return dict(re.findall(r'code="([^"]+)" value="([^"]*)"',
                           road.attrs['userData']))


def test_lanes(built):
    m, _, _ = built
    lanes = {rid: ([ln.odr_id for ln in road.sections[0].left],
                   [ln.odr_id for ln in road.sections[0].right])
             for rid, road in m.roads.items()}
    assert lanes['11'] == ([1], [-1])
    assert lanes['12'] == ([1, 2, 3], [])
    assert lanes['13'] == ([1, 2], [-1])

    def widths(road):
        sec = road.sections[0]
        return [round(ln.outer.t(0.0) - ln.inner.t(0.0), 9)
                for ln in sec.left + sec.right]

    assert widths(m.roads['11']) == [2.75, -2.75]
    assert widths(m.roads['12']) == [3.0, 3.0, 3.0]
    assert widths(m.roads['13']) == [3.0, 3.0, -3.0]
    # The carriageway is centered on the way, whatever the split.
    assert m.roads['12'].sections[0].center.t(0.0) == -4.5
    assert m.roads['13'].sections[0].center.t(0.0) == -1.5
    assert all(ln.travels_along_s for ln in m.roads['12'].lanes())
    assert not m.roads['13'].sections[0].right[0].travels_along_s
    speed = m.roads['12'].sections[0].left[0].speed[0]
    assert (speed.max, speed.unit) == (50.0, 'km/h')


def test_single_lane_two_way():
    assert yamanote_roads.lane_counts(
        {'highway': 'residential', 'lanes': '1'}) == (1, 0, True)
    assert yamanote_roads.lane_counts(
        {'highway': 'residential', 'oneway': '-1'}) == (0, 1, False)
    assert yamanote_roads.lane_counts({'highway': 'motorway'}) == (2, 0, False)


def test_shared_ends_coincide(built):
    m, _, local = built
    ref = {rid: m.roads[rid].refline for rid in ('11', '12', '13')}
    for a, b, node in (('11', '12', 3), ('12', '13', 5)):
        end = ref[a].point(m.roads[a].length, 0.0)
        start = ref[b].point(0.0, 0.0)
        assert math.dist(end, start) < 1e-6
        assert math.dist(start[:2], local[node]) < 1e-6


def test_one_node_way_skipped(built):
    m, skipped, _ = built
    assert skipped == ['14']
    assert sorted(m.roads) == ['11', '12', '13', '15', '17']


def test_segment(built, loop):
    m, _, _ = built
    first, second = loop.codes[:2]
    pair = '-'.join(sorted([first, second]))
    assert user_data(m.roads['12'])['segment'] == pair
    assert user_data(m.roads['15'])['segment'] == \
        '-'.join(sorted([loop.codes[-1], first]))
    tokyo = loop.codes.index('JY01')
    s = 0.5 * (loop.stop_s[tokyo] + loop.stop_s[tokyo + 1])
    assert loop.segment(s) == 'JY30-JY01'


def test_user_data(built):
    m, _, local = built
    data = user_data(m.roads['12'])
    assert data['osm:way'] == '12'
    assert data['osm:highway'] == 'primary'
    assert data['osm:layer'] == '1'
    s = float(re.match(r's=(\S+) node=4$', data['osm:signal'])
              .group(1))
    at = m.roads['12'].refline.point(s, 0.0)[:2]
    assert math.dist(at, local[4]) < 1e-6
    assert '<bridge ' in m.roads['12'].attrs['_unknown']
    assert user_data(m.roads['13'])['osm:name'] == 'Ginza Avenue'
    assert user_data(m.roads['11'])['osm:name'] == 'Sakura-dori'


@pytest.fixture(scope='module')
def crossing(base, loop):
    """Way 31 bends through node 42, where one-way way 32 leaves it. Way 33
    runs into the end of way 32 head to head, and starts where way 35, a
    single lane for both directions, starts too. Way 34 is a roundabout."""
    p, nrm, tan = loop_frame(loop, 0.5 * (loop.stop_s[2] + loop.stop_s[3]))
    o = p + 100.0 * nrm
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False,
                         dtype='float64')
    ring = {60 + k: o + 120.0 * tan + 30.0 * (np.cos(a) * tan
                                              + np.sin(a) * nrm)
            for k, a in enumerate(angles)}
    geo, _ = on_ellipsoid(loop, {
        41: o - 60.0 * tan + 10.0 * nrm, 42: o,
        43: o + 60.0 * tan + 10.0 * nrm, 44: o - 50.0 * nrm,
        46: o - 120.0 * nrm + 20.0 * tan, 50: o - 190.0 * nrm + 50.0 * tan,
        **ring})
    elements = [
        osm_way(geo, 31, [41, 42, 43], highway='residential'),
        osm_way(geo, 32, [42, 44], highway='residential', oneway='yes'),
        osm_way(geo, 33, [46, 44], highway='primary',
                **{'lanes:forward': '2', 'lanes:backward': '1'}),
        osm_way(geo, 34, [*ring, 60], highway='residential',
                junction='roundabout'),
        osm_way(geo, 35, [46, 50], highway='residential', lanes='1'),
    ]
    m, skipped, _ = yamanote_roads.build({'elements': elements}, base)
    assert skipped == []
    return m


def heading(road, end):
    return road.refline.pose(road.length if end == 'end' else 0.0)[3]


def lane_links(road):
    return {ln.odr_id: (ln.link.predecessors, ln.link.successors)
            for ln in road.lanes()}


def test_validate(built, crossing):
    assert built[0].validate() == []
    assert crossing.validate() == []


def test_split_through_way(crossing):
    m = crossing
    assert sorted(m.roads) == ['31', '31.1', '32', '33', '34', '34.1', '35']
    assert user_data(m.roads['31.1'])['osm:way'] == '31'
    assert m.roads['31'].successor == RoadLink('junction', 'j42')
    assert m.roads['31.1'].predecessor == RoadLink('junction', 'j42')
    assert m.roads['32'].predecessor == RoadLink('junction', 'j42')
    assert m.roads['31'].predecessor is None
    # The way bends at the junction node but its two pieces join smoothly.
    assert heading(m.roads['31'], 'end') == \
        pytest.approx(heading(m.roads['31.1'], 'start'), abs=1e-9)


def test_direct_junction(crossing):
    m = crossing
    junction = m.junctions['j42']
    assert junction.type == 'direct'
    moves = sorted((c.incoming_road, c.linked_road, c.contact_point,
                    c.lane_links) for c in junction.connections)
    # Nothing comes out of the one-way way 32, which leaves the junction.
    assert moves == [('31', '31.1', 'start', [(1, 1)]),
                     ('31', '32', 'start', [(1, 1)]),
                     ('31.1', '31', 'end', [(-1, -1)]),
                     ('31.1', '32', 'start', [(-1, 1)])]
    entering = m.roads['31'].sections[0].lane(1)
    assert sorted(ln.id for ln in m.successors(entering)) == \
        [('31.1', 0, 1), ('32', 0, 1)]


def test_head_to_head(crossing):
    m = crossing
    assert m.roads['32'].successor == RoadLink('road', '33', 'end')
    assert m.roads['33'].successor == RoadLink('road', '32', 'end')
    assert lane_links(m.roads['32']) == {1: ([], [-1])}
    # Lane 1 of way 33 would enter the one-way way 32 against its
    # direction, and lane 2 has no counterpart.
    assert lane_links(m.roads['33'])[-1][1] == [1]
    assert lane_links(m.roads['33'])[1][1] == []
    assert lane_links(m.roads['33'])[2][1] == []
    lane = m.roads['32'].sections[0].lane(1)
    assert [ln.id for ln in m.successors(lane)] == [('33', 0, -1)]
    turn = heading(m.roads['32'], 'end') - heading(m.roads['33'], 'end')
    assert math.cos(turn) == pytest.approx(-1.0, abs=1e-12)


def test_start_to_start(crossing):
    m = crossing
    assert m.roads['33'].predecessor == RoadLink('road', '35', 'start')
    assert m.roads['35'].predecessor == RoadLink('road', '33', 'start')
    # The lane for both directions takes the innermost lane each way.
    assert lane_links(m.roads['35']) == {1: ([-1, 1], [])}
    assert lane_links(m.roads['33'])[-1][0] == [1]
    assert lane_links(m.roads['33'])[1][0] == [1]
    assert lane_links(m.roads['33'])[2][0] == []


def test_closed_way(crossing):
    m = crossing
    assert m.roads['34'].successor == RoadLink('road', '34.1', 'start')
    assert m.roads['34.1'].successor == RoadLink('road', '34', 'start')
    assert lane_links(m.roads['34']) == {1: ([1], [1])}
    for a, b in (('34', '34.1'), ('34.1', '34')):
        assert heading(m.roads[a], 'end') == \
            pytest.approx(heading(m.roads[b], 'start'), abs=1e-9)


class SlopedTiles:
    """A ground plane rising 1 m every 100 m to the north, in place of the
    GSI tiles."""

    def __init__(self, loop):
        self.lat0 = loop.plane.to_geodetic(0.0, 0.0, 0.0)[0]

    def height(self, lat, lon):
        return 20.0 + 1111.0 * (lat - self.lat0), 'stub'


@pytest.fixture(scope='module')
def elevated(base, loop):
    s_mid = 0.5 * (loop.stop_s[0] + loop.stop_s[1])
    data, _ = overpass(loop, s_mid, loop.length - 50.0)
    tiles = SlopedTiles(loop)
    m, _, _ = yamanote_roads.build(data, base, tiles)

    def ground(road, s):
        x, y, _ = road.refline.point(s, 0.0)
        return tiles.height(*loop.plane.to_geodetic(x, y, 0.0)[:2])[0]

    return m, ground


def test_flat(built):
    m, _, _ = built
    assert all(not road.refline.elevation.segs for road in m.roads.values())
    assert 'source:elevation' not in m.header.attrs['userData']


def test_elevation_meets_at_nodes(elevated):
    m, ground = elevated
    for a, b in (('11', '12'), ('12', '13')):
        ra, rb = m.roads[a], m.roads[b]
        assert ra.refline.elevation(ra.length) == \
            pytest.approx(rb.refline.elevation(0.0), abs=1e-9)
    road = m.roads['11']
    for s in np.linspace(0.0, road.length, 7, dtype='float64'):
        assert road.refline.elevation(s) == \
            pytest.approx(ground(road, s), abs=0.01)
    assert 'source:elevation' in m.header.attrs['userData']


def test_bridge_ramps(elevated):
    m, ground = elevated
    road = m.roads['12']
    elev = road.refline.elevation
    # The bridge comes down to the ground roads at both ends. At 80 m it is
    # too short to climb to its full 7 m at the ramp grade, so it peaks in
    # the middle.
    grade = yamanote_roads.RAMP_GRADE
    assert 0.5 * grade * road.length < 7.0
    for f, rise in ((0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 0.0)):
        s = f * road.length
        assert elev(s) - ground(road, s) == \
            pytest.approx(grade * rise * road.length, abs=0.05)
    for s0, *_ in elev.segs[1:]:
        assert elev.derivative(s0 - 1e-9) == \
            pytest.approx(elev.derivative(s0), abs=1e-6)


def test_tile_cache(tmp_path):
    x, y = 29102, 12903
    path = tmp_path / 'dem5a' / str(_gsi.ZOOM) / str(x) / f'{y}.txt'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join([','.join(['12.5'] * 256)] * 256), 'ascii')
    n = 2 ** _gsi.ZOOM
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5)
                                                      / n))))
    tiles = _gsi.Tiles(tmp_path)
    assert tiles.height(lat, lon) == (12.5, 'dem5a')
    # The tile on disk stands in for the server.
    assert tiles.fetched == 0


def test_boxes_cover_buffer(loop):
    boxes = loop.boxes()
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(loop.xy), 200)
    ang = rng.uniform(0.0, 2.0 * np.pi, 200)
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    edge = loop.xy[k] + 0.999 * yamanote_roads.RADIUS * unit
    for x, y in edge:
        lat, lon, _ = loop.plane.to_geodetic(x, y, 0.0)
        assert any(s <= lat <= n and w <= lon <= e for s, w, n, e in boxes)
    # near() checks against the line sampled every NEAR_STEP, so it may
    # miss a point within half a step of the buffer edge.
    margin = yamanote_roads.RADIUS - 0.5 * yamanote_roads.NEAR_STEP
    assert loop.near(loop.xy[k] + margin * unit).all()
    # The middle of the loop is kilometers from the line.
    assert not loop.near(loop.xy.mean(axis=0)[None, :])[0]


def test_roundtrip(built, base, tmp_path):
    m, _, _ = built
    path = tmp_path / 'roads.xodr'
    odrcore.dump(m, path)
    back = odrcore.load(path)
    assert back.frame.geo_reference == base.frame.geo_reference
    assert back.header.name == 'JR East Yamanote Line roads'
    assert sorted(back.roads) == sorted(m.roads)
    for rid, road in m.roads.items():
        other = back.roads[rid]
        assert other.length == pytest.approx(road.length)
        assert [ln.odr_id for ln in other.lanes()] == \
            [ln.odr_id for ln in road.lanes()]
        assert user_data(other) == user_data(road)
        for s in np.linspace(0.0, road.length, 5, dtype='float64'):
            assert math.dist(other.refline.point(s, 0.0),
                             road.refline.point(s, 0.0)) < 1e-6
        center = other.sections[0].center.t(0.0)
        assert center == pytest.approx(road.sections[0].center.t(0.0))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
