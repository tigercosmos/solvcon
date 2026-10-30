# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import pathlib
import re
import sys

import numpy as np
import pytest

import odrcore

EXAMPLES = pathlib.Path(odrcore.__file__).parent / 'examples'
sys.path.insert(0, str(EXAMPLES))
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


def overpass(loop, s_mid, s_wrap):
    """Ways 100 m beside the loop at ``s_mid``: a two-way residential way,
    a straight one-way primary with three lanes and a signal in the middle,
    a secondary with lanes per direction and a width, all chained through
    shared nodes, and a way of one repeated node. Way 15 lies beside the
    loop at ``s_wrap``, just before the end of the chain."""
    p, nrm, tan = loop_frame(loop, s_mid)
    origin = p + 100.0 * nrm
    local = {1: origin - 120.0 * tan + 8.0 * nrm,
             2: origin - 60.0 * tan + 3.0 * nrm,
             3: origin,
             4: origin + 40.0 * tan,
             5: origin + 80.0 * tan,
             6: origin + 110.0 * tan + 30.0 * nrm,
             7: origin + 200.0 * nrm}
    p, nrm, tan = loop_frame(loop, s_wrap)
    local[8] = p + 80.0 * nrm - 20.0 * tan
    local[9] = p + 80.0 * nrm + 20.0 * tan
    geo = {nid: dict(zip(('lat', 'lon'), loop.plane.to_geodetic(*xy, 0.0)))
           for nid, xy in local.items()}
    # The generator drops the height, so the nodes land on the ellipsoid
    # slightly off the tangent plane.
    local = {nid: np.array(loop.plane.to_local(g['lat'], g['lon'], 0.0)[:2],
                           dtype='float64')
             for nid, g in geo.items()}

    def way(wid, ids, **tags):
        return {'type': 'way', 'id': wid, 'nodes': ids,
                'geometry': [geo[nid] for nid in ids], 'tags': tags}

    elements = [
        way(11, [1, 2, 3], highway='residential', name='Sakura-dori'),
        way(12, [3, 4, 5], highway='primary', oneway='yes', lanes='3',
            maxspeed='50', bridge='yes', layer='1'),
        way(13, [5, 6], highway='secondary', **{
            'lanes:forward': '2', 'lanes:backward': '1', 'width': '9',
            'name': 'Ginza-dori', 'name:en': 'Ginza Avenue'}),
        way(14, [7, 7], highway='tertiary'),
        way(15, [8, 9], highway='unclassified'),
        {'type': 'node', 'id': 4, **geo[4],
         'tags': {'highway': 'traffic_signals'}},
    ]
    return {'elements': elements}, local


@pytest.fixture(scope='module')
def built(base, loop):
    s_mid = 0.5 * (loop.stop_s[0] + loop.stop_s[1])
    data, local = overpass(loop, s_mid, loop.length - 50.0)
    m, skipped = yamanote_roads.build(data, base)
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
    assert sorted(m.roads) == ['11', '12', '13', '15']


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
    assert s == pytest.approx(math.dist(local[3], local[4]), abs=1e-6)
    assert '<bridge ' in m.roads['12'].attrs['_unknown']
    assert user_data(m.roads['13'])['osm:name'] == 'Ginza Avenue'
    assert user_data(m.roads['11'])['osm:name'] == 'Sakura-dori'


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
