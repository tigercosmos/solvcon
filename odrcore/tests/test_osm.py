# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import pathlib
import xml.etree.ElementTree as ET

import pytest

from odrcore import osm, parse

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'
NAMES = ['straight_three_sections', 'arc_spiral', 'junction']


def load(name):
    return parse.load(FIXTURES / f'{name}.xodr')


def tags(elem):
    return {t.get('k'): t.get('v') for t in elem.findall('tag')}


class Osm:
    """The parts of a written file the tests read back."""

    def __init__(self, text):
        self.root = ET.fromstring(text)
        self.nodes = {int(n.get('id')): n for n in self.root.iter('node')}
        self.ways = {int(w.get('id')): w for w in self.root.iter('way')}
        self.relations = {int(r.get('id')): r
                          for r in self.root.iter('relation')}

    def xy(self, way_id):
        out = []
        for nd in self.ways[way_id].findall('nd'):
            t = tags(self.nodes[int(nd.get('ref'))])
            out.append((float(t['local_x']), float(t['local_y'])))
        return out

    def members(self, rel_id):
        return [(m.get('type'), m.get('role'), int(m.get('ref')))
                for m in self.relations[rel_id].findall('member')]

    def of_type(self, kind):
        return [r for r, e in self.relations.items()
                if tags(e).get('type') == kind]


def _signed_distance(line, p):
    """Lanelet2 ``signedDistance``: positive on the left of the closest
    segment."""
    best = None
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = dx * dx + dy * dy
        u = 0.0 if seg == 0.0 else max(0.0, min(1.0, (
            (p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg))
        q = (a[0] + u * dx, a[1] + u * dy)
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if best is None or d < best[0]:
            left = dx * (p[1] - a[1]) - dy * (p[0] - a[0]) > 0.0
            best = (d, d if left else -d)
    return best[1]


def _middle(line):
    if len(line) > 2:
        return line[len(line) // 2]
    return tuple((a + b) * 0.5 for a, b in zip(line[0], line[-1]))


def align(left, right):
    """Port of Lanelet2 ``geometry::align`` that the OSM loader applies to
    the two bounds of each lanelet."""
    if _signed_distance(left, _middle(right)) >= 0.0 and len(left) > 1:
        left = left[::-1]
    if _signed_distance(right, _middle(left)) <= 0.0 and len(right) > 1:
        right = right[::-1]
    return left, right


class TestLocalTangentPlane:

    @pytest.mark.parametrize('origin', [(0.0, 0.0), (48.0, 11.0),
                                        (-33.9, 151.2), (64.1, -21.9)])
    def test_round_trip(self, origin):
        plane = osm.LocalTangentPlane(*origin)
        for x, y, z in [(0.0, 0.0, 0.0), (1234.5, -987.6, 12.3),
                        (-20000.0, 35000.0, -40.0)]:
            lat, lon, h = plane.to_geodetic(x, y, z)
            assert plane.to_local(lat, lon, h) == \
                pytest.approx((x, y, z), abs=1e-6)

    def test_origin_is_the_origin(self):
        lat, lon, h = osm.LocalTangentPlane(48.0, 11.0).to_geodetic(0, 0, 0)
        assert (lat, lon, h) == pytest.approx((48.0, 11.0, 0.0), abs=1e-9)

    def test_axes_point_east_and_north(self):
        plane = osm.LocalTangentPlane(48.0, 11.0)
        lat, lon, _ = plane.to_geodetic(100.0, 0.0, 0.0)
        assert lon > 11.0 and lat == pytest.approx(48.0, abs=1e-5)
        lat, lon, _ = plane.to_geodetic(0.0, 100.0, 0.0)
        assert lat > 48.0 and lon == pytest.approx(11.0, abs=1e-12)


class TestOrigin:

    def test_from_geo_reference(self):
        assert osm.origin_of(load('straight_three_sections')) == (48.0, 11.0)

    def test_default(self):
        m = load('straight_three_sections')
        m.frame.geo_reference = None
        assert osm.origin_of(m) == (0.0, 0.0)


class TestStructure:

    @pytest.mark.parametrize('name', NAMES)
    def test_one_relation_per_lanelet(self, name):
        m = load(name)
        out = Osm(osm.dumps(m))
        assert len(out.of_type('lanelet')) == len(m.lanes)
        assert len(out.of_type('regulatory_element')) == len(m.regulations)

    @pytest.mark.parametrize('name', NAMES)
    def test_one_way_per_boundary_and_signal(self, name):
        m = load(name)
        out = Osm(osm.dumps(m))
        signals = sum(len(r.signals) for r in m.roads.values())
        assert len(out.ways) == len(m.boundaries) + signals

    @pytest.mark.parametrize('name', NAMES)
    def test_references_resolve(self, name):
        out = Osm(osm.dumps(load(name)))
        table = {'node': out.nodes, 'way': out.ways,
                 'relation': out.relations}
        for way in out.ways.values():
            for nd in way.findall('nd'):
                assert int(nd.get('ref')) in out.nodes
        for rel in out.relations:
            for kind, _, ref in out.members(rel):
                assert ref in table[kind]

    @pytest.mark.parametrize('name', NAMES)
    def test_ids_match_view(self, name):
        m = load(name)
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        assert set(out.of_type('lanelet')) == set(v.lanelets)
        assert set(v.lines) <= set(out.ways)

    def test_shared_way(self):
        m = load('straight_three_sections')
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        center = v.by_boundary['1:0:0']
        users = [r for r in out.of_type('lanelet')
                 if ('way', 'left', center) in out.members(r)]
        assert sorted(v.lanelets[r].source for r in users) == \
            [('1', 0, -1), ('1', 0, 1)]

    def test_deterministic(self):
        assert osm.dumps(load('junction')) == osm.dumps(load('junction'))


class TestGeometry:

    @pytest.mark.parametrize('name', NAMES)
    def test_local_tags_are_model_coordinates(self, name):
        m = load(name)
        out = Osm(osm.dumps(m))
        for pt in m.lanelet2().points.values():
            if pt.id not in out.nodes:
                continue
            t = tags(out.nodes[pt.id])
            assert (float(t['local_x']), float(t['local_y'])) == \
                pt.position[:2]

    @pytest.mark.parametrize('name', NAMES)
    def test_lat_lon_project_back(self, name):
        m = load(name)
        out = Osm(osm.dumps(m))
        plane = osm.LocalTangentPlane(*osm.origin_of(m))
        for node in out.nodes.values():
            t = tags(node)
            got = plane.to_local(float(node.get('lat')),
                                 float(node.get('lon')), float(t['ele']))
            assert got[:2] == pytest.approx(
                (float(t['local_x']), float(t['local_y'])), abs=1e-6)

    def test_explicit_origin(self):
        m = load('straight_three_sections')
        out = Osm(osm.dumps(m, origin=(10.0, 20.0)))
        lats = [float(n.get('lat')) for n in out.nodes.values()]
        assert all(abs(lat - 10.0) < 0.01 for lat in lats)

    @pytest.mark.parametrize('name', NAMES)
    def test_loader_alignment_gives_travel_direction(self, name):
        """After the Lanelet2 loader aligns the bounds, the left bound
        runs in the travel direction of the lane."""
        m = load(name)
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        for rel in out.of_type('lanelet'):
            ll = v.lanelets[rel]
            left, right = align(out.xy(ll.left.id), out.xy(ll.right.id))
            want = [p.position[:2] for p in ll.left.points]
            assert left == want, ll.source
            assert right == [p.position[:2] for p in ll.right.points]


class TestTags:

    def test_participants(self):
        m = load('straight_three_sections')
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        for rel in out.of_type('lanelet'):
            t = tags(out.relations[rel])
            ll = v.lanelets[rel]
            for odr, name in osm.LANELET2_PARTICIPANTS.items():
                want = 'yes' if odr in ll.participants else 'no'
                assert t[f'participant:{name}'] == want
            assert set(filter(None, t['odr:participants'].split(','))) == \
                set(ll.participants)

    def test_lanelet_tags_come_from_view(self):
        m = load('straight_three_sections')
        out = Osm(osm.dumps(m))
        for rel, ll in m.lanelet2().lanelets.items():
            t = tags(out.relations[rel])
            assert {k: t[k] for k in ll.attrs} == ll.attrs

    def test_traffic_light_refers_a_way(self):
        m = load('junction')
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        rid = v.identifier('regulation', 'sig:tl1')
        assert tags(out.relations[rid])['subtype'] == 'traffic_light'
        refers = [(k, ref) for k, role, ref in out.members(rid)
                  if role == 'refers']
        assert len(refers) == 1 and refers[0][0] == 'way'
        way = out.ways[refers[0][1]]
        assert tags(way)['type'] == 'traffic_light'
        xs = out.xy(refers[0][1])
        assert len(xs) == 2
        assert math.dist(*xs) == pytest.approx(osm.DEFAULT_SIGNAL_WIDTH)
        mid = tuple((a + b) * 0.5 for a, b in zip(*xs))
        assert mid == pytest.approx((95.0, -5.0))
        for lane_id in (('1', 0, -1), ('3', 0, -1)):
            assert ('relation', 'regulatory_element', rid) in \
                out.members(v.by_lane[lane_id])

    def test_other_roles_keep_member_types(self):
        m = load('junction')
        reg = m.regulations['sig:tl1']
        reg.roles['yield'] = [m.lanes[('4', 0, -1)]]
        reg.roles['ref_line'] = [m.lanes[('1', 0, -1)].outer]
        out = Osm(osm.dumps(m))
        v = m.lanelet2()
        members = out.members(v.identifier('regulation', 'sig:tl1'))
        assert ('relation', 'yield', v.by_lane[('4', 0, -1)]) in members
        assert ('way', 'ref_line', v.by_boundary['1:0:-1']) in members


class TestMapApi:

    def test_lanelet2_is_cached(self):
        m = load('junction')
        assert m.lanelet2() is m.lanelet2()
        assert m.lanelet2(1.0) is not m.lanelet2()

    def test_reindex_drops_cache(self):
        m = load('junction')
        first = m.lanelet2()
        m.reindex()
        assert m.lanelet2() is not first

    def test_route_returns_lanes(self):
        m = load('junction')
        lanes = m.lanes
        path = m.route(lanes[('1', 0, -1)], lanes[('2', 0, -1)],
                       'passengerCar')
        assert [ln.id for ln in path] == [('1', 0, -1), ('3', 0, -1),
                                          ('2', 0, -1)]
        assert m.route(lanes[('1', 0, -1)], lanes[('2', 0, -1)],
                       'pedestrian') == []

    def test_dump_writes_file(self, tmp_path):
        m = load('junction')
        path = tmp_path / 'junction.osm'
        osm.dump(m, path)
        assert Osm(path.read_text()).relations.keys() == \
            Osm(osm.dumps(m)).relations.keys()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
