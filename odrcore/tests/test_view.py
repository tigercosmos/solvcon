# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import pathlib

import pytest

from odrcore import parse, serialize, view
from odrcore.compare import canonical
from odrcore.geom import PiecewisePoly3
from odrcore.model import AccessRule, MarkSegment
from odrcore.view import RelationKind

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'
NAMES = ['straight_three_sections', 'arc_spiral', 'junction']


def load(name):
    return parse.load(FIXTURES / f'{name}.xodr')


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class TestStructure:

    @pytest.mark.parametrize('name', NAMES)
    def test_one_lanelet_per_lane_and_section(self, name):
        m = load(name)
        v = view.generate(m)
        assert len(v.lanelets) == len(m.lanes)
        assert set(v.by_lane) == set(m.lanes)
        assert len(v.lines) == len(m.boundaries)

    @pytest.mark.parametrize('name', NAMES)
    def test_map_is_not_modified(self, name):
        m = load(name)
        before = canonical(m)
        view.generate(m)
        assert canonical(m) == before

    @pytest.mark.parametrize('name', NAMES)
    def test_bounds_follow_model(self, name):
        m = load(name)
        v = view.generate(m)
        for lane in m.lanes.values():
            ll = v.lanelet(lane.id)
            assert ll.left.line.source == lane.left_bound.id
            assert ll.right.line.source == lane.right_bound.id
            assert ll.left.inverted is ll.right.inverted is \
                (not lane.travels_along_s)

    @pytest.mark.parametrize('name', NAMES)
    def test_left_bound_is_on_the_left(self, name):
        v = view.generate(load(name))
        for ll in v.lanelets.values():
            lpts, rpts = ll.left.points, ll.right.points
            checked = 0
            for i in (0, -1):
                j = 1 if i == 0 else -2
                a, b = lpts[i].position, lpts[j].position
                r = rpts[i].position
                if abs(a[0] - r[0]) + abs(a[1] - r[1]) < 1e-6:
                    continue  # the lane has zero width at this end
                # direction of travel at this end of the left bound
                tip = b if i == 0 else (2 * a[0] - b[0], 2 * a[1] - b[1])
                assert cross(a, tip, r) < 0.0, ll.source
                checked += 1
            assert checked > 0, ll.source

    def test_shared_linestring_with_inversion(self):
        m = load('straight_three_sections')
        v = view.generate(m)
        right1, left1 = v.lanelet(('1', 0, -1)), v.lanelet(('1', 0, 1))
        assert right1.left.line is left1.left.line
        assert right1.left.line.source == '1:0:0'
        assert right1.left.inverted is False
        assert left1.left.inverted is True
        assert left1.left.points == right1.left.points[::-1]
        outer = v.lanelet(('1', 0, -2))
        assert outer.left.line is right1.right.line

    def test_both_and_reversed_lanes(self):
        m = load('arc_spiral')
        v = view.generate(m)
        both = v.lanelet(('7', 0, -1))
        rev = v.lanelet(('7', 0, -2))
        assert both.attrs['one_way'] == 'no'
        assert rev.attrs['one_way'] == 'yes'
        assert not both.left.inverted
        assert not rev.left.inverted
        assert rev.left.line is both.right.line
        assert rev.attrs['subtype'] == 'bicycle_lane'
        assert rev.participants == frozenset({'bicycle'})

    def test_lanelet_attributes(self):
        v = view.generate(load('straight_three_sections'))
        ll = v.lanelet(('1', 0, 1))
        assert ll.attrs['type'] == 'lanelet'
        assert ll.attrs['subtype'] == 'road'
        assert ll.attrs['location'] == 'urban'
        assert ll.attrs['speed_limit'] == '50 km/h'
        assert (ll.attrs['odr:road'], ll.attrs['odr:section'],
                ll.attrs['odr:lane']) == ('1', '0', '1')
        assert v.lanelet(('1', 0, 2)).attrs['subtype'] == 'walkway'
        assert v.lanelet(('1', 2, 1)).attrs['location'] == 'nonurban'

    def test_invalid_spacing(self):
        m = load('junction')
        for bad in (0.0, -1.0, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                view.generate(m, spacing=bad)


class TestIdentifiers:

    @pytest.mark.parametrize('name', NAMES)
    def test_deterministic_across_round_trip(self, name):
        m = load(name)
        a = view.generate(m)
        b = view.generate(parse.loads(serialize.dumps(m)))
        assert set(a.lanelets) == set(b.lanelets)
        assert set(a.lines) == set(b.lines)
        assert set(a.points) == set(b.points)
        assert set(a.regulations) == set(b.regulations)
        assert a.by_lane == b.by_lane
        assert a.routing == b.routing

    def test_ids_are_positive_63_bit(self):
        v = view.generate(load('junction'))
        ids = list(v.points) + list(v.lines) + list(v.lanelets)
        assert all(0 < i < (1 << 63) for i in ids)
        assert len(set(ids)) == len(ids)

    def test_collision_is_detected(self):
        v = view.LaneletView()
        identity = v.identifier('lanelet', '1', 0, -1)
        assert v.identifier('lanelet', '1', 0, -1) == identity
        v.provenance[identity] = ('something', 'else')
        with pytest.raises(ValueError, match='collision'):
            v.identifier('lanelet', '1', 0, -1)


class TestSampling:

    def test_breakpoints_are_stations(self):
        m = load('arc_spiral')
        v = view.generate(m, spacing=7.0)
        line = v.lines[v.by_boundary['7:0:-2']]
        st = line.stations
        assert {0.0, 30.0, 70.0, 110.0, 130.0} <= set(st)
        assert st == tuple(sorted(st))
        assert max(b - a for a, b in zip(st, st[1:])) <= 7.0 + 1e-12
        assert len(st) == len(line.points)

    def test_points_on_the_analytic_boundary(self):
        m = load('arc_spiral')
        v = view.generate(m, spacing=5.0)
        road = m.roads['7']
        for bnd in m.boundaries.values():
            line = v.lines[v.by_boundary[bnd.id]]
            for s, pt in zip(line.stations, line.points):
                assert pt.position == pytest.approx(
                    road.refline.point(s, bnd.t(s)), abs=1e-12)

    def test_mark_and_width_breaks(self):
        m = load('straight_three_sections')
        v = view.generate(m, spacing=40.0)
        center0 = v.lines[v.by_boundary['1:0:0']]
        assert 50.0 in center0.stations
        lane1 = v.lines[v.by_boundary['1:2:1']]
        assert 260.0 in lane1.stations
        # grid of 3 over 100 m plus the elevation breakpoint at 150
        center1 = v.lines[v.by_boundary['1:1:0']]
        assert center1.stations == pytest.approx(
            (100.0, 100.0 + 100.0 / 3, 150.0, 100.0 + 200.0 / 3, 200.0))

    def test_short_section_gets_two_points(self):
        v = view.generate(load('junction'), spacing=1000.0)
        line = v.lines[v.by_boundary['3:0:-1']]
        assert line.stations == (0.0, 20.0)


class TestSeams:

    def test_points_shared_across_sections(self):
        m = load('straight_three_sections')
        v = view.generate(m)
        road = m.roads['1']
        for k in (1, 2):
            prev, cur = road.sections[k - 1], road.sections[k]
            for bnd_prev, bnd_cur in zip(prev.boundaries(),
                                         cur.boundaries()):
                end = v.lines[v.by_boundary[bnd_prev.id]].points[-1]
                start = v.lines[v.by_boundary[bnd_cur.id]].points[0]
                assert end is start, (bnd_prev.id, bnd_cur.id)

    def test_successor_lanelets_touch(self):
        m = load('straight_three_sections')
        v = view.generate(m)
        a, b = v.lanelet(('1', 0, -1)), v.lanelet(('1', 1, -1))
        assert a.left.points[-1] is b.left.points[0]
        assert a.right.points[-1] is b.right.points[0]
        c, d = v.lanelet(('1', 1, 1)), v.lanelet(('1', 0, 1))
        assert c.right.points[-1] is d.right.points[0]

    def test_discontinuity_is_not_merged(self):
        m = load('straight_three_sections')
        m.roads['1'].sections[1].lane(-2).outer.t = \
            PiecewisePoly3.constant(-10.0, 100.0)
        v = view.generate(m)
        end = v.lines[v.by_boundary['1:0:-2']].points[-1]
        start = v.lines[v.by_boundary['1:1:-2']].points[0]
        assert end is not start
        assert start.position == pytest.approx(
            m.roads['1'].refline.point(100.0, -10.0))

    def test_road_ends_are_not_shared(self):
        v = view.generate(load('junction'))
        west = v.lines[v.by_boundary['1:0:0']]
        via = v.lines[v.by_boundary['3:0:0']]
        assert west.points[-1].position == pytest.approx(
            via.points[0].position)
        assert west.points[-1] is not via.points[0]


class TestLineAttributes:

    def test_mark_mapping(self):
        v = view.generate(load('straight_three_sections'))
        center = v.lines[v.by_boundary['1:0:0']].attrs
        assert (center['type'], center['subtype'], center['color']) == \
            ('line_thin', 'dashed', 'yellow')
        assert center['odr:mark_segments'] == '2'
        outer1 = v.lines[v.by_boundary['1:0:1']].attrs
        assert (outer1['type'], outer1['subtype']) == ('line_thin', 'solid')
        assert outer1['odr:lane_change'] == 'none'
        assert v.lines[v.by_boundary['1:0:2']].attrs['type'] == 'virtual'

    def test_double_line_orientation(self):
        m = load('straight_three_sections')
        sec = m.roads['1'].sections[1]
        for bnd in (sec.center, sec.lane(1).outer, sec.lane(-1).outer):
            bnd.marks = [MarkSegment(0.0, 'solid broken', weight='bold')]
        v = view.generate(m)

        def tags(bid):
            attrs = v.lines[v.by_boundary[bid]].attrs
            return attrs['type'], attrs['subtype']

        assert tags('1:1:0') == ('line_thick', 'solid_dashed')
        assert tags('1:1:-1') == ('line_thick', 'solid_dashed')
        assert tags('1:1:1') == ('line_thick', 'dashed_solid')


class TestParticipants:

    def test_lane_type_defaults(self):
        v = view.generate(load('straight_three_sections'))
        assert v.lanelet(('1', 1, -1)).participants == view.MOTOR | {
            'bicycle'}
        assert v.lanelet(('1', 0, 2)).participants == {'pedestrian'}
        assert v.lanelet(('1', 0, -2)).participants == frozenset()

    def test_access_rules(self):
        m = load('straight_three_sections')
        lane = m.lanes[('1', 0, 1)]
        assert 'truck' not in view.participants(lane)
        assert 'bus' in view.participants(lane)
        lane.access = [AccessRule(0.0, 'allow', ['bus', 'taxi'])]
        assert view.participants(lane) == {'bus', 'taxi'}
        lane.access = [AccessRule(10.0, 'allow', ['pedestrian'])]
        assert view.participants(lane) == frozenset()
        lane.access = [AccessRule(0.0, 'deny', ['trucks']),
                       AccessRule(0.0, 'deny', ['bicycle'])]
        assert view.participants(lane) == view.MOTOR - {'truck'}
        lane.access = [AccessRule(0.0, 'allow', ['none'])]
        assert view.participants(lane) == frozenset()


class TestRouting:

    def test_successors_match_link_index(self):
        for name in NAMES:
            m = load(name)
            v = view.generate(m)
            for lane in m.lanes.values():
                got = v.successors(v.by_lane[lane.id])
                want = sorted(v.by_lane[n.id] for n in m.successors(lane))
                assert got == want

    def test_route_through_junction(self):
        m = load('junction')
        v = view.generate(m)
        ids = v.by_lane
        path = v.route(ids[('1', 0, -1)], ids[('2', 0, -1)], 'passengerCar')
        assert path == (ids[('1', 0, -1)], ids[('3', 0, -1)],
                        ids[('2', 0, -1)])
        back = v.route(ids[('2', 0, 1)], ids[('1', 0, 1)], 'bus')
        assert back == (ids[('2', 0, 1)], ids[('4', 0, -1)],
                        ids[('1', 0, 1)])
        assert v.route(ids[('1', 0, -1)], ids[('2', 0, -1)],
                       'pedestrian') == ()
        assert v.route(ids[('2', 0, -1)], ids[('1', 0, -1)],
                       'passengerCar') == ()

    def test_route_respects_access(self):
        v = view.generate(load('straight_three_sections'))
        ids = v.by_lane
        src, dst = ids[('1', 2, 1)], ids[('1', 0, 1)]
        assert len(v.route(src, dst, 'passengerCar')) == 3
        assert v.route(src, dst, 'truck') == ()
        edge = next(e for e in v.routing
                    if e.source == ids[('1', 1, 1)] and e.target == dst)
        assert 'truck' not in edge.participants

    def test_lane_changes(self):
        m = load('straight_three_sections')
        v = view.generate(m)
        ids = v.by_lane
        a, b = ids[('1', 1, -1)], ids[('1', 1, -2)]
        kinds = {(e.source, e.target): e.kind for e in v.routing
                 if e.kind != RelationKind.SUCCESSOR}
        assert kinds[(a, b)] == RelationKind.RIGHT
        assert kinds[(b, a)] == RelationKind.LEFT
        # the center line separates opposite directions: adjacent only
        c = ids[('1', 1, 1)]
        assert (a, c) in v.adjacent and (a, c) not in kinds
        # the sidewalk is adjacent to lane 1 but nobody may change into it
        assert v.lane_changes(ids[('1', 0, 1)]) == []

    def test_lane_change_direction_from_marks(self):
        m = load('straight_three_sections')
        bnd = m.roads['1'].sections[1].lane(-1).outer
        bnd.marks = [MarkSegment(0.0, 'broken', lane_change='increase')]
        v = view.generate(m)
        ids = v.by_lane
        a, b = ids[('1', 1, -1)], ids[('1', 1, -2)]
        assert v.lane_changes(b) == [a]
        assert v.lane_changes(a) == []
        assert (a, b) in v.adjacent

    def test_route_with_and_without_lane_changes(self):
        v = view.generate(load('straight_three_sections'))
        ids = v.by_lane
        src, dst = ids[('1', 0, -1)], ids[('1', 2, -2)]
        path = v.route(src, dst, 'passengerCar')
        assert path[0] == src and path[-1] == dst and len(path) == 4
        assert v.route(src, dst, 'passengerCar', lane_changes=False) == ()
        assert v.predecessors(dst) == [ids[('1', 1, -2)]]


class TestRegulations:

    def test_traffic_light(self):
        m = load('junction')
        v = view.generate(m)
        rid = v.identifier('regulation', 'sig:tl1')
        reg = v.regulations[rid]
        assert reg.subtype == 'traffic_light'
        assert reg.attrs['type'] == 'regulatory_element'
        (pid,) = reg.roles['refers']
        light = v.points[pid]
        assert light.attrs['type'] == 'traffic_light'
        assert light.position == pytest.approx((95.0, -5.0, 4.5))
        for lane_id in (('1', 0, -1), ('3', 0, -1)):
            assert v.lanelet(lane_id).regulations == (rid,)
        assert v.lanelet(('1', 0, 1)).regulations == ()

    def test_speed_limit_sign(self):
        m = load('straight_three_sections')
        v = view.generate(m)
        reg = v.regulations[v.identifier('regulation', 'sig:s1')]
        assert reg.subtype == 'speed_limit'
        assert reg.attrs['sign_type'] == '50 km/h'
        sign = v.points[reg.roles['refers'][0]]
        assert sign.attrs['subtype'] == 'de274'
        attached = sorted(ll.source for ll in v.lanelets.values()
                          if reg.id in ll.regulations)
        assert attached == [('1', 1, -2), ('1', 1, -1)]

    def test_model_roles_are_translated(self):
        m = load('junction')
        reg = m.regulations['sig:tl1']
        reg.roles['yield'] = [m.lanes[('4', 0, -1)]]
        reg.roles['ref_line'] = [m.lanes[('1', 0, -1)].outer]
        v = view.generate(m)
        out = v.regulations[v.identifier('regulation', 'sig:tl1')]
        assert out.roles['yield'] == (v.by_lane[('4', 0, -1)],)
        assert out.roles['ref_line'] == (v.by_boundary['1:0:-1'],)
        assert 'right_of_way' not in out.roles

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
