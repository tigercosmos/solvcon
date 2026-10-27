# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import pathlib

import pytest

from odrcore import parse, serialize
from odrcore.compare import canonical, equivalent
from odrcore.geom import Arc, Line, ParamPoly3, PiecewisePoly3, Spiral

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'
NAMES = ['straight_three_sections', 'arc_spiral', 'junction']


def fixture(name):
    return FIXTURES / f'{name}.xodr'


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('lane_geometry', ['width', 'border'])
def test_load_dump_load(tmp_path, name, lane_geometry):
    first = parse.load(fixture(name))
    out = tmp_path / f'{name}.xodr'
    serialize.dump(first, out, lane_geometry=lane_geometry)
    second = parse.load(out)
    assert equivalent(first, second) == []
    assert canonical(first)['roads'].keys() == first.roads.keys()
    third = parse.loads(serialize.dumps(second, lane_geometry=lane_geometry))
    assert equivalent(first, third) == []


class TestStraight:

    @pytest.fixture
    def m(self):
        return parse.load(fixture('straight_three_sections'))

    def test_header_and_frame(self, m):
        assert m.header.rev_minor == 8
        assert m.header.vendor == 'odrcore tests'
        assert m.frame.geo_reference.startswith('+proj=tmerc')
        assert m.frame.offset == (1.5, -2.0, 0.0, 0.1)
        assert 'kept verbatim' in m.header.attrs['userData']

    def test_skipped_and_unknown(self, m):
        assert m.attrs['_skipped'] == 'controller'
        road = m.roads['1']
        assert '<object' in road.attrs['_unknown']
        assert '<shape' in road.attrs['_unknown_lateralProfile']
        assert '<height' in m.lanes[('1', 0, 2)].attrs['_unknown']
        assert 'vendorTag' in m.lanes[('1', 0, -1)].attrs['userData']

    def test_sections_and_layers(self, m):
        road = m.roads['1']
        assert [sec.s0 for sec in road.sections] == [0.0, 100.0, 200.0]
        assert road.sections[1].s1 == 200.0
        assert len(m.lanes) == 12
        assert len(m.boundaries) == 15
        assert road.road_type[0].speed.max == 50.0
        assert road.road_type[1].country is None

    def test_widths_become_absolute_offsets(self, m):
        sec0 = m.roads['1'].sections[0]
        assert sec0.center.t(50.0) == 0.0
        assert sec0.lane(1).outer.t(50.0) == 3.5
        assert sec0.lane(2).outer.t(50.0) == 5.5
        assert sec0.lane(-1).outer.t(50.0) == -3.5
        assert sec0.lane(-2).outer.t(50.0) == -4.5
        sec1 = m.roads['1'].sections[1]
        assert sec1.center.t(150.0) == pytest.approx(0.875)
        assert sec1.lane(1).outer.t(150.0) == pytest.approx(4.375)
        assert sec1.lane(-1).outer.t(150.0) == pytest.approx(-2.625)
        # the poly width of lane -2 is evaluated at section offset 50
        w = 1.0 + 0.075 * 50 - 0.00075 * 50 ** 2 + 0.0000025 * 50 ** 3
        assert sec1.lane(-2).outer.t(150.0) == pytest.approx(-2.625 - w)
        sec2 = m.roads['1'].sections[2]
        assert sec2.lane(1).outer.t(280.0) == pytest.approx(
            1.75 + 3.5 - 0.0875 * 20)

    def test_marks_move_to_boundaries(self, m):
        sec0 = m.roads['1'].sections[0]
        assert [mk.type for mk in sec0.center.marks] == ['broken', 'solid']
        assert sec0.center.marks[1].s_offset == 50.0
        assert sec0.center.marks[1].weight == 'bold'
        mk = sec0.lane(1).outer.marks[0]
        assert (mk.color, mk.width, mk.lane_change, mk.height) == \
            ('white', 0.12, 'none', 0.02)
        assert sec0.lane(2).outer.marks[0].type == 'none'

    def test_lane_fields(self, m):
        lane = m.lanes[('1', 0, 1)]
        assert lane.type == 'driving' and lane.level is False
        assert lane.speed[0].max == 50.0 and lane.speed[0].unit == 'km/h'
        assert lane.access[0].rule == 'deny'
        assert lane.access[0].restrictions == ['truck']
        assert lane.link.successors == [1]
        assert m.lanes[('1', 0, 2)].level is True
        assert m.lanes[('1', 1, -2)].link.predecessors == [-2]

    def test_profiles(self, m):
        rl = m.roads['1'].refline
        assert rl.elevation(100.0) == pytest.approx(101.0)
        assert rl.superelevation(150.0) == 0.02
        assert isinstance(rl.plan[0], Line)

    def test_speed_sign_regulation(self, m):
        reg = m.regulations['sig:s1']
        assert reg.kind == 'speed_limit'
        assert reg.anchor is m.signals['s1']
        assert sorted(ln.odr_id for ln in reg.roles['refers']) == [-2, -1]
        assert all(ln.section.index == 1 for ln in reg.roles['refers'])
        assert reg.roles['yield'] == [] and reg.roles['right_of_way'] == []
        assert m.signals['s1'].attrs['value'] == '50'
        assert m.signals['s1'].validity == [(-2, -1)]

    def test_width_export_restores_widths(self, m, tmp_path):
        out = tmp_path / 'w.xodr'
        serialize.dump(m, out)
        text = out.read_text()
        assert '<border' not in text
        assert 'sOffset="60.0" a="3.5" b="-0.0875"' in text
        assert text.count('<laneOffset') == 3
        again = parse.load(out)
        assert again.roads['1'].sections[2].center.t.simplified().segs == \
            [(200.0, 1.75, 0.0, 0.0, 0.0)]

    def test_width_border_width_is_exact(self, m):
        via_border = parse.loads(serialize.dumps(m, lane_geometry='border'))
        back = parse.loads(serialize.dumps(via_border))
        for lane in m.lanes.values():
            other = back.lanes[lane.id]
            width = lane.outer.t - lane.inner.t
            width2 = other.outer.t - other.inner.t
            assert width.allclose(width2, tol=1e-12)
            assert lane.outer.t.allclose(other.outer.t, tol=1e-12)
        for s in (0.0, 120.0, 150.0, 260.0, 299.0):
            assert back.roads['1'].section_at(s).center.t(s) == \
                pytest.approx(m.roads['1'].section_at(s).center.t(s),
                              abs=1e-12)

    def test_border_export(self, m, tmp_path):
        out = tmp_path / 'b.xodr'
        serialize.dump(m, out, lane_geometry='border')
        text = out.read_text()
        assert '<width' not in text
        again = parse.load(out)
        assert again.lanes[('1', 1, 1)].attrs['_geometry'] == 'border'
        assert again.lanes[('1', 1, 1)].outer.t(150.0) == pytest.approx(
            4.375)


class TestArcSpiral:

    @pytest.fixture
    def m(self):
        return parse.load(fixture('arc_spiral'))

    def test_plan_view(self, m):
        plan = m.roads['7'].refline.plan
        assert [type(g) for g in plan] == [Line, Spiral, Arc, ParamPoly3]
        assert plan[1].curv_end == 0.02
        assert plan[2].curvature == 0.02
        assert plan[3].p_range == 'normalized'
        x, y, _ = plan[1].pose(30.0)
        assert (x, y) == pytest.approx((36.32704, 34.38276))

    def test_border_is_copied(self, m):
        sec = m.roads['7'].sections[0]
        assert sec.lane(-1).outer.t(10.0) == -3.0
        assert sec.lane(-2).outer.t(50.0) == pytest.approx(-5.5)
        assert sec.lane(-2).outer.t.breakpoints() == [0.0, 30.0]
        assert sec.lane(1).outer.t(10.0) == 3.0

    def test_direction_and_lht(self, m):
        road = m.roads['7']
        assert road.rule == 'LHT'
        both, rev = m.lanes[('7', 0, -1)], m.lanes[('7', 0, -2)]
        assert both.direction == 'both' and rev.direction == 'reversed'
        assert rev.travels_along_s
        assert rev.left_bound is both.outer
        assert m.lanes[('7', 0, 1)].travels_along_s

    def test_profiles_outside_first_breakpoint(self, m):
        rl = m.roads['7'].refline
        assert rl.superelevation(10.0) == pytest.approx(-0.02)
        assert rl.superelevation(100.0) == 0.04
        assert PiecewisePoly3.constant(5.0).allclose(
            PiecewisePoly3([(0.0, 5.0, 0.0, 0.0, 0.0)]))


class TestJunction:

    @pytest.fixture
    def m(self):
        return parse.load(fixture('junction'))

    def test_junction_model(self, m):
        j = m.junctions['100']
        assert j.type == 'default'
        assert [c.connecting_road for c in j.connections] == ['3', '4']
        assert j.connections[1].lane_links == [(1, -1)]
        assert '<priority' in j.attrs['_unknown']
        assert m.roads['3'].junction == '100'
        assert m.roads['1'].junction is None
        assert m.roads['1'].successor.element_type == 'junction'
        assert m.roads['3'].predecessor.contact_point == 'end'
        assert m.attrs['_skipped'] == 'junctionGroup'

    def test_traffic_light_regulation(self, m):
        reg = m.regulations['sig:tl1']
        assert reg.kind == 'traffic_light'
        refers = {ln.id for ln in reg.roles['refers']}
        assert refers == {('1', 0, -1), ('3', 0, -1)}
        assert m.roads['3'].signal_refs[0].signal_id == 'tl1'

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
