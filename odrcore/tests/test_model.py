# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import pytest

from odrcore.geom import Line, PiecewisePoly3, RefLine
from odrcore.model import (Boundary, Lane, LaneSection, Map, Road,
                           boundary_id)


def two_by_two(rule='RHT'):
    """One section with lanes 2, 1 | 0 | -1, -2, each 3.5 m wide."""
    road = Road('1', length=100.0, rule=rule,
                refline=RefLine([Line(0.0, 100.0, 0.0, 0.0, 0.0)]))
    sec = LaneSection(0.0, road)
    road.sections.append(sec)
    sec.center = Boundary(boundary_id('1', 0, 0), sec,
                          PiecewisePoly3.constant(0.0))
    for sign, lanes in ((1, sec.left), (-1, sec.right)):
        inner = sec.center
        for k in (1, 2):
            outer = Boundary(boundary_id('1', 0, sign * k), sec,
                             PiecewisePoly3.constant(sign * 3.5 * k))
            lanes.append(Lane(sec, sign * k, inner, outer))
            inner = outer
    m = Map(roads={'1': road})
    m.reindex()
    return m


class TestBoundsResolution:

    def test_ids_and_sharing(self):
        m = two_by_two()
        assert set(m.lanes) == {('1', 0, k) for k in (2, 1, -1, -2)}
        l1, l2 = m.lanes[('1', 0, 1)], m.lanes[('1', 0, 2)]
        assert l1.outer is l2.inner
        assert m.lanes[('1', 0, -1)].inner is l1.inner
        assert l1.inner is m.roads['1'].sections[0].center

    @pytest.mark.parametrize('rule,odr_id,along', [
        ('RHT', -1, True), ('RHT', 1, False),
        ('LHT', -1, False), ('LHT', 1, True)])
    def test_standard_direction(self, rule, odr_id, along):
        m = two_by_two(rule)
        lane = m.lanes[('1', 0, odr_id)]
        assert lane.travels_along_s is along
        higher = lane.outer if odr_id > 0 else lane.inner
        lower = lane.inner if odr_id > 0 else lane.outer
        dt = lane.left_bound.t(10.0) - lane.right_bound.t(10.0)
        if along:
            assert dt > 0.0
            assert lane.left_bound is higher and lane.right_bound is lower
        else:
            assert dt < 0.0
            assert lane.left_bound is lower and lane.right_bound is higher

    def test_reversed_flips(self):
        m = two_by_two()
        std = m.lanes[('1', 0, -1)]
        left, right = std.left_bound, std.right_bound
        std.direction = 'reversed'
        assert not std.travels_along_s
        assert std.left_bound is right and std.right_bound is left

    def test_both_follows_s(self):
        m = two_by_two()
        lane = m.lanes[('1', 0, 1)]
        lane.direction = 'both'
        assert lane.travels_along_s
        assert lane.left_bound is lane.outer

    def test_section_bookkeeping(self):
        m = two_by_two()
        sec = m.roads['1'].sections[0]
        assert sec.index == 0 and sec.s1 == 100.0
        assert [ln.odr_id for ln in sec.lanes()] == [1, 2, -1, -2]
        assert sec.lane(-2).outer.id == '1:0:-2'
        with pytest.raises(KeyError):
            sec.lane(3)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
