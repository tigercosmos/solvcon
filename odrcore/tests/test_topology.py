# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import pathlib

import pytest

from odrcore import parse
from odrcore.model import RoadLink
from odrcore.tests.test_model import two_by_two

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'


def load(name):
    return parse.load(FIXTURES / f'{name}.xodr')


class TestNeighbors:

    def test_neighbor_by_shared_boundary(self):
        m = two_by_two()
        lane = m.lanes[('1', 0, -1)]
        assert m.neighbors(lane) == (m.lanes[('1', 0, 1)],
                                     m.lanes[('1', 0, -2)])
        assert m.neighbors(m.lanes[('1', 0, -2)]) == (lane, None)
        # lane 1 travels against s under RHT, so the center line is on
        # its left and lane -1 is its left neighbor.
        assert m.neighbors(m.lanes[('1', 0, 1)]) == (lane,
                                                     m.lanes[('1', 0, 2)])

    def test_neighbors_under_lht(self):
        m = two_by_two('LHT')
        lane = m.lanes[('1', 0, 1)]
        assert m.neighbors(lane) == (m.lanes[('1', 0, 2)],
                                     m.lanes[('1', 0, -1)])


class TestWithinRoad:

    @pytest.fixture
    def m(self):
        return load('straight_three_sections')

    def test_section_chain(self, m):
        r1 = m.lanes[('1', 0, -1)]
        assert m.successors(r1) == [m.lanes[('1', 1, -1)]]
        assert m.predecessors(r1) == []
        assert m.successors(m.lanes[('1', 1, -1)]) == [m.lanes[('1', 2, -1)]]
        assert m.predecessors(m.lanes[('1', 2, -1)]) == \
            [m.lanes[('1', 1, -1)]]

    def test_left_lanes_run_against_s(self, m):
        l1 = m.lanes[('1', 2, 1)]
        assert m.successors(l1) == [m.lanes[('1', 1, 1)]]
        assert m.predecessors(l1) == []
        assert m.predecessors(m.lanes[('1', 0, 1)]) == [m.lanes[('1', 1, 1)]]

    def test_links_carry_ends(self, m):
        link = m.links.at_end(m.lanes[('1', 0, -1)], 'end')[0]
        assert link.src is m.lanes[('1', 0, -1)]
        assert link.dst is m.lanes[('1', 1, -1)]
        assert (link.src_end, link.dst_end, link.contact) == \
            ('end', 'start', 'start')
        assert link.via_junction is None

    def test_validate_clean(self, m):
        assert m.validate() == []

    def test_validate_reports_asymmetry(self, m):
        m.lanes[('1', 1, -1)].link.predecessors = []
        issues = m.validate()
        assert len(issues) == 1
        assert 'does not name' in issues[0]
        m.lanes[('1', 0, -1)].link.successors = [-9]
        assert any('successor -9 missing' in msg for msg in m.validate())


class TestJunction:

    @pytest.fixture
    def m(self):
        return load('junction')

    def test_through_connecting_road(self, m):
        west = m.lanes[('1', 0, -1)]
        via = m.lanes[('3', 0, -1)]
        east = m.lanes[('2', 0, -1)]
        assert m.successors(west) == [via]
        assert m.successors(via) == [east]
        assert m.predecessors(east) == [via]
        assert m.predecessors(via) == [west]
        assert m.successors(east) == []
        link = m.links.at_end(west, 'end')[0]
        assert link.via_junction is m.junctions['100']
        assert link.contact == 'start'

    def test_reverse_direction_through_junction(self, m):
        east_l = m.lanes[('2', 0, 1)]
        via = m.lanes[('4', 0, -1)]
        west_l = m.lanes[('1', 0, 1)]
        assert m.successors(east_l) == [via]
        assert m.successors(via) == [west_l]
        assert m.predecessors(west_l) == [via]
        assert m.predecessors(via) == [east_l]
        assert m.successors(west_l) == []

    def test_lanes_not_linked_stay_isolated(self, m):
        assert m.successors(m.lanes[('1', 0, 1)]) == []
        assert m.predecessors(m.lanes[('1', 0, -1)]) == []

    def test_index_is_cached_and_rebuilt(self, m):
        first = m.links
        assert m.links is first
        m.reindex()
        assert m.links is not first

    def test_validate_clean(self, m):
        assert m.validate() == []

    def test_validate_geometry_gap(self, m):
        m.roads['2'].refline.plan[0].x0 = 125.0
        issues = m.validate()
        assert any('away from' in msg and "Road('3')" in msg
                   for msg in issues)

    def test_validate_missing_targets(self, m):
        m.junctions['100'].connections[0].lane_links = [(-1, -5)]
        m.roads['1'].successor = RoadLink('junction', '999')
        issues = m.validate()
        assert any('lane -5 missing' in msg for msg in issues)
        assert any('junction 999 missing' in msg for msg in issues)

    def test_conflicting_is_a_stub(self, m):
        with pytest.raises(NotImplementedError):
            m.links.conflicting(m.lanes[('3', 0, -1)])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
