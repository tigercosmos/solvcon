# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math

import numpy as np
import pytest

from odrcore.geom import Line, PiecewisePoly3, RefLine
from odrcore.model import Boundary, Lane, LaneSection, Road, boundary_id

from solvcon import StaticMesh
from solvcon.odrmesh import (box, carriageway, concat, ribbon, road_ribbon,
                             road_ribbons, to_static_mesh)


def straight(road_id, x0, y0, hdg, length, z=0.0):
    plan = [Line(0.0, length, x0, y0, hdg)]
    return Road(road_id, length=length,
                refline=RefLine(plan, PiecewisePoly3.constant(z)))


def laned(road_id, x0, length, offset, left, right, grade=0.0):
    """A straight road along x at ``z = grade * s`` with a center line at
    ``offset`` and lanes of the given widths to its left and its right."""
    road = Road(road_id, length=length, refline=RefLine(
        [Line(0.0, length, x0, 0.0, 0.0)],
        PiecewisePoly3([(0.0, 0.0, grade, 0.0, 0.0)])))
    sec = LaneSection(0.0, road)
    road.sections.append(sec)
    sec.center = Boundary(boundary_id(road_id, 0, 0), sec,
                          PiecewisePoly3.constant(offset))
    for sign, widths, lanes in ((1, left, sec.left), (-1, right, sec.right)):
        inner, t = sec.center, offset
        for k, width in enumerate(widths, start=1):
            t += sign * width
            outer = Boundary(boundary_id(road_id, 0, sign * k), sec,
                             PiecewisePoly3.constant(t))
            lanes.append(Lane(sec, sign * k, inner, outer))
            inner = outer
    return road


def square_loop(side=20.0):
    """Four straight roads around a square, end to start."""
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    return [straight(str(k), x, y, k * math.pi / 2, side, z=3.0)
            for k, (x, y) in enumerate(corners)]


CORNERS = (np.arange(16, dtype='int32') % 4 == 3).astype('int32')


def hex_volume(ndcrd, cell):
    """Signed volume of one hexahedron from its five-tetrahedron split."""
    p = ndcrd[cell]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7),
            (1, 3, 4, 6)]
    return sum(np.linalg.det(np.array([p[b] - p[a], p[c] - p[a],
                                       p[d] - p[a]],
                                      dtype='float64')) / 6.0
               for a, b, c, d in tets)


class TestRibbon:

    def test_open_chain_counts_and_shape(self):
        road = straight('r', 0.0, 0.0, 0.0, 10.0)
        ndcrd, clnds = ribbon([road], spacing=5.0, half_width=1.0,
                              thickness=0.5, closed=False)
        assert ndcrd.shape == (12, 3)
        assert clnds.shape == (2, 8)
        assert ndcrd[:, 0].min() == 0.0 and ndcrd[:, 0].max() == 10.0
        assert set(np.round(ndcrd[:, 1], 12)) == {-1.0, 1.0}
        assert set(np.round(ndcrd[:, 2], 12)) == {-0.5, 0.0}

    def test_closed_loop_wraps_and_is_positively_oriented(self):
        ndcrd, clnds = ribbon(square_loop(), spacing=5.0)
        assert clnds.shape == (16, 8)
        assert clnds.max() == len(ndcrd) - 1
        assert set(clnds[-1, 2:4]) == {2, 3} and set(clnds[-1, 6:8]) == {0, 1}
        volumes = np.array([hex_volume(ndcrd, cell) for cell in clnds],
                           dtype='float64')
        assert (volumes > 0.0).all()
        # The slab pinches at the four square corners; the straight cells
        # span 5 m by 3 m by 0.5 m.
        np.testing.assert_allclose(volumes[CORNERS == 0], 7.5)
        assert ndcrd[:, 2].max() == 3.0

    def test_box_is_a_unit_cube(self):
        ndcrd, clnds = box()
        assert hex_volume(ndcrd, clnds[0]) == pytest.approx(1.0)
        assert np.abs(ndcrd).max() == 0.5


class TestStaticMesh:

    def test_joined_road_ribbons_build_a_hex_mesh(self):
        roads = [laned('a', 0.0, 25.0, 0.5, [3.5], [3.0, 3.0]),
                 laned('b', 40.0, 20.0, 0.0, [2.0], [2.0])]
        mh = to_static_mesh(*concat([road_ribbon(r) for r in roads]))
        assert mh.ncell == 4 and mh.nnode == 24
        body = mh.clvol.ndarray[mh.clvol.nghost:]
        np.testing.assert_allclose(body, [12.5 * 9.5 * 0.3] * 2
                                   + [10.0 * 4.0 * 0.3] * 2)

    def test_ribbon_builds_a_hex_mesh(self):
        mh = to_static_mesh(*ribbon(square_loop(), spacing=5.0))
        assert mh.ncell == 16 and mh.nnode == 64
        assert (mh.cltpn.ndarray == StaticMesh.HEXAHEDRON).all()
        body = mh.clvol.ndarray[mh.clvol.nghost:]
        np.testing.assert_allclose(body[CORNERS == 0], 7.5)


class TestRoadRibbon:

    def test_carriageway_spans_the_outermost_boundaries(self):
        roads = [laned('a', 0.0, 30.0, 0.5, [3.5], [3.0, 3.0]),
                 laned('b', 0.0, 30.0, -1.5, [3.0], []),
                 straight('c', 0.0, 0.0, 0.0, 30.0)]
        lo, hi = carriageway(roads, np.array([0, 1, 2], dtype='int64'),
                             np.full(3, 3.0, dtype='float64'))
        np.testing.assert_allclose(lo, [-5.5, -1.5, 0.0])
        np.testing.assert_allclose(hi, [4.0, 1.5, 0.0])

    def test_one_road_is_an_open_slab_across_its_lanes(self):
        road = laned('a', 0.0, 25.0, 0.5, [3.5], [3.0, 3.0], grade=0.02)
        ndcrd, clnds = road_ribbon(road, spacing=10.0, thickness=0.3)
        assert ndcrd.shape == (12, 3) and clnds.shape == (2, 8)
        sections = ndcrd.reshape(-1, 4, 3)
        np.testing.assert_allclose(sections[:, 0, 0], [0.0, 12.5, 25.0])
        np.testing.assert_allclose(sections[:, 0, 1], 4.0)
        np.testing.assert_allclose(sections[:, 1, 1], -5.5)
        np.testing.assert_allclose(sections[:, 0, 2], [0.0, 0.25, 0.5])
        np.testing.assert_allclose(sections[:, 3, 2],
                                   sections[:, 0, 2] - 0.3)
        volumes = [hex_volume(ndcrd, cell) for cell in clnds]
        np.testing.assert_allclose(volumes, 12.5 * 9.5 * 0.3)

    def test_half_width_centers_on_the_carriageway(self):
        road = laned('a', 0.0, 10.0, 0.5, [3.5], [3.0, 3.0])
        ndcrd, _ = road_ribbon(road, half_width=1.0)
        np.testing.assert_allclose(sorted(set(ndcrd[:, 1])), [-1.75, 0.25])

    def test_many_roads_equal_their_ribbons_joined(self):
        roads = [laned('a', 0.0, 25.0, 0.5, [3.5], [3.0, 3.0], grade=0.02),
                 laned('b', 40.0, 4.0, 0.0, [2.0], [2.0]),
                 laned('c', 60.0, 31.0, -1.0, [], [2.5])]
        ndcrd, clnds = road_ribbons(roads, spacing=10.0)
        joined = concat([road_ribbon(r, spacing=10.0) for r in roads])
        np.testing.assert_allclose(ndcrd, joined[0], rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(clnds, joined[1])
        # Cells never join the last section of a road to the next road.
        assert len(clnds) == 2 + 1 + 3
        volumes = np.array([hex_volume(ndcrd, cell) for cell in clnds],
                           dtype='float64')
        assert (volumes > 0.0).all()

    def test_no_roads_give_empty_arrays(self):
        ndcrd, clnds = road_ribbons([])
        assert ndcrd.shape == (0, 3) and clnds.shape == (0, 8)


class TestConcat:

    def test_offsets_the_connectivity(self):
        first, second = box(), ribbon(square_loop(), spacing=5.0)
        ndcrd, clnds = concat([first, second])
        assert ndcrd.shape == (8 + 64, 3) and clnds.dtype == np.int32
        np.testing.assert_array_equal(clnds[:1], first[1])
        np.testing.assert_array_equal(clnds[1:], second[1] + 8)
        np.testing.assert_array_equal(ndcrd[8:], second[0])

    def test_nothing_joins_to_empty_arrays(self):
        ndcrd, clnds = concat([])
        assert ndcrd.shape == (0, 3) and clnds.shape == (0, 8)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
