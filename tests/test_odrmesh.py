# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math

import numpy as np
import pytest

from odrcore.geom import Arc, Line, PiecewisePoly3, RefLine
from odrcore.model import (Boundary, Lane, LaneSection, MarkSegment, Road,
                           RoadObject, Signal, boundary_id)

from solvcon import StaticMesh
from solvcon.odrmesh import (box, boxes, carriageway, concat,
                             crosswalk_boxes, mark_ribbons, object_poses,
                             ribbon, road_ribbon, road_ribbons,
                             signal_poses, to_static_mesh)


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
        # The road without lanes collapses onto its reference line, the
        # middle a half_width slab is centered on.
        np.testing.assert_allclose(lo, [-5.5, -1.5, 0.0])
        np.testing.assert_allclose(hi, [4.0, 1.5, 0.0])

    def test_a_road_without_lanes_draws_only_with_a_half_width(self):
        bare = straight('c', 0.0, 0.0, 0.0, 30.0)
        road = laned('a', 40.0, 20.0, 0.0, [2.0], [2.0])
        ndcrd, clnds = road_ribbons([bare, road], spacing=10.0)
        alone = road_ribbon(road, spacing=10.0)
        np.testing.assert_array_equal(ndcrd, alone[0])
        np.testing.assert_array_equal(clnds, alone[1])
        assert road_ribbon(bare)[1].shape == (0, 8)
        ndcrd, clnds = road_ribbon(bare, spacing=10.0, half_width=1.5)
        assert len(clnds) == 3
        volumes = [hex_volume(ndcrd, cell) for cell in clnds]
        np.testing.assert_allclose(volumes, 10.0 * 3.0 * 0.3)

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


def painted(road_id='a', length=30.0, grade=0.0):
    """A two-way road along x with a solid yellow center line, broken white
    lines between its two lanes of each side and solid white edges."""
    road = laned(road_id, 0.0, length, 0.0, [3.0, 3.0], [3.0, 3.0],
                 grade=grade)
    sec = road.sections[0]
    sec.center.marks = [MarkSegment(0.0, 'solid', color='yellow',
                                    width=0.3)]
    for side in (sec.left, sec.right):
        side[0].outer.marks = [MarkSegment(0.0, 'broken', color='white')]
        side[1].outer.marks = [MarkSegment(0.0, 'solid', color='white',
                                           width=0.2)]
    return road


def spans(ndcrd, clnds):
    """The x ranges the cells cover, overlapping or touching ones joined,
    so a gap in the paint is a gap between two spans."""
    cells = ndcrd[clnds]
    ranges = sorted(zip(cells[:, :, 0].min(axis=1),
                        cells[:, :, 0].max(axis=1)))
    out = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def at_offset(ndcrd, clnds, t):
    """The cells centered on lateral offset ``t``."""
    middle = ndcrd[clnds][:, :, 1].mean(axis=1)
    return clnds[np.abs(middle - t) < 1e-9]


class TestMarkRibbons:

    def test_a_broken_line_paints_dashes_and_skips_gaps(self):
        road = painted(length=30.0)
        ndcrd, clnds = mark_ribbons([road], 'white', spacing=2.0,
                                    dash=(3.0, 6.0))
        left = at_offset(ndcrd, clnds, 3.0)
        # Dashes of 3 m start every 9 m: at 0, 9, 18 and 27, the last
        # one ending with the road.
        np.testing.assert_allclose(spans(ndcrd, left),
                                   [[0, 3], [9, 12], [18, 21], [27, 30]],
                                   atol=1e-9)
        # Each dash is two cells of 1.5 m; none spans a gap.
        assert len(left) == 8
        volumes = [hex_volume(ndcrd, cell) for cell in left]
        np.testing.assert_allclose(volumes, 1.5 * 0.15 * 0.02)

    def test_a_solid_line_runs_the_whole_mark(self):
        road = painted(length=30.0)
        ndcrd, clnds = mark_ribbons([road], 'white', spacing=2.0)
        edge = at_offset(ndcrd, clnds, -6.0)
        assert len(edge) == 15
        np.testing.assert_allclose(spans(ndcrd, edge), [[0.0, 30.0]])
        np.testing.assert_allclose(sorted(set(ndcrd[edge][:, :, 1].flat)),
                                   [-6.1, -5.9])

    def test_a_mark_ends_where_the_next_one_starts(self):
        road = painted(length=30.0)
        center = road.sections[0].center
        center.marks.append(MarkSegment(10.0, 'none', color='yellow'))
        center.marks.append(MarkSegment(20.0, 'solid', color='yellow'))
        ndcrd, clnds = mark_ribbons([road], 'yellow', spacing=2.0)
        np.testing.assert_allclose(spans(ndcrd, clnds),
                                   [[0.0, 10.0], [20.0, 30.0]])

    def test_the_color_picks_the_lines(self):
        road = painted()
        ndcrd, clnds = mark_ribbons([road], 'yellow')
        np.testing.assert_allclose(sorted(set(ndcrd[:, 1])), [-0.15, 0.15])
        white = mark_ribbons([road], 'white')
        middle = white[0][white[1]][:, :, 1].mean(axis=1)
        assert set(np.round(middle, 9)) == {-6.0, -3.0, 3.0, 6.0}
        assert mark_ribbons([road], 'blue')[1].shape == (0, 8)

    def test_the_top_floats_above_the_surface(self):
        road = painted(grade=0.05)
        ndcrd, clnds = mark_ribbons([road], 'yellow', lift=0.03,
                                    thickness=0.01)
        sections = ndcrd.reshape(-1, 4, 3)
        surface = 0.05 * sections[:, 0, 0]
        np.testing.assert_allclose(sections[:, :2, 2],
                                   np.stack([surface + 0.03] * 2, axis=1))
        np.testing.assert_allclose(sections[:, 2:, 2],
                                   np.stack([surface + 0.02] * 2, axis=1))
        volumes = np.array([hex_volume(ndcrd, cell) for cell in clnds],
                           dtype='float64')
        assert (volumes > 0.0).all()


class TestBoxes:

    def test_scaled_turned_and_moved(self):
        ndcrd, clnds = boxes([(0.0, 0.0, 0.0), (10.0, 20.0, 3.0)],
                             [(1.0, 1.0, 1.0), (4.0, 2.0, 1.0)],
                             [0.0, 0.5 * math.pi])
        assert ndcrd.shape == (16, 3)
        np.testing.assert_array_equal(clnds, [range(8), range(8, 16)])
        np.testing.assert_allclose(ndcrd[:8], box()[0])
        # The corner (-2, -1, -0.5) of the 4 x 2 x 1 box turns a quarter
        # counterclockwise to (1, -2).
        np.testing.assert_allclose(ndcrd[8], (11.0, 18.0, 2.5), atol=1e-12)
        assert hex_volume(ndcrd, clnds[1]) == pytest.approx(8.0)

    def test_no_boxes_give_empty_arrays(self):
        ndcrd, clnds = boxes(np.empty((0, 3), dtype='float64'),
                             np.empty((0, 3), dtype='float64'),
                             np.empty(0, dtype='float64'))
        assert ndcrd.shape == (0, 3) and clnds.shape == (0, 8)


def arc_road(road_id='c', length=40.0, curvature=0.02, z=1.0):
    plan = [Arc(0.0, length, 5.0, -3.0, 0.3, curvature=curvature)]
    return Road(road_id, length=length,
                refline=RefLine(plan, PiecewisePoly3.constant(z)))


def assert_heading(actual, expected):
    np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=1e-12)
    np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=1e-12)


class TestPoses:

    def test_signals_on_a_straight_road(self):
        road = straight('s', 0.0, 0.0, 0.4, 50.0, z=2.0)
        road.signals = [Signal('a', road, 10.0, 3.0, z_offset=0.5),
                        Signal('b', road, 20.0, -3.0, h_offset=0.1,
                               orientation='-')]
        feet, heading = signal_poses([road])
        np.testing.assert_allclose(
            feet[:, :2], [road.refline.point(10.0, 3.0)[:2],
                          road.refline.point(20.0, -3.0)[:2]])
        np.testing.assert_allclose(feet[:, 2], [2.5, 2.0])
        assert_heading(heading, [0.4, 0.4 + math.pi + 0.1])

    def test_signals_against_an_arc(self):
        straight_road = straight('s', 0.0, 0.0, 0.0, 10.0)
        road = arc_road()
        road.signals = [Signal('a', road, 25.0, -2.0, h_offset=-0.2,
                               orientation='-')]
        feet, heading = signal_poses([straight_road, road])
        np.testing.assert_allclose(feet, [road.refline.point(25.0, -2.0)])
        # The arc turns left by its curvature times s from heading 0.3.
        assert_heading(heading, [0.3 + 0.02 * 25.0 + math.pi - 0.2])

    def test_no_signals_give_empty_poses(self):
        feet, heading = signal_poses([straight('s', 0.0, 0.0, 0.0, 10.0)])
        assert feet.shape == (0, 3) and heading.shape == (0,)

    def test_objects_of_one_type(self):
        road = arc_road()
        road.objects = [RoadObject('p', road, 5.0, 1.0, type='pole'),
                        RoadObject('w', road, 30.0, 0.5, z_offset=0.1,
                                   hdg=0.25, type='crosswalk')]
        objs, feet, heading = object_poses([road], 'crosswalk')
        assert [obj.id for obj in objs] == ['w']
        x, y, z = road.refline.point(30.0, 0.5)
        np.testing.assert_allclose(feet, [(x, y, z + 0.1)])
        assert_heading(heading, [0.3 + 0.02 * 30.0 + 0.25])


class TestCrosswalkBoxes:

    def test_a_flat_box_across_the_road(self):
        road = laned('a', 0.0, 40.0, 0.0, [4.5], [4.5])
        road.objects = [
            RoadObject('cw', road, 20.0, 0.0, hdg=0.0, type='crosswalk',
                       length=4.0, width=9.0),
            RoadObject('bare', road, 30.0, 0.0, type='crosswalk'),
            RoadObject('pole', road, 30.0, 0.0, type='pole', length=1.0,
                       width=1.0)]
        ndcrd, clnds = crosswalk_boxes([road], lift=0.03, thickness=0.02)
        assert clnds.shape == (1, 8)
        lo, hi = ndcrd.min(axis=0), ndcrd.max(axis=0)
        # Its length runs along the road and its width across it.
        np.testing.assert_allclose(lo, (18.0, -4.5, 0.01), atol=1e-12)
        np.testing.assert_allclose(hi, (22.0, 4.5, 0.03), atol=1e-12)
        assert hex_volume(ndcrd, clnds[0]) == pytest.approx(4 * 9 * 0.02)

    def test_the_heading_turns_the_box(self):
        road = straight('s', 0.0, 0.0, 0.5 * math.pi, 40.0)
        road.objects = [RoadObject('cw', road, 20.0, 1.0, type='crosswalk',
                                   length=4.0, width=9.0)]
        ndcrd, _ = crosswalk_boxes([road])
        lo, hi = ndcrd.min(axis=0), ndcrd.max(axis=0)
        # The road runs north, so the box is 9 m east to west at its left
        # offset of 1 m, which is to the west.
        np.testing.assert_allclose(lo[:2], (-5.5, 18.0), atol=1e-12)
        np.testing.assert_allclose(hi[:2], (3.5, 22.0), atol=1e-12)


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
