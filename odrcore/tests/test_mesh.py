# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math

import numpy as np
import pytest

from odrcore.geom import Line, PiecewisePoly3, RefLine
from odrcore.mesh import box, ribbon, to_static_mesh
from odrcore.model import Road


def straight(road_id, x0, y0, hdg, length, z=0.0):
    plan = [Line(0.0, length, x0, y0, hdg)]
    return Road(road_id, length=length,
                refline=RefLine(plan, PiecewisePoly3.constant(z)))


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

    def test_ribbon_builds_a_hex_mesh(self):
        solvcon = pytest.importorskip('solvcon')
        mh = to_static_mesh(*ribbon(square_loop(), spacing=5.0))
        assert mh.ncell == 16 and mh.nnode == 64
        assert (mh.cltpn.ndarray == solvcon.StaticMesh.HEXAHEDRON).all()
        body = mh.clvol.ndarray[mh.clvol.nghost:]
        np.testing.assert_allclose(body[CORNERS == 0], 7.5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
