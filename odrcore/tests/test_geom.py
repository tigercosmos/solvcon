# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math

import numpy as np
import pytest

from odrcore.geom import (Arc, Line, ParamPoly3, PiecewisePoly3, Poly3,
                          RefLine, Spiral, fresnel)


def simpson(f, a, b, n=200001):
    t = np.linspace(a, b, n)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(np.sum(w * f(t)) * (t[1] - t[0]) / 3.0)


class TestPiecewisePoly3:

    def test_eval_and_derivative(self):
        p = PiecewisePoly3([(0.0, 1.0, 2.0, 0.0, 0.0),
                            (10.0, 5.0, 0.0, 1.0, 0.0)])
        assert p(0.0) == 1.0
        assert p(4.0) == 9.0
        assert p(12.0) == 9.0
        assert p.derivative(4.0) == 2.0
        assert p.derivative(12.0) == 4.0

    def test_empty_is_zero(self):
        p = PiecewisePoly3()
        assert p(3.0) == 0.0
        assert p.derivative(3.0) == 0.0

    def test_add_sub_exact(self):
        p = PiecewisePoly3([(0.0, 1.0, 0.5, 0.0, 0.0),
                            (10.0, 6.0, 0.0, 0.25, 0.0)])
        q = PiecewisePoly3([(0.0, 2.0, 0.0, 0.0, 0.001),
                            (5.0, 0.0, 1.0, 0.0, 0.0)])
        r = p + q
        assert r.breakpoints() == [0.0, 5.0, 10.0]
        for s in (0.0, 2.5, 5.0, 7.5, 10.0, 12.0, 20.0):
            assert r(s) == pytest.approx(p(s) + q(s), abs=1e-12)
            assert (p - q)(s) == pytest.approx(p(s) - q(s), abs=1e-12)
        back = r - q
        assert back.allclose(p)

    def test_shift_is_exact_for_cubic(self):
        p = PiecewisePoly3([(0.0, 1.0, -2.0, 3.0, -0.5)])
        a, b, c, d = p.at(7.0)
        q = PiecewisePoly3([(7.0, a, b, c, d)])
        for s in (-1.0, 0.0, 7.0, 11.0):
            assert q(s) == pytest.approx(p(s), rel=1e-13)

    def test_simplified_merges_continuations(self):
        p = PiecewisePoly3([(0.0, 1.0, 2.0, 0.0, 0.0)])
        split = PiecewisePoly3([(0.0, 1.0, 2.0, 0.0, 0.0),
                                (3.0, 7.0, 2.0, 0.0, 0.0),
                                (5.0, 11.0, 2.0, 0.0, 0.0)])
        assert split.simplified().breakpoints() == [0.0]
        assert split.allclose(p)

    def test_restrict(self):
        p = PiecewisePoly3([(0.0, 0.0, 1.0, 0.0, 0.0),
                            (4.0, 4.0, 2.0, 0.0, 0.0),
                            (20.0, 0.0, 0.0, 0.0, 0.0)])
        r = p.restrict(2.0, 10.0)
        assert r.breakpoints() == [2.0, 4.0]
        for s in (2.0, 3.0, 4.0, 9.0):
            assert r(s) == pytest.approx(p(s))


class TestFresnel:

    @pytest.mark.parametrize('x', [0.1, 0.5, 1.0, 1.6, 1.7, 2.0, 3.0, 5.0])
    def test_against_quadrature(self, x):
        c, s = fresnel(x)
        assert c == pytest.approx(
            simpson(lambda t: np.cos(0.5 * np.pi * t * t), 0.0, x),
            abs=1e-12)
        assert s == pytest.approx(
            simpson(lambda t: np.sin(0.5 * np.pi * t * t), 0.0, x),
            abs=1e-12)

    def test_known_values_and_oddness(self):
        c, s = fresnel(1.0)
        assert c == pytest.approx(0.7798934003768228, abs=1e-12)
        assert s == pytest.approx(0.4382591473903548, abs=1e-12)
        cn, sn = fresnel(-1.0)
        assert (cn, sn) == (-c, -s)


class TestPlanView:

    def test_line(self):
        g = Line(5.0, 10.0, 1.0, 2.0, math.pi / 2)
        x, y, h = g.pose(8.0)
        assert (x, y, h) == pytest.approx((1.0, 5.0, math.pi / 2))

    def test_arc_quarter_circle(self):
        r = 10.0
        g = Arc(0.0, 0.5 * math.pi * r, 0.0, 0.0, 0.0, curvature=1.0 / r)
        x, y, h = g.pose(g.length)
        assert (x, y, h) == pytest.approx((r, r, math.pi / 2))
        x, y, h = g.pose(0.0)
        assert (x, y, h) == pytest.approx((0.0, 0.0, 0.0))

    def test_arc_negative_curvature_turns_right(self):
        g = Arc(0.0, 1.0, 0.0, 0.0, 0.0, curvature=-0.1)
        x, y, h = g.pose(1.0)
        assert y < 0.0 and h == pytest.approx(-0.1)

    @pytest.mark.parametrize('k0,k1', [(0.0, 0.05), (0.01, -0.05),
                                       (-0.02, -0.08), (0.03, 0.0)])
    def test_spiral_against_quadrature(self, k0, k1):
        length = 40.0
        g = Spiral(0.0, length, 1.0, 2.0, 0.3, curv_start=k0, curv_end=k1)
        kdot = (k1 - k0) / length

        def hdg(t):
            return 0.3 + k0 * t + 0.5 * kdot * t * t

        for s in (0.0, 13.0, length):
            x, y, h = g.pose(s)
            assert h == pytest.approx(hdg(s), abs=1e-12)
            if s == 0.0:
                assert (x, y) == pytest.approx((1.0, 2.0))
                continue
            assert x == pytest.approx(
                1.0 + simpson(lambda t: np.cos(hdg(t)), 0.0, s), abs=1e-10)
            assert y == pytest.approx(
                2.0 + simpson(lambda t: np.sin(hdg(t)), 0.0, s), abs=1e-10)
            assert g.curvature(s) == pytest.approx(k0 + kdot * s)

    def test_spiral_degenerates_to_arc(self):
        g = Spiral(0.0, 5.0, 0.0, 0.0, 0.0, 0.1, 0.1)
        assert g.pose(5.0) == pytest.approx(
            Arc(0.0, 5.0, 0.0, 0.0, 0.0, 0.1).pose(5.0))

    def test_poly3_arc_length_inversion(self):
        g = Poly3(0.0, 0.0, 0.0, 0.0, 0.0, a=0.0, b=0.0, c=0.01, d=0.0)
        u_end = 20.0
        g.length = simpson(lambda u: np.sqrt(1.0 + (0.02 * u) ** 2),
                           0.0, u_end)
        x, y, h = g.pose(g.length)
        assert x == pytest.approx(u_end, abs=1e-9)
        assert y == pytest.approx(0.01 * u_end ** 2, abs=1e-9)
        assert h == pytest.approx(math.atan(0.02 * u_end), abs=1e-9)

    def test_param_poly3_arclength_is_identity(self):
        g = ParamPoly3(0.0, 10.0, 0.0, 0.0, 0.0, b_u=1.0, b_v=0.0, c_v=0.01)
        x, y, h = g.pose(4.0)
        assert (x, y) == pytest.approx((4.0, 0.16))
        assert h == pytest.approx(math.atan2(0.08, 1.0))

    def test_param_poly3_normalized(self):
        g = ParamPoly3(0.0, 0.0, 0.0, 0.0, 0.0, b_u=10.0, c_v=2.0,
                       p_range='normalized')
        g.length = simpson(lambda p: np.hypot(10.0, 4.0 * p), 0.0, 1.0)
        assert g.param(g.length) == pytest.approx(1.0, abs=1e-9)
        x, y, _ = g.pose(g.length)
        assert (x, y) == pytest.approx((10.0, 2.0), abs=1e-9)


class TestRefLine:

    def make(self):
        plan = [Line(0.0, 10.0, 0.0, 0.0, 0.0),
                Arc(10.0, 5.0 * math.pi, 10.0, 0.0, 0.0, curvature=0.1)]
        elev = PiecewisePoly3([(0.0, 1.0, 0.1, 0.0, 0.0)])
        sup = PiecewisePoly3([(0.0, 0.0, 0.0, 0.0, 0.0),
                              (10.0, 0.2, 0.0, 0.0, 0.0)])
        return RefLine(plan, elev, sup)

    def test_pose_picks_segment(self):
        rl = self.make()
        assert rl.length == pytest.approx(10.0 + 5.0 * math.pi)
        x, y, z, h, roll = rl.pose(5.0)
        assert (x, y, z, h, roll) == pytest.approx((5.0, 0.0, 1.5, 0.0, 0.0))
        x, y, z, h, roll = rl.pose(10.0 + 5.0 * math.pi)
        assert (x, y, h) == pytest.approx((20.0, 10.0, 0.5 * math.pi))
        assert roll == 0.2

    def test_point_applies_superelevation(self):
        rl = self.make()
        t = 3.0
        x, y, z = rl.point(5.0, t)
        assert (x, y, z) == pytest.approx((5.0, t, 1.5))
        s = 10.0 + 5.0 * math.pi
        x, y, z = rl.point(s, t)
        assert x == pytest.approx(20.0 - t * math.cos(0.2))
        assert y == pytest.approx(10.0)
        assert z == pytest.approx(1.0 + 0.1 * s + t * math.sin(0.2))
        x2, y2, z2 = rl.point(s, -t)
        assert z2 == pytest.approx(1.0 + 0.1 * s - t * math.sin(0.2))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
