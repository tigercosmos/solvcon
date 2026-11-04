# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Analytic geometry of the OpenDRIVE reference line: the shapes only.

Nothing in the model samples a curve. Every primitive answers a pose at an
exact road s, and the piecewise cubic keeps its coefficients so that sums
and differences of cubics stay cubics. The numerical methods (arc-length
inversion, Fresnel integrals, Taylor re-anchoring) are omitted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Pose2 = tuple[float, float, float]
"""``(x, y, hdg)``."""

Pose3 = tuple[float, float, float, float, float]
"""``(x, y, z, hdg, roll)``."""

Poly3Seg = tuple[float, float, float, float, float]
"""One cubic segment ``(s_start, a, b, c, d)`` evaluated in ``s - s_start``."""


@dataclass
class PiecewisePoly3:
    """A piecewise cubic along s, as OpenDRIVE writes offsets and profiles.

    Segments are sorted by ``s_start``. The first segment also covers any s
    before it and the last one extends to infinity, so a single segment is
    a plain cubic on the whole axis. An empty instance is identically zero.

    Supports ``__call__(s)``, ``derivative(s)``, ``__add__``, ``__sub__``,
    ``restrict(s_lo, s_hi)``, ``shifted(ds)`` and ``simplified()``; all are
    exact cubic algebra, never sampling.
    """

    segs: list[Poly3Seg] = field(default_factory=list)


@dataclass
class PlanGeom:
    """One ``<planView><geometry>`` record. ``pose(s)`` takes road s."""

    s0: float
    length: float
    x0: float
    y0: float
    hdg0: float

    def pose(self, s: float) -> Pose2:
        raise NotImplementedError


@dataclass
class Line(PlanGeom):
    pass


@dataclass
class Arc(PlanGeom):
    curvature: float = 0.0


@dataclass
class Spiral(PlanGeom):
    """Clothoid; curvature is linear in s between the two end values."""

    curv_start: float = 0.0
    curv_end: float = 0.0


@dataclass
class Poly3(PlanGeom):
    """``v = a + b u + c u^2 + d u^3`` in the local (u, v) frame. The arc
    length must be inverted to find u for a given s."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0


@dataclass
class ParamPoly3(PlanGeom):
    """``u(p)``, ``v(p)`` cubics. ``p_range`` is ``arcLength`` or
    ``normalized``; the latter also needs arc-length inversion."""

    a_u: float = 0.0
    b_u: float = 0.0
    c_u: float = 0.0
    d_u: float = 0.0
    a_v: float = 0.0
    b_v: float = 0.0
    c_v: float = 0.0
    d_v: float = 0.0
    p_range: str = 'arcLength'


@dataclass
class RefLine:
    """The road reference line: plan view plus elevation and superelevation.

    Positive superelevation rolls the cross section about the s axis so the
    left side (positive t) rises, following the right-hand rule.

    ``pose(s) -> Pose3`` and ``point(s, t) -> (x, y, z)`` are evaluated
    analytically.
    """

    plan: list[PlanGeom] = field(default_factory=list)
    elevation: PiecewisePoly3 = field(default_factory=PiecewisePoly3)
    superelevation: PiecewisePoly3 = field(default_factory=PiecewisePoly3)

    @property
    def length(self) -> float:
        return self.plan[-1].s0 + self.plan[-1].length if self.plan else 0.0

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
