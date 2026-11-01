# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Analytic geometry for the OpenDRIVE reference line.

Nothing here samples a curve. Every primitive answers a pose at an exact
arc length, and the piecewise cubic keeps its coefficients so that sums and
differences of two cubics stay cubics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Poly3Seg = tuple[float, float, float, float, float]
"""One cubic segment ``(s_start, a, b, c, d)`` evaluated in ``s - s_start``."""


def _shift_poly(a: float, b: float, c: float, d: float,
                ds: float) -> tuple[float, float, float, float]:
    """Re-anchor ``a + b x + c x^2 + d x^3`` at ``x = ds`` (Taylor shift)."""
    a2 = a + ds * (b + ds * (c + ds * d))
    b2 = b + ds * (2.0 * c + 3.0 * d * ds)
    c2 = c + 3.0 * d * ds
    return a2, b2, c2, d


@dataclass
class PiecewisePoly3:
    """A piecewise cubic along s, as OpenDRIVE writes offsets and profiles.

    Segments are sorted by ``s_start``. The first segment also covers any s
    before it, and the last one extends to infinity, so a single segment is
    a plain cubic on the whole axis. An empty instance is identically zero.
    """

    segs: list[Poly3Seg] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.segs = sorted((tuple(float(v) for v in seg)
                            for seg in self.segs), key=lambda seg: seg[0])

    @classmethod
    def constant(cls, value: float, s_start: float = 0.0) -> PiecewisePoly3:
        return cls([(s_start, value, 0.0, 0.0, 0.0)])

    def _seg(self, s: float) -> Poly3Seg:
        if not self.segs:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        starts = [seg[0] for seg in self.segs]
        i = int(np.searchsorted(starts, s, side='right')) - 1
        return self.segs[max(i, 0)]

    def __call__(self, s: float) -> float:
        s0, a, b, c, d = self._seg(s)
        x = s - s0
        return a + x * (b + x * (c + x * d))

    def derivative(self, s: float) -> float:
        s0, _, b, c, d = self._seg(s)
        x = s - s0
        return b + x * (2.0 * c + 3.0 * x * d)

    def breakpoints(self) -> list[float]:
        return [seg[0] for seg in self.segs]

    def at(self, s_start: float) -> tuple[float, float, float, float]:
        """Coefficients of the cubic active at ``s_start``, anchored there."""
        s0, a, b, c, d = self._seg(s_start)
        return _shift_poly(a, b, c, d, s_start - s0)

    def restrict(self, s_lo: float, s_hi: float) -> PiecewisePoly3:
        """Segments covering ``[s_lo, s_hi)``, re-anchored to start at s_lo."""
        out = [(s_lo,) + self.at(s_lo)]
        out += [seg for seg in self.segs if s_lo < seg[0] < s_hi]
        return PiecewisePoly3(out)

    def shifted(self, ds: float) -> PiecewisePoly3:
        """The same function with every breakpoint moved by ``ds``."""
        return PiecewisePoly3([(s0 + ds, a, b, c, d)
                               for s0, a, b, c, d in self.segs])

    def _combine(self, other: PiecewisePoly3, sign: float) -> PiecewisePoly3:
        if not self.segs:
            return PiecewisePoly3([(s0, sign * a, sign * b, sign * c,
                                    sign * d)
                                   for s0, a, b, c, d in other.segs])
        if not other.segs:
            return PiecewisePoly3(list(self.segs))
        starts = sorted(set(self.breakpoints()) | set(other.breakpoints()))
        out = []
        for s0 in starts:
            p = self.at(s0)
            q = other.at(s0)
            out.append((s0,) + tuple(pi + sign * qi for pi, qi in zip(p, q)))
        return PiecewisePoly3(out)

    def __add__(self, other: PiecewisePoly3) -> PiecewisePoly3:
        return self._combine(other, 1.0)

    def __sub__(self, other: PiecewisePoly3) -> PiecewisePoly3:
        return self._combine(other, -1.0)

    def __neg__(self) -> PiecewisePoly3:
        return PiecewisePoly3() - self

    def simplified(self, tol: float = 1e-9) -> PiecewisePoly3:
        """Drop a breakpoint when the previous cubic already continues."""
        if len(self.segs) < 2:
            return PiecewisePoly3(list(self.segs))
        out = [self.segs[0]]
        for seg in self.segs[1:]:
            s0, a, b, c, d = seg
            p0, pa, pb, pc, pd = out[-1]
            cont = _shift_poly(pa, pb, pc, pd, s0 - p0)
            if all(abs(u - v) <= tol * max(1.0, abs(u))
                   for u, v in zip(cont, (a, b, c, d))):
                continue
            out.append(seg)
        return PiecewisePoly3(out)

    def allclose(self, other: PiecewisePoly3, tol: float = 1e-9) -> bool:
        a, b = self.simplified(tol).segs, other.simplified(tol).segs
        if len(a) != len(b):
            return False
        return all(abs(u - v) <= tol * max(1.0, abs(u))
                   for sa, sb in zip(a, b) for u, v in zip(sa, sb))


# Fresnel integrals C(x) = int_0^x cos(pi t^2 / 2) dt and S(x) likewise.
# The power series is exact enough below 1.6; above it Gauss-Legendre panels
# short enough for the local oscillation take over.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(20)


def _fresnel_series(x: float) -> tuple[float, float]:
    z = 0.5 * math.pi * x * x
    z2 = z * z
    c_sum, s_sum = 0.0, 0.0
    # term_c = (-1)^n z^(2n) x / (2n)!, term_s = (-1)^n z^(2n+1) x / (2n+1)!
    term_c, term_s = x, x * z
    n = 0
    while True:
        c_sum += term_c / (4 * n + 1)
        s_sum += term_s / (4 * n + 3)
        n += 1
        term_c *= -z2 / ((2 * n - 1) * (2 * n))
        term_s *= -z2 / ((2 * n) * (2 * n + 1))
        if (abs(term_c) < 1e-17 and abs(term_s) < 1e-17) or n > 60:
            return c_sum, s_sum


def _fresnel_quad(x: float) -> tuple[float, float]:
    ax = abs(x)
    panels = int(math.ceil(ax * max(2.0, ax)))
    edges = np.linspace(0.0, ax, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    t = mid[:, None] + half[:, None] * _GL_NODES[None, :]
    phase = 0.5 * math.pi * t * t
    w = half[:, None] * _GL_WEIGHTS[None, :]
    c = float(np.sum(w * np.cos(phase)))
    s = float(np.sum(w * np.sin(phase)))
    return math.copysign(c, x), math.copysign(s, x)


def fresnel(x: float) -> tuple[float, float]:
    """Return ``(C(x), S(x))``; both are odd functions of x."""
    if abs(x) <= 1.6:
        return _fresnel_series(x)
    return _fresnel_quad(x)


Pose2 = tuple[float, float, float]
"""``(x, y, hdg)``."""


@dataclass
class PlanGeom:
    """One ``<geometry>`` record. ``pose`` takes the road s, not a local s."""

    s0: float
    length: float
    x0: float
    y0: float
    hdg0: float

    @property
    def s1(self) -> float:
        return self.s0 + self.length

    def _local(self, s: float) -> float:
        return s - self.s0

    def _place(self, u: float, v: float, dhdg: float) -> Pose2:
        ch, sh = math.cos(self.hdg0), math.sin(self.hdg0)
        return (self.x0 + u * ch - v * sh, self.y0 + u * sh + v * ch,
                self.hdg0 + dhdg)

    def pose(self, s: float) -> Pose2:
        raise NotImplementedError


@dataclass
class Line(PlanGeom):

    def pose(self, s: float) -> Pose2:
        return self._place(self._local(s), 0.0, 0.0)


@dataclass
class Arc(PlanGeom):
    curvature: float = 0.0

    def pose(self, s: float) -> Pose2:
        ds = self._local(s)
        k = self.curvature
        if k == 0.0:
            return self._place(ds, 0.0, 0.0)
        th = k * ds
        return self._place(math.sin(th) / k, (1.0 - math.cos(th)) / k, th)


@dataclass
class Spiral(PlanGeom):
    """Clothoid with curvature linear in s between ``curv_start`` and
    ``curv_end``. Position comes from the Fresnel integrals in closed form:
    the curvature is zero at ``-k0 / kdot`` and the two bounds are scaled to
    the Fresnel argument from there."""

    curv_start: float = 0.0
    curv_end: float = 0.0

    @property
    def curv_dot(self) -> float:
        return (self.curv_end - self.curv_start) / self.length

    def curvature(self, s: float) -> float:
        return self.curv_start + self.curv_dot * self._local(s)

    def pose(self, s: float) -> Pose2:
        ds = self._local(s)
        k0, a = self.curv_start, self.curv_dot
        dhdg = k0 * ds + 0.5 * a * ds * ds
        if a == 0.0:
            return Arc(self.s0, self.length, self.x0, self.y0, self.hdg0,
                       k0).pose(s)
        scale = math.sqrt(math.pi / abs(a))
        sgn = 1.0 if a > 0.0 else -1.0
        x_lo = (k0 / a) / scale
        x_hi = (ds + k0 / a) / scale
        c0, s0 = fresnel(x_lo)
        c1, s1 = fresnel(x_hi)
        dc, dsn = c1 - c0, s1 - s0
        phi = k0 * k0 / (2.0 * a)
        cphi, sphi = math.cos(phi), math.sin(phi)
        u = scale * (dc * cphi + sgn * dsn * sphi)
        v = scale * (sgn * dsn * cphi - dc * sphi)
        return self._place(u, v, dhdg)


Deriv = Callable[[np.ndarray], np.ndarray]


def _arc_length_to_param(du: Deriv, dv: Deriv, length: float, s: float,
                         p_hi: float) -> float:
    """Invert ``s(p) = int_0^p sqrt(du^2 + dv^2)`` by Newton on Gauss
    panels. ``p_hi`` is the parameter value expected at ``length``."""
    def arclen(p: float) -> float:
        panels = max(1, int(math.ceil(abs(p) * 4.0 / max(p_hi, 1e-300))))
        edges = np.linspace(0.0, p, panels + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        mid = 0.5 * (edges[1:] + edges[:-1])
        t = mid[:, None] + half[:, None] * _GL_NODES[None, :]
        w = half[:, None] * _GL_WEIGHTS[None, :]
        return float(np.sum(w * np.hypot(du(t), dv(t))))

    p = s / length * p_hi if length > 0.0 else 0.0
    for _ in range(50):
        f = arclen(p) - s
        g = math.hypot(float(du(np.float64(p))), float(dv(np.float64(p))))
        if g == 0.0:
            break
        step = f / g
        p -= step
        if abs(step) < 1e-14 * max(1.0, abs(p)):
            break
    return p


@dataclass
class Poly3(PlanGeom):
    """Cubic ``v = a + b u + c u^2 + d u^3`` in the local frame, where u runs
    along the start heading. The road s is the arc length, so u(s) is found
    by inverting the arc-length integral."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    def _dv(self, u):
        return self.b + u * (2.0 * self.c + 3.0 * self.d * u)

    def pose(self, s: float) -> Pose2:
        u = _arc_length_to_param(np.ones_like, self._dv, self.length,
                                 self._local(s), self.length)
        v = self.a + u * (self.b + u * (self.c + u * self.d))
        return self._place(u, v, math.atan2(float(self._dv(u)), 1.0))


@dataclass
class ParamPoly3(PlanGeom):
    """Parametric cubic. With ``p_range="arcLength"`` the parameter is the
    local s itself. With ``"normalized"`` p runs over [0, 1] and is found
    from the arc length by inversion."""

    a_u: float = 0.0
    b_u: float = 0.0
    c_u: float = 0.0
    d_u: float = 0.0
    a_v: float = 0.0
    b_v: float = 0.0
    c_v: float = 0.0
    d_v: float = 0.0
    p_range: str = 'arcLength'

    def _du(self, p):
        return self.b_u + p * (2.0 * self.c_u + 3.0 * self.d_u * p)

    def _dv(self, p):
        return self.b_v + p * (2.0 * self.c_v + 3.0 * self.d_v * p)

    def param(self, s: float) -> float:
        ds = self._local(s)
        if self.p_range == 'arcLength':
            return ds
        return _arc_length_to_param(self._du, self._dv, self.length, ds, 1.0)

    def pose(self, s: float) -> Pose2:
        p = self.param(s)
        u = self.a_u + p * (self.b_u + p * (self.c_u + p * self.d_u))
        v = self.a_v + p * (self.b_v + p * (self.c_v + p * self.d_v))
        return self._place(u, v, math.atan2(float(self._dv(p)),
                                            float(self._du(p))))


Pose3 = tuple[float, float, float, float, float]
"""``(x, y, z, hdg, roll)``."""


@dataclass
class RefLine:
    """The road reference line: plan view plus elevation and superelevation.

    Positive superelevation rolls the cross section about the s axis so the
    left side (positive t) rises, following the right-hand rule.
    """

    plan: list[PlanGeom] = field(default_factory=list)
    elevation: PiecewisePoly3 = field(default_factory=PiecewisePoly3)
    superelevation: PiecewisePoly3 = field(default_factory=PiecewisePoly3)

    @property
    def length(self) -> float:
        return self.plan[-1].s1 if self.plan else 0.0

    def geom_at(self, s: float) -> PlanGeom:
        if not self.plan:
            raise ValueError('reference line has no plan view')
        starts = [g.s0 for g in self.plan]
        i = int(np.searchsorted(starts, s, side='right')) - 1
        return self.plan[max(i, 0)]

    def pose(self, s: float) -> Pose3:
        x, y, hdg = self.geom_at(s).pose(s)
        return x, y, self.elevation(s), hdg, self.superelevation(s)

    def point(self, s: float, t: float) -> tuple[float, float, float]:
        x, y, z, hdg, roll = self.pose(s)
        lateral = t * math.cos(roll)
        return (x - lateral * math.sin(hdg), y + lateral * math.cos(hdg),
                z + t * math.sin(roll))


# Batched evaluation: many functions or lines at many stations in one pass,
# for meshing a whole map. Each station names its owner by index; the
# results equal the scalar methods to rounding.

def locate(starts: Sequence[Sequence[float]], owner: np.ndarray,
           s: np.ndarray) -> np.ndarray:
    """Return, per station, the flat index into the concatenated ``starts``
    of the last start of its owner at or before ``s``, or the owner's first
    start before it, as ``PiecewisePoly3`` and ``RefLine`` pick a piece."""
    counts = np.array([len(st) for st in starts], dtype='int64')
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nflat = int(counts.sum())
    flat = np.fromiter((v for st in starts for v in st), dtype='float64',
                       count=nflat)
    index = np.arange(len(starts), dtype='int64')
    owners = np.concatenate([np.repeat(index, counts), owner])
    keys = np.concatenate([flat, s])
    # A start ties ahead of a station at the same s, so a station on a
    # breakpoint takes the piece that begins there.
    kind = np.concatenate([np.zeros(nflat, dtype='int8'),
                           np.ones(len(s), dtype='int8')])
    order = np.lexsort((kind, keys, owners))
    seen = np.cumsum(order < nflat) - 1
    found = np.empty(len(s), dtype='int64')
    at = order >= nflat
    found[order[at] - nflat] = seen[at]
    return np.maximum(found, first[owner])


def poly3_values(polys: Sequence[PiecewisePoly3], owner: np.ndarray,
                 s: np.ndarray) -> np.ndarray:
    """Return ``polys[owner[i]](s[i])`` for every station i."""
    segs = [p.segs or [(0.0, 0.0, 0.0, 0.0, 0.0)] for p in polys]
    table = np.array([seg for ps in segs for seg in ps], dtype='float64')
    s0, a, b, c, d = table[locate([[seg[0] for seg in ps] for ps in segs],
                                  owner, s)].T
    x = s - s0
    return a + x * (b + x * (c + x * d))


_PANELS = 4


def _cubic_deriv(b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Deriv:
    """``b + 2 c p + 3 d p^2`` with per-station coefficients, for ``p`` of
    shape ``(n, ...)``."""
    def deriv(p: np.ndarray) -> np.ndarray:
        shape = b.shape + (1,) * (p.ndim - 1)
        bb, cc, dd = b.reshape(shape), c.reshape(shape), d.reshape(shape)
        return bb + p * (2.0 * cc + 3.0 * dd * p)
    return deriv


def _arc_lengths_to_params(du: Deriv, dv: Deriv, length: np.ndarray,
                           s: np.ndarray, p_hi: np.ndarray) -> np.ndarray:
    """``_arc_length_to_param`` for one station of its own curve per entry.
    The quadrature runs on :data:`_PANELS` fixed panels over ``[0, p_hi]``
    and a partial one up to p, so every station shares one Newton loop."""
    def integral(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        half = 0.5 * (hi - lo)
        t = (0.5 * (hi + lo))[..., None] + half[..., None] * _GL_NODES
        return np.sum(_GL_WEIGHTS * np.hypot(du(t), dv(t)), axis=-1) * half

    edges = p_hi[:, None] * np.arange(_PANELS + 1, dtype='float64') / _PANELS
    cum = np.zeros_like(edges)
    cum[:, 1:] = np.cumsum(integral(edges[:, :-1], edges[:, 1:]), axis=1)
    rows = np.arange(len(s), dtype='int64')
    p = np.where(length > 0.0, s / np.where(length > 0.0, length, 1.0) * p_hi,
                 0.0)
    span = np.maximum(p_hi, 1e-300)
    for _ in range(50):
        k = np.clip(np.floor(p / span * _PANELS), 0, _PANELS - 1)
        k = k.astype('int64')
        f = cum[rows, k] + integral(edges[rows, k], p) - s
        g = np.hypot(du(p), dv(p))
        step = np.divide(f, g, out=np.zeros_like(f), where=g != 0.0)
        p = p - step
        if (np.abs(step) < 1e-14 * np.maximum(1.0, np.abs(p))).all():
            break
    return p


def _plan_local(geoms: Sequence[PlanGeom],
                ds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, v, dhdg)`` in the start frame of ``geoms[i]`` at its
    local arc length ``ds[i]``; every geometry has one type."""
    kind = type(geoms[0])

    def attr(name: str) -> np.ndarray:
        return np.array([getattr(g, name) for g in geoms], dtype='float64')

    zero = np.zeros_like(ds)
    if kind is Line:
        return ds, zero, zero
    if kind is Arc:
        k = attr('curvature')
        straight = k == 0.0
        kk = np.where(straight, 1.0, k)
        th = k * ds
        return (np.where(straight, ds, np.sin(th) / kk),
                np.where(straight, 0.0, (1.0 - np.cos(th)) / kk), th)
    if kind is Poly3:
        length = attr('length')
        b, c, d = attr('b'), attr('c'), attr('d')
        dv = _cubic_deriv(b, c, d)
        u = _arc_lengths_to_params(np.ones_like, dv, length, ds, length)
        v = attr('a') + u * (b + u * (c + u * d))
        return u, v, np.arctan2(dv(u), 1.0)
    if kind is ParamPoly3:
        bu, cu, du_ = attr('b_u'), attr('c_u'), attr('d_u')
        bv, cv, dv_ = attr('b_v'), attr('c_v'), attr('d_v')
        du, dv = _cubic_deriv(bu, cu, du_), _cubic_deriv(bv, cv, dv_)
        p = ds.copy()
        norm = np.array([g.p_range != 'arcLength' for g in geoms],
                        dtype='bool')
        if norm.any():
            sub = np.flatnonzero(norm)
            dus = _cubic_deriv(bu[sub], cu[sub], du_[sub])
            dvs = _cubic_deriv(bv[sub], cv[sub], dv_[sub])
            p[sub] = _arc_lengths_to_params(dus, dvs, attr('length')[sub],
                                            ds[sub],
                                            np.ones(len(sub), dtype='float64'))
        u = attr('a_u') + p * (bu + p * (cu + p * du_))
        v = attr('a_v') + p * (bv + p * (cv + p * dv_))
        return u, v, np.arctan2(dv(p), du(p))
    out = np.array([g.pose(g.s0 + d) for g, d in zip(geoms, ds)],
                   dtype='float64').reshape(-1, 3)
    hdg0 = attr('hdg0')
    ch, sh = np.cos(hdg0), np.sin(hdg0)
    dx, dy = out[:, 0] - attr('x0'), out[:, 1] - attr('y0')
    return dx * ch + dy * sh, dy * ch - dx * sh, out[:, 2] - hdg0


def refline_points(lines: Sequence[RefLine], owner: np.ndarray,
                   s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return ``lines[owner[i]].point(s[i], t[i, ...])`` for every station
    i as an array of shape ``t.shape + (3,)``. ``t`` may carry trailing
    axes for several offsets at one station, which share one pose."""
    owner = np.asarray(owner, dtype='int64')
    s = np.asarray(s, dtype='float64')
    planless = np.array([not line.plan for line in lines], dtype='bool')
    if planless[owner].any():
        raise ValueError('reference line has no plan view')
    flat = [g for line in lines for g in line.plan]
    gi = locate([[g.s0 for g in line.plan] for line in lines], owner, s)
    u, v, dh = (np.empty_like(s) for _ in range(3))
    kinds = np.array([id(type(g)) for g in flat], dtype='int64')[gi]
    for kind in np.unique(kinds):
        sub = np.flatnonzero(kinds == kind)
        geoms = [flat[i] for i in gi[sub]]
        s0 = np.array([g.s0 for g in geoms], dtype='float64')
        u[sub], v[sub], dh[sub] = _plan_local(geoms, s[sub] - s0)

    def attr(name: str) -> np.ndarray:
        return np.array([getattr(g, name) for g in flat],
                        dtype='float64')[gi]

    hdg0 = attr('hdg0')
    ch, sh = np.cos(hdg0), np.sin(hdg0)
    x = attr('x0') + u * ch - v * sh
    y = attr('y0') + u * sh + v * ch
    hdg = hdg0 + dh
    z = poly3_values([line.elevation for line in lines], owner, s)
    roll = poly3_values([line.superelevation for line in lines], owner, s)

    t = np.asarray(t, dtype='float64')
    shape = s.shape + (1,) * (t.ndim - 1)
    x, y, z, hdg, roll = (a.reshape(shape) for a in (x, y, z, hdg, roll))
    lateral = t * np.cos(roll)
    return np.stack([x - lateral * np.sin(hdg), y + lateral * np.cos(hdg),
                     z + t * np.sin(roll)], axis=-1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
