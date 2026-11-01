# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Hexahedral slabs along odrcore reference lines, for drawing a map as solid
geometry in the pilot 3D viewer.

A 3D viewer that shades the boundary shell of a volume mesh cannot take a
bare line or a bare triangle strip, so a track is built as a thin slab: the
reference line is sampled along s, each sample gives a cross section of
four nodes (the left and right edges at the road height and ``thickness``
below it), and consecutive sections are joined into one hexahedron.

``ribbon`` returns plain arrays and ``to_static_mesh`` turns them into a
``solvcon.StaticMesh``. ``road_ribbons`` draws a road network instead: every
road an open slab across its carriageway, sampled for all roads at once, and
``concat`` joins such arrays into one mesh.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from odrcore.geom import locate, poly3_values, refline_points
from odrcore.model import Road

from .core import StaticMesh

Arrays = tuple[np.ndarray, np.ndarray]


def _sections(road: Road, spacing: float, half_width: float,
              thickness: float, with_end: bool) -> np.ndarray:
    """Return the ``(n, 4, 3)`` cross sections along the road, ordered left
    top, right top, right bottom, left bottom."""
    n = max(int(road.length // spacing), 1)
    stations = list(np.linspace(0.0, road.length, n, endpoint=False))
    if with_end:
        stations.append(road.length)
    out = np.empty((len(stations), 4, 3), dtype='float64')
    for k, s in enumerate(stations):
        left = road.refline.point(s, half_width)
        right = road.refline.point(s, -half_width)
        out[k, 0], out[k, 1] = left, right
        out[k, 2] = right[0], right[1], right[2] - thickness
        out[k, 3] = left[0], left[1], left[2] - thickness
    return out


def ribbon(roads: Sequence[Road], spacing: float = 5.0,
           half_width: float = 1.5, thickness: float = 0.5,
           closed: bool = True) -> Arrays:
    """Return the node coordinates ``(nnode, 3)`` and the hexahedron
    connectivity ``(ncell, 8)`` of a slab along ``roads``, which run end to
    start in the given order. A closed chain joins the last section back to
    the first; an open one ends with a section at the last road's end."""
    parts = [_sections(r, spacing, half_width, thickness, False)
             for r in roads]
    if not closed:
        parts.append(_sections(roads[-1], spacing, half_width, thickness,
                               True)[-1:])
    sections = np.concatenate(parts)
    nsec = len(sections)
    ncell = nsec if closed else nsec - 1
    cells = np.arange(ncell, dtype='int64')
    return sections.reshape(-1, 3), _hexes(cells, nsec)


def _hexes(first: np.ndarray, nsec: int) -> np.ndarray:
    """Return the hexahedra that join each section in ``first`` to the next
    one, the last wrapping to section 0, out of ``nsec`` sections."""
    base = np.asarray(first, dtype='int32') * 4
    nxt = (base + 4) % (nsec * 4)
    # The bottom face runs left to right then forward, so its normal points
    # up into the cell, as the top face follows it.
    return np.stack([base + 3, base + 2, nxt + 2, nxt + 3,
                     base + 0, base + 1, nxt + 1, nxt + 0], axis=1)


def carriageway(roads: Sequence[Road], owner: np.ndarray,
                s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest and the highest lateral offset ``t`` of the lane
    boundaries of ``roads[owner[i]]`` at ``s[i]``: its right and left edge.
    A road without lanes has no width: both are 0, its reference line."""
    lo, hi = np.zeros_like(s), np.zeros_like(s)
    laned = [k for k, road in enumerate(roads) if road.sections]
    renum = np.full(len(roads), -1, dtype='int64')
    renum[laned] = np.arange(len(laned), dtype='int64')
    station = np.flatnonzero(renum[owner] >= 0)
    if not len(station):
        return lo, hi

    secs = [roads[k].sections for k in laned]
    si = locate([[sec.s0 for sec in ss] for ss in secs],
                renum[owner[station]], s[station])
    bnds = [[b.t for b in sec.boundaries()] if sec.center is not None
            else [] for ss in secs for sec in ss]
    counts = np.array([len(bs) for bs in bnds], dtype='int64')
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    keep = counts[si] > 0
    station, si = station[keep], si[keep]
    if not len(station):
        return lo, hi

    nper = counts[si]
    pair = np.repeat(np.arange(len(station), dtype='int64'), nper)
    start = np.concatenate([[0], np.cumsum(nper)[:-1]])
    within = np.arange(len(pair), dtype='int64') - np.repeat(start, nper)
    flat = [t for bs in bnds for t in bs]
    t = poly3_values(flat, first[si[pair]] + within, s[station[pair]])
    lo[station] = np.minimum.reduceat(t, start)
    hi[station] = np.maximum.reduceat(t, start)
    return lo, hi


def road_ribbons(roads: Sequence[Road], spacing: float = 10.0,
                 half_width: Optional[float] = None,
                 thickness: float = 0.3) -> Arrays:
    """Return the node and hexahedron arrays of an open slab along every
    road of ``roads``, one after the other: across the carriageway from
    :func:`carriageway`, or ``half_width`` to either side of its middle,
    with the top on the road surface and ``thickness`` below it. A road
    without lane boundaries has no carriageway and is left out, unless
    ``half_width`` is given to center its slab on the reference line."""
    if half_width is None:
        roads = [road for road in roads
                 if any(sec.center is not None for sec in road.sections)]
    if not roads:
        return (np.empty((0, 3), dtype='float64'),
                np.empty((0, 8), dtype='int32'))
    length = np.array([road.length for road in roads], dtype='float64')
    nseg = np.maximum((length // spacing).astype('int64'), 1)
    owner = np.repeat(np.arange(len(roads), dtype='int64'), nseg + 1)
    start = np.concatenate([[0], np.cumsum(nseg + 1)[:-1]])
    k = np.arange(len(owner), dtype='int64') - start[owner]
    s = np.where(k == nseg[owner], length[owner],
                 k * (length / nseg)[owner])

    lo, hi = carriageway(roads, owner, s)
    if half_width is not None:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - half_width, mid + half_width
    top = refline_points([road.refline for road in roads], owner, s,
                         np.stack([hi, lo], axis=1))
    sections = np.concatenate([top, top[:, ::-1]], axis=1)
    sections[:, 2:, 2] -= thickness

    last = start + nseg
    first = np.setdiff1d(np.arange(len(s), dtype='int64'), last,
                         assume_unique=True)
    return sections.reshape(-1, 3), _hexes(first, len(s))


def road_ribbon(road: Road, spacing: float = 10.0,
                half_width: Optional[float] = None,
                thickness: float = 0.3) -> Arrays:
    """Return the node and hexahedron arrays of an open slab along one
    road, as :func:`road_ribbons` draws it."""
    return road_ribbons([road], spacing, half_width, thickness)


def concat(parts: Sequence[Arrays]) -> Arrays:
    """Join several ``(ndcrd, clnds)`` pairs into one, renumbering the
    connectivity of each part past the nodes of the parts before it."""
    parts = list(parts)
    if not parts:
        return (np.empty((0, 3), dtype='float64'),
                np.empty((0, 8), dtype='int32'))
    sizes = [len(ndcrd) for ndcrd, _ in parts]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ndcrd = np.concatenate([nd for nd, _ in parts])
    clnds = np.concatenate([cl + off for (_, cl), off in zip(parts, offsets)])
    return ndcrd, clnds.astype('int32')


def box() -> Arrays:
    """Return a unit cube centered at the origin, one hexahedron."""
    ndcrd = np.array([(x, y, z) for z in (-0.5, 0.5)
                      for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5),
                                   (-0.5, 0.5))], dtype='float64')
    return ndcrd, np.arange(8, dtype='int32').reshape(1, 8)


def to_static_mesh(ndcrd: np.ndarray, clnds: np.ndarray):
    """Build a ``solvcon.StaticMesh`` of hexahedra from ``ribbon`` or ``box``
    output, with its interior, boundary and ghosts."""
    mh = StaticMesh(ndim=3, nnode=len(ndcrd), nface=0, ncell=len(clnds))
    mh.ndcrd.ndarray[:, :] = ndcrd
    mh.cltpn.ndarray[:] = StaticMesh.HEXAHEDRON
    mh.clnds.ndarray[:, 0] = 8
    mh.clnds.ndarray[:, 1:9] = clnds
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
