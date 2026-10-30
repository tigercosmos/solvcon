# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Hexahedral slabs along reference lines, for drawing a map as solid geometry.

A 3D viewer that shades the boundary shell of a volume mesh cannot take a
bare line or a bare triangle strip, so a track is built as a thin slab: the
reference line is sampled along s, each sample gives a cross section of
four nodes (the left and right edges at the road height and ``thickness``
below it), and consecutive sections are joined into one hexahedron.

``ribbon`` returns plain arrays so it is testable without the solvcon
extension; ``to_static_mesh`` turns them into a ``solvcon.StaticMesh``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .model import Road

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
    base = np.arange(ncell, dtype='int32') * 4
    nxt = (base + 4) % (nsec * 4)
    # The bottom face runs left to right then forward, so its normal points
    # up into the cell, as the top face follows it.
    clnds = np.stack([base + 3, base + 2, nxt + 2, nxt + 3,
                      base + 0, base + 1, nxt + 1, nxt + 0], axis=1)
    return sections.reshape(-1, 3), clnds


def box() -> Arrays:
    """Return a unit cube centered at the origin, one hexahedron."""
    ndcrd = np.array([(x, y, z) for z in (-0.5, 0.5)
                      for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5),
                                   (-0.5, 0.5))], dtype='float64')
    return ndcrd, np.arange(8, dtype='int32').reshape(1, 8)


def to_static_mesh(ndcrd: np.ndarray, clnds: np.ndarray):
    """Build a ``solvcon.StaticMesh`` of hexahedra from ``ribbon`` or ``box``
    output, with its interior, boundary and ghosts."""
    import solvcon
    mh = solvcon.StaticMesh(ndim=3, nnode=len(ndcrd), nface=0,
                            ncell=len(clnds))
    mh.ndcrd.ndarray[:, :] = ndcrd
    mh.cltpn.ndarray[:] = solvcon.StaticMesh.HEXAHEDRON
    mh.clnds.ndarray[:, 0] = 8
    mh.clnds.ndarray[:, 1:9] = clnds
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
