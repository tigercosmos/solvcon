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
``concat`` joins such arrays into one mesh. The details on the roads are
hexahedra too: ``mark_ribbons`` lays a thin slab along each painted lane
boundary, and ``boxes`` bakes many small boxes into one mesh, posed by
``signal_poses`` and ``object_poses`` as ``crosswalk_boxes`` does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from odrcore.geom import locate, poly3_values, refline_points
from odrcore.model import Road, RoadObject

from .core import StaticMesh

Arrays = tuple[np.ndarray, np.ndarray]
#: The road mark types :func:`mark_ribbons` draws.
MARK_TYPES = ('solid', 'broken')
#: The width of a road mark that does not give one, in meters.
MARK_WIDTH = 0.15


def _empty() -> Arrays:
    return (np.empty((0, 3), dtype='float64'),
            np.empty((0, 8), dtype='int32'))


def _sections(road: Road, spacing: float, half_width: float,
              thickness: float, with_end: bool) -> np.ndarray:
    """Return the ``(n, 4, 3)`` cross sections along the road, ordered left
    top, right top, right bottom, left bottom."""
    n = max(int(road.length // spacing), 1)
    stations = list(np.linspace(0.0, road.length, n, endpoint=False,
                                dtype='float64'))
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


def _chains(lo: np.ndarray, hi: np.ndarray,
            nseg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the owner chain and the station ``s`` of every section of the
    chains from ``lo`` to ``hi`` in ``nseg`` equal steps, one after the
    other, and the sections that start a cell: all but the last of each."""
    owner = np.repeat(np.arange(len(lo), dtype='int64'), nseg + 1)
    start = np.concatenate([[0], np.cumsum(nseg + 1)[:-1]]).astype('int64')
    k = np.arange(len(owner), dtype='int64') - start[owner]
    s = np.where(k == nseg[owner], hi[owner],
                 lo[owner] + k * ((hi - lo) / nseg)[owner])
    first = np.setdiff1d(np.arange(len(s), dtype='int64'), start + nseg,
                         assume_unique=True)
    return owner, s, first


def _slab(roads: Sequence[Road], owner: np.ndarray, s: np.ndarray,
          left: np.ndarray, right: np.ndarray, thickness: float,
          lift: float = 0.0) -> np.ndarray:
    """Return the ``(n, 4, 3)`` cross sections, ordered as ``_sections``
    orders them, from offset ``left`` to ``right`` of ``roads[owner]`` at
    ``s``, the top ``lift`` above the road surface."""
    top = refline_points([road.refline for road in roads], owner, s,
                         np.stack([left, right], axis=1))
    sections = np.concatenate([top, top[:, ::-1]], axis=1)
    sections[:, :, 2] += lift
    sections[:, 2:, 2] -= thickness
    return sections


def road_ribbons(roads: Sequence[Road], spacing: float = 10.0,
                 half_width: float | None = None,
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
        return _empty()
    length = np.array([road.length for road in roads], dtype='float64')
    nseg = np.maximum((length // spacing).astype('int64'), 1)
    owner, s, first = _chains(np.zeros_like(length), length, nseg)

    lo, hi = carriageway(roads, owner, s)
    if half_width is not None:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - half_width, mid + half_width
    sections = _slab(roads, owner, s, hi, lo, thickness)
    return sections.reshape(-1, 3), _hexes(first, len(s))


def _mark_spans(roads: Sequence[Road], colors: set[str]):
    """Yield the road index, the boundary, the mark and the start and end
    s of every mark of ``roads`` in ``colors`` of a type in
    :data:`MARK_TYPES`."""
    for k, road in enumerate(roads):
        for sec in road.sections:
            if sec.center is None:
                continue
            for bnd in sec.boundaries():
                ends = [sec.s0 + mk.s_offset for mk in bnd.marks[1:]]
                for mk, end in zip(bnd.marks, ends + [sec.s1]):
                    if mk.color in colors and mk.type in MARK_TYPES:
                        yield k, bnd, mk, sec.s0 + mk.s_offset, end


def _painted(roads: Sequence[Road], color: str,
             dash: tuple[float, float]) -> tuple[list, list, np.ndarray]:
    """Return the road index and the boundary of every painted stretch of
    the marks of ``color``, and their ``(n, 3)`` start s, end s and
    width."""
    colors = {color, 'standard'} if color == 'white' else {color}
    owner, bnds, rows = [], [], []
    for k, bnd, mk, start, end in _mark_spans(roads, colors):
        if end <= start:
            continue
        if mk.type == 'broken':
            # No dash shorter than a micrometer at the end.
            lo = np.arange(start, end - 1e-6, dash[0] + dash[1],
                           dtype='float64')
            hi = np.minimum(lo + dash[0], end)
        else:
            lo = np.array([start], dtype='float64')
            hi = np.array([end], dtype='float64')
        width = MARK_WIDTH if mk.width is None else mk.width
        owner += [k] * len(lo)
        bnds += [bnd] * len(lo)
        rows.append(np.stack([lo, hi, np.full_like(lo, width)], axis=1))
    table = np.concatenate(rows) if rows \
        else np.empty((0, 3), dtype='float64')
    return owner, bnds, table


def mark_ribbons(roads: Sequence[Road], color: str, spacing: float = 2.0,
                 lift: float = 0.02, thickness: float = 0.02,
                 dash: tuple[float, float] = (3.0, 6.0)) -> Arrays:
    """Return the node and hexahedron arrays of a thin slab along every lane
    boundary of ``roads`` painted ``color`` (``'white'``, which also takes
    the OpenDRIVE ``'standard'``, or ``'yellow'``): the mark's width
    centered on the boundary, the top ``lift`` above the road surface and
    ``thickness`` below that.

    A mark runs from its start to the next mark of the boundary or the end
    of its lane section. A ``solid`` mark paints all of it, a ``broken``
    one ``dash[0]`` meters out of every ``dash[0] + dash[1]`` from its
    start, the last dash cut at its end, and every other type is left
    out. Each painted stretch, a solid mark or one dash, is an open chain
    of its own with sections at most ``spacing`` apart, so the ends of a
    dash are always sections and no cell spans a gap."""
    owner, bnds, table = _painted(roads, color, dash)
    if not len(table):
        return _empty()
    lo, hi, width = table.T
    nseg = np.maximum(np.ceil((hi - lo) / spacing - 1e-9), 1)
    piece, s, first = _chains(lo, hi, nseg.astype('int64'))

    t = poly3_values([bnd.t for bnd in bnds], piece, s)
    half = 0.5 * width[piece]
    road = np.array(owner, dtype='int64')[piece]
    sections = _slab(roads, road, s, t + half, t - half, thickness, lift)
    return sections.reshape(-1, 3), _hexes(first, len(s))


def road_ribbon(road: Road, spacing: float = 10.0,
                half_width: float | None = None,
                thickness: float = 0.3) -> Arrays:
    """Return the node and hexahedron arrays of an open slab along one
    road, as :func:`road_ribbons` draws it."""
    return road_ribbons([road], spacing, half_width, thickness)


def concat(parts: Sequence[Arrays]) -> Arrays:
    """Join several ``(ndcrd, clnds)`` pairs into one, renumbering the
    connectivity of each part past the nodes of the parts before it."""
    parts = list(parts)
    if not parts:
        return _empty()
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


def boxes(centers: np.ndarray, sizes: np.ndarray,
          headings: np.ndarray) -> Arrays:
    """Return the node and hexahedron arrays of ``n`` boxes, the unit cube
    of :func:`box` scaled by ``sizes`` ``(n, 3)`` along x, y and z, turned
    by ``headings`` ``(n,)`` in radians counterclockwise about z and moved
    to ``centers`` ``(n, 3)``. Box ``i`` has nodes ``8 i`` to ``8 i + 7``
    in the order of :func:`box`."""
    centers = np.asarray(centers, dtype='float64').reshape(-1, 3)
    sizes = np.asarray(sizes, dtype='float64').reshape(-1, 3)
    headings = np.asarray(headings, dtype='float64').reshape(-1)
    nodes = box()[0][None] * sizes[:, None, :]
    cos, sin = np.cos(headings)[:, None], np.sin(headings)[:, None]
    x, y = nodes[..., 0], nodes[..., 1]
    nodes = np.stack([x * cos - y * sin, x * sin + y * cos, nodes[..., 2]],
                     axis=-1) + centers[:, None, :]
    clnds = np.arange(8 * len(centers), dtype='int32').reshape(-1, 8)
    return nodes.reshape(-1, 3), clnds


def _poses(roads: Sequence[Road], owner: np.ndarray, s: np.ndarray,
           t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the points ``(n, 3)`` at ``(s, t)`` of ``roads[owner]`` and
    the heading ``(n,)`` of the reference line there."""
    if not len(owner):
        return (np.empty((0, 3), dtype='float64'),
                np.empty(0, dtype='float64'))
    pts = refline_points([road.refline for road in roads], owner, s,
                         np.stack([t, t + 1.0], axis=1))
    # The offset runs along the left normal (-sin hdg, cos hdg), scaled by
    # the cosine of the superelevation, which is positive.
    across = pts[:, 1] - pts[:, 0]
    return pts[:, 0], np.arctan2(-across[:, 0], across[:, 1])


def signal_poses(roads: Sequence[Road]) -> tuple[np.ndarray, np.ndarray]:
    """Return the foot ``(n, 3)`` and the heading ``(n,)`` of every signal
    of ``roads``, road by road in their order: the road surface at its
    ``(s, t)`` raised by its ``z_offset``, and the heading of the reference
    line there, turned by pi for orientation ``'-'`` and by its
    ``h_offset``."""
    sigs = [(k, sig) for k, road in enumerate(roads) for sig in road.signals]
    owner = np.array([k for k, _ in sigs], dtype='int64')
    s, t, z, turn = (np.array([getattr(sig, name) for _, sig in sigs],
                              dtype='float64')
                     for name in ('s', 't', 'z_offset', 'h_offset'))
    back = np.array([sig.orientation == '-' for _, sig in sigs],
                    dtype='bool')
    feet, heading = _poses(roads, owner, s, t)
    feet[:, 2] += z
    return feet, heading + turn + np.where(back, np.pi, 0.0)


def object_poses(roads: Sequence[Road], type: str
                 ) -> tuple[list[RoadObject], np.ndarray, np.ndarray]:
    """Return the objects of ``roads`` of ``type``, road by road, their foot
    ``(n, 3)``, the road surface at ``(s, t)`` raised by ``z_offset``, and
    their heading ``(n,)``, that of the reference line there plus
    ``hdg``."""
    found = [(k, obj) for k, road in enumerate(roads)
             for obj in road.objects if obj.type == type]
    objs = [obj for _, obj in found]
    owner = np.array([k for k, _ in found], dtype='int64')
    s, t, z, hdg = (np.array([getattr(obj, name) for obj in objs],
                             dtype='float64')
                    for name in ('s', 't', 'z_offset', 'hdg'))
    feet, heading = _poses(roads, owner, s, t)
    feet[:, 2] += z
    return objs, feet, heading + hdg


def crosswalk_boxes(roads: Sequence[Road], lift: float = 0.02,
                    thickness: float = 0.02) -> Arrays:
    """Return the node and hexahedron arrays of a flat box for every
    ``crosswalk`` object of ``roads`` with a length and a width: its
    ``length`` along its heading, its ``width`` across and ``thickness``
    high, the top ``lift`` above its foot."""
    objs, feet, heading = object_poses(roads, 'crosswalk')
    sized = np.array([obj.length is not None and obj.width is not None
                      for obj in objs], dtype='bool').reshape(-1)
    sizes = np.array([(obj.length, obj.width, thickness)
                      for obj, keep in zip(objs, sized) if keep],
                     dtype='float64').reshape(-1, 3)
    feet = feet[sized]
    feet[:, 2] += lift - 0.5 * thickness
    return boxes(feet, sizes, heading[sized])


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
