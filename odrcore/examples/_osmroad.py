# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The parts the OpenStreetMap example generators share: the Overpass fetch,
the vertex thinning, the Hermite fit of a polyline into normalized
``paramPoly3`` segments, the bridge or tunnel ``<objects>`` of a way and the
road chain of a Yamanote Line loop.
"""

import json
import math
import os
import urllib.parse
import urllib.request
from xml.sax.saxutils import quoteattr

import numpy as np

from odrcore.geom import ParamPoly3
from odrcore.model import Map, Road

OVERPASS_URL = os.environ.get('OVERPASS_URL',
                              'https://overpass-api.de/api/interpreter')
MIN_SPACING = 2.0

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def fetch(query: str, timeout: float = 180.0) -> dict:
    body = urllib.parse.urlencode({'data': query}).encode()
    req = urllib.request.Request(OVERPASS_URL, body,
                                 {'User-Agent': 'solvcon-odrcore'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def thin(xy: np.ndarray, keep: set[int]) -> list[int]:
    """Indices of the points kept: every index in ``keep`` and the interior
    points at least ``MIN_SPACING`` from the previous kept one."""
    idx = [0]
    for i in range(1, len(xy)):
        if i in keep or np.hypot(*(xy[i] - xy[idx[-1]])) >= MIN_SPACING:
            idx.append(i)
    return idx


def tangents(xy: np.ndarray) -> np.ndarray:
    prev, nxt = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
    h0 = np.hypot(*(xy - prev).T)[:, None]
    h1 = np.hypot(*(nxt - xy).T)[:, None]
    t = (h1 * (xy - prev) / h0 + h0 * (nxt - xy) / h1) / (h0 + h1)
    return t / np.hypot(*t.T)[:, None]


def open_tangents(xy: np.ndarray) -> np.ndarray:
    """The Bessel tangents of an open polyline, with the chord direction
    at the two ends."""
    t = np.empty_like(xy)
    if len(xy) > 2:
        t[1:-1] = tangents(xy)[1:-1]
    t[0], t[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
    t[[0, -1]] /= np.hypot(*t[[0, -1]].T)[:, None]
    return t


def hermite(s0: float, p0, p1, t0, t1) -> ParamPoly3:
    """Normalized paramPoly3 from ``p0`` to ``p1`` leaving along unit
    tangent ``t0`` and arriving along ``t1``, both scaled by the chord."""
    hdg = math.atan2(t0[1], t0[0])
    ch, sh = math.cos(hdg), math.sin(hdg)
    chord = float(np.hypot(*(p1 - p0)))

    def local(v):
        return np.array([v[0] * ch + v[1] * sh, -v[0] * sh + v[1] * ch])

    q1, m0, m1 = local(p1 - p0), local(t0) * chord, local(t1) * chord
    b, c, d = m0, 3.0 * q1 - 2.0 * m0 - m1, -2.0 * q1 + m0 + m1
    p = 0.5 * (_GL_NODES + 1.0)
    du = b[0] + p * (2.0 * c[0] + 3.0 * d[0] * p)
    dv = b[1] + p * (2.0 * c[1] + 3.0 * d[1] * p)
    length = 0.5 * float(np.sum(_GL_WEIGHTS * np.hypot(du, dv)))
    return ParamPoly3(s0, length, float(p0[0]), float(p0[1]), hdg,
                      0.0, float(b[0]), float(c[0]), float(d[0]),
                      0.0, float(b[1]), float(c[1]), float(d[1]),
                      p_range='normalized')


def structure(way: dict, length: float, road_id: str = '') -> str:
    """The bridge or tunnel ``<objects>`` of a way, or of the road
    ``road_id`` cut from it."""
    tags = way['tags']
    if tags.get('tunnel'):
        kind = 'tunnel'
        typ = ('underpass' if tags['tunnel'] == 'building_passage'
               else 'standard')
    elif tags.get('bridge'):
        kind, typ = 'bridge', 'concrete'
    else:
        return ''
    name = tags.get(f'{kind}:name', '')
    return (f'<objects><{kind} s="0.0" length="{length!r}" '
            f'id="{road_id or way["id"]}.{kind}" name={quoteattr(name)} '
            f'type="{typ}"/></objects>')


def loop_chain(m: Map, label: str) -> list[Road]:
    """The roads of one Yamanote Line loop in successor order."""
    first = next(r for r in m.roads.values() if r.name.startswith(label))
    chain, road = [], first
    while True:
        chain.append(road)
        road = m.roads[road.successor.element_id]
        if road is first:
            return chain

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
