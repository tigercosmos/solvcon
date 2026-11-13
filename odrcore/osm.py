# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Write the Lanelet2 form of a map as a Lanelet2 OSM file.

The output is what ``lanelet2.io.load`` reads. Each linestring of the view
becomes one way oriented along s. The Lanelet2 loader orients the two
bounds of every lanelet by geometry (``geometry::align``), so lanelets of
opposite travel direction share a way without a copy.

The model x, y, z are taken as east, north and up of a local tangent plane
at ``origin``. Node ``lat``, ``lon`` and ``ele`` are the exact WGS84
inverse of that plane, so ``LocalCartesianProjector(Origin(lat, lon))``
gives back the model coordinates. Every node also has ``local_x`` and
``local_y`` tags. The header ``<offset>`` is not applied.

Lanelet2 traffic signs and lights refer to a linestring, not a point, so
each signal becomes a two-node way across the road at its s, as wide as
the signal ``width`` (0.6 m when absent), at the bottom of the signal.

There is no reader: the map is edited in its OpenDRIVE form only.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Union

from .model import Map, Signal
from .view import PARTICIPANTS, LaneletView

Origin = tuple[float, float]
"""``(lat, lon)`` in degrees."""

LANELET2_PARTICIPANTS = {
    'passengerCar': 'vehicle:car', 'bus': 'vehicle:bus',
    'truck': 'vehicle:truck', 'taxi': 'vehicle:taxi',
    'emergency': 'vehicle:emergency', 'motorcycle': 'vehicle:motorcycle',
    'bicycle': 'bicycle', 'pedestrian': 'pedestrian',
}
"""OpenDRIVE road users with a Lanelet2 participant. ``delivery`` has
none and appears only in the ``odr:participants`` tag."""

DEFAULT_SIGNAL_WIDTH = 0.6

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def _ecef(lat: float, lon: float, h: float) -> tuple[float, float, float]:
    sl, cl = math.sin(lat), math.cos(lat)
    n = _A / math.sqrt(1.0 - _E2 * sl * sl)
    return ((n + h) * cl * math.cos(lon), (n + h) * cl * math.sin(lon),
            (n * (1.0 - _E2) + h) * sl)


class LocalTangentPlane:
    """East-north-up coordinates about a WGS84 origin, as GeographicLib
    ``LocalCartesian`` defines them."""

    def __init__(self, lat: float, lon: float, h: float = 0.0) -> None:
        self.lat0, self.lon0 = math.radians(lat), math.radians(lon)
        self.center = _ecef(self.lat0, self.lon0, h)
        sp, cp = math.sin(self.lat0), math.cos(self.lat0)
        sl, cl = math.sin(self.lon0), math.cos(self.lon0)
        self.east = (-sl, cl, 0.0)
        self.north = (-sp * cl, -sp * sl, cp)
        self.up = (cp * cl, cp * sl, sp)

    def to_geodetic(self, x: float, y: float,
                    z: float) -> tuple[float, float, float]:
        """``(lat, lon, h)`` in degrees and meters."""
        px, py, pz = (c + x * e + y * n + z * u for c, e, n, u in
                      zip(self.center, self.east, self.north, self.up))
        lon = math.atan2(py, px)
        p = math.hypot(px, py)
        lat = math.atan2(pz, p * (1.0 - _E2))
        h = 0.0
        for _ in range(16):
            sl = math.sin(lat)
            n = _A / math.sqrt(1.0 - _E2 * sl * sl)
            h = p / math.cos(lat) - n
            lat = math.atan2(pz, p * (1.0 - _E2 * n / (n + h)))
        return math.degrees(lat), math.degrees(lon), h

    def to_local(self, lat: float, lon: float,
                 h: float) -> tuple[float, float, float]:
        q = _ecef(math.radians(lat), math.radians(lon), h)
        d = [a - b for a, b in zip(q, self.center)]
        return tuple(sum(a * b for a, b in zip(d, axis))  # type: ignore
                     for axis in (self.east, self.north, self.up))


def origin_of(m: Map) -> Origin:
    """``+lat_0`` and ``+lon_0`` of the geoReference, else ``(0, 0)``."""
    text = m.frame.geo_reference or ''
    found = []
    for name in ('lat_0', 'lon_0'):
        hit = re.search(rf'\+{name}=([-+0-9.eE]+)', text)
        found.append(float(hit.group(1)) if hit else 0.0)
    return found[0], found[1]


def _num(value: float) -> str:
    return repr(float(value))


class Writer:

    def __init__(self, m: Map, spacing: float = 2.0,
                 origin: Optional[Origin] = None) -> None:
        self.map = m
        self.view: LaneletView = m.lanelet2(spacing)
        self.origin = origin_of(m) if origin is None else origin
        self.plane = LocalTangentPlane(*self.origin)
        self.root = ET.Element('osm', version='0.6', generator='odrcore')
        self._nodes: dict[int, ET.Element] = {}
        self._ways: list[ET.Element] = []
        self._relations: list[ET.Element] = []
        self._signals = {sig.id: sig for road in m.roads.values()
                         for sig in road.signals}
        self._signal_ways: set[int] = set()

    def write(self) -> ET.Element:
        for line in self.view.lines.values():
            way = self._way(line.id, [p.id for p in line.points], line.attrs)
            for pt in line.points:
                self._node(pt.id, pt.position, pt.attrs)
            self._ways.append(way)
        for ll in self.view.lanelets.values():
            self._lanelet(ll)
        for reg in self.view.regulations.values():
            self._regulation(reg)
        self.root.extend(self._nodes[k] for k in sorted(self._nodes))
        self.root.extend(self._ways)
        self.root.extend(self._relations)
        return self.root

    @staticmethod
    def _tags(elem: ET.Element, attrs: dict[str, str]) -> None:
        for k, v in attrs.items():
            ET.SubElement(elem, 'tag', k=k, v=v)

    def _node(self, identity: int, position: tuple[float, float, float],
              attrs: Optional[dict[str, str]] = None) -> None:
        if identity in self._nodes:
            return
        x, y, z = position
        lat, lon, h = self.plane.to_geodetic(x, y, z)
        elem = ET.Element('node', id=str(identity), lat=_num(lat),
                          lon=_num(lon))
        self._tags(elem, {'ele': _num(h), 'local_x': _num(x),
                          'local_y': _num(y)})
        self._tags(elem, attrs or {})
        self._nodes[identity] = elem

    def _way(self, identity: int, refs: list[int],
             attrs: dict[str, str]) -> ET.Element:
        elem = ET.Element('way', id=str(identity))
        for ref in refs:
            ET.SubElement(elem, 'nd', ref=str(ref))
        self._tags(elem, attrs)
        return elem

    def _relation(self, identity: int, members: list[tuple[str, str, int]],
                  attrs: dict[str, str]) -> None:
        elem = ET.Element('relation', id=str(identity))
        for kind, role, ref in members:
            ET.SubElement(elem, 'member', type=kind, role=role, ref=str(ref))
        self._tags(elem, attrs)
        self._relations.append(elem)

    def _lanelet(self, ll) -> None:
        members = [('way', 'left', ll.left.id), ('way', 'right', ll.right.id)]
        members += [('relation', 'regulatory_element', r)
                    for r in ll.regulations]
        attrs = dict(ll.attrs)
        for odr, name in LANELET2_PARTICIPANTS.items():
            attrs[f'participant:{name}'] = \
                'yes' if odr in ll.participants else 'no'
        attrs['odr:participants'] = ','.join(
            sorted(ll.participants & PARTICIPANTS))
        self._relation(ll.id, members, attrs)

    def _signal_way(self, sig: Signal, attrs: dict[str, str]) -> int:
        identity = self.view.identifier('signal_line', sig.id)
        if identity in self._signal_ways:
            return identity
        width = float(sig.attrs.get('width', DEFAULT_SIGNAL_WIDTH))
        rl = sig.road.refline
        refs = []
        for i, dt in enumerate((-0.5 * width, 0.5 * width)):
            x, y, z = rl.point(sig.s, sig.t + dt)
            nid = self.view.identifier('point', 'signal_line', sig.id, i)
            self._node(nid, (x, y, z + sig.z_offset))
            refs.append(nid)
        tags = dict(attrs)
        if 'height' in sig.attrs:
            tags['height'] = sig.attrs['height']
        self._ways.append(self._way(identity, refs, tags))
        self._signal_ways.add(identity)
        return identity

    def _member(self, identity: int) -> tuple[str, int]:
        view = self.view
        if identity in view.lanelets:
            return 'relation', identity
        if identity in view.lines:
            return 'way', identity
        pt = view.points[identity]
        sig = self._signals.get(pt.attrs.get('odr:signal', ''))
        if sig is not None:
            return 'way', self._signal_way(sig, pt.attrs)
        self._node(pt.id, pt.position, pt.attrs)
        return 'node', identity

    def _regulation(self, reg) -> None:
        members = []
        for role, ids in reg.roles.items():
            for identity in ids:
                kind, ref = self._member(identity)
                members.append((kind, role, ref))
        self._relation(reg.id, members, reg.attrs)


def to_element(m: Map, spacing: float = 2.0,
               origin: Optional[Origin] = None) -> ET.Element:
    return Writer(m, spacing, origin).write()


def dumps(m: Map, spacing: float = 2.0,
          origin: Optional[Origin] = None) -> str:
    root = to_element(m, spacing, origin)
    ET.indent(root)
    return ET.tostring(root, encoding='unicode', xml_declaration=True)


def dump(m: Map, path: Union[str, 'os.PathLike[str]'], spacing: float = 2.0,
         origin: Optional[Origin] = None) -> None:
    root = to_element(m, spacing, origin)
    ET.indent(root)
    ET.ElementTree(root).write(os.fspath(path), encoding='UTF-8',
                               xml_declaration=True)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
