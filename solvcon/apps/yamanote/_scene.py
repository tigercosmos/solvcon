# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Yamanote Line as solid geometry: the two loops as slab meshes and the
stations as points along them. Qt-free, so the scene builds headlessly.
"""

import pathlib
import re

import odrcore
from odrcore import mesh as odrmesh

__all__ = [
    'COLORS',
    'MAP_PATH',
    'STATION_SIZE',
    'Scene',
    'Station',
]

MAP_PATH = pathlib.Path(odrcore.__file__).parent / 'examples' / 'yamanote.xodr'
LOOPS = ('Outer', 'Inner')
STOP_RE = re.compile(r'osm:stop" value="(\w+) (.+) (\S+) s=([-0-9.e+]+)')
COLORS = {
    'Outer': (0.55, 0.78, 0.25, 1.0),
    'Inner': (0.20, 0.55, 0.30, 1.0),
    'station': (0.85, 0.15, 0.15, 1.0),
}
#: A stop marker box in meters: x, y, z.
STATION_SIZE = (30.0, 30.0, 12.0)


class Station(object):
    """One stop of one loop: its code, names, position in map meters and
    arc length ``s`` along the loop from the start of its first road."""

    def __init__(self, code, name_en, name_ja, loop, xyz, s=0.0):
        self.code = code
        self.name_en = name_en
        self.name_ja = name_ja
        self.loop = loop
        self.x, self.y, self.z = xyz
        self.s = s

    @property
    def name(self):
        return f'{self.code}.{self.loop}'


class Scene(object):
    """The loaded map, each loop's road chain and the stations on it."""

    SPACING = 5.0
    HALF_WIDTH = 1.5
    THICKNESS = 0.5
    #: Half width of the slab drawn in place of the true one from afar.
    WIDE_HALF_WIDTH = 13.0

    def __init__(self, path=MAP_PATH):
        self.map = odrcore.load(path)
        self.chains = {loop: self._chain(loop) for loop in LOOPS}
        self.stations = [st for loop in LOOPS for st in self._stops(loop)]

    def _chain(self, loop):
        first = next(r for r in self.map.roads.values()
                     if r.name.startswith(loop))
        chain, road = [], first
        while True:
            chain.append(road)
            road = self.map.roads[road.successor.element_id]
            if road is first:
                return chain

    def _stops(self, loop):
        offset = 0.0
        for road in self.chains[loop]:
            for code, en, ja, s in STOP_RE.findall(
                    road.attrs.get('userData', '')):
                s = float(s)
                yield Station(code, en, ja, loop,
                              road.refline.point(s, 0.0), offset + s)
            offset += road.length

    @staticmethod
    def wide_name(loop):
        return f'{loop}.wide'

    def loop_arrays(self, loop, half_width=None):
        """Return the node and hexahedron arrays of one loop's slab,
        :attr:`HALF_WIDTH` to either side unless ``half_width`` is given."""
        if half_width is None:
            half_width = self.HALF_WIDTH
        return odrmesh.ribbon(self.chains[loop], self.SPACING, half_width,
                              self.THICKNESS)

    def loop_mesh(self, loop, half_width=None):
        return odrmesh.to_static_mesh(*self.loop_arrays(loop, half_width))

    @staticmethod
    def marker_mesh():
        return odrmesh.to_static_mesh(*odrmesh.box())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
