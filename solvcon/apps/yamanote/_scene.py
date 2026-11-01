# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Yamanote Line as solid geometry: the two loops as slab meshes and the
stations as points along them, and the streets around the line as slabs
across their carriageways when their map is there. Qt-free, so the scene
builds headlessly.
"""

import pathlib
import re
import time

import odrcore
from odrcore import mesh as odrmesh

__all__ = [
    'COLORS',
    'MAP_PATH',
    'ROADS_PATH',
    'STATION_SIZE',
    'Scene',
    'Station',
]

MAP_PATH = pathlib.Path(odrcore.__file__).parent / 'examples' / 'yamanote.xodr'
#: The OSM streets within 500 m of the line. It is built locally and not
#: checked in, so a scene without it draws the line alone.
ROADS_PATH = MAP_PATH.with_name('yamanote_roads.xodr')
LOOPS = ('Outer', 'Inner')
STOP_RE = re.compile(r'osm:stop" value="(\w+) (.+) (\S+) s=([-0-9.e+]+)')
HIGHWAY_RE = re.compile(r'osm:highway" value="([^"]+)"')
SEGMENT_RE = re.compile(r'"segment" value="([^"]+)"')
#: The OSM highway classes drawn as major roads, with their ``_link`` ramps;
#: every other class is minor.
MAJOR_HIGHWAYS = frozenset(['motorway', 'trunk', 'primary', 'secondary'])
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
    """The loaded map, each loop's road chain and the stations on it, and
    the streets grouped for drawing.

    :ivar roads_map: The street map, or None without one.
    :ivar road_groups: The streets by object name,
        ``roads.<segment>.major`` or ``roads.<segment>.minor``, where the
        segment is the pair of stations a street lies by.
    :ivar roads_seconds: The seconds loading the street map took.
    """

    SPACING = 5.0
    HALF_WIDTH = 1.5
    THICKNESS = 0.5
    #: Half width of the slab drawn in place of the true one from afar.
    WIDE_HALF_WIDTH = 13.0
    ROAD_SPACING = 10.0
    ROAD_THICKNESS = 0.3

    def __init__(self, path=MAP_PATH, roads_path=None):
        self.map = odrcore.load(path)
        self.chains = {loop: self._chain(loop) for loop in LOOPS}
        self.stations = [st for loop in LOOPS for st in self._stops(loop)]
        if roads_path is None:
            roads_path = ROADS_PATH
        self.roads_map, self.road_groups = None, {}
        self.roads_seconds = 0.0
        if pathlib.Path(roads_path).exists():
            start = time.perf_counter()
            self.roads_map = odrcore.load(roads_path)
            self.roads_seconds = time.perf_counter() - start
            self.road_groups = self._group(self.roads_map.roads.values())

    @staticmethod
    def _group(roads):
        groups = {}
        for road in roads:
            data = road.attrs.get('userData', '')
            segment = SEGMENT_RE.search(data)
            segment = segment.group(1) if segment else 'other'
            highway = HIGHWAY_RE.search(data)
            major = (highway is not None and highway.group(1).removesuffix(
                '_link') in MAJOR_HIGHWAYS)
            name = f'roads.{segment}.{"major" if major else "minor"}'
            groups.setdefault(name, []).append(road)
        return dict(sorted(groups.items()))

    @property
    def road_count(self):
        return sum(len(roads) for roads in self.road_groups.values())

    @staticmethod
    def is_minor(group):
        return group.endswith('.minor')

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

    def road_arrays(self, group):
        """Return the node and hexahedron arrays of the streets of
        ``group``, each a slab across its carriageway with the top on the
        road surface."""
        return odrmesh.road_ribbons(self.road_groups[group],
                                    self.ROAD_SPACING,
                                    thickness=self.ROAD_THICKNESS)

    def road_mesh(self, group):
        return odrmesh.to_static_mesh(*self.road_arrays(group))

    @staticmethod
    def marker_mesh():
        return odrmesh.to_static_mesh(*odrmesh.box())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
