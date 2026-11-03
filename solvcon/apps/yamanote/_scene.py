# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Yamanote Line as solid geometry: the two loops as slab meshes and the
stations as points along them, and the streets around the line as slabs
across their carriageways when their map is there, with their lane marks,
crosswalks and traffic signals. Qt-free, so the scene builds headlessly.
"""

import pathlib
import time

import odrcore
from odrcore.parse import user_data
from odrcore.topology import loop_chain

from ... import odrmesh

__all__ = [
    'COLORS',
    'DETAIL_KINDS',
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
#: The details drawn on the streets of each segment: the kind, the object
#: name with the segment for ``{}``, and the :class:`Scene` method that
#: builds the arrays from the segment and the extra arguments.
DETAIL_KINDS = (
    ('marks.white', 'marks.{}.white', 'mark_arrays', ('white',)),
    ('marks.yellow', 'marks.{}.yellow', 'mark_arrays', ('yellow',)),
    ('crosswalks', 'crosswalks.{}', 'crosswalk_arrays', ()),
    ('signals.heads', 'signals.{}.heads', 'signal_arrays', ('heads',)),
    ('signals.poles', 'signals.{}.poles', 'signal_arrays', ('poles',)),
)


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
    :ivar segments: The streets of both classes by segment.
    :ivar roads_seconds: The seconds loading the street map took.
    """

    SPACING = 5.0
    HALF_WIDTH = 1.5
    THICKNESS = 0.5
    #: Half width of the slab drawn in place of the true one from afar.
    WIDE_HALF_WIDTH = 13.0
    ROAD_SPACING = 10.0
    ROAD_THICKNESS = 0.3
    #: The marks sample the boundaries as the slabs do their streets.
    MARK_SPACING = 10.0
    #: The gap of the top of a mark over the street, and its thickness.
    MARK_LIFT = 0.05
    MARK_THICKNESS = 0.03
    #: A crosswalk tops the marks it crosses.
    CROSSWALK_LIFT = 0.08
    CROSSWALK_THICKNESS = 0.03
    #: A signal head in meters, along the way it faces, across and up; its
    #: bottom sits on top of the pole.
    SIGNAL_HEAD_SIZE = (0.4, 1.2, 0.4)
    SIGNAL_POLE_SIZE = (0.15, 0.15, 5.0)

    def __init__(self, path=MAP_PATH, roads_path=None):
        self.map = odrcore.load(path)
        self.chains = {loop: loop_chain(self.map, loop) for loop in LOOPS}
        self.stations = [st for loop in LOOPS for st in self._stops(loop)]
        if roads_path is None:
            roads_path = ROADS_PATH
        self.roads_map, self.road_groups, self.segments = None, {}, {}
        self.roads_seconds = 0.0
        self._detail_groups, self._detail_arrays = None, {}
        if pathlib.Path(roads_path).exists():
            start = time.perf_counter()
            self.roads_map = odrcore.load(roads_path)
            self.roads_seconds = time.perf_counter() - start
            self.road_groups, self.segments = self._group(
                self.roads_map.roads.values())

    @staticmethod
    def _group(roads):
        groups, segments = {}, {}
        for road in roads:
            data = dict(user_data(road.attrs.get('userData')))
            segment = data.get('segment', 'other')
            major = (data.get('osm:highway', '').removesuffix('_link')
                     in MAJOR_HIGHWAYS)
            name = f'roads.{segment}.{"major" if major else "minor"}'
            groups.setdefault(name, []).append(road)
            segments.setdefault(segment, []).append(road)
        return dict(sorted(groups.items())), dict(sorted(segments.items()))

    @property
    def road_count(self):
        return sum(len(roads) for roads in self.road_groups.values())

    @property
    def signal_count(self):
        return 0 if self.roads_map is None else len(self.roads_map.signals)

    @property
    def crosswalk_count(self):
        return sum(obj.type == 'crosswalk'
                   for roads in self.segments.values() for road in roads
                   for obj in road.objects)

    @property
    def detail_groups(self):
        """The detail objects with any cell by name, ``marks.<segment>.white``
        and the others of :data:`DETAIL_KINDS`, each to its kind. The first
        access builds the arrays of every detail and keeps them for
        :meth:`detail_mesh`."""
        if self._detail_groups is None:
            groups = {}
            for segment in self.segments:
                for kind, pattern, method, args in DETAIL_KINDS:
                    arrays = getattr(self, method)(segment, *args)
                    if len(arrays[1]):
                        name = pattern.format(segment)
                        groups[name] = kind
                        self._detail_arrays[name] = arrays
            self._detail_groups = dict(sorted(groups.items()))
        return self._detail_groups

    @staticmethod
    def is_minor(group):
        return group.endswith('.minor')

    def _stops(self, loop):
        offset = 0.0
        for road in self.chains[loop]:
            for kind, value in user_data(road.attrs.get('userData')):
                if kind != 'osm:stop':
                    continue
                # "<code> <name:en> <name> s=<s> node=<id>"
                head, s, _ = value.rsplit(' ', 2)
                code, names = head.split(' ', 1)
                en, ja = names.rsplit(' ', 1)
                s = float(s.removeprefix('s='))
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

    def mark_arrays(self, segment, color):
        """Return the node and hexahedron arrays of the lane marks of
        ``color``, ``'white'`` or ``'yellow'``, on the streets of
        ``segment``, :attr:`MARK_LIFT` above the surface."""
        return odrmesh.mark_ribbons(self.segments[segment], color,
                                    self.MARK_SPACING, self.MARK_LIFT,
                                    self.MARK_THICKNESS)

    def mark_mesh(self, segment, color):
        return odrmesh.to_static_mesh(*self.mark_arrays(segment, color))

    def crosswalk_arrays(self, segment):
        """Return the node and hexahedron arrays of a flat box for each
        crosswalk on the streets of ``segment``."""
        return odrmesh.crosswalk_boxes(self.segments[segment],
                                       self.CROSSWALK_LIFT,
                                       self.CROSSWALK_THICKNESS)

    def crosswalk_mesh(self, segment):
        return odrmesh.to_static_mesh(*self.crosswalk_arrays(segment))

    def signal_arrays(self, segment, part):
        """Return the node and hexahedron arrays of one ``part`` of every
        traffic signal on the streets of ``segment``: ``'poles'`` stand at
        the signals, ``'heads'`` sit on top of them, both turned the way
        the signal faces."""
        feet, heading = odrmesh.signal_poses(self.segments[segment])
        pole = self.SIGNAL_POLE_SIZE
        if part == 'poles':
            size, rise = pole, 0.5 * pole[2]
        else:
            size = self.SIGNAL_HEAD_SIZE
            rise = pole[2] + 0.5 * size[2]
        centers = feet + (0.0, 0.0, rise)
        return odrmesh.boxes(centers, [size] * len(feet), heading)

    def signal_mesh(self, segment, part):
        return odrmesh.to_static_mesh(*self.signal_arrays(segment, part))

    def detail_mesh(self, name):
        """Return the mesh of one object of :attr:`detail_groups`."""
        if name not in self.detail_groups:
            raise KeyError(name)
        return odrmesh.to_static_mesh(*self._detail_arrays[name])

    @staticmethod
    def marker_mesh():
        return odrmesh.to_static_mesh(*odrmesh.box())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
