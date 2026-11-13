# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Draw the streets around the Yamanote Line into the viewer, one object
per group of the scene, and the lane marks, crosswalks and traffic signals
on them, one object per detail of a segment."""

import functools
import time
import typing

__all__ = [
    'DETAIL_SWITCHES',
    'ROAD_COLORS',
    'RoadLayer',
]

#: The color of the major and the minor streets and of each detail kind of
#: the scene.
ROAD_COLORS = {
    'major': (0.45, 0.45, 0.48, 1.0),
    'minor': (0.70, 0.70, 0.72, 1.0),
    'marks.white': (0.92, 0.92, 0.92, 1.0),
    'marks.yellow': (0.95, 0.75, 0.15, 1.0),
    'crosswalks': (0.92, 0.92, 0.92, 1.0),
    'signals.heads': (0.85, 0.15, 0.15, 1.0),
    'signals.poles': (0.25, 0.25, 0.27, 1.0),
}
#: The switch of :class:`RoadLayer` that shows each detail kind.
DETAIL_SWITCHES = {
    'marks.white': 'marks',
    'marks.yellow': 'marks',
    'crosswalks': 'marks',
    'signals.heads': 'signals',
    'signals.poles': 'signals',
}


class _Item(typing.NamedTuple):
    """One viewer object of the layer. ``rank`` is ``'major'``,
    ``'minor'`` or ``'detail'``; ``switch`` is ``'roads'``, ``'marks'`` or
    ``'signals'``."""

    name: str
    rank: str
    switch: str
    color: tuple
    build: typing.Callable


class RoadLayer(object):
    """The street groups of a scene as viewer objects, gray, the major
    streets darker, and the details on them in their colors. The minor
    streets and the details drop out from afar, where they would blur into
    one gray sheet. Three switches show the streets, the marks with the
    crosswalks, and the signals.

    The meshes are built on the first attach and kept for the next viewer.

    :ivar shown: Whether each object was last set visible, by name.
    :ivar build_seconds: The seconds building the meshes took.
    """

    def __init__(self, scene):
        self._scene = scene
        self._viewer = None
        self._items = None
        self._meshes = {}
        self.visible = True
        self.marks_visible = True
        self.signals_visible = True
        self.wide = False
        self.shown = {}
        self.build_seconds = 0.0

    @property
    def items(self):
        """Every object of the layer, the street groups first. Listing the
        details builds their arrays."""
        if self._items is None:
            scene = self._scene
            items = []
            for name in scene.road_groups:
                rank = 'minor' if scene.is_minor(name) else 'major'
                items.append(_Item(name, rank, 'roads', ROAD_COLORS[rank],
                                   functools.partial(scene.road_mesh, name)))
            for name, kind in scene.detail_groups.items():
                items.append(_Item(name, 'detail', DETAIL_SWITCHES[kind],
                                   ROAD_COLORS[kind],
                                   functools.partial(scene.detail_mesh,
                                                     name)))
            self._items = items
        return self._items

    @property
    def names(self):
        return [item.name for item in self.items]

    def attach(self, viewer):
        """Add every object to ``viewer`` and show what the switches
        allow."""
        if not self._meshes:
            start = time.perf_counter()
            self._meshes = {item.name: item.build() for item in self.items}
            self.build_seconds = time.perf_counter() - start
        for item in self.items:
            viewer.addObject(item.name, self._meshes[item.name])
            viewer.setObjectColor(item.name, *item.color)
        self._viewer = viewer
        self._apply()

    def detach(self):
        """Let go of the viewer before Qt frees it."""
        self._viewer = None

    def set_visible(self, visible):
        self.visible = visible
        self._apply()

    def set_marks_visible(self, visible):
        self.marks_visible = visible
        self._apply()

    def set_signals_visible(self, visible):
        self.signals_visible = visible
        self._apply()

    def set_wide(self, wide):
        """Hide the minor streets and the details while ``wide``, as the
        eye is far."""
        self.wide = wide
        self._apply()

    def _apply(self):
        viewer = self._viewer
        if viewer is None:
            return
        switches = {'roads': self.visible, 'marks': self.marks_visible,
                    'signals': self.signals_visible}
        for item in self.items:
            shown = switches[item.switch] and not (
                self.wide and item.rank != 'major')
            viewer.setObjectVisible(item.name, shown)
            self.shown[item.name] = shown

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
