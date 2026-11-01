# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Draw the streets around the Yamanote Line into the viewer, one object
per group of the scene."""

import time

__all__ = [
    'ROAD_COLORS',
    'RoadLayer',
]

ROAD_COLORS = {
    'major': (0.45, 0.45, 0.48, 1.0),
    'minor': (0.70, 0.70, 0.72, 1.0),
}


class RoadLayer(object):
    """The street groups of a scene as viewer objects, gray, the major
    streets darker.  The minor streets drop out from afar, where they would
    blur into one gray sheet.

    The meshes are built on the first attach and kept for the next viewer.

    :ivar shown: Whether each object was last set visible, by name.
    :ivar build_seconds: The seconds building the meshes took.
    """

    def __init__(self, scene):
        self._scene = scene
        self._viewer = None
        self._meshes = {}
        self.visible = True
        self.wide = False
        self.shown = {}
        self.build_seconds = 0.0

    @property
    def names(self):
        return list(self._scene.road_groups)

    def attach(self, viewer):
        """Add every group to ``viewer`` and show what the switches allow."""
        if not self._meshes:
            start = time.perf_counter()
            self._meshes = {name: self._scene.road_mesh(name)
                            for name in self.names}
            self.build_seconds = time.perf_counter() - start
        for name, mesh in self._meshes.items():
            viewer.addObject(name, mesh)
            rank = 'minor' if self._scene.is_minor(name) else 'major'
            viewer.setObjectColor(name, *ROAD_COLORS[rank])
        self._viewer = viewer
        self._apply()

    def detach(self):
        """Let go of the viewer before Qt frees it."""
        self._viewer = None

    def set_visible(self, visible):
        self.visible = visible
        self._apply()

    def set_wide(self, wide):
        """Hide the minor streets while ``wide``, as the eye is far."""
        self.wide = wide
        self._apply()

    def _apply(self):
        viewer = self._viewer
        if viewer is None:
            return
        for name in self.names:
            shown = self.visible and not (
                self.wide and self._scene.is_minor(name))
            viewer.setObjectVisible(name, shown)
            self.shown[name] = shown

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
