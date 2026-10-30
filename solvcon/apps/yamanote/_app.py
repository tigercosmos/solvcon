# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The pilot feature that draws the Yamanote Line in a 3D sub-window and
runs its trains on the timetable from a dock panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget

from ...pilot import base

from . import _scene
from ._control import TrainController
from ._panel import TrainPanel

__all__ = [  # noqa: F822
    'YamanoteApp',
    'build_app',
]


class YamanoteApp(base.PilotFeature):
    """Open the Yamanote Line scene: both loops as slabs at their surveyed
    height, a marker at every stop and the trains of both loops, in a
    domain viewer, with the dock panel that runs the trains."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._scene = None
        self._viewer = None
        self._dock = None
        self._control = None
        # Held for the viewer's lifetime: a throwaway mdiArea wrapper is
        # garbage-collected right after use, invalidating the sub-window.
        self._mdi = None
        self._subwin = None
        self._close_filter = None

    def populate_menu(self):
        self.add_action(
            "View", "Yamanote Line",
            "Draw the JR East Yamanote Line in a 3D sub-window",
            self.run, id="view.yamanote", weight=80)

    @property
    def scene(self):
        if self._scene is None:
            self._scene = _scene.Scene()
        return self._scene

    @property
    def viewer(self):
        """The open 3D widget, or None once its sub-window closed."""
        return self._viewer

    @property
    def control(self):
        return self._control

    def run(self):
        """Show the train panel and open the viewer if it is not open."""
        self._ensure_dock()
        self._dock.show()
        if self._viewer is None:
            self._open_viewer()

    def _ensure_dock(self):
        if self._dock is not None:
            return
        panel = TrainPanel()
        self._control = TrainController(panel, self.scene)
        self._dock = QDockWidget("Yamanote Line")
        self._dock.setWidget(panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, self._dock)

    def _open_viewer(self):
        scene = self.scene
        viewer = self._mgr.add3DWidget()
        for loop in _scene.LOOPS:
            viewer.addObject(loop, scene.loop_mesh(loop))
            viewer.setObjectColor(loop, *_scene.COLORS[loop])
        marker = scene.marker_mesh()
        for st in scene.stations:
            name = f'{st.code}.{st.loop}'
            viewer.addObject(name, marker)
            viewer.setObjectTransform(name, st.x, st.y, st.z,
                                      *_scene.STATION_SIZE)
            viewer.setObjectColor(name, *_scene.COLORS['station'])
        # A stop sits on both loops; label it once.
        for st in {st.code: st for st in scene.stations}.values():
            viewer.addLabel(st.code, f"{st.code} {st.name_en}",
                            st.x, st.y, st.z)
        self._control.attach(viewer, marker)
        viewer.showAxis(True)
        viewer.setView("iso")
        self._watch_close()
        self._viewer = viewer
        self._pycon.writeToHistory(
            f"yamanote: {len(scene.stations)} stops and "
            f"{len(self._control.names)} trains on "
            f"{len(_scene.LOOPS)} loops\n")

    def _watch_close(self):
        """Delete the sub-window on close and hear of that close before Qt
        frees the widget, so the trains stop drawing into it first."""
        if self._mdi is None:
            self._mdi = self._mgr.mdiArea
        self._subwin = self._mdi.activeSubWindow()
        if self._subwin is None:
            return
        self._subwin.setAttribute(Qt.WA_DeleteOnClose, True)
        self._close_filter = base.SubWindowCloseFilter(self._on_closed,
                                                       self._subwin)
        self._subwin.installEventFilter(self._close_filter)

    def _on_closed(self):
        self._control.detach()
        self._viewer = None
        self._subwin = None
        self._close_filter = None


def build_app(controller):
    return YamanoteApp(mgr=controller.mgr)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
