# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Run the trains of both loops on the timetable clock into the viewer."""

import math

from PySide6.QtCore import QElapsedTimer, QTimer

from . import _scene
from ._timetable import LoopLine, Timetable

__all__ = [  # noqa: F822
    'TrainController',
]

#: An E235 runs 11 cars, each a box of its length, width and height in
#: meters, one pitch apart; the pitch less the length is the coupling gap.
CAR_COUNT = 11
CAR_PITCH = 20.0
CAR_SIZE = (19.5, 3.0, 4.0)
TRAIN_COLORS = {
    'Outer': (0.95, 0.55, 0.10, 1.0),
    'Inner': (0.20, 0.45, 0.90, 1.0),
}

#: The eye-to-pivot distance in meters at which the station markers and the
#: car cross sections start to grow with it, and the most they grow by.
LOD_NEAR = 1500.0
LOD_MAX_FACTOR = 12.0
#: Beyond this eye-to-pivot distance the wide slabs replace the true ones.
LOD_WIDE_DISTANCE = 3000.0
#: The relative change of the distance that re-evaluates the detail.
LOD_TOLERANCE = 0.03


def lod_scale(distance):
    """Return the scale of the markers and the car cross sections at an
    eye-to-pivot ``distance`` in meters."""
    return min(max(distance / LOD_NEAR, 1.0), LOD_MAX_FACTOR)


class TrainController(object):
    """Advance the clock and place every train in the viewer.

    The clock runs on the frame timer while playing: each frame adds the
    real time since the last one times the speed factor.  The viewer is
    optional; while none is attached the clock and the poses still run.

    A train is :data:`CAR_COUNT` boxes, each at its own place on the loop,
    so it bends with the track.  The geometry is to scale, which leaves the
    track, the markers and the trains too thin to see from afar: the level
    of detail swaps in wide slabs and fattens the markers and the cars as
    the eye backs off the pivot.

    :ivar time: Simulated seconds since :attr:`START`.
    :ivar poses: ``(x, y, z, heading)`` of the head of every train by train
        name, as last placed.
    :ivar lod_distance: The eye-to-pivot distance the detail was last
        evaluated at, or None before a viewer is attached.
    :ivar lod_factor: The scale of the markers and the car cross sections.
    :ivar lod_wide: Whether the wide slabs show in place of the true ones.
    :ivar wide_changed: Owner-supplied callback taking :attr:`lod_wide`
        each time the detail re-evaluates it, or None.
    """

    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50
    #: Interval of the level-of-detail check while paused, in milliseconds.
    LOD_INTERVAL_MS = 200
    #: The time of day the clock starts at, in seconds after midnight.
    START = 5 * 3600.0

    def __init__(self, panel, scene):
        self._panel = panel
        self._scene = scene
        self._viewer = None
        self.lines = {}
        self.timetables = {}
        for loop in _scene.LOOPS:
            line = LoopLine(scene.chains[loop])
            stops = [st.s for st in scene.stations if st.loop == loop]
            self.lines[loop] = line
            self.timetables[loop] = Timetable(line.length, stops)
        self.time = 0.0
        self.factor = panel.factor()
        self.poses = {}
        self.lod_distance = None
        self.lod_factor = 1.0
        self.lod_wide = False
        self.wide_changed = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._lod_timer = QTimer()
        self._lod_timer.timeout.connect(self.update_lod)
        self._elapsed = QElapsedTimer()
        panel.play_toggled = self._on_play
        panel.factor_changed = self._on_factor
        self._place()

    @property
    def names(self):
        return [self.train_name(loop, k) for loop in _scene.LOOPS
                for k in range(self.timetables[loop].ntrain)]

    @property
    def car_names(self):
        """Return the object names of every car of every train."""
        return [self.car_name(name, i) for name in self.names
                for i in range(CAR_COUNT)]

    @staticmethod
    def train_name(loop, train):
        return f'train.{loop}.{train:02d}'

    @staticmethod
    def car_name(train_name, car):
        return f'{train_name}.car{car:02d}'

    @property
    def playing(self):
        return self._timer.isActive()

    def attach(self, viewer, mesh):
        """Add a wide slab for every loop and a box of ``mesh`` for every
        car to ``viewer``, place them and start checking the detail."""
        for loop in _scene.LOOPS:
            name = self._scene.wide_name(loop)
            viewer.addObject(name, self._scene.loop_mesh(
                loop, self._scene.WIDE_HALF_WIDTH))
            viewer.setObjectColor(name, *_scene.COLORS[loop])
            for k in range(self.timetables[loop].ntrain):
                train = self.train_name(loop, k)
                for i in range(CAR_COUNT):
                    car = self.car_name(train, i)
                    viewer.addObject(car, mesh)
                    viewer.setObjectColor(car, *TRAIN_COLORS[loop])
        self._viewer = viewer
        self.update_lod(force=True)
        self._lod_timer.start(self.LOD_INTERVAL_MS)

    def detach(self):
        """Stop and let go of the viewer before Qt frees it."""
        self.pause()
        self._lod_timer.stop()
        self._viewer = None
        self.lod_distance = None

    def play(self):
        self._elapsed.start()
        self._timer.start(self.INTERVAL_MS)
        self._panel.set_playing(True)

    def pause(self):
        self._timer.stop()
        self._panel.set_playing(False)

    def advance(self, seconds):
        """Move the clock ``seconds`` of simulated time on and place the
        trains there."""
        self.time += seconds
        self._place()

    def update_lod(self, force=False):
        """Fit the detail to the eye-to-pivot distance of the viewer, unless
        it changed by no more than :data:`LOD_TOLERANCE` and not ``force``.

        :return: Whether the detail was re-evaluated.
        """
        viewer = self._viewer
        if viewer is None:
            return False
        eye, pivot = viewer.cameraPosition, viewer.cameraTarget
        distance = math.dist(eye, pivot)
        last = self.lod_distance
        if (not force and last is not None
                and abs(distance - last) <= LOD_TOLERANCE * last):
            return False
        self.lod_distance = distance
        factor = lod_scale(distance)
        wide = distance > LOD_WIDE_DISTANCE
        if force or wide != self.lod_wide:
            self.lod_wide = wide
            for loop in _scene.LOOPS:
                viewer.setObjectVisible(loop, not wide)
                viewer.setObjectVisible(self._scene.wide_name(loop), wide)
            if self.wide_changed is not None:
                self.wide_changed(wide)
        if force or factor != self.lod_factor:
            self.lod_factor = factor
            sx, sy, sz = (factor * v for v in _scene.STATION_SIZE)
            for st in self._scene.stations:
                viewer.setObjectTransform(st.name, st.x, st.y, st.z,
                                          sx, sy, sz)
            if not self.playing:
                self._place()
        return True

    def _tick(self):
        self.update_lod()
        self.advance(self._elapsed.restart() / 1000.0 * self.factor)

    def _on_play(self, checked):
        if checked:
            self.play()
        else:
            self.pause()

    def _on_factor(self, factor):
        self.factor = factor

    def _place(self):
        viewer = self._viewer
        length, width, height = CAR_SIZE
        width *= self.lod_factor
        height *= self.lod_factor
        lift = height / 2.0
        for loop in _scene.LOOPS:
            line, tt = self.lines[loop], self.timetables[loop]
            for k in range(tt.ntrain):
                name = self.train_name(loop, k)
                head = tt.position(k, self.time)
                self.poses[name] = line.pose(head)
                if viewer is None:
                    continue
                cars = line.car_poses(head, CAR_COUNT, CAR_PITCH)
                for i, (x, y, z, hdg) in enumerate(cars):
                    viewer.setObjectTransform(
                        self.car_name(name, i), x, y, z + lift,
                        length, width, height, hdg)
        self._panel.show_clock(self.START + self.time)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
