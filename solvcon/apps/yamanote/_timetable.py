# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
A synthetic timetable for the trains running around one loop.

A loop's roads, laid end to end, give one global arc length ``s`` that wraps
at the loop length.  Every train runs the same piecewise-linear schedule in
the direction of ``s``: it cruises at one speed between stations and stands
for a fixed dwell at each.  The trains follow one another a fixed headway
apart.  Qt-free, so the timetable runs headlessly.
"""

import bisect

import numpy as np

__all__ = [
    'LoopLine',
    'Timetable',
]


class LoopLine(object):
    """A closed chain of roads on one global arc length.

    :ivar offsets: The global ``s`` at the start of every road, followed by
        the loop length.
    """

    def __init__(self, chain):
        self.chain = list(chain)
        self.offsets = np.cumsum(
            [0.0] + [road.length for road in self.chain], dtype='float64')

    @property
    def length(self):
        return float(self.offsets[-1])

    def locate(self, s):
        """Return the road a global ``s`` (taken modulo the loop length) falls
        on and the local ``s`` along that road."""
        s = s % self.length
        i = min(bisect.bisect_right(self.offsets, s) - 1, len(self.chain) - 1)
        road = self.chain[i]
        return road, min(s - float(self.offsets[i]), road.length)

    def pose(self, s):
        """Return ``(x, y, z, heading)`` in map meters and radians at global
        ``s``."""
        road, local = self.locate(s)
        x, y, z, hdg, _ = road.refline.pose(local)
        return x, y, z, hdg

    def car_poses(self, head, ncar, pitch):
        """Return the :meth:`pose` at the middle of each of ``ncar`` cars
        ``pitch`` long, nose to tail behind a train head at global ``head``."""
        return [self.pose(head - (i + 0.5) * pitch) for i in range(ncar)]


class Timetable(object):
    """The schedule every train of one loop runs, a headway apart.

    The cruise speed makes one lap, dwells included, take ``lap_time``.
    Train ``k`` runs the schedule of train 0 delayed by ``k`` headways, and
    as many trains as fit in a lap are on the line.

    :ivar times: Knot times in seconds of one lap of train 0, from its
        arrival at the first station at time 0 to ``lap_time``.
    :ivar places: The unwrapped global ``s`` at every knot; a dwell is a
        pair of knots at one ``s``.
    """

    def __init__(self, length, stops, lap_time=3660.0, dwell=30.0,
                 headway=240.0):
        stops = sorted(s % length for s in stops)
        running = lap_time - len(stops) * dwell
        if running <= 0.0:
            raise ValueError(
                f"a lap of {lap_time} s leaves no time to run between "
                f"{len(stops)} dwells of {dwell} s")
        self.length = length
        self.lap_time = lap_time
        self.dwell = dwell
        self.headway = headway
        self.speed = length / running
        self.ntrain = max(1, int(lap_time // headway))

        first = stops[0] if stops else 0.0
        times, places, t = [0.0], [first], 0.0
        for here, there in zip(stops, stops[1:] + [first + length]):
            t += dwell
            times.append(t)
            places.append(here)
            t += (there - here) / self.speed
            times.append(t)
            places.append(there)
        if not stops:
            times.append(lap_time)
            places.append(length)
        # Pin the lap end against round-off, so a lap closes exactly.
        times[-1] = lap_time
        self.times = np.array(times, dtype='float64')
        self.places = np.array(places, dtype='float64')

    def position(self, train, t):
        """Return the unwrapped global ``s`` of ``train`` at time ``t`` in
        seconds; it is continuous in ``t`` and grows by one loop length a
        lap."""
        laps, tau = divmod(t - train * self.headway, self.lap_time)
        return float(np.interp(tau, self.times, self.places)
                     + laps * self.length)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
