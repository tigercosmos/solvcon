# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The synthetic Yamanote timetable on a square loop of straight roads, and
once on the real map."""

import math
import unittest

from odrcore.geom import Line, PiecewisePoly3, RefLine
from odrcore.model import Road

from solvcon.apps.yamanote import LoopLine, Scene, Timetable

SIDE = 1000.0
STOPS = (100.0, 1500.0, 3200.0)


def square_loop():
    """Four straight roads around a square, end to start, at z = 3."""
    corners = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE)]
    return LoopLine(
        Road(str(k), length=SIDE,
             refline=RefLine([Line(0.0, SIDE, x, y, k * math.pi / 2)],
                             PiecewisePoly3.constant(3.0)))
        for k, (x, y) in enumerate(corners))


class LoopLineTC(unittest.TestCase):

    def setUp(self):
        self.line = square_loop()

    def test_length_is_the_sum_of_the_roads(self):
        self.assertEqual(4 * SIDE, self.line.length)

    def test_global_s_lands_on_its_road(self):
        for s, rid, local in ((0.0, '0', 0.0), (999.0, '0', 999.0),
                              (1000.0, '1', 0.0), (2500.0, '2', 500.0),
                              (4000.0, '0', 0.0), (-250.0, '3', 750.0),
                              (9300.0, '1', 300.0)):
            road, got = self.line.locate(s)
            self.assertEqual(rid, road.id)
            self.assertAlmostEqual(local, got)
            self.assertTrue(0.0 <= got <= road.length)

    def test_pose_follows_the_road_heading(self):
        x, y, z, hdg = self.line.pose(1250.0)
        self.assertAlmostEqual(SIDE, x)
        self.assertAlmostEqual(250.0, y)
        self.assertAlmostEqual(3.0, z)
        self.assertAlmostEqual(math.pi / 2, hdg)

    def test_cars_trail_the_head_a_pitch_apart(self):
        # The head is 500 m down the third road, which runs to -x.
        cars = self.line.car_poses(2500.0, 11, 20.0)
        self.assertEqual(11, len(cars))
        for i, (x, y, z, hdg) in enumerate(cars):
            self.assertAlmostEqual(500.0 + (i + 0.5) * 20.0, x)
            self.assertAlmostEqual(SIDE, y)
            self.assertAlmostEqual(3.0, z)
            self.assertAlmostEqual(math.pi, hdg)

    def test_cars_follow_the_loop_around_the_wrap(self):
        # The head is just past the start on its second lap, so the cars
        # behind the first are still on the last road, which runs to -y.
        cars = self.line.car_poses(4 * SIDE + 25.0, 4, 20.0)
        expected = [(15.0, 0.0, 0.0), (0.0, 5.0, 3 * math.pi / 2),
                    (0.0, 25.0, 3 * math.pi / 2),
                    (0.0, 45.0, 3 * math.pi / 2)]
        for (x, y, _, hdg), (ex, ey, ehdg) in zip(cars, expected):
            self.assertAlmostEqual(ex, x)
            self.assertAlmostEqual(ey, y)
            self.assertAlmostEqual(ehdg, hdg)


class TimetableTC(unittest.TestCase):

    def setUp(self):
        self.tt = Timetable(4 * SIDE, STOPS, lap_time=600.0, dwell=30.0,
                            headway=120.0)

    def test_speed_fills_the_lap_after_the_dwells(self):
        self.assertAlmostEqual(4 * SIDE / (600.0 - 3 * 30.0), self.tt.speed)
        self.assertEqual(5, self.tt.ntrain)

    def test_train_stands_exactly_during_a_dwell(self):
        tt = self.tt
        arrive = tt.dwell + (STOPS[1] - STOPS[0]) / tt.speed
        for t in (arrive, arrive + 10.0, arrive + tt.dwell):
            self.assertAlmostEqual(STOPS[1], tt.position(0, t))
        self.assertGreater(tt.position(0, arrive + tt.dwell + 1.0),
                           STOPS[1])
        self.assertLess(tt.position(0, arrive - 1.0), STOPS[1])

    def test_train_cruises_between_stations(self):
        tt = self.tt
        t0 = tt.dwell + 5.0
        ds = tt.position(0, t0 + 20.0) - tt.position(0, t0)
        self.assertAlmostEqual(tt.speed * 20.0, ds)

    def test_one_lap_takes_the_lap_time(self):
        tt = self.tt
        for t in (0.0, 17.0, 333.3):
            self.assertAlmostEqual(tt.position(0, t) + tt.length,
                                   tt.position(0, t + tt.lap_time))

    def test_position_is_continuous_across_the_lap(self):
        tt = self.tt
        before = tt.position(0, tt.lap_time - 1e-6)
        after = tt.position(0, tt.lap_time + 1e-6)
        self.assertAlmostEqual(before, after, places=3)

    def test_trains_follow_a_headway_apart(self):
        tt = self.tt
        for k in range(tt.ntrain - 1):
            for t in (0.0, 45.0, 500.0):
                self.assertAlmostEqual(tt.position(k, t),
                                       tt.position(k + 1, t + tt.headway))

    def test_no_time_to_run_is_refused(self):
        with self.assertRaises(ValueError):
            Timetable(4 * SIDE, STOPS, lap_time=90.0, dwell=30.0)


class RealMapTC(unittest.TestCase):

    def test_trains_run_on_both_loops(self):
        scene = Scene()
        for loop, chain in scene.chains.items():
            line = LoopLine(chain)
            stops = [st.s for st in scene.stations if st.loop == loop]
            tt = Timetable(line.length, stops)
            self.assertEqual(30, len(stops))
            self.assertEqual(15, tt.ntrain)
            self.assertTrue(30000.0 < line.length < 40000.0)
            self.assertTrue(40.0 < tt.speed * 3.6 < 50.0)
            first = next(st for st in scene.stations if st.loop == loop)
            x, y, _, _ = line.pose(tt.position(0, 0.0))
            self.assertAlmostEqual(first.x, x, places=6)
            self.assertAlmostEqual(first.y, y, places=6)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
