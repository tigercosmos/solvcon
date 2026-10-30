# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.yamanote import _app, _control, _scene
except ImportError:
    pilot = None


@testing.requires_live_window
class YamanoteAppTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.feature = _app.YamanoteApp(mgr=self.mgr)
        self.feature.populate_menu()

    def tearDown(self):
        if self.feature._subwin is not None:
            self.feature._subwin.close()

    def test_run_draws_both_loops_every_stop_and_the_trains(self):
        feature = self.feature
        self.assertEqual("Yamanote Line",
                         self.mgr.menu_model.action("view.yamanote").text())
        feature.run()
        viewer = self.mgr.currentR3DWidget()
        self.assertIs(viewer, feature.viewer)
        names = set(viewer.objectNames())
        self.assertTrue(set(_scene.LOOPS) <= names)
        stops = {f"{st.code}.{st.loop}" for st in feature.scene.stations}
        self.assertEqual(len(stops), 60)
        trains = set(feature.control.names)
        self.assertEqual(30, len(trains))
        self.assertIn("train.Outer.03", trains)
        self.assertIn("train.Inner.14", trains)
        cars = set(feature.control.car_names)
        self.assertEqual(30 * _control.CAR_COUNT, len(cars))
        self.assertIn("train.Outer.03.car07", cars)
        wide = {feature.scene.wide_name(loop) for loop in _scene.LOOPS}
        self.assertEqual(names, set(_scene.LOOPS) | wide | stops | cars)

    def test_every_station_is_labeled_once(self):
        feature = self.feature
        feature.run()
        codes = {st.code for st in feature.scene.stations}
        self.assertEqual(30, len(codes))
        self.assertEqual(codes, set(feature.viewer.labelNames()))
        self.assertTrue(feature.viewer.labelsShown)

    def test_detail_follows_the_eye_to_pivot_distance(self):
        feature = self.feature
        feature.run()
        control, viewer = feature.control, feature.viewer
        self.assertTrue(control.update_lod(force=True))
        fitted = control.lod_distance
        self.assertGreater(fitted, _control.LOD_WIDE_DISTANCE)
        self.assertTrue(control.lod_wide)
        self.assertGreater(control.lod_factor, 1.0)
        self.assertFalse(control.update_lod())

        viewer.zoomCamera(40.0)
        self.assertTrue(control.update_lod())
        self.assertLess(control.lod_distance, _control.LOD_NEAR)
        self.assertFalse(control.lod_wide)
        self.assertEqual(1.0, control.lod_factor)

        viewer.setView("iso")
        self.assertTrue(control.update_lod())
        self.assertAlmostEqual(fitted, control.lod_distance, delta=1.0)
        self.assertTrue(control.lod_wide)

    def test_trains_move_as_the_clock_advances(self):
        feature = self.feature
        feature.run()
        control = feature.control
        panel = feature._dock.widget()
        self.assertEqual("05:00:00", panel.clock_text())
        control.play()
        self.assertTrue(control.playing)
        before = dict(control.poses)
        control.advance(90.0)
        self.assertEqual("05:01:30", panel.clock_text())
        moved = [name for name in control.names
                 if control.poses[name] != before[name]]
        # A train standing at a station for the whole step keeps its pose.
        self.assertGreater(len(moved), len(control.names) // 2)
        control.pause()
        self.assertFalse(control.playing)

    def test_closing_the_viewer_stops_the_trains(self):
        feature = self.feature
        feature.run()
        control = feature.control
        control.play()
        feature._subwin.close()
        self.assertIsNone(feature.viewer)
        self.assertFalse(control.playing)
        self.assertFalse(control._lod_timer.isActive())
        self.assertFalse(control.update_lod(force=True))
        control.advance(60.0)
        self.assertEqual(30, len(control.poses))
        feature.run()
        self.assertIsNotNone(feature.viewer)
        self.assertIn("train.Outer.00.car00", feature.viewer.objectNames())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
