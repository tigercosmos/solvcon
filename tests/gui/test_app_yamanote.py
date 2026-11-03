# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import importlib.util
import pathlib
import tempfile
import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.yamanote import _app, _control, _scene
except ImportError:
    pilot = None


def _load_synthetic_roads():
    """Import the street map builder the scene tests share, which lives
    with them rather than on the path of this directory."""
    path = (pathlib.Path(__file__).parents[1] / 'apps' / 'yamanote'
            / 'synthetic_roads.py')
    spec = importlib.util.spec_from_file_location('synthetic_roads', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synthetic_roads = _load_synthetic_roads()


@testing.requires_live_window
class YamanoteAppTC(unittest.TestCase):
    """Run the app on the line map and the synthetic street map, so the
    tests neither need nor wait for the full street map."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.roads_path = synthetic_roads.write_roads(pathlib.Path(
            cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.feature = _app.YamanoteApp(mgr=self.mgr)
        self.feature.populate_menu()
        self.feature._scene = _scene.Scene(roads_path=self.roads_path)

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
        roads = set(synthetic_roads.GROUPS)
        details = set(synthetic_roads.DETAILS)
        self.assertEqual(names, set(_scene.LOOPS) | wide | stops | cars
                         | roads | details)

    def _near(self):
        """Run the app with the eye close enough for every detail."""
        feature = self.feature
        feature.run()
        feature.viewer.zoomCamera(40.0)
        feature.control.update_lod()
        return feature.roads, feature._dock.widget()

    def test_the_roads_switch_hides_every_street(self):
        roads, panel = self._near()
        names = set(synthetic_roads.GROUPS) | set(synthetic_roads.DETAILS)
        self.assertEqual(names, set(roads.names))
        self.assertTrue(names <= set(self.feature.viewer.objectNames()))
        self.assertTrue(panel._roads.isEnabled())
        self.assertTrue(panel.roads_shown())
        self.assertEqual(dict.fromkeys(names, True), roads.shown)

        panel._roads.setChecked(False)
        streets = set(synthetic_roads.GROUPS)
        self.assertEqual({name: name not in streets for name in names},
                         roads.shown)
        panel._roads.setChecked(True)
        self.assertEqual(dict.fromkeys(names, True), roads.shown)

    def test_the_detail_switches_hide_their_kinds(self):
        roads, panel = self._near()
        self.assertTrue(panel._marks.isEnabled())
        self.assertTrue(panel._signals.isEnabled())
        self.assertTrue(panel.marks_shown() and panel.signals_shown())
        kinds = synthetic_roads.DETAILS

        panel._marks.setChecked(False)
        self.assertFalse(panel.marks_shown())
        hidden = {name for name, shown in roads.shown.items() if not shown}
        self.assertEqual({name for name, kind in kinds.items()
                          if kind.startswith(('marks.', 'crosswalks'))},
                         hidden)
        panel._marks.setChecked(True)
        panel._signals.setChecked(False)
        hidden = {name for name, shown in roads.shown.items() if not shown}
        self.assertEqual({'signals.JY01-JY02.heads',
                          'signals.JY01-JY02.poles'}, hidden)
        panel._signals.setChecked(True)
        self.assertTrue(all(roads.shown.values()))

    def test_minor_streets_and_details_drop_out_from_afar(self):
        feature = self.feature
        feature.run()
        control, roads = feature.control, feature.roads
        self.assertTrue(control.update_lod(force=True))
        self.assertTrue(control.lod_wide)
        self.assertTrue(roads.wide)
        minor = {name for name in roads.names
                 if feature.scene.is_minor(name)}
        self.assertEqual({'roads.JY01-JY02.minor', 'roads.JY02-JY03.minor'},
                         minor)
        major = {'roads.JY01-JY02.major', 'roads.JY02-JY03.major'}
        self.assertEqual({name: name in major for name in roads.names},
                         roads.shown)

        feature.viewer.zoomCamera(40.0)
        self.assertTrue(control.update_lod())
        self.assertFalse(roads.wide)
        self.assertTrue(all(roads.shown.values()))

    def test_the_line_runs_without_a_street_map(self):
        feature = self.feature
        feature._scene = _scene.Scene(
            roads_path=pathlib.Path(self._tmp.name) / 'missing.xodr')
        feature.run()
        self.assertIsNone(feature.roads)
        panel = feature._dock.widget()
        self.assertFalse(panel._roads.isEnabled())
        self.assertFalse(panel._marks.isEnabled())
        self.assertFalse(panel._signals.isEnabled())
        names = set(feature.viewer.objectNames())
        self.assertFalse({name for name in names if name.startswith(
            ('roads.', 'marks.', 'crosswalks.', 'signals.'))})
        self.assertTrue(feature.control.update_lod(force=True))

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
