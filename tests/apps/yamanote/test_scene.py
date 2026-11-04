# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The streets of the Yamanote scene, from the synthetic street map of
:mod:`synthetic_roads` next to this file."""

import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import solvcon
from solvcon.apps.yamanote import _scene

import synthetic_roads


class SceneRoadsTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.roads_path = synthetic_roads.write_roads(pathlib.Path(
            cls._tmp.name))
        cls.scene = _scene.Scene(roads_path=cls.roads_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_groups_split_each_segment_into_major_and_minor(self):
        groups = {name: [road.id for road in roads]
                  for name, roads in self.scene.road_groups.items()}
        self.assertEqual(synthetic_roads.GROUPS, groups)
        self.assertEqual(5, self.scene.road_count)
        self.assertGreater(self.scene.roads_seconds, 0.0)
        self.assertEqual(
            ['roads.JY01-JY02.minor', 'roads.JY02-JY03.minor'],
            [name for name in groups if self.scene.is_minor(name)])

    def test_a_slab_spans_the_carriageway_under_the_surface(self):
        ndcrd, clnds = self.scene.road_arrays('roads.JY01-JY02.major')
        # 120 m at 10 m spacing: 12 cells over 13 sections of 4 nodes.
        self.assertEqual((52, 3), ndcrd.shape)
        self.assertEqual((12, 8), clnds.shape)
        self.assertEqual((-6.0, 6.0), (ndcrd[:, 1].min(), ndcrd[:, 1].max()))
        sections = ndcrd.reshape(-1, 4, 3)
        np.testing.assert_allclose(sections[:, :2, 2], 2.0)
        np.testing.assert_allclose(sections[:, 2:, 2],
                                   2.0 - _scene.Scene.ROAD_THICKNESS)

    def test_a_climbing_street_keeps_its_top_on_the_surface(self):
        ndcrd, _ = self.scene.road_arrays('roads.JY02-JY03.minor')
        top = ndcrd.reshape(-1, 4, 3)[:, 0]
        climbing = top[top[:, 1] > 60.0]
        np.testing.assert_allclose(
            climbing[:, 2], 2.0 + 0.04 * (climbing[:, 0] - 200.0))

    def test_a_group_builds_one_hex_mesh(self):
        mh = self.scene.road_mesh('roads.JY02-JY03.minor')
        # u1 is 45 m (4 cells), r2 is 50 m (5 cells).
        self.assertEqual(9, mh.ncell)
        self.assertTrue((mh.cltpn.ndarray
                         == solvcon.StaticMesh.HEXAHEDRON).all())

    def test_details_of_a_segment_cover_both_classes(self):
        self.assertEqual(synthetic_roads.DETAILS, self.scene.detail_groups)
        self.assertEqual(['JY01-JY02', 'JY02-JY03'],
                         list(self.scene.segments))
        self.assertEqual(['p1', 'r1'], [road.id for road in
                                        self.scene.segments['JY01-JY02']])
        self.assertEqual(1, self.scene.crosswalk_count)
        self.assertEqual(2, self.scene.signal_count)
        mh = self.scene.detail_mesh('marks.JY02-JY03.yellow')
        self.assertTrue((mh.cltpn.ndarray
                         == solvcon.StaticMesh.HEXAHEDRON).all())
        with self.assertRaises(KeyError):
            self.scene.detail_mesh('crosswalks.JY02-JY03')

    def test_marks_float_over_the_slab(self):
        ndcrd, clnds = self.scene.mark_arrays('JY01-JY02', 'yellow')
        # The center lines of p1 (120 m) and r1 (35 m), in 10 m steps.
        self.assertEqual(12 + 4, len(clnds))
        top = 2.0 + _scene.Scene.MARK_LIFT
        sections = ndcrd.reshape(-1, 4, 3)
        np.testing.assert_allclose(sections[:, :2, 2], top)
        np.testing.assert_allclose(sections[:, 2:, 2],
                                   top - _scene.Scene.MARK_THICKNESS)
        np.testing.assert_allclose(sorted(set(np.round(ndcrd[:, 1], 9))),
                                   [-0.075, 0.075, 39.925, 40.075])

    def test_broken_lines_only_part_lanes_of_one_side(self):
        ndcrd, clnds = self.scene.mark_arrays('JY01-JY02', 'white')
        middle = np.round(ndcrd[clnds][:, :, 1].mean(axis=1), 9)
        # p1 has two lanes a side 3 m wide, r1 one lane a side 2.5 m wide.
        self.assertEqual({-6.0, -3.0, 3.0, 6.0, 37.5, 42.5}, set(middle))
        # 120 m of dashes of 3 m every 9 m.
        self.assertEqual(14, np.count_nonzero(middle == 3.0))
        self.assertEqual(12, np.count_nonzero(middle == 6.0))

    def test_a_crosswalk_tops_the_marks(self):
        ndcrd, clnds = self.scene.crosswalk_arrays('JY01-JY02')
        self.assertEqual((1, 8), clnds.shape)
        np.testing.assert_allclose(ndcrd.min(axis=0),
                                   (108.0, -6.0, 2.05), atol=1e-12)
        np.testing.assert_allclose(ndcrd.max(axis=0),
                                   (112.0, 6.0, 2.08), atol=1e-12)
        self.assertEqual((0, 8),
                         self.scene.crosswalk_arrays('JY02-JY03')[1].shape)

    def test_signal_heads_sit_on_their_poles(self):
        poles, _ = self.scene.signal_arrays('JY01-JY02', 'poles')
        heads, _ = self.scene.signal_arrays('JY01-JY02', 'heads')
        self.assertEqual((16, 3), poles.shape)
        np.testing.assert_allclose(poles[:, 2].min(), 2.0)
        np.testing.assert_allclose(poles[:, 2].max(), 7.0)
        np.testing.assert_allclose(heads[:, 2].min(), 7.0)
        np.testing.assert_allclose(heads[:, 2].max(), 7.4)
        # The head is 1.2 m across the road and 0.4 m along it.
        left = heads[:8]
        np.testing.assert_allclose(np.ptp(left[:, :2], axis=0), (0.4, 1.2),
                                   atol=1e-12)
        np.testing.assert_allclose(left[:, 1].mean(), 6.0)
        np.testing.assert_allclose(heads[8:, 1].mean(), -6.0)

    def test_the_default_street_map_sits_next_to_the_line(self):
        with mock.patch.object(_scene, 'ROADS_PATH', self.roads_path):
            scene = _scene.Scene()
        self.assertEqual(set(synthetic_roads.GROUPS), set(scene.road_groups))


class SceneWithoutRoadsTC(unittest.TestCase):

    def test_a_missing_street_map_leaves_the_line_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = pathlib.Path(tmp) / 'yamanote_roads.xodr'
            with mock.patch.object(_scene, 'ROADS_PATH', missing):
                scene = _scene.Scene()
        self.assertIsNone(scene.roads_map)
        self.assertEqual({}, scene.road_groups)
        self.assertEqual({}, scene.detail_groups)
        self.assertEqual((0, 0), (scene.signal_count,
                                  scene.crosswalk_count))
        self.assertEqual(0, scene.road_count)
        self.assertEqual(60, len(scene.stations))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
