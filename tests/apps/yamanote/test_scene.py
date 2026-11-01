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
        self.assertEqual(0, scene.road_count)
        self.assertEqual(60, len(scene.stations))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
