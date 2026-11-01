# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A small stand-in for ``yamanote_roads.xodr``: five straight roads of two
station segments, major and minor, so no test needs the 52 MB map.

The GUI tests live in another directory and load this module by its path.
"""

import odrcore
from odrcore.geom import Line, PiecewisePoly3, RefLine
from odrcore.model import (Boundary, Lane, LaneSection, Map, Road,
                           boundary_id)

#: ``(id, highway, segment, x0, y0, length, lane width, lanes per side,
#: grade)``; every road runs along x at z = 2 m plus its grade.
ROADS = (
    ('p1', 'primary', 'JY01-JY02', 0.0, 0.0, 120.0, 3.0, 2, 0.0),
    ('r1', 'residential', 'JY01-JY02', 0.0, 40.0, 35.0, 2.5, 1, 0.0),
    ('l1', 'secondary_link', 'JY02-JY03', 200.0, 0.0, 60.0, 3.5, 1, 0.0),
    ('u1', 'unclassified', 'JY02-JY03', 200.0, 40.0, 45.0, 2.75, 1, 0.0),
    ('r2', 'residential', 'JY02-JY03', 200.0, 80.0, 50.0, 2.5, 1, 0.04),
)
#: The ids of :data:`ROADS` in each group, by object name.
GROUPS = {
    'roads.JY01-JY02.major': ['p1'],
    'roads.JY01-JY02.minor': ['r1'],
    'roads.JY02-JY03.major': ['l1'],
    'roads.JY02-JY03.minor': ['u1', 'r2'],
}


def _road(road_id, highway, segment, x0, y0, length, width, nlane, grade):
    road = Road(road_id, length=length, refline=RefLine(
        [Line(0.0, length, x0, y0, 0.0)],
        PiecewisePoly3([(0.0, 2.0, grade, 0.0, 0.0)])))
    road.attrs['userData'] = (
        f'<userData code="osm:highway" value="{highway}" />\n'
        f'<userData code="segment" value="{segment}" />')
    sec = LaneSection(0.0, road)
    road.sections.append(sec)
    sec.center = Boundary(boundary_id(road_id, 0, 0), sec,
                          PiecewisePoly3.constant(0.0))
    for sign, lanes in ((1, sec.left), (-1, sec.right)):
        inner = sec.center
        for k in range(1, nlane + 1):
            outer = Boundary(boundary_id(road_id, 0, sign * k), sec,
                             PiecewisePoly3.constant(sign * width * k))
            lanes.append(Lane(sec, sign * k, inner, outer))
            inner = outer
    return road


def write_roads(directory):
    """Write the roads of :data:`ROADS` to ``yamanote_roads.xodr`` in
    ``directory`` and return its path."""
    m = Map(roads={spec[0]: _road(*spec) for spec in ROADS})
    m.reindex()
    path = directory / 'yamanote_roads.xodr'
    odrcore.dump(m, path)
    return path

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
