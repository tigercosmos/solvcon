# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Write the Lanelet2 form of a map as a Lanelet2 OSM file.

Each view linestring becomes one way oriented along s. The Lanelet2 loader
orients the two bounds of every lanelet by geometry, so lanelets of
opposite travel direction share a way without a copy.

Model x, y, z are east, north and up of a local tangent plane at
``origin``; ``lat``, ``lon`` and ``ele`` are the exact WGS84 inverse, so
``LocalCartesianProjector(Origin(lat, lon))`` gives back the model
coordinates. Each signal becomes a two-node way across the road, because
Lanelet2 signs and lights refer to a linestring.

There is no reader: the map is edited in its OpenDRIVE form only.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from .model import Map

Origin = tuple[float, float]
"""``(lat, lon)`` in degrees; default ``+lat_0``/``+lon_0`` of the
geoReference, else ``(0, 0)``."""

LANELET2_PARTICIPANTS = {
    'passengerCar': 'vehicle:car', 'bus': 'vehicle:bus',
    'truck': 'vehicle:truck', 'taxi': 'vehicle:taxi',
    'emergency': 'vehicle:emergency', 'motorcycle': 'vehicle:motorcycle',
    'bicycle': 'bicycle', 'pedestrian': 'pedestrian',
}
"""Written as ``participant:<name>=yes|no`` on every lanelet."""


def dumps(m: Map, spacing: float = 2.0,
          origin: Optional[Origin] = None) -> str:
    raise NotImplementedError


def dump(m: Map, path: Union[str, 'os.PathLike[str]'], spacing: float = 2.0,
         origin: Optional[Origin] = None) -> None:
    raise NotImplementedError

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
