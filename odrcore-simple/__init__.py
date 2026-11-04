# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Data-structure skeleton of ``odrcore``: a normalized subset of OpenDRIVE
1.8 that is a superset of the Lanelet2 core model. See README.md.
"""

from . import geom, model, osm, topology, view  # noqa: F401
from .model import Map  # noqa: F401

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
