# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The JR East Yamanote Line drawn as a 3D map.

The scene reads the OpenDRIVE map that ``odrcore/examples`` carries, turns
each loop into a slab mesh at its surveyed height and places a marker at
every stop. The scene imports without the pilot; the feature that shows it
in a domain viewer rides behind the pilot toggle.
"""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_app': ['YamanoteApp'],
    '_scene': ['Scene', 'Station'],
    '_timetable': ['LoopLine', 'Timetable'],
}, ungated=('_scene', '_timetable'))

if pilot.enable:
    from . import _app
    pilot.register_app("yamanote", _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
