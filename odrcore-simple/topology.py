# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Derived connectivity. The file's links are the single source of truth;
``LinkIndex`` is built from them, cached on the ``Map`` and never written.

Sources, in order: ``Lane.link`` between consecutive sections, ``Road.link``
of type ``road`` joined with the lane links of the end section, and junction
connections for the road end whose ``Road.link`` names the junction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import End, Junction, Lane, Map


@dataclass(frozen=True)
class Link:
    """``src`` at ``src_end`` touches ``dst`` at ``dst_end`` in the s sense,
    which also covers head-to-head joins. ``contact`` is the end of ``dst``
    that is touched, as OpenDRIVE's contactPoint."""

    src: Lane
    dst: Lane
    via_junction: Optional[Junction]
    src_end: End
    dst_end: End

    @property
    def contact(self) -> End:
        return self.dst_end


class LinkIndex:
    """Symmetric: every link is also known from ``dst``'s side.

    ``links`` is the flat list; ``at_end(lane, end)`` the links touching
    one end of a lane. ``successors``/``predecessors`` read the index in
    travel direction (``Lane.travels_along_s``). ``neighbors(lane)`` is
    ``(left, right)``: the other lane holding the same ``Boundary`` object
    as ``left_bound``/``right_bound``. ``conflicting`` (crossing lanes in
    a junction) is a stub.
    """

    def __init__(self, m: Map) -> None:
        self.map = m
        self.links: list[Link] = []

    def at_end(self, lane: Lane, end: End) -> list[Link]:
        raise NotImplementedError

    def successors(self, lane: Lane) -> list[Lane]:
        raise NotImplementedError

    def predecessors(self, lane: Lane) -> list[Lane]:
        raise NotImplementedError

    def neighbors(self, lane: Lane) -> tuple[Optional[Lane], Optional[Lane]]:
        raise NotImplementedError

    def conflicting(self, lane: Lane) -> list[Lane]:
        raise NotImplementedError


def validate(m: Map, tol: float = 1e-3) -> list[str]:
    """Reports: lane links naming a missing lane, lane-link asymmetry
    between sections, road links whose partner does not link back (outside
    junctions), a positional gap between reference lines at a road link,
    and junction connections naming missing roads or lanes."""
    raise NotImplementedError

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
