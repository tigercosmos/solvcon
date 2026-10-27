# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Lanelet2 view over the map: disposable, generated, never written back.

This is the only layer that samples geometry. One lanelet is made per lane
per lane section. Each section ``Boundary`` becomes one ``LineString``
oriented along s; a lanelet that travels against s holds it through an
inverted ``OrientedLine``, so the two lanes on either side of a boundary
share one linestring object. Points at a section seam are shared.

Identifiers are 63-bit hashes of a key built from model ids, for example
``("lanelet", road, section, lane)`` or ``("linestring", boundary id)``, so
the same map always yields the same view ids.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .model import LaneId, Map

Vec3 = tuple[float, float, float]
Attrs = dict[str, str]

PARTICIPANTS = frozenset({'passengerCar', 'bus', 'truck', 'delivery',
                          'emergency', 'taxi', 'motorcycle', 'bicycle',
                          'pedestrian'})
"""Road users, named after OpenDRIVE ``e_accessRestrictionType``. Lane type
gives a default set; ``<access>`` rules narrow it."""


@dataclass(frozen=True)
class Point:
    id: int
    position: Vec3
    attrs: Attrs = field(default_factory=dict)


@dataclass(frozen=True)
class LineString:
    """A sampled ``Boundary``, oriented along increasing s. ``stations``
    holds the road s of each point. ``source`` is the boundary id."""

    id: int
    source: str
    points: tuple[Point, ...]
    stations: tuple[float, ...]
    attrs: Attrs = field(default_factory=dict)


@dataclass(frozen=True)
class OrientedLine:
    line: LineString
    inverted: bool = False

    @property
    def id(self) -> int:
        return self.line.id

    @property
    def points(self) -> tuple[Point, ...]:
        return self.line.points[::-1] if self.inverted else self.line.points


@dataclass(frozen=True)
class Lanelet:
    """Bounds are ``Lane.left_bound``/``right_bound`` in travel direction.
    A ``both`` lane yields one lanelet tagged ``one_way=no``."""

    id: int
    source: LaneId
    left: OrientedLine
    right: OrientedLine
    participants: frozenset[str]
    attrs: Attrs = field(default_factory=dict)
    regulations: tuple[int, ...] = ()


@dataclass(frozen=True)
class RegulatoryElement:
    """Lanelet2 roles: ``refers`` holds the sign or light point, the other
    roles hold lanelet or linestring ids. The model's ``refers`` lanes
    receive the regulation in ``Lanelet.regulations`` instead."""

    id: int
    source: str
    subtype: str
    roles: dict[str, tuple[int, ...]]
    attrs: Attrs = field(default_factory=dict)


class RelationKind(enum.StrEnum):
    SUCCESSOR = 'successor'
    LEFT = 'left'
    RIGHT = 'right'


@dataclass(frozen=True)
class RoutingEdge:
    """Successor edges come from ``LinkIndex``; left/right edges join
    same-direction neighbors whose shared boundary marks permit crossing.
    ``participants`` are those allowed on both lanelets."""

    source: int
    target: int
    kind: RelationKind
    participants: frozenset[str]


@dataclass
class LaneletView:
    points: dict[int, Point] = field(default_factory=dict)
    lines: dict[int, LineString] = field(default_factory=dict)
    lanelets: dict[int, Lanelet] = field(default_factory=dict)
    regulations: dict[int, RegulatoryElement] = field(default_factory=dict)
    routing: set[RoutingEdge] = field(default_factory=set)
    adjacent: set[tuple[int, int]] = field(default_factory=set)
    """Ordered pairs of lanelets that share a bound, passable or not."""
    by_lane: dict[LaneId, int] = field(default_factory=dict)
    by_boundary: dict[str, int] = field(default_factory=dict)
    provenance: dict[int, tuple] = field(default_factory=dict)
    """View id to the model key it was hashed from."""


def generate(m: Map, spacing: float = 2.0,
             seam_tol: float = 1e-6) -> LaneletView:
    """Sample every boundary on a grid no coarser than ``spacing`` plus
    every s where a formula changes; merge seam points within
    ``seam_tol``."""
    raise NotImplementedError

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
