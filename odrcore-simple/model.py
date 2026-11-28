# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The in-memory map model: a reshaped subset of OpenDRIVE 1.8.

Storage follows OpenDRIVE: the s axis of the reference line, signed lane
ids, lateral offsets measured from the reference line. The Lanelet2 notions
(left and right bound in travel direction, adjacency through a shared
linestring) are derived from that storage and never written back into it.

Entities that are shared by identity (``Boundary``, ``Lane``) disable
dataclass equality so that two lanes sharing a ``Boundary`` object are
adjacent by construction. Structural comparison is a separate concern.

Ownership and indexing::

    Map
     |-- header: Header
     |-- frame: Frame
     |-- roads: {id: Road}                      (owns)
     |    |-- refline: RefLine
     |    |-- sections: [LaneSection]
     |    |    |-- center: Boundary              (lane 0, laneOffset)
     |    |    |-- left:  [Lane]  odr_id > 0
     |    |    |-- right: [Lane]  odr_id < 0
     |    |         |-- inner, outer: Boundary   (outer owned by this lane,
     |    |                                       inner shared with the
     |    |                                       lane one step toward 0)
     |    |-- predecessor, successor: RoadLink
     |    |-- signals: [Signal], signal_refs: [SignalReference]
     |-- junctions: {id: Junction}              (owns)
     |    |-- connections: [Connection]
     |-- lanes: {LaneId: Lane}                  (index into roads)
     |-- boundaries: {str: Boundary}            (index into roads)
     |-- signals: {str: Signal}                 (index into roads)
     |-- regulations: {str: Regulation}         (owns; roles hold objects)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

from .geom import PiecewisePoly3, RefLine

if TYPE_CHECKING:
    from .topology import LinkIndex
    from .view import LaneletView

LaneId = tuple[str, int, int]
"""``(road_id, section_index, odr_lane_id)``."""

Direction = Literal['standard', 'reversed', 'both']
TrafficHand = Literal['RHT', 'LHT']
Side = Literal['left', 'right']
End = Literal['start', 'end']
Attrs = dict[str, str]
"""Verbatim leftovers: ``userData``, unknown children, skipped tags."""


@dataclass
class MarkSegment:
    """One ``<roadMark>``; ``s_offset`` is relative to the section start."""

    s_offset: float
    type: str = 'none'
    weight: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    width: Optional[float] = None
    lane_change: Optional[str] = None
    height: Optional[float] = None
    attrs: Attrs = field(default_factory=dict)


@dataclass(eq=False)
class Boundary:
    """An analytic lane border, the Lanelet2 LineString.

    ``t`` is the absolute lateral offset from the reference line as a
    function of road s (breakpoints are road s, not section offsets).
    ``<width>`` is turned into absolute ``t`` on load and recovered as
    ``outer.t - inner.t`` on dump. The same object is referenced by the two
    lanes it separates. Id is ``"road:section:lane"`` of the lane whose
    outer border it is; lane 0 names the section center line.
    """

    id: str
    section: LaneSection
    t: PiecewisePoly3 = field(default_factory=PiecewisePoly3)
    marks: list[MarkSegment] = field(default_factory=list)
    attrs: Attrs = field(default_factory=dict)


@dataclass
class SpeedLimit:
    s_offset: float
    max: float
    unit: Optional[str] = None


@dataclass
class AccessRule:
    """``<access>``: the 1.8 form carries ``<restriction type>`` children,
    the 1.7 form an attribute; both land in ``restrictions``."""

    s_offset: float
    rule: str
    restrictions: list[str] = field(default_factory=list)


@dataclass
class LaneLink:
    """``<lane><link>``: odr ids of the lanes this one continues from and
    into, in the s direction, in the neighboring section or road."""

    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)


@dataclass(eq=False)
class Lane:
    """A lane of one section, the Lanelet2 Lanelet.

    Stored in OpenDRIVE terms (``inner``/``outer``, signed ``odr_id``).
    ``left_bound``/``right_bound`` are derived from the side, the lane
    ``direction`` and the road ``rule``; an inverted lanelet is never
    stored.
    """

    section: LaneSection
    odr_id: int
    inner: Boundary
    outer: Boundary
    type: str = 'driving'
    level: bool = False
    direction: Direction = 'standard'
    speed: list[SpeedLimit] = field(default_factory=list)
    access: list[AccessRule] = field(default_factory=list)
    link: LaneLink = field(default_factory=LaneLink)
    attrs: Attrs = field(default_factory=dict)

    @property
    def id(self) -> LaneId:
        return (self.section.road.id, self.section.index, self.odr_id)

    @property
    def road(self) -> Road:
        return self.section.road

    @property
    def side(self) -> Side:
        return 'left' if self.odr_id > 0 else 'right'

    @property
    def travels_along_s(self) -> bool:
        """Right lanes under RHT and left lanes under LHT run with s;
        ``reversed`` flips that; ``both`` is taken to run with s so that
        its bounds are well defined."""
        if self.direction == 'both':
            return True
        rht = self.road.rule == 'RHT'
        along = (self.side == 'right') == rht
        return along != (self.direction == 'reversed')

    @property
    def _higher_t(self) -> Boundary:
        return self.outer if self.side == 'left' else self.inner

    @property
    def _lower_t(self) -> Boundary:
        return self.inner if self.side == 'left' else self.outer

    @property
    def left_bound(self) -> Boundary:
        return self._higher_t if self.travels_along_s else self._lower_t

    @property
    def right_bound(self) -> Boundary:
        return self._lower_t if self.travels_along_s else self._higher_t


@dataclass(eq=False)
class LaneSection:
    """``<laneSection>``. ``s1`` is the next section's ``s0`` or the road
    length. ``index`` is the position in ``road.sections``."""

    s0: float
    road: Road
    center: Boundary = None  # type: ignore[assignment]
    left: list[Lane] = field(default_factory=list)
    right: list[Lane] = field(default_factory=list)
    single_side: Optional[str] = None
    attrs: Attrs = field(default_factory=dict)

    @property
    def index(self) -> int:
        return self.road.sections.index(self)

    @property
    def s1(self) -> float:
        secs = self.road.sections
        i = self.index
        return secs[i + 1].s0 if i + 1 < len(secs) else self.road.length

    def lanes(self) -> Iterator[Lane]:
        yield from self.left
        yield from self.right

    def boundaries(self) -> Iterator[Boundary]:
        yield self.center
        for lane in self.lanes():
            yield lane.outer


@dataclass
class RoadLink:
    """``<road><link><predecessor|successor>``."""

    element_type: Literal['road', 'junction']
    element_id: str
    contact_point: Optional[End] = None
    element_s: Optional[float] = None
    element_dir: Optional[str] = None


@dataclass
class RoadType:
    s: float
    type: str
    country: Optional[str] = None
    speed: Optional[SpeedLimit] = None


@dataclass(eq=False)
class Road:
    id: str
    name: str = ''
    length: float = 0.0
    refline: RefLine = field(default_factory=RefLine)
    sections: list[LaneSection] = field(default_factory=list)
    junction: Optional[str] = None
    predecessor: Optional[RoadLink] = None
    successor: Optional[RoadLink] = None
    road_type: list[RoadType] = field(default_factory=list)
    rule: TrafficHand = 'RHT'
    signals: list[Signal] = field(default_factory=list)
    signal_refs: list[SignalReference] = field(default_factory=list)
    attrs: Attrs = field(default_factory=dict)


@dataclass
class Connection:
    """``<junction><connection>`` with its ``<laneLink>`` pairs
    ``(from, to)``. ``connecting_road`` or, for a direct junction,
    ``linked_road`` is set; ``target_road`` hides the difference."""

    id: str
    incoming_road: str
    connecting_road: Optional[str] = None
    contact_point: Optional[End] = None
    lane_links: list[tuple[int, int]] = field(default_factory=list)
    type: Optional[str] = None
    linked_road: Optional[str] = None
    attrs: Attrs = field(default_factory=dict)

    @property
    def target_road(self) -> str:
        return self.connecting_road or self.linked_road  # type: ignore


@dataclass(eq=False)
class Junction:
    id: str
    name: str = ''
    type: Optional[str] = None
    connections: list[Connection] = field(default_factory=list)
    attrs: Attrs = field(default_factory=dict)


Validity = tuple[int, int]
"""``(from_lane, to_lane)`` of ``<validity>``."""


@dataclass(eq=False)
class Signal:
    """``<signal>`` placed on a road at ``(s, t)``. ``h_offset`` is the
    spec name of the heading offset. Other spec attributes stay in
    ``attrs``."""

    id: str
    road: Road
    s: float
    t: float
    z_offset: float = 0.0
    h_offset: float = 0.0
    type: str = '-1'
    subtype: str = '-1'
    country: Optional[str] = None
    dynamic: bool = False
    orientation: str = '+'
    validity: list[Validity] = field(default_factory=list)
    attrs: Attrs = field(default_factory=dict)


@dataclass(eq=False)
class SignalReference:
    """``<signalReference>``: the same signal seen from another road."""

    signal_id: str
    road: Road
    s: float
    t: float
    orientation: str = '+'
    validity: list[Validity] = field(default_factory=list)
    attrs: Attrs = field(default_factory=dict)


RegulationKind = Literal['traffic_light', 'traffic_sign', 'speed_limit',
                         'stop_line', 'right_of_way', 'all_way_stop']
ROLE_NAMES = ('refers', 'ref_line', 'yield', 'right_of_way', 'cancels',
              'cancel_line')
Participant = Union[Lane, Boundary, Signal]


@dataclass(eq=False)
class Regulation:
    """The Lanelet2 RegulatoryElement. Roles hold model objects, never ids,
    so a regulation stays valid under renumbering.

    One is made per ``<signal>`` (id ``sig:<signal id>``): ``anchor`` is
    the signal, ``refers`` the lanes its ``<validity>`` names in the
    section at the signal's s. A ``<signalReference>`` adds lanes of its
    road to the same regulation. ``kind`` is chosen from the signal type,
    unit and ``dynamic``.
    """

    id: str
    kind: RegulationKind
    roles: dict[str, list[Participant]] = field(default_factory=dict)
    anchor: Optional[Signal] = None
    attrs: Attrs = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ROLE_NAMES:
            self.roles.setdefault(name, [])


@dataclass
class Frame:
    """``<header><geoReference>`` and ``<offset>``: the projector origin."""

    geo_reference: Optional[str] = None
    offset: Optional[tuple[float, float, float, float]] = None


@dataclass
class Header:
    rev_major: int = 1
    rev_minor: int = 8
    name: str = ''
    version: str = ''
    date: str = ''
    north: float = 0.0
    south: float = 0.0
    east: float = 0.0
    west: float = 0.0
    vendor: str = ''
    attrs: Attrs = field(default_factory=dict)


@dataclass(eq=False)
class Map:
    """The Lanelet2 LaneletMap: dict layers keyed by id. ``roads`` and
    ``junctions`` own the structure; ``lanes``, ``boundaries``, ``signals``
    and ``regulations`` index into it. ``reindex()`` rebuilds the index
    layers from the roads and drops the derived caches.

    Edits happen on this OpenDRIVE form only. The link index and the
    sampled Lanelet2 form are derived on demand and cached until
    ``reindex()``, which must follow an edit."""

    header: Header = field(default_factory=Header)
    frame: Frame = field(default_factory=Frame)
    roads: dict[str, Road] = field(default_factory=dict)
    junctions: dict[str, Junction] = field(default_factory=dict)
    lanes: dict[LaneId, Lane] = field(default_factory=dict)
    boundaries: dict[str, Boundary] = field(default_factory=dict)
    signals: dict[str, Signal] = field(default_factory=dict)
    regulations: dict[str, Regulation] = field(default_factory=dict)
    attrs: Attrs = field(default_factory=dict)
    _links: Optional['LinkIndex'] = field(default=None, repr=False)
    _lanelet2: dict[tuple[float, float], 'LaneletView'] = field(
        default_factory=dict, repr=False)

    def reindex(self) -> None:
        self.lanes.clear()
        self.boundaries.clear()
        for road in self.roads.values():
            for sec in road.sections:
                for bnd in sec.boundaries():
                    self.boundaries[bnd.id] = bnd
                for lane in sec.lanes():
                    self.lanes[lane.id] = lane
        self._links = None
        self._lanelet2.clear()

    @property
    def links(self) -> 'LinkIndex':
        """Derived and cached; never stored or written to the file."""
        if self._links is None:
            from .topology import LinkIndex
            self._links = LinkIndex(self)
        return self._links

    def lanelet2(self, spacing: float = 2.0,
                 seam_tol: float = 1e-6) -> 'LaneletView':
        """The Lanelet2 form sampled every ``spacing`` meters at most,
        cached per sampling setting."""
        raise NotImplementedError

    def route(self, source: Lane, target: Lane, participant: str,
              lane_changes: bool = True) -> list[Lane]:
        """Fewest-lane path usable by ``participant``, or ``[]``."""
        raise NotImplementedError


def boundary_id(road_id: str, section_index: int, odr_lane_id: int) -> str:
    """Id of the outer boundary of a lane (``0`` names the center)."""
    return f'{road_id}:{section_index}:{odr_lane_id}'

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
