# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Lanelet2 view over the odrcore map.

The view is generated from a ``Map`` and is disposable: nothing here writes
back into the model. It is the only place in odrcore that samples geometry;
the model keeps every curve analytic.

One lanelet is made per lane per lane section. Each section ``Boundary``
becomes one ``LineString`` oriented along s, and a lanelet that travels
against s holds it through an inverted ``OrientedLine``, so the two lanes
on either side of a boundary share the same linestring object. Points at a
section seam are shared between the linestrings that meet there.

Identifiers are 63-bit hashes of a key built from model ids, so the same
map always yields the same view ids, independent of iteration order.
"""

from __future__ import annotations

import bisect
import collections
import enum
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from .model import (Boundary, Lane, LaneId, LaneSection, Map, MarkSegment,
                    Road, Signal)

Vec3 = tuple[float, float, float]
Attrs = dict[str, str]

PARTICIPANTS = frozenset({'passengerCar', 'bus', 'truck', 'delivery',
                          'emergency', 'taxi', 'motorcycle', 'bicycle',
                          'pedestrian'})
"""Road users, named after OpenDRIVE ``e_accessRestrictionType``."""

MOTOR = PARTICIPANTS - {'bicycle', 'pedestrian'}

_ROAD_LANE = MOTOR | {'bicycle'}
LANE_TYPE_ACCESS: dict[str, frozenset[str]] = {
    'driving': _ROAD_LANE, 'entry': _ROAD_LANE, 'exit': _ROAD_LANE,
    'onRamp': _ROAD_LANE, 'offRamp': _ROAD_LANE,
    'connectingRamp': _ROAD_LANE, 'slipLane': _ROAD_LANE,
    'bidirectional': _ROAD_LANE, 'HOV': MOTOR,
    'bus': frozenset({'bus', 'emergency'}),
    'taxi': frozenset({'taxi', 'emergency'}),
    'biking': frozenset({'bicycle'}),
    'sidewalk': frozenset({'pedestrian'}),
    'walking': frozenset({'pedestrian'}),
}
"""Who may use a lane of each type before ``<access>`` rules apply. Types
not listed (shoulder, border, median, parking, ...) admit nobody."""

_RESTRICTION_ALIASES = {'trucks': 'truck'}

_LANE_SUBTYPE = {'driving': 'road', 'entry': 'road', 'exit': 'road',
                 'onRamp': 'road', 'offRamp': 'road',
                 'connectingRamp': 'road', 'slipLane': 'road',
                 'bidirectional': 'road', 'HOV': 'road', 'bus': 'bus_lane',
                 'biking': 'bicycle_lane', 'sidewalk': 'walkway',
                 'walking': 'walkway'}

_MARK_SUBTYPE = {'solid': 'solid', 'broken': 'dashed',
                 'solid solid': 'solid_solid', 'broken broken': 'dashed',
                 'solid broken': 'solid_dashed',
                 'broken solid': 'dashed_solid'}
_MARK_FLIPPED = {'solid_dashed': 'dashed_solid',
                 'dashed_solid': 'solid_dashed'}


@dataclass(frozen=True)
class Point:
    id: int
    position: Vec3
    attrs: Attrs = field(default_factory=dict)


@dataclass(frozen=True)
class LineString:
    """A sampled ``Boundary``, oriented along increasing s. ``stations``
    holds the road s of each point."""

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
    roles hold lanelet or linestring ids."""

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

    def identifier(self, *key) -> int:
        encoded = json.dumps(key, separators=(',', ':'), ensure_ascii=True)
        digest = hashlib.blake2b(encoded.encode(), digest_size=8).digest()
        identity = int.from_bytes(digest, 'big') & ((1 << 63) - 1) or 1
        known = self.provenance.get(identity)
        if known is not None and known != key:
            raise ValueError(f'derived id collision between {known!r} and '
                             f'{key!r}')
        self.provenance[identity] = key
        return identity

    def lanelet(self, lane_id: LaneId) -> Lanelet:
        return self.lanelets[self.by_lane[lane_id]]

    def _targets(self, identity: int,
                 kinds: Iterable[RelationKind]) -> list[int]:
        kinds = set(kinds)
        return sorted(e.target for e in self.routing
                      if e.source == identity and e.kind in kinds)

    def successors(self, identity: int) -> list[int]:
        return self._targets(identity, [RelationKind.SUCCESSOR])

    def predecessors(self, identity: int) -> list[int]:
        return sorted(e.source for e in self.routing
                      if e.target == identity
                      and e.kind == RelationKind.SUCCESSOR)

    def lane_changes(self, identity: int) -> list[int]:
        return self._targets(identity, [RelationKind.LEFT,
                                        RelationKind.RIGHT])

    def route(self, source: int, target: int, participant: str,
              lane_changes: bool = True) -> tuple[int, ...]:
        """Fewest-lanelet path usable by ``participant``, or ``()``."""
        if participant not in self.lanelets[source].participants or \
                participant not in self.lanelets[target].participants:
            return ()
        out: dict[int, set[int]] = collections.defaultdict(set)
        for edge in self.routing:
            if not lane_changes and edge.kind != RelationKind.SUCCESSOR:
                continue
            if participant in edge.participants:
                out[edge.source].add(edge.target)
        previous: dict[int, Optional[int]] = {source: None}
        queue = collections.deque([source])
        while queue:
            current = queue.popleft()
            if current == target:
                path = []
                node: Optional[int] = current
                while node is not None:
                    path.append(node)
                    node = previous[node]
                return tuple(reversed(path))
            for nxt in sorted(out[current]):
                if nxt not in previous:
                    previous[nxt] = current
                    queue.append(nxt)
        return ()


def participants(lane: Lane) -> frozenset[str]:
    """Road users allowed on the whole lane.

    ``<access>`` records sharing an ``sOffset`` form one rule set: ``allow``
    records replace the lane-type default with what they list, ``deny``
    records remove from it. A participant must be allowed by every rule set
    in the section, and by the lane-type default before the first one.
    """
    default = LANE_TYPE_ACCESS.get(lane.type, frozenset())
    groups = collections.defaultdict(list)
    for rule in lane.access:
        groups[rule.s_offset].append(rule)
    if not groups:
        return default
    allowed = set(default if min(groups) > 0.0 else PARTICIPANTS)
    for s_offset in sorted(groups):
        listed_allow, deny, has_allow = set(), set(), False
        for rule in groups[s_offset]:
            names = {_RESTRICTION_ALIASES.get(r, r)
                     for r in rule.restrictions}
            if rule.rule == 'allow':
                has_allow = True
                listed_allow |= names
            else:
                deny |= names
        here = (listed_allow & PARTICIPANTS) if has_allow else set(default)
        allowed &= here - deny
    return frozenset(allowed)


def crossable(marks: list[MarkSegment], increasing: bool) -> bool:
    """Whether every mark of a boundary lets a vehicle cross it towards
    the lane with the higher OpenDRIVE id (``increasing``) or the lower.
    A missing ``laneChange`` is ``both``, the spec default."""
    ok = {'both', 'increase'} if increasing else {'both', 'decrease'}
    return all((mk.lane_change or 'both') in ok for mk in marks)


def line_attrs(marks: list[MarkSegment], left_side: bool) -> Attrs:
    """Lanelet2 line tags from the first mark of a boundary.

    OpenDRIVE lists the parts of a double line from the inner to the outer
    lane, and from left to right on the center lane. A linestring runs
    along s, so Lanelet2 reads its first part on the +t side. For a
    boundary on the left of the reference line the inner part is on the
    -t side, so the double-line subtype is mirrored there.
    """
    if not marks:
        return {'type': 'virtual'}
    mk = marks[0]
    attrs: Attrs = {'odr:mark_type': mk.type}
    if mk.type in _MARK_SUBTYPE:
        attrs['type'] = 'line_thick' if mk.weight == 'bold' \
            else 'line_thin'
        sub = _MARK_SUBTYPE[mk.type]
        attrs['subtype'] = _MARK_FLIPPED.get(sub, sub) if left_side else sub
    elif mk.type == 'curb':
        attrs.update(type='curbstone', subtype='high')
    elif mk.type == 'edge':
        attrs['type'] = 'road_border'
    else:
        attrs['type'] = 'virtual'
    if mk.color is not None:
        attrs['color'] = mk.color
    if mk.lane_change is not None:
        attrs['odr:lane_change'] = mk.lane_change
    if len(marks) > 1:
        attrs['odr:mark_segments'] = str(len(marks))
    return attrs


def _location(road: Road, s: float) -> Optional[str]:
    kind = None
    for rt in road.road_type:
        if rt.s <= s:
            kind = rt.type
    if kind is None:
        return None
    urban = kind.startswith('town') or kind in ('lowSpeed', 'pedestrian',
                                                'bicycle')
    return 'urban' if urban else 'nonurban'


class Generator:

    def __init__(self, m: Map, spacing: float = 2.0,
                 seam_tol: float = 1e-6) -> None:
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError('sampling spacing must be positive and finite')
        self.map = m
        self.spacing = spacing
        self.seam_tol = seam_tol
        self.view = LaneletView()
        self._ends: dict[tuple[str, str], Point] = {}

    def generate(self) -> LaneletView:
        for road in self.map.roads.values():
            self._seams(road)
        for road in self.map.roads.values():
            for sec in road.sections:
                self._section(sec)
        self._routing()
        self._regulations()
        return self.view

    def _point(self, key: tuple, position: Vec3,
               attrs: Optional[Attrs] = None) -> Point:
        identity = self.view.identifier('point', *key)
        pt = self.view.points.get(identity)
        if pt is None:
            pt = Point(identity, position, attrs or {})
            self.view.points[identity] = pt
        return pt

    def _seams(self, road: Road) -> None:
        """Share one point among the boundary ends that meet at a section
        seam. Ends closer than ``seam_tol`` in t are one cluster; the point
        sits at the lowest t of its cluster."""
        for k in range(1, len(road.sections)):
            prev, cur = road.sections[k - 1], road.sections[k]
            s = cur.s0
            ends = [(bnd.t(s), 0, bnd.id, 'end') for bnd in prev.boundaries()]
            ends += [(bnd.t(s), 1, bnd.id, 'start')
                     for bnd in cur.boundaries()]
            ends.sort()
            clusters: list[list[tuple]] = []
            for end in ends:
                if clusters and end[0] - clusters[-1][-1][0] <= self.seam_tol:
                    clusters[-1].append(end)
                else:
                    clusters.append([end])
            for i, cluster in enumerate(clusters):
                pt = self._point(('seam', road.id, k, i),
                                 road.refline.point(s, cluster[0][0]))
                for _, _, bid, which in cluster:
                    self._ends[(bid, which)] = pt

    def stations(self, sec: LaneSection, bnd: Boundary) -> list[float]:
        """Sample stations of a boundary over its section: a uniform grid
        no coarser than ``spacing`` plus every s where the boundary offset,
        the plan view, the elevation, the superelevation or the road mark
        changes its formula."""
        s0, s1 = sec.s0, sec.s1
        rl = sec.road.refline
        cuts = list(bnd.t.breakpoints())
        cuts += [g.s0 for g in rl.plan]
        cuts += rl.elevation.breakpoints() + rl.superelevation.breakpoints()
        cuts += [s0 + mk.s_offset for mk in bnd.marks]
        exact = sorted({s0, s1} | {s for s in cuts if s0 < s < s1})
        n = max(1, math.ceil((s1 - s0) / self.spacing))
        eps = 1e-9 * max(1.0, abs(s1))
        out = list(exact)
        for i in range(1, n):
            s = s0 + (s1 - s0) * i / n
            j = bisect.bisect_left(exact, s)
            near = exact[max(j - 1, 0):j + 1]
            if all(abs(s - e) > eps for e in near):
                out.append(s)
        out.sort()
        kept = [out[0]]
        for s in out[1:]:
            if s - kept[-1] > eps:
                kept.append(s)
        kept[-1] = s1
        return kept

    def _line(self, sec: LaneSection, bnd: Boundary,
              left_side: bool) -> LineString:
        rl = sec.road.refline
        stations = self.stations(sec, bnd)
        pts = []
        for i, s in enumerate(stations):
            which = 'start' if i == 0 else \
                'end' if i == len(stations) - 1 else None
            pt = self._ends.get((bnd.id, which)) if which else None
            if pt is None:
                pt = self._point(('boundary', bnd.id, s),
                                 rl.point(s, bnd.t(s)))
            pts.append(pt)
        identity = self.view.identifier('linestring', bnd.id)
        line = LineString(identity, bnd.id, tuple(pts), tuple(stations),
                          line_attrs(bnd.marks, left_side))
        self.view.lines[identity] = line
        self.view.by_boundary[bnd.id] = identity
        return line

    def _section(self, sec: LaneSection) -> None:
        lines = {sec.center.id: self._line(sec, sec.center, False)}
        for lane in sec.lanes():
            lines[lane.outer.id] = self._line(sec, lane.outer,
                                              lane.odr_id > 0)
        for lane in sec.lanes():
            self._lanelet(lane, lines)

    def _lanelet(self, lane: Lane, lines: dict[str, LineString]) -> None:
        inverted = not lane.travels_along_s
        road, sec = lane.road, lane.section
        attrs: Attrs = {
            'type': 'lanelet',
            'subtype': _LANE_SUBTYPE.get(lane.type, lane.type),
            'one_way': 'no' if lane.direction == 'both' else 'yes',
            'odr:road': road.id, 'odr:section': str(sec.index),
            'odr:lane': str(lane.odr_id), 'odr:type': lane.type,
        }
        if attrs['subtype'] == 'road' and any(
                rt.type == 'motorway' and rt.s <= sec.s0
                for rt in road.road_type):
            attrs['subtype'] = 'highway'
        location = _location(road, sec.s0)
        if location is not None:
            attrs['location'] = location
        if lane.speed:
            sp = lane.speed[0]
            attrs['speed_limit'] = f'{sp.max:g} {sp.unit or "m/s"}'
        identity = self.view.identifier('lanelet', *lane.id)
        self.view.lanelets[identity] = Lanelet(
            identity, lane.id,
            OrientedLine(lines[lane.left_bound.id], inverted),
            OrientedLine(lines[lane.right_bound.id], inverted),
            participants(lane), attrs)
        self.view.by_lane[lane.id] = identity

    def _connect(self, source: int, target: int,
                 kind: RelationKind) -> None:
        allowed = (self.view.lanelets[source].participants
                   & self.view.lanelets[target].participants)
        self.view.routing.add(RoutingEdge(source, target, kind, allowed))

    def _routing(self) -> None:
        m, by_lane = self.map, self.view.by_lane
        for lane in m.lanes.values():
            src = by_lane[lane.id]
            for nxt in m.successors(lane):
                self._connect(src, by_lane[nxt.id], RelationKind.SUCCESSOR)
            left, right = m.neighbors(lane)
            for other, kind, bnd in ((left, RelationKind.LEFT,
                                      lane.left_bound),
                                     (right, RelationKind.RIGHT,
                                      lane.right_bound)):
                if other is None:
                    continue
                dst = by_lane[other.id]
                self.view.adjacent.add((src, dst))
                if other.travels_along_s != lane.travels_along_s:
                    continue
                if crossable(bnd.marks, other.odr_id > lane.odr_id):
                    self._connect(src, dst, kind)

    def _signal(self, sig: Signal) -> int:
        x, y, z = sig.road.refline.point(sig.s, sig.t)
        country = (sig.country or '').lower()
        attrs = {'type': 'traffic_light' if sig.dynamic else 'traffic_sign',
                 'subtype': f'{country}{sig.type}',
                 'odr:signal': sig.id, 'odr:subtype': sig.subtype}
        return self._point(('signal', sig.id), (x, y, z + sig.z_offset),
                           attrs).id

    def _regulations(self) -> None:
        view = self.view
        attached: dict[int, list[int]] = collections.defaultdict(list)
        for reg in self.map.regulations.values():
            identity = view.identifier('regulation', reg.id)
            refers: list[int] = []
            if reg.anchor is not None:
                refers.append(self._signal(reg.anchor))
            roles: dict[str, tuple[int, ...]] = {}
            for name, members in reg.roles.items():
                ids: list[int] = []
                for p in members:
                    if isinstance(p, Lane):
                        target = view.by_lane[p.id]
                        if name == 'refers':
                            attached[target].append(identity)
                        else:
                            ids.append(target)
                    elif isinstance(p, Boundary):
                        ids.append(view.by_boundary[p.id])
                    elif isinstance(p, Signal):
                        sid = self._signal(p)
                        (refers if name == 'refers' else ids).append(sid)
                if name != 'refers' and ids:
                    roles[name] = tuple(ids)
            if refers:
                roles['refers'] = tuple(dict.fromkeys(refers))
            attrs: Attrs = {'type': 'regulatory_element',
                            'subtype': reg.kind}
            attrs.update({k: v for k, v in reg.attrs.items()
                          if not k.startswith('_')})
            if reg.kind == 'speed_limit' and reg.anchor is not None:
                value = reg.anchor.attrs.get('value')
                if value is not None:
                    unit = reg.anchor.attrs.get('unit', '')
                    attrs['sign_type'] = f'{value} {unit}'.strip()
            view.regulations[identity] = RegulatoryElement(
                identity, reg.id, reg.kind, roles, attrs)
        for target, regs in attached.items():
            ll = view.lanelets[target]
            view.lanelets[target] = replace(
                ll, regulations=ll.regulations + tuple(dict.fromkeys(regs)))


def generate(m: Map, spacing: float = 2.0,
             seam_tol: float = 1e-6) -> LaneletView:
    """Build the Lanelet2 view of ``m``. Sample spacing is in meters."""
    return Generator(m, spacing, seam_tol).generate()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
