# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Derived lane connectivity, the Lanelet2 routing graph.

The source of truth is what the file says: ``Road.predecessor`` and
``Road.successor``, ``Lane.link`` and ``Junction.connections``. This module
builds a cached index of directed ``Link`` records from them. A link joins
one end of a source lane to one end of a destination lane in the s sense;
``successors`` and ``predecessors`` then read it in travel direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import End, Junction, Lane, LaneSection, Map, Road


@dataclass(frozen=True)
class Link:
    """``src`` at ``src_end`` touches ``dst`` at ``dst_end``. ``contact`` is
    the end of ``dst`` that is touched, as OpenDRIVE's contactPoint."""

    src: Lane
    dst: Lane
    via_junction: Optional[Junction]
    src_end: End
    dst_end: End

    @property
    def contact(self) -> End:
        return self.dst_end


def _other(end: End) -> End:
    return 'end' if end == 'start' else 'start'


def _ids(lane: Lane, end: End) -> list[int]:
    return lane.link.successors if end == 'end' \
        else lane.link.predecessors


class LinkIndex:

    def __init__(self, m: Map) -> None:
        self.map = m
        self.links: list[Link] = []
        self._by_lane: dict[tuple[int, End], list[Link]] = {}
        self._by_boundary: dict[int, list[Lane]] = {}
        self._seen: dict[tuple[int, End, int, End], int] = {}
        self._build()

    def _add(self, src: Lane, dst: Lane, via: Optional[Junction],
             src_end: End, dst_end: End) -> None:
        key = (id(src), src_end, id(dst), dst_end)
        rkey = (id(dst), dst_end, id(src), src_end)
        if key in self._seen or rkey in self._seen:
            i = self._seen.get(key, self._seen.get(rkey))
            old = self.links[i]
            if old.via_junction is None and via is not None:
                self.links[i] = Link(old.src, old.dst, via, old.src_end,
                                     old.dst_end)
            return
        link = Link(src, dst, via, src_end, dst_end)
        self._seen[key] = len(self.links)
        self.links.append(link)
        self._by_lane.setdefault((id(src), src_end), []).append(link)
        self._by_lane.setdefault((id(dst), dst_end), []).append(link)

    def _build(self) -> None:
        m = self.map
        for lane in m.lanes.values():
            for bnd in (lane.inner, lane.outer):
                self._by_boundary.setdefault(id(bnd), []).append(lane)
        for road in m.roads.values():
            self._within_road(road)
            if road.successor is not None:
                self._road_end(road, 'end')
            if road.predecessor is not None:
                self._road_end(road, 'start')

    def _within_road(self, road: Road) -> None:
        secs = road.sections
        for a, b in zip(secs, secs[1:]):
            for lane in a.lanes():
                for sid in lane.link.successors:
                    dst = _find(b, sid)
                    if dst is not None:
                        self._add(lane, dst, None, 'end', 'start')
            for lane in b.lanes():
                for pid in lane.link.predecessors:
                    src = _find(a, pid)
                    if src is not None:
                        self._add(src, lane, None, 'end', 'start')

    def _road_end(self, road: Road, end: End) -> None:
        link = road.successor if end == 'end' else road.predecessor
        sec = road.end_section(end)
        if link.element_type == 'road':
            other = self.map.roads.get(link.element_id)
            if other is None or not other.sections or \
                    link.contact_point is None:
                return
            osec = other.end_section(link.contact_point)
            for lane in sec.lanes():
                for oid in _ids(lane, end):
                    dst = _find(osec, oid)
                    if dst is not None:
                        self._add(lane, dst, None, end, link.contact_point)
            return
        junction = self.map.junctions.get(link.element_id)
        if junction is None:
            return
        for conn in junction.connections:
            if conn.incoming_road != road.id:
                continue
            target = self.map.roads.get(conn.target_road)
            if target is None or not target.sections:
                continue
            contact = conn.contact_point or 'start'
            back = target.predecessor if contact == 'start' \
                else target.successor
            # The connecting road's own link says which end of the incoming
            # road it touches; without it assume this end.
            if back is not None and back.element_type == 'road' and \
                    back.element_id == road.id and \
                    back.contact_point not in (None, end):
                continue
            tsec = target.end_section(contact)
            for src_id, dst_id in conn.lane_links:
                src, dst = _find(sec, src_id), _find(tsec, dst_id)
                if src is not None and dst is not None:
                    self._add(src, dst, junction, end, contact)

    def at_end(self, lane: Lane, end: End) -> list[Link]:
        return list(self._by_lane.get((id(lane), end), []))

    def _across(self, lane: Lane, end: End) -> list[Lane]:
        out = []
        for link in self.at_end(lane, end):
            other = link.dst if link.src is lane else link.src
            if other not in out:
                out.append(other)
        return out

    def successors(self, lane: Lane) -> list[Lane]:
        """Lanes entered when leaving ``lane`` in its travel direction."""
        return self._across(lane, 'end' if lane.travels_along_s
                            else 'start')

    def predecessors(self, lane: Lane) -> list[Lane]:
        return self._across(lane, 'start' if lane.travels_along_s
                            else 'end')

    def neighbors(self, lane: Lane) -> tuple[Optional[Lane], Optional[Lane]]:
        """``(left, right)`` in travel direction, through the shared
        boundary objects."""
        return (self._sharing(lane, lane.left_bound),
                self._sharing(lane, lane.right_bound))

    def _sharing(self, lane: Lane, bnd) -> Optional[Lane]:
        for other in self._by_boundary.get(id(bnd), []):
            if other is not lane:
                return other
        return None

    def conflicting(self, lane: Lane) -> list[Lane]:
        """Lanes whose path crosses this one inside a junction.

        Not in this milestone. It needs either a geometric sweep of the
        connecting roads or the junction ``<priority>`` records; neither is
        modeled yet.
        """
        raise NotImplementedError('conflict detection is a later milestone')


def _find(sec: LaneSection, odr_id: int) -> Optional[Lane]:
    for lane in sec.lanes():
        if lane.odr_id == odr_id:
            return lane
    return None


def validate(m: Map, tol: float = 1e-3) -> list[str]:
    """Consistency of the links the file declares. Returns messages; an
    empty list means no problem was found."""
    issues: list[str] = []
    for road in m.roads.values():
        _validate_sections(road, issues)
        for end in ('start', 'end'):
            _validate_road_link(m, road, end, tol, issues)
    for junction in m.junctions.values():
        _validate_junction(m, junction, issues)
    return issues


def _validate_sections(road: Road, issues: list[str]) -> None:
    secs = road.sections
    for a, b in zip(secs, secs[1:]):
        for lane in a.lanes():
            for sid in lane.link.successors:
                dst = _find(b, sid)
                if dst is None:
                    issues.append(f'{lane!r}: successor {sid} missing in '
                                  f'section {b.index}')
                elif lane.odr_id not in dst.link.predecessors:
                    issues.append(f'{dst!r} does not name {lane!r} as a '
                                  f'predecessor')
        for lane in b.lanes():
            for pid in lane.link.predecessors:
                src = _find(a, pid)
                if src is None:
                    issues.append(f'{lane!r}: predecessor {pid} missing in '
                                  f'section {a.index}')
                elif lane.odr_id not in src.link.successors:
                    issues.append(f'{src!r} does not name {lane!r} as a '
                                  f'successor')


def _validate_road_link(m: Map, road: Road, end: End, tol: float,
                        issues: list[str]) -> None:
    link = road.successor if end == 'end' else road.predecessor
    if link is None:
        return
    if link.element_type == 'junction':
        if link.element_id not in m.junctions:
            issues.append(f'{road!r} {end}: junction {link.element_id} '
                          f'missing')
        return
    other = m.roads.get(link.element_id)
    if other is None:
        issues.append(f'{road!r} {end}: road {link.element_id} missing')
        return
    if link.contact_point is None:
        issues.append(f'{road!r} {end}: road link without contactPoint')
        return
    back = other.predecessor if link.contact_point == 'start' \
        else other.successor
    if back is None or back.element_type != 'road' or \
            back.element_id != road.id or back.contact_point != end:
        # A connecting road links to a road that only knows the junction;
        # that is the normal asymmetry inside a junction.
        if road.junction is None:
            issues.append(f'{road!r} {end}: {other!r} does not link back')
    s_here = road.length if end == 'end' else 0.0
    s_there = other.length if link.contact_point == 'end' else 0.0
    if road.refline.plan and other.refline.plan:
        x0, y0, *_ = road.refline.pose(s_here)
        x1, y1, *_ = other.refline.pose(s_there)
        if abs(x0 - x1) > tol or abs(y0 - y1) > tol:
            issues.append(f'{road!r} {end} is {abs(x0 - x1):.3g}, '
                          f'{abs(y0 - y1):.3g} away from {other!r} '
                          f'{link.contact_point}')
    sec = road.end_section(end)
    osec = other.end_section(link.contact_point)
    for lane in sec.lanes():
        for oid in _ids(lane, end):
            if _find(osec, oid) is None:
                issues.append(f'{lane!r}: lane {oid} missing in {other!r}')


def _validate_junction(m: Map, junction: Junction,
                       issues: list[str]) -> None:
    for conn in junction.connections:
        incoming = m.roads.get(conn.incoming_road)
        target = m.roads.get(conn.target_road)
        if incoming is None:
            issues.append(f'{junction!r} connection {conn.id}: incoming '
                          f'road {conn.incoming_road} missing')
        if target is None:
            issues.append(f'{junction!r} connection {conn.id}: road '
                          f'{conn.target_road} missing')
        if incoming is None or target is None:
            continue
        if conn.connecting_road is not None and \
                target.junction != junction.id:
            issues.append(f'{target!r} does not belong to {junction!r}')
        tsec = target.end_section(conn.contact_point or 'start')
        for src_id, dst_id in conn.lane_links:
            if all(_find(sec, src_id) is None
                   for sec in (incoming.sections[0], incoming.sections[-1])):
                issues.append(f'{junction!r} connection {conn.id}: lane '
                              f'{src_id} missing in {incoming!r}')
            if _find(tsec, dst_id) is None:
                issues.append(f'{junction!r} connection {conn.id}: lane '
                              f'{dst_id} missing in {target!r}')

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
