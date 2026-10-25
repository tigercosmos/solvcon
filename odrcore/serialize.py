# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Write the model back to ``.xodr``.

Boundaries are turned back into lane ``<width>`` records by default
(``outer.t - inner.t``), or into ``<border>`` records on request. Road
marks return from the boundary to the lane that owns it.
"""

from __future__ import annotations

import os
from typing import Literal, Union

from . import _xml
from ._xml import ET, Element, set_attrs
from .geom import (Arc, Line, ParamPoly3, PiecewisePoly3, PlanGeom, Poly3,
                   Spiral)
from .model import (Boundary, Junction, Lane, LaneSection, Map, MarkSegment,
                    Road, RoadLink, Signal, SignalReference, SpeedLimit)

LaneGeometry = Literal['width', 'border']


def dump(m: Map, path: Union[str, 'os.PathLike[str]'],
         lane_geometry: LaneGeometry = 'width') -> None:
    root = to_element(m, lane_geometry)
    tree = ET.ElementTree(root)
    if _xml.HAVE_LXML:
        tree.write(os.fspath(path), encoding='UTF-8', xml_declaration=True,
                   pretty_print=True)
    else:  # pragma: no cover - depends on the environment
        ET.indent(tree)
        tree.write(os.fspath(path), encoding='UTF-8', xml_declaration=True)


def dumps(m: Map, lane_geometry: LaneGeometry = 'width') -> str:
    return _xml.to_string(to_element(m, lane_geometry))


def to_element(m: Map, lane_geometry: LaneGeometry = 'width') -> Element:
    root = ET.Element('OpenDRIVE')
    _header(root, m)
    for road in m.roads.values():
        _road(root, road, lane_geometry)
    for junction in m.junctions.values():
        _junction(root, junction)
    return root


def _sub(parent: Element, tag: str, **attrs) -> Element:
    elem = ET.SubElement(parent, tag)
    set_attrs(elem, attrs)
    return elem


def _header(root: Element, m: Map) -> None:
    h = m.header
    elem = _sub(root, 'header', revMajor=h.rev_major, revMinor=h.rev_minor,
                name=h.name, version=h.version, date=h.date, north=h.north,
                south=h.south, east=h.east, west=h.west, vendor=h.vendor)
    set_attrs(elem, {k: v for k, v in h.attrs.items()
                     if not k.startswith('_') and k != 'userData'})
    if m.frame.geo_reference is not None:
        geo = ET.SubElement(elem, 'geoReference')
        if _xml.HAVE_LXML:
            geo.text = ET.CDATA(m.frame.geo_reference)
        else:  # pragma: no cover - depends on the environment
            geo.text = m.frame.geo_reference
    if m.frame.offset is not None:
        x, y, z, hdg = m.frame.offset
        _sub(elem, 'offset', x=x, y=y, z=z, hdg=hdg)
    _xml.restore_unknown(elem, h.attrs)


def _plain_attrs(attrs: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in attrs.items()
            if not k.startswith('_') and k != 'userData'}


def _road_link(parent: Element, tag: str, link: RoadLink) -> None:
    _sub(parent, tag, elementType=link.element_type,
         elementId=link.element_id, contactPoint=link.contact_point,
         elementS=link.element_s, elementDir=link.element_dir)


def _speed(parent: Element, sp: SpeedLimit, with_offset: bool) -> None:
    attrs = {'sOffset': sp.s_offset} if with_offset else {}
    attrs.update(max=sp.max, unit=sp.unit)
    _sub(parent, 'speed', **attrs)


def _plan_geom(parent: Element, g: PlanGeom) -> None:
    elem = _sub(parent, 'geometry', s=g.s0, x=g.x0, y=g.y0, hdg=g.hdg0,
                length=g.length)
    if isinstance(g, Arc):
        _sub(elem, 'arc', curvature=g.curvature)
    elif isinstance(g, Spiral):
        _sub(elem, 'spiral', curvStart=g.curv_start, curvEnd=g.curv_end)
    elif isinstance(g, Poly3):
        _sub(elem, 'poly3', a=g.a, b=g.b, c=g.c, d=g.d)
    elif isinstance(g, ParamPoly3):
        _sub(elem, 'paramPoly3', aU=g.a_u, bU=g.b_u, cU=g.c_u, dU=g.d_u,
             aV=g.a_v, bV=g.b_v, cV=g.c_v, dV=g.d_v, pRange=g.p_range)
    elif isinstance(g, Line):
        ET.SubElement(elem, 'line')
    else:
        raise TypeError(f'unknown plan geometry {type(g).__name__}')


def _poly_list(parent: Element, tag: str, s_name: str, poly: PiecewisePoly3,
               shift: float = 0.0) -> None:
    for s0, a, b, c, d in poly.segs:
        _sub(parent, tag, **{s_name: s0 - shift, 'a': a, 'b': b, 'c': c,
                             'd': d})


def _road(root: Element, road: Road, lane_geometry: LaneGeometry) -> None:
    elem = _sub(root, 'road', name=road.name, length=road.length, id=road.id,
                junction='-1' if road.junction is None else road.junction,
                rule=road.rule)
    set_attrs(elem, _plain_attrs(road.attrs))

    if road.predecessor is not None or road.successor is not None:
        link = ET.SubElement(elem, 'link')
        if road.predecessor is not None:
            _road_link(link, 'predecessor', road.predecessor)
        if road.successor is not None:
            _road_link(link, 'successor', road.successor)

    for rt in road.road_type:
        t = _sub(elem, 'type', s=rt.s, type=rt.type, country=rt.country)
        if rt.speed is not None:
            _speed(t, rt.speed, with_offset=False)

    plan = ET.SubElement(elem, 'planView')
    for g in road.refline.plan:
        _plan_geom(plan, g)

    if road.refline.elevation.segs:
        _poly_list(ET.SubElement(elem, 'elevationProfile'), 'elevation', 's',
                   road.refline.elevation)
    lat_unknown = road.attrs.get('_unknown_lateralProfile')
    if road.refline.superelevation.segs or lat_unknown:
        lat = ET.SubElement(elem, 'lateralProfile')
        _poly_list(lat, 'superelevation', 's', road.refline.superelevation)
        _xml.restore_unknown(lat, road.attrs, key='_unknown_lateralProfile')

    _lanes(elem, road, lane_geometry)

    if road.signals or road.signal_refs:
        signals = ET.SubElement(elem, 'signals')
        for sig in road.signals:
            _signal(signals, sig)
        for ref in road.signal_refs:
            _signal_ref(signals, ref)

    _xml.restore_unknown(elem, road.attrs)


def _lane_offset(road: Road) -> PiecewisePoly3:
    segs = []
    for sec in road.sections:
        segs += sec.center.t.restrict(sec.s0, sec.s1).segs
    poly = PiecewisePoly3(segs).simplified()
    if len(poly.segs) == 1 and all(v == 0.0 for v in poly.segs[0][1:]):
        return PiecewisePoly3()
    return poly


def _lanes(parent: Element, road: Road,
           lane_geometry: LaneGeometry) -> None:
    elem = ET.SubElement(parent, 'lanes')
    _poly_list(elem, 'laneOffset', 's', _lane_offset(road))
    for sec in road.sections:
        sec_elem = _sub(elem, 'laneSection', s=sec.s0,
                        singleSide=sec.single_side)
        if sec.left:
            left = ET.SubElement(sec_elem, 'left')
            for lane in sorted(sec.left, key=lambda ln: -ln.odr_id):
                _lane(left, lane, lane_geometry)
        center = ET.SubElement(sec_elem, 'center')
        c = _sub(center, 'lane', id=0, type='none', level=False)
        set_attrs(c, _plain_attrs(sec.center.attrs))
        _marks(c, sec.center)
        _xml.restore_unknown(c, sec.center.attrs)
        if sec.right:
            right = ET.SubElement(sec_elem, 'right')
            for lane in sorted(sec.right, key=lambda ln: -ln.odr_id):
                _lane(right, lane, lane_geometry)
        _xml.restore_unknown(sec_elem, sec.attrs)


def _lane(parent: Element, lane: Lane, lane_geometry: LaneGeometry) -> None:
    elem = _sub(parent, 'lane', id=lane.odr_id, type=lane.type,
                level=lane.level)
    if lane.direction != 'standard':
        elem.set('direction', lane.direction)
    set_attrs(elem, _plain_attrs(lane.attrs))

    if lane.link.predecessors or lane.link.successors:
        link = ET.SubElement(elem, 'link')
        for pid in lane.link.predecessors:
            _sub(link, 'predecessor', id=pid)
        for sid in lane.link.successors:
            _sub(link, 'successor', id=sid)

    sec: LaneSection = lane.section
    if lane_geometry == 'border':
        poly = lane.outer.t.restrict(sec.s0, sec.s1).simplified()
        _poly_list(elem, 'border', 'sOffset', poly, shift=sec.s0)
    else:
        sign = 1.0 if lane.odr_id > 0 else -1.0
        width = (lane.outer.t - lane.inner.t) if sign > 0 \
            else (lane.inner.t - lane.outer.t)
        poly = width.restrict(sec.s0, sec.s1).simplified()
        _poly_list(elem, 'width', 'sOffset', poly, shift=sec.s0)

    _marks(elem, lane.outer)
    for sp in lane.speed:
        _speed(elem, sp, with_offset=True)
    for ac in lane.access:
        a = _sub(elem, 'access', sOffset=ac.s_offset, rule=ac.rule)
        for r in ac.restrictions:
            _sub(a, 'restriction', type=r)
    _xml.restore_unknown(elem, lane.attrs)


def _marks(parent: Element, bnd: Boundary) -> None:
    for mk in bnd.marks:
        _mark(parent, mk)


def _mark(parent: Element, mk: MarkSegment) -> None:
    elem = _sub(parent, 'roadMark', sOffset=mk.s_offset, type=mk.type,
                weight=mk.weight, color=mk.color, material=mk.material,
                width=mk.width, laneChange=mk.lane_change, height=mk.height)
    set_attrs(elem, _plain_attrs(mk.attrs))
    _xml.restore_unknown(elem, mk.attrs)


def _validity(parent: Element, validity: list[tuple[int, int]]) -> None:
    for lo, hi in validity:
        _sub(parent, 'validity', fromLane=lo, toLane=hi)


def _signal(parent: Element, sig: Signal) -> None:
    elem = _sub(parent, 'signal', s=sig.s, t=sig.t, id=sig.id,
                dynamic='yes' if sig.dynamic else 'no',
                orientation=sig.orientation, zOffset=sig.z_offset,
                country=sig.country, type=sig.type, subtype=sig.subtype,
                hOffset=sig.h_offset)
    set_attrs(elem, _plain_attrs(sig.attrs))
    _validity(elem, sig.validity)
    _xml.restore_unknown(elem, sig.attrs)


def _signal_ref(parent: Element, ref: SignalReference) -> None:
    elem = _sub(parent, 'signalReference', s=ref.s, t=ref.t,
                id=ref.signal_id, orientation=ref.orientation)
    set_attrs(elem, _plain_attrs(ref.attrs))
    _validity(elem, ref.validity)
    _xml.restore_unknown(elem, ref.attrs)


def _junction(root: Element, j: Junction) -> None:
    elem = _sub(root, 'junction', id=j.id, name=j.name, type=j.type)
    set_attrs(elem, _plain_attrs(j.attrs))
    for conn in j.connections:
        c = _sub(elem, 'connection', id=conn.id,
                 incomingRoad=conn.incoming_road,
                 connectingRoad=conn.connecting_road,
                 linkedRoad=conn.linked_road,
                 contactPoint=conn.contact_point, type=conn.type)
        set_attrs(c, _plain_attrs(conn.attrs))
        for src, dst in conn.lane_links:
            _sub(c, 'laneLink', **{'from': src, 'to': dst})
        _xml.restore_unknown(c, conn.attrs)
    _xml.restore_unknown(elem, j.attrs)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
