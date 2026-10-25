# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Read an ``.xodr`` file into the model.

The reshaping happens here: lane widths become absolute boundary offsets,
road marks move from the lane to the boundary they paint, and signals with
their validity become regulations.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Union

from . import _xml
from ._xml import ET, Element, child, children, get_bool, get_float, get_int
from .geom import (Arc, Line, ParamPoly3, PiecewisePoly3, PlanGeom, Poly3,
                   RefLine, Spiral)
from .model import (AccessRule, Boundary, Connection, Frame, Header,
                    Junction, Lane, LaneLink, LaneSection, Map, MarkSegment,
                    Regulation, RegulationKind, Road, RoadLink, RoadType,
                    Signal, SignalReference, SpeedLimit, boundary_id)

SPEED_UNITS = {'m/s', 'mph', 'km/h'}
TOP_LEVEL = {'header', 'road', 'junction'}
ROAD_CHILDREN = {'link', 'type', 'planView', 'elevationProfile',
                 'lateralProfile', 'lanes', 'signals'}
LANE_CHILDREN = {'link', 'width', 'border', 'roadMark', 'speed', 'access'}
SIGNAL_ATTRS = {'id', 's', 't', 'zOffset', 'hOffset', 'type', 'subtype',
                'country', 'dynamic', 'orientation'}


class ParseError(ValueError):
    pass


def load(path: Union[str, 'os.PathLike[str]']) -> Map:
    tree = ET.parse(os.fspath(path))
    return from_element(tree.getroot())


def loads(text: Union[str, bytes]) -> Map:
    if isinstance(text, str):
        text = text.encode('utf-8')
    return from_element(ET.fromstring(text))


def from_element(root: Element) -> Map:
    if root.tag != 'OpenDRIVE':
        raise ParseError(f'root element is <{root.tag}>, not <OpenDRIVE>')
    m = Map()
    skipped: list[str] = []
    for elem in children(root):
        if elem.tag == 'header':
            m.header, m.frame = _header(elem)
        elif elem.tag == 'road':
            road = _road(elem)
            m.roads[road.id] = road
        elif elem.tag == 'junction':
            junction = _junction(elem)
            m.junctions[junction.id] = junction
        else:
            skipped.append(elem.tag)
    if skipped:
        m.attrs['_skipped'] = ','.join(skipped)
    m.reindex()
    for road in m.roads.values():
        for sig in road.signals:
            m.signals[sig.id] = sig
    _regulations(m)
    return m


def _poly_list(elems: Iterable[Element], s_name: str) -> PiecewisePoly3:
    return PiecewisePoly3([(float(e.get(s_name)), get_float(e, 'a', 0.0),
                            get_float(e, 'b', 0.0), get_float(e, 'c', 0.0),
                            get_float(e, 'd', 0.0)) for e in elems])


def _header(elem: Element) -> tuple[Header, Frame]:
    known = {'revMajor', 'revMinor', 'name', 'version', 'date', 'north',
             'south', 'east', 'west', 'vendor'}
    h = Header(rev_major=get_int(elem, 'revMajor', 1),
               rev_minor=get_int(elem, 'revMinor', 8),
               name=elem.get('name', ''), version=elem.get('version', ''),
               date=elem.get('date', ''),
               north=get_float(elem, 'north', 0.0),
               south=get_float(elem, 'south', 0.0),
               east=get_float(elem, 'east', 0.0),
               west=get_float(elem, 'west', 0.0),
               vendor=elem.get('vendor', ''),
               attrs=_xml.extra_attrs(elem, known))
    frame = Frame()
    geo = child(elem, 'geoReference')
    if geo is not None and geo.text is not None:
        frame.geo_reference = geo.text.strip()
    off = child(elem, 'offset')
    if off is not None:
        frame.offset = (get_float(off, 'x', 0.0), get_float(off, 'y', 0.0),
                        get_float(off, 'z', 0.0), get_float(off, 'hdg', 0.0))
    _xml.stash_unknown(elem, {'geoReference', 'offset'}, h.attrs)
    return h, frame


def _road_link(elem: Element) -> RoadLink:
    return RoadLink(element_type=elem.get('elementType'),
                    element_id=elem.get('elementId'),
                    contact_point=elem.get('contactPoint'),
                    element_s=get_float(elem, 'elementS'),
                    element_dir=elem.get('elementDir'))


def _speed(elem: Optional[Element]) -> Optional[SpeedLimit]:
    if elem is None:
        return None
    return SpeedLimit(get_float(elem, 'sOffset', 0.0),
                      get_float(elem, 'max', 0.0), elem.get('unit'))


def _plan_geom(elem: Element) -> PlanGeom:
    base = (float(elem.get('s')), float(elem.get('length')),
            float(elem.get('x')), float(elem.get('y')),
            float(elem.get('hdg')))
    shape = next(children(elem), None)
    if shape is None or shape.tag == 'line':
        return Line(*base)
    if shape.tag == 'arc':
        return Arc(*base, curvature=float(shape.get('curvature')))
    if shape.tag == 'spiral':
        return Spiral(*base, curv_start=float(shape.get('curvStart')),
                      curv_end=float(shape.get('curvEnd')))
    if shape.tag == 'poly3':
        return Poly3(*base, a=get_float(shape, 'a', 0.0),
                     b=get_float(shape, 'b', 0.0),
                     c=get_float(shape, 'c', 0.0),
                     d=get_float(shape, 'd', 0.0))
    if shape.tag == 'paramPoly3':
        vals = {k: get_float(shape, k, 0.0)
                for k in ('aU', 'bU', 'cU', 'dU', 'aV', 'bV', 'cV', 'dV')}
        return ParamPoly3(*base, a_u=vals['aU'], b_u=vals['bU'],
                          c_u=vals['cU'], d_u=vals['dU'], a_v=vals['aV'],
                          b_v=vals['bV'], c_v=vals['cV'], d_v=vals['dV'],
                          p_range=shape.get('pRange', 'arcLength'))
    raise ParseError(f'unknown plan geometry <{shape.tag}>')


def _road(elem: Element) -> Road:
    junction = elem.get('junction', '-1')
    road = Road(id=elem.get('id'), name=elem.get('name', ''),
                length=float(elem.get('length')),
                junction=None if junction == '-1' else junction,
                rule=elem.get('rule', 'RHT'),
                attrs=_xml.extra_attrs(
                    elem, {'id', 'name', 'length', 'junction', 'rule'}))

    link = child(elem, 'link')
    if link is not None:
        pred, succ = child(link, 'predecessor'), child(link, 'successor')
        road.predecessor = _road_link(pred) if pred is not None else None
        road.successor = _road_link(succ) if succ is not None else None

    for t in children(elem):
        if t.tag == 'type':
            road.road_type.append(RoadType(
                float(t.get('s')), t.get('type'), t.get('country'),
                _speed(child(t, 'speed'))))

    plan = child(elem, 'planView')
    geoms = [_plan_geom(g) for g in children(plan)] if plan is not None \
        else []
    elev = child(elem, 'elevationProfile')
    elevation = _poly_list(
        (e for e in children(elev) if e.tag == 'elevation'), 's') \
        if elev is not None else PiecewisePoly3()
    lat = child(elem, 'lateralProfile')
    superelevation = PiecewisePoly3()
    if lat is not None:
        superelevation = _poly_list(
            (e for e in children(lat) if e.tag == 'superelevation'), 's')
        _xml.stash_unknown(lat, {'superelevation'}, road.attrs,
                           key='_unknown_lateralProfile')
    road.refline = RefLine(geoms, elevation, superelevation)

    lanes = child(elem, 'lanes')
    if lanes is not None:
        _lanes(lanes, road)

    signals = child(elem, 'signals')
    if signals is not None:
        for s in children(signals):
            if s.tag == 'signal':
                road.signals.append(_signal(s, road))
            elif s.tag == 'signalReference':
                road.signal_refs.append(_signal_ref(s, road))

    _xml.stash_unknown(elem, ROAD_CHILDREN, road.attrs)
    return road


def _lanes(elem: Element, road: Road) -> None:
    lane_offset = _poly_list(
        (e for e in children(elem) if e.tag == 'laneOffset'), 's')
    secs = [e for e in children(elem) if e.tag == 'laneSection']
    starts = [float(e.get('s')) for e in secs]
    for i, (sec_elem, s0) in enumerate(zip(secs, starts)):
        s1 = starts[i + 1] if i + 1 < len(starts) else road.length
        sec = LaneSection(s0, road, single_side=sec_elem.get('singleSide'))
        road.sections.append(sec)
        sec.center = Boundary(boundary_id(road.id, i, 0), sec,
                              lane_offset.restrict(s0, s1))
        center = child(sec_elem, 'center')
        if center is not None:
            for ln in children(center):
                if ln.tag == 'lane':
                    sec.center.marks = _marks(ln)
                    sec.center.attrs.update(_xml.extra_attrs(
                        ln, {'id', 'type', 'level'}))
                    _xml.stash_unknown(ln, {'roadMark'}, sec.center.attrs)
        for side_tag, sign, target in (('left', 1, sec.left),
                                       ('right', -1, sec.right)):
            side = child(sec_elem, side_tag)
            if side is None:
                continue
            lane_elems = sorted(
                (ln for ln in children(side) if ln.tag == 'lane'),
                key=lambda ln: abs(int(ln.get('id'))))
            inner = sec.center
            for ln in lane_elems:
                lane = _lane(ln, sec, inner, sign)
                target.append(lane)
                inner = lane.outer
        _xml.stash_unknown(sec_elem, {'left', 'center', 'right'}, sec.attrs)


def _lane(elem: Element, sec: LaneSection, inner: Boundary,
          sign: int) -> Lane:
    odr_id = int(elem.get('id'))
    if (odr_id > 0) != (sign > 0):
        raise ParseError(f'lane {odr_id} is on the wrong side of road '
                         f'{sec.road.id}')
    s0 = sec.s0
    borders = [e for e in children(elem) if e.tag == 'border']
    if borders:
        t = _poly_list(borders, 'sOffset').shifted(s0)
    else:
        widths = [e for e in children(elem) if e.tag == 'width']
        w = _poly_list(widths, 'sOffset').shifted(s0) if widths \
            else PiecewisePoly3.constant(0.0, s0)
        t = inner.t + w if sign > 0 else inner.t - w
    outer = Boundary(boundary_id(sec.road.id, sec.index, odr_id), sec, t,
                     marks=_marks(elem))

    lane = Lane(sec, odr_id, inner, outer, type=elem.get('type', 'driving'),
                level=get_bool(elem, 'level'),
                direction=elem.get('direction', 'standard'),
                attrs=_xml.extra_attrs(
                    elem, {'id', 'type', 'level', 'direction'}))
    if borders:
        lane.attrs['_geometry'] = 'border'

    link = child(elem, 'link')
    if link is not None:
        lane.link = LaneLink(
            [int(e.get('id')) for e in children(link)
             if e.tag == 'predecessor'],
            [int(e.get('id')) for e in children(link)
             if e.tag == 'successor'])
    for e in children(elem):
        if e.tag == 'speed':
            lane.speed.append(_speed(e))
        elif e.tag == 'access':
            restr = [r.get('type') for r in children(e)
                     if r.tag == 'restriction']
            if e.get('restriction') is not None:
                restr.insert(0, e.get('restriction'))
            lane.access.append(AccessRule(get_float(e, 'sOffset', 0.0),
                                          e.get('rule', 'allow'), restr))
    _xml.stash_unknown(elem, LANE_CHILDREN, lane.attrs)
    return lane


def _marks(elem: Element) -> list[MarkSegment]:
    known = {'sOffset', 'type', 'weight', 'color', 'material', 'width',
             'laneChange', 'height'}
    out = []
    for e in children(elem):
        if e.tag != 'roadMark':
            continue
        mark = MarkSegment(get_float(e, 'sOffset', 0.0),
                           type=e.get('type', 'none'),
                           weight=e.get('weight'), color=e.get('color'),
                           material=e.get('material'),
                           width=get_float(e, 'width'),
                           lane_change=e.get('laneChange'),
                           height=get_float(e, 'height'),
                           attrs=_xml.extra_attrs(e, known))
        _xml.stash_unknown(e, set(), mark.attrs)
        out.append(mark)
    out.sort(key=lambda mk: mk.s_offset)
    return out


def _validity(elem: Element) -> list[tuple[int, int]]:
    return [(int(v.get('fromLane')), int(v.get('toLane')))
            for v in children(elem) if v.tag == 'validity']


def _signal(elem: Element, road: Road) -> Signal:
    sig = Signal(id=elem.get('id'), road=road, s=float(elem.get('s')),
                 t=float(elem.get('t')),
                 z_offset=get_float(elem, 'zOffset', 0.0),
                 h_offset=get_float(elem, 'hOffset', 0.0),
                 type=elem.get('type', '-1'),
                 subtype=elem.get('subtype', '-1'),
                 country=elem.get('country'),
                 dynamic=get_bool(elem, 'dynamic'),
                 orientation=elem.get('orientation', '+'),
                 validity=_validity(elem),
                 attrs=_xml.extra_attrs(elem, SIGNAL_ATTRS))
    _xml.stash_unknown(elem, {'validity'}, sig.attrs)
    return sig


def _signal_ref(elem: Element, road: Road) -> SignalReference:
    ref = SignalReference(signal_id=elem.get('id'), road=road,
                          s=float(elem.get('s')), t=float(elem.get('t')),
                          orientation=elem.get('orientation', '+'),
                          validity=_validity(elem),
                          attrs=_xml.extra_attrs(
                              elem, {'id', 's', 't', 'orientation'}))
    _xml.stash_unknown(elem, {'validity'}, ref.attrs)
    return ref


def _junction(elem: Element) -> Junction:
    j = Junction(id=elem.get('id'), name=elem.get('name', ''),
                 type=elem.get('type'),
                 attrs=_xml.extra_attrs(elem, {'id', 'name', 'type'}))
    known = {'id', 'incomingRoad', 'connectingRoad', 'contactPoint', 'type',
             'linkedRoad'}
    for c in children(elem):
        if c.tag != 'connection':
            continue
        conn = Connection(id=c.get('id'), incoming_road=c.get('incomingRoad'),
                          connecting_road=c.get('connectingRoad'),
                          contact_point=c.get('contactPoint'),
                          lane_links=[(int(ll.get('from')), int(ll.get('to')))
                                      for ll in children(c)
                                      if ll.tag == 'laneLink'],
                          type=c.get('type'), linked_road=c.get('linkedRoad'),
                          attrs=_xml.extra_attrs(c, known))
        _xml.stash_unknown(c, {'laneLink'}, conn.attrs)
        j.connections.append(conn)
    _xml.stash_unknown(elem, {'connection'}, j.attrs)
    return j


def regulation_kind(sig: Signal) -> RegulationKind:
    if sig.dynamic:
        return 'traffic_light'
    if sig.attrs.get('unit') in SPEED_UNITS or sig.type == '274':
        return 'speed_limit'
    return 'traffic_sign'


def valid_lanes(road: Road, s: float,
                validity: list[tuple[int, int]]) -> list[Lane]:
    sec = road.section_at(s)
    lanes = list(sec.lanes())
    if not validity:
        return lanes
    out = []
    for lo, hi in validity:
        lo, hi = min(lo, hi), max(lo, hi)
        out += [ln for ln in lanes if lo <= ln.odr_id <= hi
                and ln not in out]
    return out


def _regulations(m: Map) -> None:
    for sig in m.signals.values():
        reg = Regulation(id=f'sig:{sig.id}', kind=regulation_kind(sig),
                         anchor=sig)
        reg.roles['refers'] = valid_lanes(sig.road, sig.s, sig.validity)
        m.regulations[reg.id] = reg
    for road in m.roads.values():
        for ref in road.signal_refs:
            reg = m.regulations.get(f'sig:{ref.signal_id}')
            if reg is None:
                continue
            for ln in valid_lanes(road, ref.s, ref.validity):
                if ln not in reg.roles['refers']:
                    reg.roles['refers'].append(ln)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
