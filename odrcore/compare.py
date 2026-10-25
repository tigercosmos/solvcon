# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Structural comparison of two maps, for round-trip checks.

``canonical`` flattens a map into plain containers keyed by stable ids, so
that two maps built by different parses compare by content while shared
``Boundary`` and ``Lane`` objects keep identity semantics in the model.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from .geom import PiecewisePoly3, RefLine
from .model import Boundary, Lane, Map, Regulation, Signal

Canon = Any


def _ref(obj: Any) -> Any:
    if isinstance(obj, Lane):
        return ('lane',) + obj.id
    if isinstance(obj, Boundary):
        return ('boundary', obj.id)
    if isinstance(obj, Signal):
        return ('signal', obj.id)
    return obj


def canonical(obj: Any, tol: float = 1e-9) -> Canon:
    if isinstance(obj, Map):
        return {
            'header': canonical(obj.header, tol),
            'frame': canonical(obj.frame, tol),
            'roads': {k: canonical(v, tol) for k, v in obj.roads.items()},
            'junctions': {k: canonical(v, tol)
                          for k, v in obj.junctions.items()},
            'regulations': {k: canonical(v, tol)
                            for k, v in obj.regulations.items()},
            'attrs': _attrs(obj.attrs),
        }
    if isinstance(obj, PiecewisePoly3):
        return [tuple(seg) for seg in obj.simplified(tol).segs]
    if isinstance(obj, RefLine):
        return {'plan': [(type(g).__name__,) + canonical(
                    tuple(getattr(g, f.name) for f in fields(g)), tol)
                         for g in obj.plan],
                'elevation': canonical(obj.elevation, tol),
                'superelevation': canonical(obj.superelevation, tol)}
    if isinstance(obj, Regulation):
        return {'kind': obj.kind, 'anchor': _ref(obj.anchor),
                'roles': {k: sorted(_ref(p) for p in v)
                          for k, v in obj.roles.items()},
                'attrs': dict(obj.attrs)}
    if isinstance(obj, Lane):
        out = _dataclass(obj, tol, skip={'section', 'link'})
        out['inner'] = obj.inner.id
        out['outer'] = canonical(obj.outer, tol)
        out['link'] = (list(obj.link.predecessors),
                       list(obj.link.successors))
        return out
    if isinstance(obj, (Boundary, Signal)):
        return _dataclass(obj, tol, skip={'section', 'road'})
    if is_dataclass(obj) and not isinstance(obj, type):
        return _dataclass(obj, tol, skip={'road', 'section', '_links'})
    if isinstance(obj, dict):
        return {k: canonical(v, tol) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(canonical(v, tol) for v in obj)
    return obj


META_ATTRS = {'_skipped', '_geometry'}
"""attrs keys that describe how a file was read, not what it contains."""


def _attrs(attrs: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in attrs.items() if k not in META_ATTRS}


def _dataclass(obj: Any, tol: float, skip: set[str]) -> dict[str, Canon]:
    out = {}
    for f in fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        out[f.name] = _attrs(value) if f.name == 'attrs' \
            else canonical(value, tol)
    return out


def diff(a: Canon, b: Canon, tol: float = 1e-9,
         path: str = '') -> list[str]:
    """Human-readable differences between two canonical forms."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f'{path}/{k}: only in {"b" if k not in a else "a"}')
            else:
                out += diff(a[k], b[k], tol, f'{path}/{k}')
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        out = []
        for i, (u, v) in enumerate(zip(a, b)):
            out += diff(u, v, tol, f'{path}[{i}]')
        return out
    if isinstance(a, float) and isinstance(b, (int, float)):
        if abs(a - b) <= tol * max(1.0, abs(a)):
            return []
        return [f'{path}: {a!r} != {b!r}']
    if a != b:
        return [f'{path}: {a!r} != {b!r}']
    return []


def equivalent(a: Map, b: Map, tol: float = 1e-9) -> list[str]:
    return diff(canonical(a, tol), canonical(b, tol), tol)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
