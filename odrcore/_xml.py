# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Thin XML helpers shared by the parser and the serializer.

lxml is the intended backend. When it is absent the standard library
ElementTree stands in, so the package still loads and round-trips; only
CDATA wrapping of ``<geoReference>`` is lost in that case.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

try:
    from lxml import etree as ET
    HAVE_LXML = True
except ImportError:  # pragma: no cover - depends on the environment
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    HAVE_LXML = False

Element = Any

UNKNOWN_KEY = '_unknown'
"""attrs key holding verbatim XML of child elements the model has no
field for, so that a dump reproduces them."""


def children(elem: Element) -> Iterator[Element]:
    """Element children only; lxml also yields comments and PIs."""
    for child in elem:
        if isinstance(child.tag, str):
            yield child


def child(elem: Element, tag: str) -> Optional[Element]:
    for c in children(elem):
        if c.tag == tag:
            return c
    return None


def _strip_whitespace(elem: Element) -> None:
    """Drop whitespace-only text so pretty printing cannot change the
    verbatim XML kept in attrs."""
    if elem.text is not None and not elem.text.strip():
        elem.text = None
    for c in children(elem):
        if c.tail is not None and not c.tail.strip():
            c.tail = None
        _strip_whitespace(c)


def to_string(elem: Element) -> str:
    _strip_whitespace(elem)
    elem.tail = None
    text = ET.tostring(elem, encoding='unicode')
    return text.strip()


def from_string(text: str) -> Element:
    return ET.fromstring(text)


def stash_unknown(elem: Element, known: set[str], attrs: dict[str, str],
                  key: str = UNKNOWN_KEY) -> None:
    """Keep the XML of children outside ``known`` under ``attrs[key]``.
    ``<userData>`` is stashed under its own key."""
    unknown, user = [], []
    for c in children(elem):
        if c.tag == 'userData':
            user.append(to_string(c))
        elif c.tag not in known:
            unknown.append(to_string(c))
    if unknown:
        attrs[key] = '\n'.join(unknown)
    if user:
        attrs['userData'] = '\n'.join(user)


def restore_unknown(elem: Element, attrs: dict[str, str],
                    key: str = UNKNOWN_KEY) -> None:
    for k in (key, 'userData'):
        text = attrs.get(k)
        if not text:
            continue
        wrapped = from_string(f'<_>{text}</_>')
        for c in children(wrapped):
            elem.append(c)


def set_attrs(elem: Element, pairs: dict[str, Any]) -> None:
    """Set attributes, skipping ``None`` and formatting numbers so that the
    shortest exact decimal is written."""
    for name, value in pairs.items():
        if value is None:
            continue
        elem.set(name, fmt(value))


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return repr(value)
    return str(value)


def get_float(elem: Element, name: str, default: Optional[float] = None
              ) -> Optional[float]:
    raw = elem.get(name)
    return default if raw is None else float(raw)


def get_int(elem: Element, name: str, default: Optional[int] = None
            ) -> Optional[int]:
    raw = elem.get(name)
    return default if raw is None else int(raw)


def get_bool(elem: Element, name: str, default: bool = False) -> bool:
    raw = elem.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('true', '1', 'yes')


def extra_attrs(elem: Element, known: set[str]) -> dict[str, str]:
    """Attributes the model has no field for, kept verbatim."""
    return {k: v for k, v in elem.attrib.items() if k not in known}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
