# odrcore-simple

The data-structure design of `odrcore`, stripped of implementation. Every
dataclass and type alias from `odrcore` is kept with its fields; the
parser, serializer, comparison, numerical geometry, link-index build and
view sampling are reduced to signatures and docstrings.

## Layers

| Module | Content |
| --- | --- |
| `geom.py` | `PiecewisePoly3`, `PlanGeom` family, `RefLine`. Analytic shapes. |
| `model.py` | Entities and the `Map` container. The core of the design. |
| `topology.py` | Derived `Link` and the `LinkIndex` interface. |
| `view.py` | Disposable Lanelet2 view entities and `LaneletView`. |
| `osm.py` | Lanelet2 OSM writer interface. |

Both formats are supported for querying and output. `Map.lanelet2()`
returns the cached Lanelet2 form, `Map.route()` routes between model
lanes, and `osm.dump()` writes a Lanelet2 OSM file next to the `.xodr`
writer. Edits happen on the OpenDRIVE form only, and there is no Lanelet2
reader: sampled linestrings cannot become the analytic model without
fitting, which would break the lossless round trip.

Dropped entirely: `parse.py`, `serialize.py`, `compare.py`, `_xml.py`,
tests and fixtures.

## Rules the model follows

1. Storage is OpenDRIVE: road s, signed lane ids, lateral offsets `t` from
   the reference line. Lanelet2 left/right bounds are properties computed
   from lane side, lane `direction` and road `rule` (RHT/LHT). Inverted
   lanelets are never stored.
2. The model is analytic. Only `view.py` samples.
3. Lossless round trip for the supported subset. Unknown children live in
   `attrs["_unknown"]`; `<userData>` in `attrs["userData"]`.
4. The model reshapes the file (widths become absolute offsets, each
   `<signal>` becomes a `Regulation`); it is not a mirror of the XML.

## Mapping

| OpenDRIVE 1.8 | Model | Lanelet2 |
| --- | --- | --- |
| `<header>` | `Header`, `Frame` | map metadata, projector origin |
| `<road>` | `Road` | none (grouping) |
| `<road><link>` | `RoadLink` | routing edge source |
| `<planView><geometry>` | `Line`, `Arc`, `Spiral`, `Poly3`, `ParamPoly3` | linestring geometry |
| `<elevationProfile>`, `<lateralProfile>` | `RefLine.elevation`, `.superelevation` | point z |
| `<lanes><laneOffset>` | `LaneSection.center.t` | center linestring |
| `<laneSection>` | `LaneSection` | none (grouping) |
| `<lane>` (id != 0) | `Lane` | Lanelet |
| `<lane><width>` / `<border>` | `Boundary.t` of the outer boundary | LineString |
| `<lane><roadMark>` | `MarkSegment` on the painted boundary | LineString `type`, `subtype` |
| `<lane><link>` | `LaneLink` | routing edge source |
| `<lane><speed>`, `<access>` | `SpeedLimit`, `AccessRule` | `speed_limit`, participants |
| `<junction><connection>` | `Junction`, `Connection` | routing edges inside a junction |
| `<signal>`, `<signalReference>` | `Signal`, `SignalReference` | sign or light point |
| `<signal>` plus `<validity>` | `Regulation` | RegulatoryElement |

Ids: `Lane.id` is `(road_id, section_index, odr_lane_id)`. A boundary is
`"road:section:lane"` named after the lane whose outer border it is;
`"road:section:0"` is the section center line.

## Identity and sharing

- `Boundary` and `Lane` use `eq=False`: adjacency is object identity. Two
  lanes are neighbors because they hold the same `Boundary` object.
- `Regulation.roles` hold model objects, not ids, so renumbering cannot
  break them.
- `Map.roads` and `Map.junctions` own; `Map.lanes`, `.boundaries`,
  `.signals` are rebuilt by `reindex()`. `Map.links` and
  `Map.lanelet2()` are derived and cached until `reindex()`; neither is
  stored in the `.xodr`.
- View ids are 63-bit hashes of model keys, stable across load/dump/load.

## Open decisions (carried over)

- Road marks stay piecewise on a boundary; the view takes line tags from
  the first mark and does not split lanelets at mark changes.
- A missing `laneChange` permits crossing both ways, even on a solid line.
- A `both` lane runs with s for bound resolution and is routed along s only.
- Section seams share points; road-to-road joins do not.
- Stop lines, `right_of_way` and `yield` roles are representable but never
  filled by import.
- `LinkIndex.conflicting` is a stub.
- The OSM writer ignores the header `<offset>` and treats the model plane
  as a local tangent plane, which matches a transverse Mercator
  geoReference near the origin only.
