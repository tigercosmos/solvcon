# odrcore design

`odrcore` holds an in-memory map model that is a normalized subset of ASAM
OpenDRIVE 1.8, limited to what Lanelet2's core needs: linestrings, lanelets,
attributes, regulatory elements and a routing graph. The `.xodr` file is the
storage format. Lanelet2 is a view over the model; nothing in the model is
stored in Lanelet2 terms.

## Layers

| Module | Content |
| --- | --- |
| `geom.py` | `PiecewisePoly3`, plan-view primitives, `RefLine`. Analytic only. |
| `model.py` | Entities and the `Map` container. |
| `parse.py` | `load(path)`, `loads(text)`: `.xodr` to `Map`, with the reshaping. |
| `serialize.py` | `dump(map, path, lane_geometry="width"|"border")`, `dumps`. |
| `topology.py` | Derived `Link` index, travel-direction accessors, `validate`. |
| `compare.py` | Canonical form and `equivalent(a, b)` for round-trip checks. |
| `_xml.py` | lxml with a stdlib ElementTree fallback, verbatim sidecars. |

## Rules the model follows

1. Storage is OpenDRIVE: the s axis of the reference line, signed lane ids,
   lateral offsets from the reference line. Lanelet2 left and right bounds
   are properties computed from the lane side, the lane `direction` and the
   road `rule` (RHT or LHT). Inverted lanelets are never stored.
2. No sampling. `PlanGeom.pose(s)`, `RefLine.point(s, t)` and every lateral
   function are evaluated analytically. `Poly3` and normalized `ParamPoly3`
   need the arc length inverted; that is done by Newton iteration on a
   Gauss-Legendre quadrature of the exact integrand, not by a polyline.
   The clothoid uses Fresnel integrals (power series below 1.6, panelled
   Gauss-Legendre above), so `scipy` is not needed.
3. Lossless round trip for the supported subset. Parse, dump and parse again
   gives a map that `compare.equivalent` reports as identical to 1e-9.
   Element order may change. Children the model has no field for are kept
   verbatim in `attrs["_unknown"]` and written back. `<userData>` is kept in
   `attrs["userData"]`.
4. The model reshapes the file; it is not a mirror of the XML.

## Mapping table

| OpenDRIVE 1.8 | Model | Lanelet2 |
| --- | --- | --- |
| `<header>` attributes | `Header` | map metadata |
| `<header><geoReference>`, `<offset>` | `Frame` | projector origin |
| `<road>` | `Road` | none (grouping) |
| `<road><link>` | `Road.predecessor`, `Road.successor` (`RoadLink`) | routing graph edge source |
| `<road><type>` with `<speed>` | `RoadType` | lanelet attribute source |
| `<planView><geometry>` | `Line`, `Arc`, `Spiral`, `Poly3`, `ParamPoly3` | linestring geometry source |
| `<elevationProfile><elevation>` | `RefLine.elevation` | point z |
| `<lateralProfile><superelevation>` | `RefLine.superelevation` | point z of offset points |
| `<lanes><laneOffset>` | `LaneSection.center.t` | center linestring |
| `<laneSection>` | `LaneSection` | none (grouping) |
| `<lane>` (id != 0) | `Lane` | Lanelet |
| `<lane><width>` | `Boundary.t` of the outer boundary (`inner.t +- width`) | LineString |
| `<lane><border>` | `Boundary.t` copied | LineString |
| `<lane><roadMark>` | `MarkSegment` on the boundary the lane paints | LineString `type`, `subtype` |
| `<lane><link>` | `LaneLink` | routing graph edge source |
| `<lane><speed>` | `SpeedLimit` | `speed_limit` regulatory element source |
| `<lane><access>` | `AccessRule` | participant attributes |
| `<lane>` `type`, `level`, `direction` | `Lane.type`, `Lane.level`, `Lane.direction` | lanelet attributes, `one_way` |
| `<junction>` | `Junction` | none (grouping) |
| `<junction><connection>` with `<laneLink>` | `Connection` | routing graph edges inside a junction |
| `<signal>` | `Signal` | traffic sign or light primitive |
| `<signalReference>` | `SignalReference` | extra `refers` lanes |
| `<signal>` plus `<validity>` | `Regulation` (`kind` from type and `dynamic`) | RegulatoryElement |
| `<userData>` anywhere | `attrs["userData"]` | attributes |
| unsupported top-level element | tag name in `Map.attrs["_skipped"]` | dropped |

Ids: `Lane.id` is `(road_id, section_index, odr_lane_id)`. A boundary is
named after the lane whose outer border it is, `"road:section:lane"`, and
`"road:section:0"` is the section's center line.

## Reshaping details

- Widths are turned into absolute offsets on load: for right lanes
  `t_k = t_{k+1} - width_k`, for left lanes `t_k = t_{k-1} + width_k`, with
  `t_0 = laneOffset` restricted to the section. All breakpoints are road s.
  `PiecewisePoly3.__add__` re-anchors each cubic by a Taylor shift, which
  is exact up to floating-point rounding, so the sum is a cubic again.
- Dumping emits `<width>` as `outer.t - inner.t` (sign by side) by default,
  or `<border>` with `lane_geometry="border"`. `simplified()` drops the
  breakpoints that the laneOffset sum introduced where the cubic continues.
  `<laneOffset>` is rebuilt from the center boundaries of all sections and
  omitted when identically zero.
- Lane 0's `<roadMark>` goes to the center boundary; lane k's `<roadMark>`
  goes to lane k's outer boundary. `MarkSegment.s_offset` stays relative to
  the section start as in the file.
- Each `<signal>` makes one `Regulation` with id `sig:<signal id>`, `anchor`
  the signal and `refers` the lanes of the section at the signal's s that
  the `<validity>` ranges name (all lanes of that section when there is no
  validity). A `<signalReference>` on another road adds that road's valid
  lanes to the same regulation.
- `direction` is the OpenDRIVE 1.8 lane attribute. The design draft used
  the same three values, so nothing was renamed.
- `travels_along_s`: right lanes under RHT and left lanes under LHT run with
  s; `reversed` flips that; `both` is treated as running with s so that
  `left_bound` and `right_bound` are well defined.

## Where the spec differs from the design draft

- The signal heading offset is `hOffset` in the spec; the field is named
  `h_offset`, not `hdg`. The road s and t of a signal are `s`, `t` as
  drafted. Other signal attributes (`name`, `value`, `unit`, `height`,
  `width`, `text`, `pitch`, `roll`, `countryRevision`) stay in `attrs`.
- `<access>` in 1.8 carries `<restriction type="..."/>` children; 1.7 wrote
  a `restriction` attribute. Both parse into `AccessRule.restrictions`; the
  1.8 form is written.
- `<lane><link>` allows several `<predecessor>` and `<successor>` entries, so
  `LaneLink` holds lists rather than a single id each.
- `<connection>` may name `linkedRoad` instead of `connectingRoad` (direct
  junction). `Connection.target_road` hides the difference.
- The signal `dynamic` attribute is `yes`/`no`, not a boolean literal. It is
  parsed leniently and written as `yes`/`no`.
- The verbatim sidecar is appended at the end of its parent element. The
  schema orders `<objects>` before `<signals>`, so a dumped road that kept
  an `<objects>` block is semantically complete but not schema-ordered.

## Topology

`Map.links` builds a `LinkIndex` lazily and caches it until `reindex()`.
Each `Link(src, dst, via_junction, src_end, dst_end)` says that one end of
`src` touches one end of `dst` in the s sense, which also covers head-to-head
joins. The sources are, in order: `Lane.link` between consecutive sections,
`Road.link` of type `road` with the lane links of the end section, and
junction connections for the road end whose `Road.link` names the junction.
The connecting road's own link decides which end of the incoming road a
connection touches.

`successors(lane)` and `predecessors(lane)` read the index in travel
direction. `neighbors(lane)` returns `(left, right)` as the other lane that
holds the same `Boundary` object as `left_bound` or `right_bound`.

`validate()` reports: lane links that name a missing lane, lane link
asymmetry between sections, road links whose partner road does not link
back (outside junctions), a positional gap between the two reference lines
at a road link, and junction connections that name missing roads or lanes.

## Out of scope this milestone

Ignored on parse and never breaking it: `lateralProfile/shape`, `surface`,
`railroad`, `controller`, `junctionGroup`, `station`, `objects`, lane
`height`, `material`, `rule`, `<include>`. Road-level ones are kept verbatim
in the sidecar; top-level ones are only recorded by tag name.

Not provided: a Lanelet2 exporter, any polyline sampling, routing
algorithms.

## Default decisions (open)

- **open** Road marks stay piecewise on the boundary; lanes are not split at
  mark changes on import. A later Lanelet2 view layer does the splitting.
- **open** Stop lines are not imported. `kind="stop_line"` and the `ref_line`
  role exist in `Regulation` only.
- **open** `right_of_way` and `yield` roles are representable but left empty
  by import because OpenDRIVE has no source for them. Junction `<priority>`
  is kept verbatim and could feed them later.
- **open** The single source of truth for connectivity is the file's links;
  the `Link` index is derived and cached, never stored or written.
- **open** Conflicting-lane detection in junctions is a stub
  (`LinkIndex.conflicting` raises `NotImplementedError`).
- **open** A `both` lane is treated as running with s for bound resolution.
- **open** A signal is a `speed_limit` when its `unit` is a speed unit or its
  type is `274`; `dynamic="yes"` makes a `traffic_light`; everything else
  is a `traffic_sign`.
- **open** For `pRange="normalized"` the parameter is found from the arc
  length by inversion rather than taken as `s / length`.

## Environment note

`lxml` is the intended XML backend. This checkout did not have it or
`scipy` installed, so `_xml.py` falls back to the standard library when
`lxml` is missing, and the Fresnel integral is implemented in `geom.py`.
With `lxml` present the `<geoReference>` text is written as CDATA.
