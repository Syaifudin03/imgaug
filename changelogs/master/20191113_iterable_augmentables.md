# Simplified Access to Coordinats and Items in Augmentables

* [rarely breaking] Changed the normalization routines for coordinate-based
  augmentables (e.g. keypoints, bounding boxes) to only support lists where
  previously all iterables were supported as list-likes. This does not affect
  being able to provide xy-coordinates as tuples (or xyxy in case of bounding
  boxes). This change was necessary to allow the corresponding augmentables
  to be iterable. Otherwise they would be interpreted as list-likes, causing
  confusion during the normalization.
  This change might affect some use cases where generaters were used
* Added ability to iterate over coordinate-based `*OnImage` instances
  (keypoints, bounding boxes, polygons, line strings), e.g.
  `bbsoi = BoundingBoxesOnImage(bbs, shape=...); for bb in bbsoi: ...`.
  would iterate now over `bbs`.
* Added ability to iterate over coordinates of `BoundingBox` (top-left,
  bottom-right), `Polygon` and `LineString` via `for xy in obj: ...`.
* Added ability to access coordinates of `BoundingBox`, `Polygon` and
  `LineString` using indices or slices, e.g. `line_string[1:]` to get an
  array of all coordinates except the first one.
* Added property `Keypoint.xy`.
* Added property `Keypoint.xy_int`.
