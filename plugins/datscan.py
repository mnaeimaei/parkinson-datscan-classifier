"""DaT-SPECT-specific CaliVox plugin.

Registers one preprocessing step:

    roi_slice_window   whole-volume axial window localised on the frozen striatal ROI

This is the E4/E7 2.5D input rule of the original study
(research/scripts/experiments_script/two_point_five_d_input.py), expressed as a CaliVox
step so the 2.5D experiments run through the framework unchanged:

  1. map the 8 corners of the subject's ROI crop (ROI voxel -> world -> whole voxel),
  2. take the ROI's axial (z) extent in the whole volume,
  3. keep `slices` consecutive axial slices centred on that extent,
  4. shift the window only to stay inside the volume,
  5. fail loudly if the complete ROI extent does not fit in the window.

Only NIfTI headers of the ROI are read; no intensity operation is performed and no label
is used.

Use in a config:

    plugins: [../plugins/datscan.py]
    preprocess:
      whole:
        - {op: roi_slice_window, roi_column: roi_path, slices: 40}
"""
from __future__ import annotations

import numpy as np
from calivox.volumes import Volume, register_step

TOLERANCE = 0.51          # voxel round-off allowed by the original rule


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ np.asarray(affine, float).T)[:, :3]


def localized_window(whole_affine, whole_shape, roi_affine, roi_shape, slices: int = 40) -> tuple[int, int]:
    """(start, stop) z indices of the window in the whole volume."""
    corners = np.array([[x, y, z] for x in (0.0, roi_shape[0] - 1.0) for y in (0.0, roi_shape[1] - 1.0)
                        for z in (0.0, roi_shape[2] - 1.0)])
    try:
        inverse = np.linalg.inv(np.asarray(whole_affine, float))
    except np.linalg.LinAlgError as exc:
        raise ValueError("whole-volume affine is singular") from exc
    z = _apply_affine(inverse, _apply_affine(roi_affine, corners))[:, 2]
    if not np.isfinite(z).all():
        raise ValueError("non-finite ROI-to-whole mapped z coordinate")
    z_low, z_high = float(z.min()), float(z.max())
    depth = whole_shape[2]
    if z_low < -TOLERANCE or z_high > depth - 1 + TOLERANCE:
        raise ValueError(f"ROI crop maps outside the whole volume: z=[{z_low:.3f}, {z_high:.3f}], depth={depth}")
    if z_high - z_low + 1.0 > slices + TOLERANCE:
        raise ValueError(f"ROI axial extent ({z_high - z_low + 1:.2f} slices) is wider than the {slices}-slice window")
    if slices > depth:
        raise ValueError(f"window of {slices} slices is deeper than the volume ({depth})")
    start = int(np.floor(0.5 * (z_low + z_high) - (slices - 1) / 2.0 + 0.5))
    start = max(0, min(start, depth - slices))
    if z_low < start - TOLERANCE or z_high > start + slices - 1 + TOLERANCE:
        raise RuntimeError(f"no {slices}-slice window contains the ROI extent z=[{z_low:.3f}, {z_high:.3f}]")
    return start, start + slices


@register_step("roi_slice_window")
def roi_slice_window(vol: Volume, params: dict, row: dict, resolve) -> Volume:
    import nibabel as nib

    column = params.get("roi_column", "roi_path")
    if column not in row:
        raise ValueError(f"roi_slice_window: manifest has no column {column!r}")
    roi = nib.load(str(resolve(str(row[column]))))
    start, stop = localized_window(vol.affine, vol.array.shape, roi.affine, roi.shape[:3],
                                   int(params.get("slices", 40)))
    affine = vol.affine.copy()
    affine[:3, 3] = affine[:3, :3] @ np.array([0.0, 0.0, float(start)]) + affine[:3, 3]
    return Volume(np.ascontiguousarray(vol.array[:, :, start:stop]), affine)
