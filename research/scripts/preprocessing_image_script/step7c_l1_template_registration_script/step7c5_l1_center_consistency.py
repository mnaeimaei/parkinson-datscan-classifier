from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def similarity_rescue_uids(project_root: Path) -> set[str]:
    path = (
        project_root
        / "data/preprocessing_image_data/"
        / "step6c_finalize_registration_data/"
        / "step6c_finalize_all_registration_transforms/"
        / "final_registration_manifest.csv"
    )

    if not path.exists():
        return set()

    reg = pd.read_csv(path)

    uid_col = next(
        (
            column
            for column in ["uid", "subject_uid", "subject_id"]
            if column in reg.columns
        ),
        None,
    )
    source_col = next(
        (
            column
            for column in ["final_source_type", "source_type"]
            if column in reg.columns
        ),
        None,
    )

    if uid_col is None or source_col is None:
        return set()

    mask = (
        reg[source_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == "similarity_rescue"
    )

    return set(
        reg.loc[mask, uid_col].astype(str)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mapped-mask centroid against directly "
            "transformed template striatal center."
        )
    )

    parser.add_argument(
        "--localization-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c1_roi_l1_template_transform/l1_localization.csv"
        ),
    )

    parser.add_argument(
        "--template-mask",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step5d_normalization_strategy_analysis_data/"
            "build_striatal_masks/"
            "striatum_mask.nii.gz"
        ),
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
        ),
    )

    parser.add_argument(
        "--transform-root",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step6c_finalize_registration_data/"
            "step6c_finalize_all_registration_transforms/transforms"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c5_l1_center_consistency/l1_center_consistency.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c5_l1_center_consistency/summary.json"
        ),
    )

    args = parser.parse_args()

    localization_csv = resolve_path(args.localization_csv)
    template_mask_path = resolve_path(args.template_mask)
    normalized_dir = resolve_path(args.normalized_dir)
    transform_root = resolve_path(args.transform_root)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)

    df = pd.read_csv(
        localization_csv
    )

    if df.empty:
        raise RuntimeError(
            "Localization CSV is empty."
        )

    # ---------------------------------------------------------
    # Template bilateral center
    # ---------------------------------------------------------

    template_mask = sitk.ReadImage(
        str(template_mask_path)
    )

    mask_array = (
        sitk.GetArrayFromImage(template_mask) > 0
    )

    # SimpleITK array order = Z, Y, X
    coords_zyx = np.argwhere(mask_array)

    if coords_zyx.size == 0:
        raise RuntimeError(
            "Template striatal mask is empty."
        )

    center_zyx = coords_zyx.mean(axis=0)

    center_xyz = (
        float(center_zyx[2]),
        float(center_zyx[1]),
        float(center_zyx[0]),
    )

    # Continuous voxel/index -> SimpleITK physical point
    template_center_physical = (
        template_mask.TransformContinuousIndexToPhysicalPoint(
            center_xyz
        )
    )

    print()
    print("=" * 80)
    print(
        "STEP 7E-2 — L1 DIRECT-CENTER CONSISTENCY"
    )
    print("=" * 80)

    print(
        "Template center index XYZ:",
        tuple(round(v, 4) for v in center_xyz),
    )

    print(
        "Template center physical:",
        tuple(
            round(v, 4)
            for v in template_center_physical
        ),
    )

    print()

    rows = []
    failures = []

    # ---------------------------------------------------------
    # Each subject
    # ---------------------------------------------------------

    for i, row in df.iterrows():

        uid = str(row["uid"])

        subject_path = (
            normalized_dir
            / f"{uid}.nii.gz"
        )

        transform_path = (
            transform_root
            / uid
            / "final_transform.h5"
        )

        try:
            subject = sitk.ReadImage(
                str(subject_path)
            )

            transform = sitk.ReadTransform(
                str(transform_path)
            )

            # IMPORTANT:
            #
            # Frozen Step-6 transform maps:
            #
            # template physical point
            #          ->
            # subject physical point
            #
            direct_subject_physical = (
                transform.TransformPoint(
                    template_center_physical
                )
            )

            direct_subject_index = (
                subject.TransformPhysicalPointToContinuousIndex(
                    direct_subject_physical
                )
            )

            mapped_centroid = np.asarray(
                [
                    row["l1_center_x"],
                    row["l1_center_y"],
                    row["l1_center_z"],
                ],
                dtype=np.float64,
            )

            direct_index = np.asarray(
                direct_subject_index,
                dtype=np.float64,
            )

            delta = (
                mapped_centroid
                - direct_index
            )

            distance_voxels = float(
                np.linalg.norm(delta)
            )

            spacing = np.asarray(
                subject.GetSpacing(),
                dtype=np.float64,
            )

            delta_mm = (
                delta * spacing
            )

            distance_mm = float(
                np.linalg.norm(delta_mm)
            )

            size = np.asarray(
                subject.GetSize(),
                dtype=np.float64,
            )

            direct_inside = bool(
                np.all(direct_index >= 0)
                and np.all(
                    direct_index <= size - 1
                )
            )

            rows.append(
                {
                    "uid": uid,

                    "mapped_center_x": float(
                        mapped_centroid[0]
                    ),
                    "mapped_center_y": float(
                        mapped_centroid[1]
                    ),
                    "mapped_center_z": float(
                        mapped_centroid[2]
                    ),

                    "direct_center_x": float(
                        direct_index[0]
                    ),
                    "direct_center_y": float(
                        direct_index[1]
                    ),
                    "direct_center_z": float(
                        direct_index[2]
                    ),

                    "delta_x_voxels": float(
                        delta[0]
                    ),
                    "delta_y_voxels": float(
                        delta[1]
                    ),
                    "delta_z_voxels": float(
                        delta[2]
                    ),

                    "distance_voxels": (
                        distance_voxels
                    ),
                    "distance_mm": (
                        distance_mm
                    ),

                    "direct_center_inside_image": (
                        direct_inside
                    ),

                    "transform_type": (
                        transform.GetName()
                    ),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "error": str(exc),
                }
            )

        if (
            i == 0
            or (i + 1) % 100 == 0
            or i + 1 == len(df)
        ):
            print(
                f"[{i + 1:4d}/{len(df)}] processed"
            )

    if failures:
        for failure in failures[:20]:
            print(
                failure["uid"],
                failure["error"],
            )

        raise RuntimeError(
            f"{len(failures)} consistency checks failed."
        )

    out = pd.DataFrame(rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    distance_voxels = (
        out["distance_voxels"].to_numpy()
    )

    distance_mm = (
        out["distance_mm"].to_numpy()
    )

    summary = {
        "processed": int(len(out)),
        "failed": int(len(failures)),
        "direct_centers_outside_image": int(
            (~out[
                "direct_center_inside_image"
            ]).sum()
        ),
        "distance_voxels": describe(
            distance_voxels
        ),
        "distance_mm": describe(
            distance_mm
        ),
        "similarity_rescue_cases": (
            out[
                out["uid"].astype(str).isin(
                    similarity_rescue_uids(PROJECT_ROOT)
                )
            ]
            .to_dict(
                orient="records"
            )
        ),
    }

    summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Successful:              {len(out)}"
    )
    print(
        f"Failed:                  {len(failures)}"
    )
    print(
        "Direct centers outside: ",
        summary[
            "direct_centers_outside_image"
        ],
    )

    print()
    print("Mapped centroid vs direct center:")

    print(
        "Median distance [vox]: ",
        f"{summary['distance_voxels']['median']:.4f}",
    )

    print(
        "P95 distance [vox]:    ",
        f"{summary['distance_voxels']['p95']:.4f}",
    )

    print(
        "Maximum distance [vox]:",
        f"{summary['distance_voxels']['max']:.4f}",
    )

    print()

    print(
        "Median distance [mm]:  ",
        f"{summary['distance_mm']['median']:.4f}",
    )

    print(
        "P95 distance [mm]:     ",
        f"{summary['distance_mm']['p95']:.4f}",
    )

    print(
        "Maximum distance [mm]: ",
        f"{summary['distance_mm']['max']:.4f}",
    )

    print()

    rescue = out[
        out["uid"].astype(str).isin(
            similarity_rescue_uids(PROJECT_ROOT)
        )
    ]

    if len(rescue) > 0:
        print("Similarity-rescue cases:")

        for _, r in rescue.iterrows():
            print(
                f"  {r['uid']}  distance vox: "
                f"{r['distance_voxels']:.4f}  "
                f"distance mm: {r['distance_mm']:.4f}"
            )

    print()
    print("CSV:")
    print(
        f"  {output_csv.resolve()}"
    )

    print()
    print("Summary:")
    print(
        f"  {summary_json.resolve()}"
    )

    print()
    print(
        "STEP 7E-2 CONSISTENCY CHECK: COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
