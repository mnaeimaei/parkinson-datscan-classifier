from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def finite_center(values) -> bool:
    x = np.asarray(values, dtype=np.float64)

    return bool(
        x.shape == (3,)
        and np.all(np.isfinite(x))
    )


def inside_image(
    center: np.ndarray,
    shape: tuple[int, int, int],
) -> bool:

    size = np.asarray(
        shape,
        dtype=np.float64,
    )

    return bool(
        np.all(center >= 0)
        and np.all(center <= size - 1)
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Step 7H — freeze final striatal "
            "localization policy."
        )
    )

    parser.add_argument(
        "--l1-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c5_l1_center_consistency/l1_center_consistency.csv"
        ),
    )

    parser.add_argument(
        "--l0-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7b_l0_central_baseline_data/"
            "step7b_roi_l0_fixed_center/l0_centers.csv"
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
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7e_localizer_comparison_and_freeze_data/"
            "step7e2_freeze_final_localization/final_localization.csv"
        ),
    )

    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7e_localizer_comparison_and_freeze_data/"
            "step7e2_freeze_final_localization/localization_policy.json"
        ),
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    l1_csv = resolve_path(args.l1_csv)
    l0_csv = resolve_path(args.l0_csv)
    normalized_dir = resolve_path(args.normalized_dir)
    output_csv = resolve_path(args.output_csv)
    policy_json = resolve_path(args.policy_json)

    # ---------------------------------------------------------
    # Read L1
    # ---------------------------------------------------------

    l1 = pd.read_csv(
        l1_csv
    )

    if args.expected_count is None:
        args.expected_count = int(len(l1))

    if len(l1) != args.expected_count:
        raise RuntimeError(
            f"L1 expected {args.expected_count} rows, "
            f"found {len(l1)}."
        )

    if l1["uid"].duplicated().any():
        raise RuntimeError(
            "Duplicate UID in L1 CSV."
        )

    # ---------------------------------------------------------
    # Read L0
    # ---------------------------------------------------------

    l0 = pd.read_csv(
        l0_csv
    )

    if len(l0) != args.expected_count:
        raise RuntimeError(
            f"L0 expected {args.expected_count} rows, "
            f"found {len(l0)}."
        )

    if l0["uid"].duplicated().any():
        raise RuntimeError(
            "Duplicate UID in L0 CSV."
        )

    required_l0 = [
        "l0_center_x",
        "l0_center_y",
        "l0_center_z",
    ]

    missing = [
        c
        for c in required_l0
        if c not in l0.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing L0 columns: {missing}"
        )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    df = l1.merge(
        l0[
            [
                "uid",
                "l0_center_x",
                "l0_center_y",
                "l0_center_z",
            ]
        ],
        on="uid",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != args.expected_count:
        raise RuntimeError(
            f"Merged rows = {len(df)}, "
            f"expected {args.expected_count}."
        )

    # ---------------------------------------------------------
    # Freeze final localization
    # ---------------------------------------------------------

    rows = []

    l1_used = 0
    l0_used = 0
    failures = []

    print()
    print("=" * 80)
    print(
        "STEP 7H — FREEZE FINAL LOCALIZATION POLICY"
    )
    print("=" * 80)

    print()
    print("Policy:")
    print("  PRIMARY  = L1")
    print("  FALLBACK = L0")
    print("  L2       = excluded from final policy")
    print()

    for index, row in df.iterrows():

        uid = str(row["uid"])

        image_path = (
            normalized_dir
            / f"{uid}.nii.gz"
        )

        try:
            img = nib.load(
                str(image_path)
            )

            orientation = nib.aff2axcodes(
                img.affine
            )

            if orientation != (
                "R",
                "A",
                "S",
            ):
                raise RuntimeError(
                    f"{uid}: expected RAS, "
                    f"got {orientation}."
                )

            shape = img.shape

            if len(shape) != 3:
                raise RuntimeError(
                    f"{uid}: expected 3-D image, "
                    f"got {shape}."
                )

            # -------------------------------------------------
            # PRIMARY: validated directly transformed L1 center
            # -------------------------------------------------

            l1_center = np.asarray(
                [
                    row[
                        "direct_center_x"
                    ],
                    row[
                        "direct_center_y"
                    ],
                    row[
                        "direct_center_z"
                    ],
                ],
                dtype=np.float64,
            )

            l1_valid = (
                finite_center(l1_center)
                and inside_image(
                    l1_center,
                    shape,
                )
            )

            # If Step 7E-2 contains explicit inside-image
            # validation, also respect it.
            if (
                "direct_center_inside_image"
                in row.index
            ):
                l1_valid = (
                    l1_valid
                    and bool(
                        row[
                            "direct_center_inside_image"
                        ]
                    )
                )

            fallback_used = False

            if l1_valid:

                center = l1_center

                method = (
                    "L1_template_transform"
                )

                # Categorical technical confidence,
                # NOT a calibrated probability.
                confidence = "HIGH"

                qc_status = "PASS"

                l1_used += 1

            else:

                # ---------------------------------------------
                # FALLBACK: deterministic array center
                # ---------------------------------------------

                l0_center = np.asarray(
                    [
                        row["l0_center_x"],
                        row["l0_center_y"],
                        row["l0_center_z"],
                    ],
                    dtype=np.float64,
                )

                if not (
                    finite_center(l0_center)
                    and inside_image(
                        l0_center,
                        shape,
                    )
                ):
                    raise RuntimeError(
                        "Both L1 and L0 localization invalid."
                    )

                center = l0_center

                method = (
                    "L0_array_center_fallback"
                )

                confidence = "LOW"

                qc_status = "FALLBACK_REVIEW"

                fallback_used = True

                l0_used += 1

            # -------------------------------------------------
            # Subject-space physical RAS coordinates
            # -------------------------------------------------

            physical = (
                nib.affines.apply_affine(
                    img.affine,
                    center,
                )
            )

            rows.append(
                {
                    "uid": uid,

                    "center_x": float(
                        center[0]
                    ),
                    "center_y": float(
                        center[1]
                    ),
                    "center_z": float(
                        center[2]
                    ),

                    "physical_x_mm": float(
                        physical[0]
                    ),
                    "physical_y_mm": float(
                        physical[1]
                    ),
                    "physical_z_mm": float(
                        physical[2]
                    ),

                    "localization_method": (
                        method
                    ),

                    "localization_confidence": (
                        confidence
                    ),

                    "fallback_used": (
                        fallback_used
                    ),

                    "qc_status": (
                        qc_status
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
            index == 0
            or (index + 1) % 100 == 0
            or index + 1 == len(df)
        ):
            print(
                f"[{index + 1:4d}/{len(df)}] "
                f"processed"
            )

    # ---------------------------------------------------------
    # Fail if any training localization cannot be produced
    # ---------------------------------------------------------

    if failures:

        print()
        print("FAILURES")
        print("-" * 80)

        for failure in failures[:20]:
            print(
                failure["uid"],
                "->",
                failure["error"],
            )

        raise RuntimeError(
            f"{len(failures)} final localization "
            f"failures."
        )

    final_df = pd.DataFrame(
        rows
    )

    if len(final_df) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} final rows, "
            f"found {len(final_df)}."
        )

    if final_df["uid"].duplicated().any():
        raise RuntimeError(
            "Duplicate final localization UIDs."
        )

    # ---------------------------------------------------------
    # Save final localization CSV
    # ---------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        output_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # Freeze policy
    # ---------------------------------------------------------

    policy = {
        "step": "7H",

        "status": "FROZEN_PENDING_RUNTIME_GATE",

        "primary_localizer": {
            "method": (
                "L1_template_transform"
            ),

            "definition": (
                "Direct transformation of the "
                "template bilateral striatal center "
                "into subject space using the "
                "validated Step-6 registration transform."
            ),
        },

        "fallback_localizer": {
            "method": (
                "L0_array_center"
            ),

            "definition": (
                "Deterministic array center used only "
                "if L1 cannot produce a technically "
                "valid center."
            ),

            "warning": (
                "L0 is an emergency fallback only; "
                "Step-7G showed large disagreement "
                "with validated L1 due to FOV variation."
            ),
        },

        "excluded_from_final_policy": {
            "L2_v1": (
                "Rejected due to diffuse candidate "
                "regions and poor localization accuracy."
            ),

            "L2_v2": (
                "Rejected as automatic localizer due "
                "to catastrophic high-confidence "
                "localization outliers."
            ),
        },

        "training_dataset_result": {
            "number_of_scans": int(
                len(final_df)
            ),

            "L1_used": int(
                l1_used
            ),

            "L0_fallback_used": int(
                l0_used
            ),

            "failures": int(
                len(failures)
            ),
        },

        "important": [
            (
                "The final ROI center uses L1 direct "
                "continuous coordinates, not the "
                "nearest-neighbor mapped-mask centroid."
            ),

            (
                "Mapped L1 masks remain available for "
                "visual QC and extent analysis."
            ),

            (
                "Similarity-rescue L1 centers remain valid; their "
                "similarity-scaled mapped masks must not "
                "be used to infer anatomical extent."
            ),

            (
                "This policy is not approved for final "
                "competition inference until Step 7I "
                "runtime testing passes."
            ),
        ],
    }

    policy_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with policy_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            policy,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7H RESULTS")
    print("=" * 80)

    print(
        f"Final localizations:      "
        f"{len(final_df)}"
    )

    print(
        f"L1 primary used:          "
        f"{l1_used}"
    )

    print(
        f"L0 fallback used:         "
        f"{l0_used}"
    )

    print(
        f"Failures:                 "
        f"{len(failures)}"
    )

    print()

    print("Method counts:")
    print(
        final_df[
            "localization_method"
        ].value_counts()
    )

    print()

    print("Final localization CSV:")
    print(
        output_csv.resolve()
    )

    print()

    print("Frozen policy:")
    print(
        policy_json.resolve()
    )

    print()
    print(
        "STEP 7H STATUS: "
        "FROZEN PENDING STEP-7I RUNTIME GATE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
