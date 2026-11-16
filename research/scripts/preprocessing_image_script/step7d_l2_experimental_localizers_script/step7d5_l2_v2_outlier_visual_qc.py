from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def display_image(image):
    values = image[
        np.isfinite(image)
        & (image > 0)
    ]

    if values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    lo = np.percentile(
        values,
        5,
    )

    hi = np.percentile(
        values,
        99.7,
    )

    out = (
        image - lo
    ) / max(
        hi - lo,
        1e-8,
    )

    out = np.clip(
        out,
        0,
        1,
    )

    return out ** 1.4


def crosshair(
    ax,
    x,
    y,
    label,
):
    ax.scatter(
        [x],
        [y],
        marker="+",
        s=170,
        linewidths=2,
        label=label,
    )


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--l1-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c5_l1_center_consistency/l1_center_consistency.csv"
        ),
    )

    parser.add_argument(
        "--l2-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d4_roi_l2_v2_bilateral_peak_local/l2_v2_localization.csv"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d5_l2_v2_outlier_visual_qc"
        ),
    )

    args = parser.parse_args()

    l1_path = resolve_path(args.l1_csv)
    l2_path = resolve_path(args.l2_csv)
    image_dir = resolve_path(args.image_dir)
    output_dir = resolve_path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    l1 = pd.read_csv(
        l1_path
    )

    l2 = pd.read_csv(
        l2_path
    )

    df = l1[
        [
            "uid",
            "direct_center_x",
            "direct_center_y",
            "direct_center_z",
        ]
    ].merge(
        l2,
        on="uid",
        validate="one_to_one",
    )

    # --------------------------------------------
    # Calculate disagreement
    # --------------------------------------------

    for axis in [
        "x",
        "y",
        "z",
    ]:

        df[
            f"delta_{axis}"
        ] = (
            df[
                f"l2v2_center_{axis}"
            ]
            - df[
                f"direct_center_{axis}"
            ]
        )

    df[
        "distance_vox"
    ] = np.sqrt(
        df["delta_x"] ** 2
        + df["delta_y"] ** 2
        + df["delta_z"] ** 2
    )

    df[
        "distance_mm"
    ] = (
        df["distance_vox"]
        * 2.46
    )

    # --------------------------------------------
    # Select cases
    # --------------------------------------------

    worst = (
        df.sort_values(
            "distance_mm",
            ascending=False,
        )
        .head(15)
        .copy()
    )

    # Also guarantee similarity-rescue diagnostic cases.
    special = similarity_rescue_uids(PROJECT_ROOT)

    selected = worst.copy()

    selected = pd.concat(
        [
            selected,
            df[
                df["uid"].isin(
                    special
                )
            ],
        ],
        ignore_index=True,
    )

    selected = (
        selected
        .drop_duplicates(
            subset="uid"
        )
        .sort_values(
            "distance_mm",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "STEP 7F — L2-v2 OUTLIER VISUAL QC"
    )
    print()

    print(
        f"Selected cases: "
        f"{len(selected)}"
    )

    print()

    manifest = []

    # ============================================
    # QC
    # ============================================

    for index, row in (
        selected.iterrows()
    ):

        uid = row["uid"]

        path = (
            image_dir
            / f"{uid}.nii.gz"
        )

        img = nib.load(
            str(path)
        )

        image = np.asarray(
            img.dataobj,
            dtype=np.float32,
        )

        image = display_image(
            image
        )

        # ----------------------------------------
        # Centers
        # ----------------------------------------

        l1_center = np.array(
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
            dtype=float,
        )

        l2_center = np.array(
            [
                row[
                    "l2v2_center_x"
                ],
                row[
                    "l2v2_center_y"
                ],
                row[
                    "l2v2_center_z"
                ],
            ],
            dtype=float,
        )

        left_peak = np.array(
            [
                row["left_peak_x"],
                row["left_peak_y"],
                row["left_peak_z"],
            ]
        )

        right_peak = np.array(
            [
                row["right_peak_x"],
                row["right_peak_y"],
                row["right_peak_z"],
            ]
        )

        l1_z = int(
            np.clip(
                round(
                    l1_center[2]
                ),
                0,
                image.shape[2] - 1,
            )
        )

        l2_z = int(
            np.clip(
                round(
                    l2_center[2]
                ),
                0,
                image.shape[2] - 1,
            )
        )

        l1_y = int(
            np.clip(
                round(
                    l1_center[1]
                ),
                0,
                image.shape[1] - 1,
            )
        )

        l2_y = int(
            np.clip(
                round(
                    l2_center[1]
                ),
                0,
                image.shape[1] - 1,
            )
        )

        # ========================================
        # Plot
        # ========================================

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(
                16,
                10,
            ),
        )

        # ----------------------------------------
        # Axial at validated L1 Z
        # ----------------------------------------

        ax = axes[
            0,
            0,
        ]

        ax.imshow(
            image[
                :,
                :,
                l1_z,
            ].T,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l1_center[0],
            l1_center[1],
            "L1 validated center",
        )

        crosshair(
            ax,
            l2_center[0],
            l2_center[1],
            "L2-v2 center",
        )

        ax.set_title(
            f"Axial at L1 z={l1_z}"
        )

        ax.legend(
            fontsize=7,
        )

        ax.axis("off")

        # ----------------------------------------
        # Axial at L2 Z
        # ----------------------------------------

        ax = axes[
            0,
            1,
        ]

        ax.imshow(
            image[
                :,
                :,
                l2_z,
            ].T,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l2_center[0],
            l2_center[1],
            "L2-v2 midpoint",
        )

        ax.scatter(
            left_peak[0],
            left_peak[1],
            marker="o",
            facecolors="none",
            s=100,
            linewidths=2,
            label="L2 left peak",
        )

        ax.scatter(
            right_peak[0],
            right_peak[1],
            marker="o",
            facecolors="none",
            s=100,
            linewidths=2,
            label="L2 right peak",
        )

        ax.set_title(
            f"Axial at L2 z={l2_z}"
        )

        ax.legend(
            fontsize=7,
        )

        ax.axis("off")

        # ----------------------------------------
        # Axial MIP
        # ----------------------------------------

        ax = axes[
            0,
            2,
        ]

        axial_mip = np.max(
            image,
            axis=2,
        ).T

        ax.imshow(
            axial_mip,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l1_center[0],
            l1_center[1],
            "L1",
        )

        crosshair(
            ax,
            l2_center[0],
            l2_center[1],
            "L2",
        )

        ax.scatter(
            [
                left_peak[0],
                right_peak[0],
            ],
            [
                left_peak[1],
                right_peak[1],
            ],
            marker="o",
            facecolors="none",
            s=90,
            linewidths=2,
        )

        ax.set_title(
            "Axial whole-volume MIP"
        )

        ax.axis("off")

        # ----------------------------------------
        # Coronal at L1
        # ----------------------------------------

        ax = axes[
            1,
            0,
        ]

        ax.imshow(
            image[
                :,
                l1_y,
                :,
            ].T,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l1_center[0],
            l1_center[2],
            "L1",
        )

        crosshair(
            ax,
            l2_center[0],
            l2_center[2],
            "L2",
        )

        ax.set_title(
            f"Coronal at L1 y={l1_y}"
        )

        ax.axis("off")

        # ----------------------------------------
        # Coronal at L2
        # ----------------------------------------

        ax = axes[
            1,
            1,
        ]

        ax.imshow(
            image[
                :,
                l2_y,
                :,
            ].T,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l2_center[0],
            l2_center[2],
            "L2 midpoint",
        )

        ax.scatter(
            [
                left_peak[0],
                right_peak[0],
            ],
            [
                left_peak[2],
                right_peak[2],
            ],
            marker="o",
            facecolors="none",
            s=90,
            linewidths=2,
        )

        ax.set_title(
            f"Coronal at L2 y={l2_y}"
        )

        ax.axis("off")

        # ----------------------------------------
        # Sagittal MIP
        # horizontal = Y
        # vertical   = Z
        # ----------------------------------------

        ax = axes[
            1,
            2,
        ]

        sagittal_mip = np.max(
            image,
            axis=0,
        ).T

        ax.imshow(
            sagittal_mip,
            cmap="gray",
            origin="lower",
        )

        crosshair(
            ax,
            l1_center[1],
            l1_center[2],
            "L1",
        )

        crosshair(
            ax,
            l2_center[1],
            l2_center[2],
            "L2",
        )

        ax.scatter(
            [
                left_peak[1],
                right_peak[1],
            ],
            [
                left_peak[2],
                right_peak[2],
            ],
            marker="o",
            facecolors="none",
            s=90,
            linewidths=2,
        )

        ax.set_title(
            "Sagittal whole-volume MIP"
        )

        ax.axis("off")

        fig.suptitle(
            (
                f"L2-v2 diagnostic QC — {uid}\n"
                f"L1↔L2 distance = "
                f"{row['distance_mm']:.2f} mm | "
                f"confidence = "
                f"{row['l2v2_confidence']:.3f}"
            ),
            fontsize=14,
        )

        plt.tight_layout()

        out_path = (
            output_dir
            / f"{uid}_l2v2_outlier_qc.png"
        )

        fig.savefig(
            out_path,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        manifest.append(
            {
                "uid": uid,
                "distance_mm": (
                    row[
                        "distance_mm"
                    ]
                ),
                "confidence": (
                    row[
                        "l2v2_confidence"
                    ]
                ),
                "qc_image": str(
                    out_path
                ),
            }
        )

        print(
            f"[{index + 1:2d}/"
            f"{len(selected)}] "
            f"{uid}: "
            f"{row['distance_mm']:.2f} mm"
        )

    manifest_path = (
        output_dir
        / "manifest.csv"
    )

    pd.DataFrame(
        manifest
    ).to_csv(
        manifest_path,
        index=False,
    )

    print()
    print(
        "Saved QC directory:"
    )
    print(
        output_dir
    )


if __name__ == "__main__":
    main()
