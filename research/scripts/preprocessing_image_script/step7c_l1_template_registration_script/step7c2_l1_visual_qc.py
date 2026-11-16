from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def uid_from_path(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


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


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros_like(image)

    low = np.percentile(finite, 1)
    high = np.percentile(finite, 99.5)

    if high <= low:
        return np.zeros_like(image)

    out = (image - low) / (high - low)
    return np.clip(out, 0, 1)


def plot_crosshair(
    ax,
    x: float,
    y: float,
) -> None:
    ax.axvline(
        x=x,
        linewidth=1.2,
    )

    ax.axhline(
        y=y,
        linewidth=1.2,
    )


def make_qc(
    uid: str,
    normalized_path: Path,
    mask_path: Path,
    center: np.ndarray,
    output_path: Path,
    special_note: str | None = None,
) -> None:

    image_img = nib.load(str(normalized_path))
    mask_img = nib.load(str(mask_path))

    image = np.asarray(
        image_img.dataobj,
        dtype=np.float32,
    )

    mask = (
        np.asarray(mask_img.dataobj) > 0
    )

    if image.shape != mask.shape:
        raise RuntimeError(
            f"{uid}: image/mask shape mismatch."
        )

    image_display = normalize_for_display(image)

    cx, cy, cz = center

    ix = int(round(cx))
    iy = int(round(cy))
    iz = int(round(cz))

    ix = int(np.clip(ix, 0, image.shape[0] - 1))
    iy = int(np.clip(iy, 0, image.shape[1] - 1))
    iz = int(np.clip(iz, 0, image.shape[2] - 1))

    # ---------------------------------------------------------
    # Views
    # ---------------------------------------------------------

    axial_img = image_display[:, :, iz].T
    axial_mask = mask[:, :, iz].T

    coronal_img = image_display[:, iy, :].T
    coronal_mask = mask[:, iy, :].T

    sagittal_img = image_display[ix, :, :].T
    sagittal_mask = mask[ix, :, :].T

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    views = [
        (
            "Axial",
            axial_img,
            axial_mask,
            cx,
            cy,
        ),
        (
            "Coronal",
            coronal_img,
            coronal_mask,
            cx,
            cz,
        ),
        (
            "Sagittal",
            sagittal_img,
            sagittal_mask,
            cy,
            cz,
        ),
    ]

    # ---------------------------------------------------------
    # Top row — normalized image + center
    # Bottom row — normalized image + mask + center
    # ---------------------------------------------------------

    for col, (
        title,
        view_img,
        view_mask,
        center_x,
        center_y,
    ) in enumerate(views):

        # -----------------------------
        # Image + center
        # -----------------------------

        ax = axes[0, col]

        ax.imshow(
            view_img,
            cmap="gray",
            origin="lower",
        )

        plot_crosshair(
            ax,
            center_x,
            center_y,
        )

        ax.set_title(
            f"{title} — center"
        )
        ax.axis("off")

        # -----------------------------
        # Image + mask + center
        # -----------------------------

        ax = axes[1, col]

        ax.imshow(
            view_img,
            cmap="gray",
            origin="lower",
        )

        ax.imshow(
            np.ma.masked_where(
                ~view_mask,
                view_mask,
            ),
            alpha=0.35,
            origin="lower",
        )

        plot_crosshair(
            ax,
            center_x,
            center_y,
        )

        ax.set_title(
            f"{title} — mapped mask"
        )
        ax.axis("off")

    title = (
        f"Step 7E — L1 Visual QC — {uid}\n"
        f"Center voxel = "
        f"({cx:.2f}, {cy:.2f}, {cz:.2f})"
    )

    if special_note:
        title += f"\n{special_note}"

    fig.suptitle(
        title,
        fontsize=14,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--localization-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c1_roi_l1_template_transform/l1_localization.csv"
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
        "--mask-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7c_l1_template_registration_data/step7c1_roi_l1_template_transform"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7c_l1_template_registration_data/"
            "step7c2_l1_visual_qc"
        ),
    )

    parser.add_argument(
        "--random-count",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    localization_csv = resolve_path(args.localization_csv)
    normalized_dir = resolve_path(args.normalized_dir)
    mask_dir = resolve_path(args.mask_dir)
    output_dir = resolve_path(args.output_dir)

    df = pd.read_csv(
        localization_csv
    )

    if df.empty:
        raise RuntimeError(
            "Localization CSV is empty."
        )

    # ---------------------------------------------------------
    # Select QC cases
    # ---------------------------------------------------------

    selections = {}

    def add(uid: str, reason: str):
        if uid not in selections:
            selections[uid] = reason
        else:
            selections[uid] += f"; {reason}"

    # ---------------------------------------------------------
    # Similarity-rescue registrations (invalid mapped extent)
    # ---------------------------------------------------------

    for uid in similarity_rescue_uids(PROJECT_ROOT):
        if uid in set(df["uid"].astype(str)):
            add(
                uid,
                (
                    "Similarity rescue; "
                    "scale-distorted mask extent"
                ),
            )

    # ---------------------------------------------------------
    # Mask-size extremes
    # ---------------------------------------------------------

    for _, row in (
        df.nsmallest(
            5,
            "mapped_mask_voxels",
        ).iterrows()
    ):
        add(
            row["uid"],
            "lowest mapped-mask voxel count",
        )

    for _, row in (
        df.nlargest(
            5,
            "mapped_mask_voxels",
        ).iterrows()
    ):
        add(
            row["uid"],
            "highest mapped-mask voxel count",
        )

    # ---------------------------------------------------------
    # Center-coordinate extremes
    # ---------------------------------------------------------

    for axis in ["x", "y", "z"]:
        col = f"l1_center_{axis}"

        for _, row in (
            df.nsmallest(3, col).iterrows()
        ):
            add(
                row["uid"],
                f"lowest L1 center {axis.upper()}",
            )

        for _, row in (
            df.nlargest(3, col).iterrows()
        ):
            add(
                row["uid"],
                f"highest L1 center {axis.upper()}",
            )

    # ---------------------------------------------------------
    # Border-distance extremes
    # ---------------------------------------------------------

    for _, row in (
        df.nsmallest(
            5,
            "minimum_border_distance_voxels",
        ).iterrows()
    ):
        add(
            row["uid"],
            "closest mapped ROI to image border",
        )

    # ---------------------------------------------------------
    # Random representative cases
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    remaining = df[
        ~df["uid"].isin(
            selections.keys()
        )
    ]

    random_n = min(
        args.random_count,
        len(remaining),
    )

    random_indices = rng.choice(
        remaining.index.to_numpy(),
        size=random_n,
        replace=False,
    )

    for idx in random_indices:
        add(
            df.loc[idx, "uid"],
            "random representative case",
        )

    # ---------------------------------------------------------
    # Generate QC images
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print("STEP 7E — L1 VISUAL QC")
    print("=" * 76)

    print(
        f"Selected cases: {len(selections)}"
    )
    print()

    failures = []

    manifest_rows = []

    for i, (uid, reason) in enumerate(
        selections.items(),
        start=1,
    ):

        row = df[
            df["uid"] == uid
        ].iloc[0]

        normalized_path = (
            normalized_dir
            / f"{uid}.nii.gz"
        )

        mask_path = (
            mask_dir
            / f"{uid}.nii.gz"
        )

        output_path = (
            output_dir
            / f"{uid}_l1_qc.png"
        )

        try:
            if not normalized_path.exists():
                raise FileNotFoundError(
                    normalized_path
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    mask_path
                )

            center = np.array(
                [
                    row["l1_center_x"],
                    row["l1_center_y"],
                    row["l1_center_z"],
                ],
                dtype=np.float64,
            )

            special_note = reason

            make_qc(
                uid=uid,
                normalized_path=normalized_path,
                mask_path=mask_path,
                center=center,
                output_path=output_path,
                special_note=special_note,
            )

            manifest_rows.append(
                {
                    "uid": uid,
                    "reason": reason,
                    "mapped_mask_voxels": int(
                        row["mapped_mask_voxels"]
                    ),
                    "bbox_size_x": int(
                        row["bbox_size_x"]
                    ),
                    "bbox_size_y": int(
                        row["bbox_size_y"]
                    ),
                    "bbox_size_z": int(
                        row["bbox_size_z"]
                    ),
                    "minimum_border_distance_voxels": int(
                        row[
                            "minimum_border_distance_voxels"
                        ]
                    ),
                    "qc_image": str(
                        output_path
                    ),
                    "manual_qc": "",
                    "manual_notes": "",
                }
            )

            print(
                f"[{i:2d}/{len(selections)}] "
                f"{uid}"
            )
            print(
                f"    reason: {reason}"
            )

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "error": str(exc),
                }
            )

    # ---------------------------------------------------------
    # Save QC manifest
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "l1_visual_qc_manifest.csv"
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        manifest_path,
        index=False,
    )

    print()
    print("=" * 76)
    print("STEP 7E GENERATION RESULTS")
    print("=" * 76)

    print(
        f"QC images generated: {len(manifest_rows)}"
    )
    print(
        f"Generation failures: {len(failures)}"
    )

    print()
    print("QC directory:")
    print(
        f"  {output_dir.resolve()}"
    )

    print()
    print("QC manifest:")
    print(
        f"  {manifest_path.resolve()}"
    )

    if failures:
        print()
        print("Failures:")

        for failure in failures:
            print(
                failure["uid"],
                failure["error"],
            )

        raise RuntimeError(
            "One or more QC images failed."
        )

    print()
    print(
        "NEXT: manually inspect images and "
        "fill manual_qc as PASS / REVIEW / FAIL."
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
