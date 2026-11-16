from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from src.registration.affine_registrar import (
    register_affine,
    save_transform,
)

from src.registration.reference_region_mapper import (
    analyze_reference_region,
    map_template_mask_to_subject,
)


def find_nifti_files(
    directory: Path,
) -> list[Path]:

    files = (
        list(directory.rglob("*.nii"))
        + list(directory.rglob("*.nii.gz"))
    )

    return sorted(set(files))


def add_prefixed(
    row: dict,
    prefix: str,
    values: dict,
) -> None:

    for key, value in values.items():
        row[
            f"{prefix}_{key}"
        ] = value


def analyze_anatomical_references(
    input_dir: str | Path,
    template_path: str | Path,
    occipital_mask_path: str | Path,
    cerebellar_mask_path: str | Path,
    output_dir: str | Path,
    transform_dir: str | Path,
    limit: int = 0,
) -> dict:

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    transform_dir = Path(transform_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = sitk.ReadImage(
        str(template_path),
        sitk.sitkFloat32,
    )

    occipital_mask = sitk.ReadImage(
        str(occipital_mask_path),
        sitk.sitkUInt8,
    )

    cerebellar_mask = sitk.ReadImage(
        str(cerebellar_mask_path),
        sitk.sitkUInt8,
    )

    files = find_nifti_files(
        input_dir
    )

    if limit > 0:
        files = files[:limit]

    results: list[dict] = []
    failures: list[dict] = []

    total = len(files)

    for index, path in enumerate(
        files,
        start=1,
    ):
        print(
            f"[{index}/{total}] "
            f"{path.name}"
        )

        try:
            subject = sitk.ReadImage(
                str(path),
                sitk.sitkFloat32,
            )

            transform, registration = (
                register_affine(
                    fixed_template=template,
                    moving_subject=subject,
                )
            )

            uid = (
                path.name
                .replace(".nii.gz", "")
                .replace(".nii", "")
            )

            transform_path = (
                transform_dir
                / f"{uid}_affine.tfm"
            )

            save_transform(
                transform,
                transform_path,
            )

            occipital_subject = (
                map_template_mask_to_subject(
                    template_mask=occipital_mask,
                    subject_image=subject,
                    subject_to_template_transform=transform,
                )
            )

            cerebellar_subject = (
                map_template_mask_to_subject(
                    template_mask=cerebellar_mask,
                    subject_image=subject,
                    subject_to_template_transform=transform,
                )
            )

            occipital_stats = (
                analyze_reference_region(
                    subject_image=subject,
                    subject_mask=occipital_subject,
                    template_mask=occipital_mask,
                )
            )

            cerebellar_stats = (
                analyze_reference_region(
                    subject_image=subject,
                    subject_mask=cerebellar_subject,
                    template_mask=cerebellar_mask,
                )
            )

            row = {
                "file_name": path.name,
                "uid": uid,
                "transform_path": str(
                    transform_path
                ),

                "rigid_metric": (
                    registration[
                        "rigid_metric"
                    ]
                ),

                "affine_metric": (
                    registration[
                        "affine_metric"
                    ]
                ),

                "head_mask_dice": (
                    registration[
                        "head_mask_dice"
                    ]
                ),
            }

            add_prefixed(
                row,
                "occipital",
                occipital_stats,
            )

            add_prefixed(
                row,
                "cerebellar",
                cerebellar_stats,
            )

            results.append(row)

            print(
                "  Dice="
                f"{registration['head_mask_dice']:.3f}"
            )

            print(
                "  Occipital positive="
                f"{occipital_stats.get('positive_fraction')}"
            )

            print(
                "  Cerebellar positive="
                f"{cerebellar_stats.get('positive_fraction')}"
            )

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": path.name,
                    "error": str(exc),
                }
            )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        output_dir
        / "anatomical_reference_per_scan.csv",
        index=False,
    )

    summary = {
        "analysis_type": (
            "anatomical_reference_feasibility"
        ),

        "number_of_files": total,

        "successful": len(results),

        "failed": len(failures),

        "template": str(
            template_path
        ),

        "occipital_mask": str(
            occipital_mask_path
        ),

        "cerebellar_mask": str(
            cerebellar_mask_path
        ),

        "registration": {
            "type": "rigid_then_affine",
            "nonlinear_warping": False,
            "deterministic_sampling_seed": 42,
        },

        "important_note": (
            "Reference values are measured in the "
            "original resampled subject image. "
            "Registration-scaled images are used only "
            "for spatial localization."
        ),

        "failures": failures,
    }

    with (
        output_dir
        / "anatomical_reference_statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--occipital-mask",
        required=True,
    )

    parser.add_argument(
        "--cerebellar-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--transform-dir",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "0 = all scans. "
            "Use a small value for initial QC."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ANATOMICAL REFERENCE FEASIBILITY ANALYSIS")
    print("=" * 70)

    summary = analyze_anatomical_references(
        input_dir=args.input_dir,
        template_path=args.template,
        occipital_mask_path=args.occipital_mask,
        cerebellar_mask_path=args.cerebellar_mask,
        output_dir=args.output_dir,
        transform_dir=args.transform_dir,
        limit=args.limit,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"Successful: {summary['successful']}"
    )

    print(
        f"Failed: {summary['failed']}"
    )

    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()