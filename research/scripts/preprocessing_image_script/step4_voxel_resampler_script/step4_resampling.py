# dat-scan-classifier/scripts/preprocessing_image_script/step4_voxel_resampler_script/step4_resampling.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def resample_nifti(
    input_path: str | Path,
    output_path: str | Path,
    target_spacing: tuple[float, float, float],
    interpolation_order: int = 1,
) -> dict[str, Any]:
    """
    Resample one 3D NIfTI image to a target voxel spacing.

    Parameters
    ----------
    input_path:
        Input NIfTI file.

    output_path:
        Path where the resampled NIfTI will be saved.

    target_spacing:
        Target voxel spacing in millimeters:
        (x, y, z).

    interpolation_order:
        0 = nearest-neighbor
        1 = linear
        3 = cubic

        Linear interpolation is used by default for
        continuous medical-image intensity data.

    This function does NOT:
    - crop
    - pad
    - normalize intensities
    - force a common image shape
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {input_path}"
        )

    if len(target_spacing) != 3:
        raise ValueError(
            "target_spacing must contain exactly "
            "three values: (x, y, z)."
        )

    if any(value <= 0 for value in target_spacing):
        raise ValueError(
            f"Invalid target spacing: {target_spacing}"
        )

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = nib.load(input_path)

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D image, but "
            f"{input_path.name} has shape {image.shape}"
        )

    original_shape = tuple(
        int(value)
        for value in image.shape
    )

    original_spacing = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]
    )

    original_orientation = tuple(
        str(value)
        for value in nib.aff2axcodes(image.affine)
    )

    # --------------------------------------------------
    # Safety check
    #
    # This pipeline expects orientation standardization
    # to have already been performed.
    # --------------------------------------------------

    if original_orientation != ("R", "A", "S"):
        raise ValueError(
            f"{input_path.name} is not RAS oriented. "
            f"Found orientation: {original_orientation}"
        )

    # --------------------------------------------------
    # Resample
    # --------------------------------------------------

    resampled_image = resample_to_output(
        image,
        voxel_sizes=target_spacing,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    )

    # Convert resampled image data to float32.
    #
    # Interpolation creates continuous values anyway,
    # so retaining uint16 is not appropriate.
    data = resampled_image.get_fdata(
        dtype=np.float32
    )

    header = resampled_image.header.copy()

    output_image = nib.Nifti1Image(
        data,
        resampled_image.affine,
        header=header,
    )

    output_image.set_data_dtype(
        np.float32
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        output_image,
        output_path,
    )

    # --------------------------------------------------
    # Validate output
    # --------------------------------------------------

    final_shape = tuple(
        int(value)
        for value in output_image.shape
    )

    final_spacing = tuple(
        float(value)
        for value in output_image.header.get_zooms()[:3]
    )

    final_orientation = tuple(
        str(value)
        for value in nib.aff2axcodes(
            output_image.affine
        )
    )

    spacing_difference = tuple(
        abs(
            final_spacing[index]
            - target_spacing[index]
        )
        for index in range(3)
    )

    tolerance = 1e-4

    spacing_valid = all(
        difference <= tolerance
        for difference in spacing_difference
    )

    return {
        "file_name": input_path.name,

        "input_path": str(input_path),

        "output_path": str(output_path),

        "original_shape": list(
            original_shape
        ),

        "resampled_shape": list(
            final_shape
        ),

        "original_spacing": list(
            original_spacing
        ),

        "target_spacing": [
            float(value)
            for value in target_spacing
        ],

        "resampled_spacing": list(
            final_spacing
        ),

        "original_orientation": list(
            original_orientation
        ),

        "resampled_orientation": list(
            final_orientation
        ),

        "interpolation_order": (
            interpolation_order
        ),

        "spacing_valid": spacing_valid,

        "output_dtype": "float32",
    }


def find_nifti_files(
    input_dir: Path,
) -> list[Path]:
    """
    Find all .nii and .nii.gz files.
    """

    nii_files = list(
        input_dir.rglob("*.nii")
    )

    nii_gz_files = list(
        input_dir.rglob("*.nii.gz")
    )

    return sorted(
        set(nii_files + nii_gz_files)
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Resample RAS-standardized NIfTI scans "
            "to a common voxel spacing."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing RAS-standardized "
            "NIfTI images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where resampled NIfTI "
            "images will be saved."
        ),
    )

    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        required=True,
        metavar=(
            "X",
            "Y",
            "Z",
        ),
        help=(
            "Target voxel spacing in mm. "
            "Example: "
            "--target-spacing 2.46 2.46 2.46"
        ),
    )

    parser.add_argument(
        "--interpolation-order",
        type=int,
        default=1,
        choices=[
            0,
            1,
            3,
        ],
        help=(
            "Interpolation order: "
            "0=nearest, 1=linear, 3=cubic. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
        help=(
            "Path where the resampling summary "
            "JSON will be saved."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Validate directories
    # --------------------------------------------------

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{args.input_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_spacing = tuple(
        float(value)
        for value in args.target_spacing
    )

    # --------------------------------------------------
    # Find files
    # --------------------------------------------------

    nifti_files = find_nifti_files(
        args.input_dir
    )

    if not nifti_files:
        raise RuntimeError(
            f"No NIfTI files found in: "
            f"{args.input_dir}"
        )

    print()
    print("=" * 60)
    print("NIfTI VOXEL RESAMPLING")
    print("=" * 60)

    print(
        f"Number of scans: "
        f"{len(nifti_files)}"
    )

    print(
        "Target spacing: "
        f"{target_spacing[0]} × "
        f"{target_spacing[1]} × "
        f"{target_spacing[2]} mm"
    )

    print(
        "Interpolation order:",
        args.interpolation_order,
    )

    print()

    # --------------------------------------------------
    # Process images
    # --------------------------------------------------

    results = []
    failures = []

    for index, input_path in enumerate(
        nifti_files,
        start=1,
    ):

        relative_path = input_path.relative_to(
            args.input_dir
        )

        output_path = (
            args.output_dir
            / relative_path
        )

        print(
            f"[{index}/{len(nifti_files)}] "
            f"{input_path.name}"
        )

        try:

            result = resample_nifti(
                input_path=input_path,
                output_path=output_path,
                target_spacing=target_spacing,
                interpolation_order=(
                    args.interpolation_order
                ),
            )

            results.append(
                result
            )

            print(
                "    spacing: "
                f"{result['original_spacing']} "
                "-> "
                f"{result['resampled_spacing']}"
            )

            print(
                "    shape:   "
                f"{result['original_shape']} "
                "-> "
                f"{result['resampled_shape']}"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            failures.append(
                {
                    "file_name": (
                        input_path.name
                    ),
                    "input_path": str(
                        input_path
                    ),
                    "error": str(exc),
                }
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    invalid_spacing = [
        result
        for result in results
        if not result["spacing_valid"]
    ]

    non_ras = [
        result
        for result in results
        if result[
            "resampled_orientation"
        ] != ["R", "A", "S"]
    ]

    summary = {
        "target_spacing": list(
            target_spacing
        ),

        "interpolation_order": (
            args.interpolation_order
        ),

        "number_of_files": len(
            nifti_files
        ),

        "successful": len(
            results
        ),

        "failed": len(
            failures
        ),

        "spacing_validation_failed": len(
            invalid_spacing
        ),

        "non_ras_after_resampling": len(
            non_ras
        ),

        "files": results,

        "failures": failures,
    }

    with args.summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("RESAMPLING COMPLETED")
    print("=" * 60)

    print(
        "Successful:",
        len(results),
    )

    print(
        "Failed:",
        len(failures),
    )

    print(
        "Spacing validation failures:",
        len(invalid_spacing),
    )

    print(
        "Non-RAS outputs:",
        len(non_ras),
    )

    print(
        "Summary:",
        args.summary_output,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
