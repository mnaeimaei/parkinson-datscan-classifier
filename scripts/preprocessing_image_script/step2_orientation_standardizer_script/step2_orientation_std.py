import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib


def standardize_orientation(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    Convert a NIfTI image to canonical RAS+ orientation.

    This operation only:
    - reorders axes
    - flips axes when necessary
    - updates the affine matrix

    It does NOT:
    - resample voxel spacing
    - interpolate voxel intensities
    - crop the image
    - normalize intensities
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {input_path}"
        )

    # Load original image
    image = nib.load(input_path)

    original_orientation = nib.aff2axcodes(image.affine)
    original_shape = image.shape
    original_spacing = image.header.get_zooms()[:3]

    # Convert to closest canonical orientation = RAS+
    standardized_image = nib.as_closest_canonical(image)

    standardized_orientation = nib.aff2axcodes(
        standardized_image.affine
    )

    standardized_shape = standardized_image.shape
    standardized_spacing = standardized_image.header.get_zooms()[:3]

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save standardized NIfTI
    nib.save(
        standardized_image,
        output_path,
    )

    return {
        "file_name": input_path.name,
        "input_path": str(input_path),
        "output_path": str(output_path),

        "original_orientation": [
            str(x) for x in original_orientation
        ],

        "standardized_orientation": [
            str(x) for x in standardized_orientation
        ],

        "original_shape": [
            int(x) for x in original_shape
        ],

        "standardized_shape": [
            int(x) for x in standardized_shape
        ],

        "original_spacing": [
            float(x) for x in original_spacing
        ],

        "standardized_spacing": [
            float(x) for x in standardized_spacing
        ],
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
            "Standardize NIfTI images "
            "to RAS+ anatomical orientation."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw NIfTI files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where standardized "
            "NIfTI files will be saved."
        ),
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing "
            "orientation conversion information."
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nifti_files = find_nifti_files(
        input_dir
    )

    if not nifti_files:
        raise RuntimeError(
            f"No NIfTI files found in: "
            f"{input_dir}"
        )

    print(
        f"Found {len(nifti_files)} "
        f"NIfTI files."
    )

    results = []
    failed = []

    for index, input_path in enumerate(
        nifti_files,
        start=1,
    ):
        relative_path = input_path.relative_to(
            input_dir
        )

        output_path = (
            output_dir / relative_path
        )

        print(
            f"[{index}/{len(nifti_files)}] "
            f"{input_path.name}"
        )

        try:
            result = standardize_orientation(
                input_path=input_path,
                output_path=output_path,
            )

            results.append(result)

            print(
                "    "
                f"{''.join(result['original_orientation'])}"
                " -> "
                f"{''.join(result['standardized_orientation'])}"
            )

        except Exception as exc:
            print(
                f"    ERROR: {exc}"
            )

            failed.append(
                {
                    "file_name": input_path.name,
                    "error": str(exc),
                }
            )

    print()
    print(
        f"Successfully processed: "
        f"{len(results)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = {
            "target_orientation": "RAS",
            "number_of_files": len(
                nifti_files
            ),
            "successful": len(results),
            "failed": len(failed),
            "files": results,
            "failures": failed,
        }

        with args.metadata_output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
            )

        print(
            f"Metadata saved to: "
            f"{args.metadata_output}"
        )


if __name__ == "__main__":
    main()
