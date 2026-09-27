import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def read_nifti_metadata(nifti_path: str | Path) -> dict[str, Any]:
    """
    Load a NIfTI image and extract its main metadata.

    Extracted information:
    - file name
    - shape
    - voxel spacing
    - orientation
    - affine matrix
    - data type
    """

    nifti_path = Path(nifti_path)

    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    # Load NIfTI image
    image = nib.load(nifti_path)

    header = image.header
    affine = image.affine

    # Image dimensions
    shape = image.shape

    # Voxel spacing
    voxel_spacing = header.get_zooms()[:3]

    # Orientation codes, e.g. ('R', 'A', 'S')
    orientation = nib.aff2axcodes(affine)

    # Data type
    dtype = header.get_data_dtype()

    metadata = {
        "file_name": nifti_path.name,
        "file_path": str(nifti_path),
        "shape": [int(dim) for dim in shape],
        "voxel_spacing": [float(spacing) for spacing in voxel_spacing],
        "orientation": list(orientation),
        "affine": np.asarray(affine).tolist(),
        "dtype": str(dtype),
    }

    return metadata


def find_nifti_files(input_dir: Path) -> list[Path]:
    """
    Find all .nii and .nii.gz files.
    """
    nii_files = list(input_dir.rglob("*.nii"))
    nii_gz_files = list(input_dir.rglob("*.nii.gz"))

    return sorted(set(nii_files + nii_gz_files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read NIfTI image metadata."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing NIfTI files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metadata will be saved.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = find_nifti_files(input_dir)

    if not nifti_files:
        raise RuntimeError(
            f"No NIfTI files found in: {input_dir}"
        )

    print(f"Found {len(nifti_files)} NIfTI files.")

    all_metadata = []

    for index, nifti_path in enumerate(nifti_files, start=1):
        print(
            f"[{index}/{len(nifti_files)}] "
            f"Reading {nifti_path.name}"
        )

        try:
            metadata = read_nifti_metadata(nifti_path)
            all_metadata.append(metadata)

        except Exception as exc:
            print(
                f"Error reading {nifti_path}: {exc}"
            )

    output_file = output_dir / "nifti_metadata.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            all_metadata,
            file,
            indent=4,
        )

    print()
    print(f"Metadata saved to: {output_file}")


if __name__ == "__main__":
    main()
