from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

LEFT_REGION_NAMES = {
    "caudate_l",
    "putamen_l",
}

RIGHT_REGION_NAMES = {
    "caudate_r",
    "putamen_r",
}


def _parse_integer(token: str) -> int | None:
    token = token.strip()

    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)

    return None


def parse_aal_lut(
    lut_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Parse the AAL lookup table.

    Expected structure is similar to:

        71 Caudate_L 7001

    where:
        71   = ordinal stored in this atlas
        7001 = AAL region code
    """
    lut_path = Path(lut_path)

    entries: list[dict[str, Any]] = []

    with lut_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = [
                token
                for token in re.split(
                    r"[\s,;\t]+",
                    line,
                )
                if token
            ]

            integer_positions: list[
                tuple[int, int]
            ] = []

            for index, token in enumerate(tokens):
                value = _parse_integer(token)

                if value is not None:
                    integer_positions.append(
                        (index, value)
                    )

            if not integer_positions:
                continue

            if len(integer_positions) >= 2:
                first_position, ordinal = (
                    integer_positions[0]
                )

                last_position, code = (
                    integer_positions[-1]
                )

                name_tokens = tokens[
                    first_position + 1:
                    last_position
                ]

            else:
                position, code = (
                    integer_positions[0]
                )

                ordinal = code

                name_tokens = (
                    tokens[:position]
                    + tokens[position + 1:]
                )

            if not name_tokens:
                continue

            name = "_".join(name_tokens)

            entries.append(
                {
                    "ordinal": int(ordinal),
                    "code": int(code),
                    "name": name,
                }
            )

    if not entries:
        raise RuntimeError(
            f"No AAL labels parsed from {lut_path}"
        )

    return entries


def _integer_unique_values(
    array: np.ndarray,
) -> set[int]:
    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return set()

    values = np.unique(
        np.rint(finite).astype(np.int64)
    )

    return {
        int(value)
        for value in values
        if int(value) != 0
    }


def detect_aal_volume_and_scheme(
    atlas_img: nib.Nifti1Image,
    lut_entries: list[dict[str, Any]],
) -> tuple[np.ndarray, str, int | None]:
    """
    Detect the AAL label volume and whether voxel values
    use ordinal IDs or AAL codes.
    """
    atlas_data = np.asanyarray(
        atlas_img.dataobj
    )

    ordinals = {
        int(entry["ordinal"])
        for entry in lut_entries
    }

    codes = {
        int(entry["code"])
        for entry in lut_entries
    }

    if atlas_data.ndim == 3:
        volumes = [
            (None, atlas_data)
        ]

    elif atlas_data.ndim == 4:
        volumes = [
            (
                index,
                atlas_data[..., index],
            )
            for index in range(
                atlas_data.shape[3]
            )
        ]

    else:
        raise ValueError(
            "Expected 3D or 4D AAL atlas, "
            f"got {atlas_data.shape}"
        )

    best_score = -1
    best_volume = None
    best_scheme = None
    best_index = None

    for volume_index, volume in volumes:
        values = _integer_unique_values(
            volume
        )

        ordinal_matches = len(
            values & ordinals
        )

        code_matches = len(
            values & codes
        )

        if ordinal_matches >= code_matches:
            score = ordinal_matches
            scheme = "ordinal"
        else:
            score = code_matches
            scheme = "code"

        if score > best_score:
            best_score = score
            best_volume = volume
            best_scheme = scheme
            best_index = volume_index

    if (
        best_volume is None
        or best_scheme is None
        or best_score <= 0
    ):
        raise RuntimeError(
            "Could not identify AAL label volume."
        )

    return (
        np.rint(best_volume).astype(np.int32),
        best_scheme,
        best_index,
    )


def _normalize_name(
    name: str,
) -> str:
    return (
        name
        .strip()
        .lower()
        .replace(" ", "_")
    )


def select_regions(
    lut_entries: list[dict[str, Any]],
    requested_names: set[str],
) -> list[dict[str, Any]]:
    """
    Select exact region names.

    Exact matching is deliberate so that similarly named
    structures cannot accidentally enter the mask.
    """
    selected = []

    for entry in lut_entries:
        normalized = _normalize_name(
            entry["name"]
        )

        if normalized in requested_names:
            selected.append(entry)

    return selected


def region_values(
    regions: list[dict[str, Any]],
    id_scheme: str,
) -> list[int]:
    if id_scheme == "ordinal":
        return [
            int(region["ordinal"])
            for region in regions
        ]

    if id_scheme == "code":
        return [
            int(region["code"])
            for region in regions
        ]

    raise ValueError(
        f"Unknown ID scheme: {id_scheme}"
    )


def make_binary_mask(
    atlas_volume: np.ndarray,
    values: list[int],
) -> np.ndarray:
    return np.isin(
        atlas_volume,
        values,
    ).astype(np.uint8)


def resample_mask_to_template(
    mask: np.ndarray,
    atlas_affine: np.ndarray,
    template_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """
    Resample a label mask to FP-CIT template space.

    Nearest-neighbor interpolation is used because this
    is a categorical/binary image.
    """
    source = nib.Nifti1Image(
        mask.astype(np.uint8),
        atlas_affine,
    )

    resampled = resample_from_to(
        source,
        (
            template_img.shape,
            template_img.affine,
        ),
        order=0,
        mode="constant",
        cval=0,
    )

    data = (
        resampled.get_fdata() > 0.5
    ).astype(np.uint8)

    header = template_img.header.copy()
    header.set_data_dtype(np.uint8)

    return nib.Nifti1Image(
        data,
        template_img.affine,
        header,
    )


def build_striatal_masks(
    atlas_path: str | Path,
    lut_path: str | Path,
    template_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:

    atlas_path = Path(atlas_path)
    lut_path = Path(lut_path)
    template_path = Path(template_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    atlas_img = nib.load(
        atlas_path
    )

    template_img = nib.load(
        template_path
    )

    if len(template_img.shape) != 3:
        raise ValueError(
            "FP-CIT template must be 3D."
        )

    lut_entries = parse_aal_lut(
        lut_path
    )

    (
        atlas_volume,
        id_scheme,
        atlas_volume_index,
    ) = detect_aal_volume_and_scheme(
        atlas_img,
        lut_entries,
    )

    left_regions = select_regions(
        lut_entries,
        LEFT_REGION_NAMES,
    )

    right_regions = select_regions(
        lut_entries,
        RIGHT_REGION_NAMES,
    )

    if len(left_regions) != 2:
        raise RuntimeError(
            "Expected exactly two left striatal regions "
            f"(Caudate_L, Putamen_L), found: "
            f"{[x['name'] for x in left_regions]}"
        )

    if len(right_regions) != 2:
        raise RuntimeError(
            "Expected exactly two right striatal regions "
            f"(Caudate_R, Putamen_R), found: "
            f"{[x['name'] for x in right_regions]}"
        )

    left_values = region_values(
        left_regions,
        id_scheme,
    )

    right_values = region_values(
        right_regions,
        id_scheme,
    )

    left_native = make_binary_mask(
        atlas_volume,
        left_values,
    )

    right_native = make_binary_mask(
        atlas_volume,
        right_values,
    )

    bilateral_native = np.logical_or(
        left_native > 0,
        right_native > 0,
    ).astype(np.uint8)

    if left_native.sum() == 0:
        raise RuntimeError(
            "Left striatal mask is empty."
        )

    if right_native.sum() == 0:
        raise RuntimeError(
            "Right striatal mask is empty."
        )

    left_template = resample_mask_to_template(
        left_native,
        atlas_img.affine,
        template_img,
    )

    right_template = resample_mask_to_template(
        right_native,
        atlas_img.affine,
        template_img,
    )

    bilateral_template = resample_mask_to_template(
        bilateral_native,
        atlas_img.affine,
        template_img,
    )

    left_data = (
        left_template.get_fdata() > 0
    )

    right_data = (
        right_template.get_fdata() > 0
    )

    bilateral_data = (
        bilateral_template.get_fdata() > 0
    )

    overlap = np.logical_and(
        left_data,
        right_data,
    )

    overlap_count = int(
        overlap.sum()
    )

    if overlap_count > 0:
        raise RuntimeError(
            "Left and right striatal masks overlap "
            f"after resampling: {overlap_count} voxels."
        )

    left_path = (
        output_dir
        / "left_striatum_mask.nii.gz"
    )

    right_path = (
        output_dir
        / "right_striatum_mask.nii.gz"
    )

    bilateral_path = (
        output_dir
        / "striatum_mask.nii.gz"
    )

    nib.save(
        left_template,
        left_path,
    )

    nib.save(
        right_template,
        right_path,
    )

    nib.save(
        bilateral_template,
        bilateral_path,
    )

    manifest = {
        "analysis_type": (
            "striatal_template_mask_build"
        ),

        "atlas": str(atlas_path),
        "atlas_shape": [
            int(x)
            for x in atlas_img.shape
        ],

        "selected_atlas_volume": (
            atlas_volume_index
        ),

        "atlas_id_scheme": (
            id_scheme
        ),

        "template": str(template_path),

        "template_shape": [
            int(x)
            for x in template_img.shape
        ],

        "template_orientation": [
            str(x)
            for x in nib.aff2axcodes(
                template_img.affine
            )
        ],

        "definition": {
            "left": [
                region["name"]
                for region in left_regions
            ],

            "right": [
                region["name"]
                for region in right_regions
            ],

            "pallidum_included": False,
        },

        "left": {
            "regions": left_regions,
            "atlas_values": left_values,
            "native_voxels": int(
                left_native.sum()
            ),
            "template_voxels": int(
                left_data.sum()
            ),
        },

        "right": {
            "regions": right_regions,
            "atlas_values": right_values,
            "native_voxels": int(
                right_native.sum()
            ),
            "template_voxels": int(
                right_data.sum()
            ),
        },

        "bilateral": {
            "native_voxels": int(
                bilateral_native.sum()
            ),
            "template_voxels": int(
                bilateral_data.sum()
            ),
        },

        "left_right_overlap_voxels": (
            overlap_count
        ),

        "outputs": {
            "left": str(left_path),
            "right": str(right_path),
            "bilateral": str(
                bilateral_path
            ),
        },
    }

    with (
        output_dir
        / "striatal_mask_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build left, right and bilateral striatal "
            "masks from the AAL atlas in FP-CIT "
            "template space."
        )
    )

    parser.add_argument(
        "--atlas",
        required=True,
    )

    parser.add_argument(
        "--lut",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("BUILDING STRIATAL MASKS")
    print("=" * 72)

    summary = build_striatal_masks(
        atlas_path=args.atlas,
        lut_path=args.lut,
        template_path=args.template,
        output_dir=args.output_dir,
    )

    print()
    print("Atlas:")
    print(
        f"  Selected volume: "
        f"{summary['selected_atlas_volume']}"
    )

    print(
        f"  ID scheme: "
        f"{summary['atlas_id_scheme']}"
    )

    print()
    print("Left striatum:")

    for region in summary[
        "definition"
    ]["left"]:
        print(
            f"  {region}"
        )

    print(
        f"  Voxels in template: "
        f"{summary['left']['template_voxels']}"
    )

    print()
    print("Right striatum:")

    for region in summary[
        "definition"
    ]["right"]:
        print(
            f"  {region}"
        )

    print(
        f"  Voxels in template: "
        f"{summary['right']['template_voxels']}"
    )

    print()
    print(
        "Bilateral striatal voxels: "
        f"{summary['bilateral']['template_voxels']}"
    )

    print(
        "Left/right overlap: "
        f"{summary['left_right_overlap_voxels']}"
    )

    print()
    print("=" * 72)
    print("STRIATAL MASK BUILD COMPLETED")
    print("=" * 72)

    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
