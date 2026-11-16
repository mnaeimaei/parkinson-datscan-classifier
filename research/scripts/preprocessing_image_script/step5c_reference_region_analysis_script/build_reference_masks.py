from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ---------------------------------------------------------
# Region definitions
# ---------------------------------------------------------

# Exact prefixes are used intentionally.
# This prevents "Precuneus" from being accidentally selected
# by the "Cuneus" rule.
OCCIPITAL_PREFIXES = (
    "calcarine_",
    "cuneus_",
    "lingual_",
    "occipital_sup_",
    "occipital_mid_",
    "occipital_inf_",
)

CEREBELLAR_PREFIXES = (
    "cerebelum_",
    "cerebellum_",
    "vermis_",
)


# ---------------------------------------------------------
# LUT parsing
# ---------------------------------------------------------

def parse_integer(token: str) -> int | None:
    """
    Return integer if token is an integer, otherwise None.
    """
    token = token.strip()

    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)

    return None


def parse_lut(path: Path) -> list[dict]:
    """
    Parse AAL lookup table.

    The downloaded MNI_AAL.txt commonly contains information
    equivalent to:

        43  Calcarine_L  5001

    where:
        43   = ROI ordinal/index
        5001 = AAL label code

    We preserve both because we will inspect the NIfTI itself
    and determine which convention it actually uses.
    """
    entries: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # Split on whitespace, comma, semicolon, or tab.
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
                value = parse_integer(token)

                if value is not None:
                    integer_positions.append(
                        (index, value)
                    )

            if not integer_positions:
                continue

            # Typical format:
            #
            # ordinal  name  code
            #
            # Example:
            # 43 Calcarine_L 5001
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
                # Fallback for a LUT containing only one
                # numerical identifier.
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
            f"No valid atlas labels found in: {path}"
        )

    return entries


# ---------------------------------------------------------
# Atlas volume detection
# ---------------------------------------------------------

def integer_unique_values(
    volume: np.ndarray,
) -> set[int]:
    """
    Return finite rounded integer values from a label volume.
    """
    finite = volume[
        np.isfinite(volume)
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


def detect_aal_volume_and_id_scheme(
    atlas_img: nib.Nifti1Image,
    lut_entries: list[dict],
) -> tuple[
    np.ndarray,
    str,
    int | None,
]:
    """
    Find which atlas volume contains the AAL parcellation.

    MNI_AAL_AndMore may be 4D because it contains additional
    volumes. We compare every volume with the LUT and choose
    the volume with the strongest overlap.

    We also determine whether voxel labels correspond to:
        - ROI ordinals (1..116-like)
        - AAL codes (e.g. 5001, 5002)
    """
    data = np.asanyarray(
        atlas_img.dataobj
    )

    ordinal_values = {
        entry["ordinal"]
        for entry in lut_entries
    }

    code_values = {
        entry["code"]
        for entry in lut_entries
    }

    print()
    print("Atlas information:")
    print(f"  Shape: {data.shape}")
    print(f"  Dimensions: {data.ndim}")

    # -----------------------------------------------------
    # 3D atlas
    # -----------------------------------------------------

    if data.ndim == 3:

        unique_values = integer_unique_values(
            data
        )

        ordinal_overlap = len(
            unique_values & ordinal_values
        )

        code_overlap = len(
            unique_values & code_values
        )

        if ordinal_overlap == 0 and code_overlap == 0:
            raise RuntimeError(
                "The 3D atlas does not contain values "
                "matching either LUT ordinals or LUT codes."
            )

        if code_overlap > ordinal_overlap:
            scheme = "code"
            matches = code_overlap
        else:
            scheme = "ordinal"
            matches = ordinal_overlap

        print("  Atlas is already 3D.")
        print(
            f"  Detected label convention: {scheme}"
        )
        print(
            f"  Matching LUT labels: {matches}"
        )

        return (
            np.rint(data).astype(np.int32),
            scheme,
            None,
        )

    # -----------------------------------------------------
    # 4D atlas
    # -----------------------------------------------------

    if data.ndim != 4:
        raise RuntimeError(
            "Expected a 3D or 4D atlas, "
            f"but got shape {data.shape}."
        )

    best_score = -1
    best_volume = None
    best_scheme = None

    print()
    print("Inspecting atlas volumes:")

    for volume_index in range(
        data.shape[3]
    ):

        volume = data[
            ...,
            volume_index
        ]

        unique_values = (
            integer_unique_values(volume)
        )

        ordinal_overlap = len(
            unique_values & ordinal_values
        )

        code_overlap = len(
            unique_values & code_values
        )

        print(
            f"  Volume {volume_index}: "
            f"ordinal matches={ordinal_overlap}, "
            f"code matches={code_overlap}, "
            f"unique nonzero values="
            f"{len(unique_values)}"
        )

        if code_overlap > ordinal_overlap:
            score = code_overlap
            scheme = "code"
        else:
            score = ordinal_overlap
            scheme = "ordinal"

        if score > best_score:
            best_score = score
            best_volume = volume_index
            best_scheme = scheme

    if (
        best_volume is None
        or best_scheme is None
        or best_score <= 0
    ):
        raise RuntimeError(
            "Could not identify the AAL label volume "
            "from the 4D atlas."
        )

    print()
    print(
        f"Selected atlas volume: {best_volume}"
    )
    print(
        f"Detected label convention: "
        f"{best_scheme}"
    )
    print(
        f"Number of matching LUT labels: "
        f"{best_score}"
    )

    selected_volume = data[
        ...,
        best_volume
    ]

    return (
        np.rint(
            selected_volume
        ).astype(np.int32),
        best_scheme,
        int(best_volume),
    )


# ---------------------------------------------------------
# Region selection
# ---------------------------------------------------------

def normalize_region_name(
    name: str,
) -> str:

    return (
        name
        .strip()
        .lower()
        .replace(" ", "_")
    )


def select_regions(
    lut_entries: list[dict],
    prefixes: tuple[str, ...],
) -> list[dict]:
    """
    Select anatomical regions by exact beginning of region name.

    Using startswith avoids:
        "cuneus" matching "precuneus".
    """
    selected = []

    for entry in lut_entries:

        normalized_name = (
            normalize_region_name(
                entry["name"]
            )
        )

        if any(
            normalized_name.startswith(prefix)
            for prefix in prefixes
        ):
            selected.append(entry)

    return selected


def get_region_values(
    regions: list[dict],
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
        f"Unknown atlas ID scheme: {id_scheme}"
    )


# ---------------------------------------------------------
# Mask creation
# ---------------------------------------------------------

def build_binary_mask(
    atlas_volume: np.ndarray,
    region_values: list[int],
) -> np.ndarray:

    mask = np.isin(
        atlas_volume,
        region_values,
    )

    return mask.astype(
        np.uint8
    )


def resample_mask_to_template(
    mask: np.ndarray,
    atlas_affine: np.ndarray,
    template_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """
    Resample a 3D binary atlas mask into the FP-CIT
    template grid.

    Nearest-neighbour interpolation is mandatory for
    label/binary masks.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Mask must be 3D, got {mask.shape}"
        )

    if len(template_img.shape) != 3:
        raise ValueError(
            "FP-CIT template must be 3D, "
            f"got {template_img.shape}"
        )

    source_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        atlas_affine,
    )

    resampled = resample_from_to(
        source_img,
        (
            template_img.shape,
            template_img.affine,
        ),
        order=0,
        mode="constant",
        cval=0,
    )

    data = (
        resampled.get_fdata()
        > 0.5
    ).astype(np.uint8)

    output_header = (
        template_img.header.copy()
    )

    output_header.set_data_dtype(
        np.uint8
    )

    output = nib.Nifti1Image(
        data,
        template_img.affine,
        output_header,
    )

    return output


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build candidate occipital and cerebellar "
            "reference masks in FP-CIT template space."
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

    atlas_path = Path(
        args.atlas
    )

    lut_path = Path(
        args.lut
    )

    template_path = Path(
        args.template
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    atlas_img = nib.load(
        atlas_path
    )

    template_img = nib.load(
        template_path
    )

    print("=" * 70)
    print("BUILDING ANATOMICAL REFERENCE MASKS")
    print("=" * 70)

    print(
        f"AAL atlas: {atlas_path}"
    )

    print(
        f"FP-CIT template: {template_path}"
    )

    print(
        f"Template shape: "
        f"{template_img.shape}"
    )

    print(
        f"Template orientation: "
        f"{nib.aff2axcodes(template_img.affine)}"
    )

    # -----------------------------------------------------
    # Parse LUT
    # -----------------------------------------------------

    lut_entries = parse_lut(
        lut_path
    )

    print(
        f"LUT regions parsed: "
        f"{len(lut_entries)}"
    )

    # -----------------------------------------------------
    # Select correct 3D AAL volume
    # -----------------------------------------------------

    (
        atlas_volume,
        id_scheme,
        selected_volume_index,
    ) = detect_aal_volume_and_id_scheme(
        atlas_img=atlas_img,
        lut_entries=lut_entries,
    )

    # -----------------------------------------------------
    # Select anatomical regions
    # -----------------------------------------------------

    occipital_regions = select_regions(
        lut_entries,
        OCCIPITAL_PREFIXES,
    )

    cerebellar_regions = select_regions(
        lut_entries,
        CEREBELLAR_PREFIXES,
    )

    if not occipital_regions:
        raise RuntimeError(
            "No occipital regions were selected."
        )

    if not cerebellar_regions:
        raise RuntimeError(
            "No cerebellar regions were selected."
        )

    print()
    print("Occipital regions:")

    for region in occipital_regions:
        print(
            f"  ordinal={region['ordinal']:3d} | "
            f"code={region['code']:4d} | "
            f"{region['name']}"
        )

    print()
    print("Cerebellar regions:")

    for region in cerebellar_regions:
        print(
            f"  ordinal={region['ordinal']:3d} | "
            f"code={region['code']:4d} | "
            f"{region['name']}"
        )

    # Precuneus should NOT appear.
    if any(
        "precuneus"
        in normalize_region_name(
            region["name"]
        )
        for region in occipital_regions
    ):
        raise RuntimeError(
            "Precuneus was unexpectedly included "
            "in the occipital mask."
        )

    occipital_values = (
        get_region_values(
            occipital_regions,
            id_scheme,
        )
    )

    cerebellar_values = (
        get_region_values(
            cerebellar_regions,
            id_scheme,
        )
    )

    # -----------------------------------------------------
    # Create atlas-space masks
    # -----------------------------------------------------

    occipital_native = (
        build_binary_mask(
            atlas_volume,
            occipital_values,
        )
    )

    cerebellar_native = (
        build_binary_mask(
            atlas_volume,
            cerebellar_values,
        )
    )

    occipital_native_count = int(
        occipital_native.sum()
    )

    cerebellar_native_count = int(
        cerebellar_native.sum()
    )

    print()
    print(
        "Native atlas mask voxel counts:"
    )

    print(
        f"  Occipital: "
        f"{occipital_native_count}"
    )

    print(
        f"  Cerebellar: "
        f"{cerebellar_native_count}"
    )

    if occipital_native_count == 0:
        raise RuntimeError(
            "Occipital mask is empty. "
            "Atlas label convention is incorrect."
        )

    if cerebellar_native_count == 0:
        raise RuntimeError(
            "Cerebellar mask is empty. "
            "Atlas label convention is incorrect."
        )

    # -----------------------------------------------------
    # Resample masks to FP-CIT template
    # -----------------------------------------------------

    occipital_template = (
        resample_mask_to_template(
            mask=occipital_native,
            atlas_affine=atlas_img.affine,
            template_img=template_img,
        )
    )

    cerebellar_template = (
        resample_mask_to_template(
            mask=cerebellar_native,
            atlas_affine=atlas_img.affine,
            template_img=template_img,
        )
    )

    occipital_template_count = int(
        np.count_nonzero(
            occipital_template.get_fdata()
        )
    )

    cerebellar_template_count = int(
        np.count_nonzero(
            cerebellar_template.get_fdata()
        )
    )

    if occipital_template_count == 0:
        raise RuntimeError(
            "Occipital mask became empty after "
            "resampling to FP-CIT template."
        )

    if cerebellar_template_count == 0:
        raise RuntimeError(
            "Cerebellar mask became empty after "
            "resampling to FP-CIT template."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    occipital_path = (
        output_dir
        / "occipital_mask.nii.gz"
    )

    cerebellar_path = (
        output_dir
        / "cerebellar_mask.nii.gz"
    )

    nib.save(
        occipital_template,
        occipital_path,
    )

    nib.save(
        cerebellar_template,
        cerebellar_path,
    )

    manifest = {
        "atlas": str(
            atlas_path
        ),

        "atlas_original_shape": [
            int(value)
            for value in atlas_img.shape
        ],

        "selected_atlas_volume": (
            selected_volume_index
        ),

        "atlas_id_scheme": (
            id_scheme
        ),

        "template": str(
            template_path
        ),

        "template_shape": [
            int(value)
            for value in template_img.shape
        ],

        "occipital": {
            "regions": (
                occipital_regions
            ),

            "atlas_values_used": (
                occipital_values
            ),

            "native_voxel_count": (
                occipital_native_count
            ),

            "template_voxel_count": (
                occipital_template_count
            ),
        },

        "cerebellar": {
            "regions": (
                cerebellar_regions
            ),

            "atlas_values_used": (
                cerebellar_values
            ),

            "native_voxel_count": (
                cerebellar_native_count
            ),

            "template_voxel_count": (
                cerebellar_template_count
            ),
        },
    }

    manifest_path = (
        output_dir
        / "reference_mask_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("REFERENCE MASK BUILD COMPLETED")
    print("=" * 70)

    print(
        f"ID convention used: "
        f"{id_scheme}"
    )

    print(
        f"Occipital mask voxels: "
        f"{occipital_template_count}"
    )

    print(
        f"Cerebellar mask voxels: "
        f"{cerebellar_template_count}"
    )

    print()
    print(
        f"Occipital mask: "
        f"{occipital_path}"
    )

    print(
        f"Cerebellar mask: "
        f"{cerebellar_path}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()