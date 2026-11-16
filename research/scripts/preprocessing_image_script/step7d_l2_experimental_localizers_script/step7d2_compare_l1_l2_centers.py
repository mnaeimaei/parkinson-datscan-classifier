import pandas as pd
import numpy as np
from pathlib import Path

REGISTRATION_MANIFEST = Path(
    "data/preprocessing_image_data/"
    "step6c_finalize_registration_data/"
    "step6c_finalize_all_registration_transforms/"
    "final_registration_manifest.csv"
)


def similarity_rescue_uids() -> set[str]:
    if not REGISTRATION_MANIFEST.exists():
        return set()

    reg = pd.read_csv(REGISTRATION_MANIFEST)

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


l1_path = Path(
    "data/preprocessing_image_data/"
    "step7c_l1_template_registration_data/"
    "step7c5_l1_center_consistency/"
    "l1_center_consistency.csv"
)

l2_path = Path(
    "data/preprocessing_image_data/"
    "step7d_l2_experimental_localizers_data/"
    "step7d1_roi_l2_intensity_localizer/"
    "l2_localization.csv"
)

output = Path(
    "data/preprocessing_image_data/"
    "step7d_l2_experimental_localizers_data/"
    "step7d2_compare_l1_l2_centers/"
    "l1_l2_center_comparison.csv"
)

l1 = pd.read_csv(l1_path)
l2 = pd.read_csv(l2_path)

# Use the analytically transformed template center,
# which was already validated in the L1 center consistency step.
l1 = l1[
    [
        "uid",
        "direct_center_x",
        "direct_center_y",
        "direct_center_z",
    ]
].copy()

m = l1.merge(
    l2,
    on="uid",
    how="inner",
    validate="one_to_one",
)

    if len(m) != len(l1) or len(m) != len(l2):
        raise RuntimeError(
            "L1 and L2 UID sets do not match. "
            f"L1={len(l1)}, L2={len(l2)}, merged={len(m)}"
        )

# ---------------------------------------------------------
# L1 versus raw L2
# ---------------------------------------------------------

m["delta_x_vox"] = (
    m["l2_center_x"]
    - m["direct_center_x"]
)

m["delta_y_vox"] = (
    m["l2_center_y"]
    - m["direct_center_y"]
)

m["delta_z_vox"] = (
    m["l2_center_z"]
    - m["direct_center_z"]
)

m["l1_l2_distance_vox"] = np.sqrt(
    m["delta_x_vox"] ** 2
    + m["delta_y_vox"] ** 2
    + m["delta_z_vox"] ** 2
)

# All normalized scans are 2.46-mm isotropic.
m["l1_l2_distance_mm"] = (
    m["l1_l2_distance_vox"] * 2.46
)

x = m["l1_l2_distance_mm"]

print()
print("=" * 72)
print("RAW L2-v1 vs VALIDATED L1 CENTER")
print("=" * 72)

print(f"Cases:      {len(m)}")
print()
print(f"min:        {x.min():8.3f} mm")
print(f"p05:        {x.quantile(0.05):8.3f} mm")
print(f"p25:        {x.quantile(0.25):8.3f} mm")
print(f"median:     {x.median():8.3f} mm")
print(f"mean:       {x.mean():8.3f} mm")
print(f"p75:        {x.quantile(0.75):8.3f} mm")
print(f"p95:        {x.quantile(0.95):8.3f} mm")
print(f"p99:        {x.quantile(0.99):8.3f} mm")
print(f"max:        {x.max():8.3f} mm")

print()
for threshold in [5, 10, 15, 20, 30, 40, 50]:
    n = int((x > threshold).sum())
    pct = 100 * n / len(m)

    print(
        f"> {threshold:2d} mm: "
        f"{n:4d}  ({pct:6.2f}%)"
    )

print()
print("Similarity-rescue cases:")
rescue = m.loc[
    m["uid"].astype(str).isin(similarity_rescue_uids()),
    [
        "uid",
        "l1_l2_distance_mm",
        "l2_confidence",
        "candidate_voxels",
        "bilateral_balance",
    ],
]
if rescue.empty:
    print("  none")
else:
    print(rescue.to_string(index=False))

print()
print("20 LARGEST L1-L2 DISAGREEMENTS")
print("-" * 72)

cols = [
    "uid",
    "l1_l2_distance_mm",
    "delta_x_vox",
    "delta_y_vox",
    "delta_z_vox",
    "l2_confidence",
    "candidate_voxels",
    "bilateral_balance",
]

print(
    m.sort_values(
        "l1_l2_distance_mm",
        ascending=False,
    )[cols]
    .head(20)
    .to_string(index=False)
)

# Create the output directory if it does not already exist.
output.parent.mkdir(parents=True, exist_ok=True)

m.to_csv(
    output,
    index=False,
)

print()
print("Saved:")
print(output)

print("=" * 72)
