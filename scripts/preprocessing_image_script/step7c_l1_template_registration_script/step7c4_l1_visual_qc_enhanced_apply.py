import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

input_csv = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step7c_l1_template_registration_data/"
    / "step7c3_l1_visual_qc_enhanced"
    / "l1_visual_qc_enhanced_manifest.csv"
)

output_dir = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step7c_l1_template_registration_data/"
    / "step7c4_l1_visual_qc_enhanced_apply"
)

output_csv = (
    output_dir
    / "l1_visual_qc_enhanced_apply_manifest.csv"
)

df = pd.read_csv(input_csv)

# Default: all visually acceptable.
df["manual_qc"] = "PASS"
df["manual_notes"] = (
    "L1 center and mapped bilateral target visually acceptable."
)

reviews = {
    "313fpoyi": (
        "Localization plausible, but unusual scan appearance/FOV "
        "and strong edge activity reduce visual confidence."
    ),
    "c11ertj5": (
        "L1 center visually acceptable; mapped ROI extent is very "
        "close to the inferior FOV boundary."
    ),
    "sexggoc5": (
        "L1 center visually acceptable; mapped ROI extent is very "
        "close to the inferior FOV boundary."
    ),
    "qmvqwpqo": (
        "L1 center visually acceptable; mapped ROI extent is very "
        "close to the inferior FOV boundary."
    ),
    "aragdx5o": (
        "L1 center visually acceptable; mapped ROI extent is very "
        "close to the inferior FOV boundary."
    ),
}

for uid, note in reviews.items():
    df.loc[df["uid"] == uid, "manual_qc"] = "REVIEW"
    df.loc[df["uid"] == uid, "manual_notes"] = note

# Special similarity-rescue registrations: center PASS,
# mapped-mask extent invalid for anatomical analysis.
rescue_manifest = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/"
    / "step6c_finalize_registration_data/"
    / "step6c_finalize_all_registration_transforms/"
    / "final_registration_manifest.csv"
)

if rescue_manifest.exists():
    reg = pd.read_csv(rescue_manifest)
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

    if uid_col is not None and source_col is not None:
        rescue_uids = set(
            reg.loc[
                reg[source_col]
                .astype(str)
                .str.strip()
                .str.lower()
                == "similarity_rescue",
                uid_col,
            ].astype(str)
        )

        for uid in rescue_uids:
            df.loc[df["uid"].astype(str) == uid, "manual_qc"] = "PASS"
            df.loc[df["uid"].astype(str) == uid, "manual_notes"] = (
                "L1 localization center PASS. Mapped-mask extent is "
                "invalid for anatomical extent analysis because the "
                "registration is a similarity rescue."
            )

output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)

print(df["manual_qc"].value_counts())
    print()
print("Read: ", input_csv)
print("Saved:", output_csv)
