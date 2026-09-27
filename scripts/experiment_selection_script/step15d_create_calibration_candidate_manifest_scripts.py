from pathlib import Path
import sys

import pandas as pd

# ==================================================================
# STEP 14D — CREATE CALIBRATION-CANDIDATE MANIFEST (ALL 60 EXPERIMENTS)
# ==================================================================
#
# Deliberately does NOT create a fixed F1-F7 shortlist.
# All 60 already-trained predictors advance to cross-fitted calibration
# because calibration is cheap and Log Loss is the competition primary
# metric. The final competition shortlist is created AFTER calibration.


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_file = (
        project_root / "data" / "experiment_selection_data" /
        "rank_experiment_results_data" / "ranked_all_60_experiment_results.csv"
    )
    output_dir = (
        project_root / "data" / "experiment_selection_data" /
        "calibration_candidate_manifest_data"
    )
    output_file = output_dir / "all_60_calibration_candidates.csv"
    preview_file = output_dir / "top_10_uncalibrated_log_loss_preview.csv"

    print("\n" + "=" * 78)
    print("STEP 14D — CREATE ALL-60 CALIBRATION CANDIDATE MANIFEST")
    print("=" * 78)
    print(f"Input  : {input_file}")
    print(f"Output : {output_file}\n")

    if not input_file.is_file():
        print(f"ERROR: ranked results not found: {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)
    if len(df) != 60:
        raise RuntimeError(f"Expected 60 experiments, found {len(df)}")

    required = [
        "Model", "Scenario ID", "Overall Log Loss Rank", "Global OOF Log Loss",
        "Global OOF AUROC", "OOF Subjects", "OOF Prediction File",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if df.duplicated(["Model", "Scenario ID"]).any():
        raise RuntimeError("Duplicate model/scenario candidates detected")
    if not (df["OOF Subjects"] == 1362).all():
        raise RuntimeError("Every calibration candidate must contain 1362 OOF subjects")

    missing_oof = [p for p in df["OOF Prediction File"].astype(str) if not Path(p).is_file()]
    if missing_oof:
        raise FileNotFoundError(
            "One or more OOF prediction files do not exist:\n" +
            "\n".join(f"  - {p}" for p in missing_oof[:20])
        )

    out = df.copy()
    out.insert(0, "Calibration Candidate ID", [f"C{i:02d}" for i in range(1, 61)])
    out["Advance to Cross-Fitted Calibration"] = "YES"
    out["Selection Stage"] = "Pre-calibration competition ranking"
    out["Selection Policy"] = (
        "All 60 advance; final selection after cross-fitted calibration by Log Loss"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False, float_format="%.9f")
    out.head(10).to_csv(preview_file, index=False, float_format="%.9f")

    print("Candidates                          : 60 / 60")
    print("OOF subjects per candidate          : 1362")
    print("Advance to cross-fitted calibration : 60 / 60")
    print(f"Saved manifest                      : {output_file}")
    print(f"Top-10 reporting preview            : {preview_file}")
    print("\nSTATUS: PASS")
    print("No candidate is excluded before calibration.")
    print("The definitive competition shortlist must be created from cross-fitted calibrated Log Loss.\n")


if __name__ == "__main__":
    main()
