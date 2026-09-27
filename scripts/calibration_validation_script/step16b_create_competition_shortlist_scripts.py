#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/cluster/home/miladna/PythonProject/dat-scan-classifier"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "calibration_validation_data"
    / "cross_fitted_calibration_data"
    / "best_calibration_method_per_experiment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "calibration_validation_data"
    / "competition_shortlist_data"
)

OUTPUT_FILE = OUTPUT_DIR / "competition_shortlist_8.csv"

N_SHORTLIST = 8


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Calibration Candidate ID",
    "Model",
    "Scenario ID",
    "Calibration Method",
    "Cross-Fitted OOF Log Loss",
    "Cross-Fitted OOF AUROC",
    "Cross-Fitted OOF Brier Score",
    "Cross-Fitted OOF ECE",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 88)
    print("CREATE POST-CALIBRATION COMPETITION SHORTLIST")
    print("=" * 88)

    print(f"Input       : {INPUT_FILE}")
    print(f"Output      : {OUTPUT_FILE}")
    print(f"Shortlist   : Top {N_SHORTLIST}")
    print("Primary     : Cross-Fitted OOF Log Loss (lower is better)")
    print("Tie-breaker : Cross-Fitted OOF AUROC (higher is better)")
    print()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if len(df) != 60:
        raise ValueError(
            f"Expected exactly 60 best-per-experiment rows, "
            f"but found {len(df)}."
        )

    missing_columns = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Validate uniqueness
    # --------------------------------------------------------

    if df["Calibration Candidate ID"].duplicated().any():
        duplicated = df.loc[
            df["Calibration Candidate ID"].duplicated(keep=False),
            "Calibration Candidate ID"
        ].tolist()

        raise ValueError(
            f"Duplicate calibration candidates found: {duplicated}"
        )

    # --------------------------------------------------------
    # Validate metrics
    # --------------------------------------------------------

    numeric_columns = [
        "Cross-Fitted OOF Log Loss",
        "Cross-Fitted OOF AUROC",
        "Cross-Fitted OOF Brier Score",
        "Cross-Fitted OOF ECE",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isna().any():
            raise ValueError(
                f"NaN/non-numeric values detected in '{column}'."
            )

    # --------------------------------------------------------
    # Competition ranking
    #
    # PRIMARY:
    #   Log Loss ascending
    #
    # TIE BREAKERS:
    #   AUROC descending
    #   Brier ascending
    #   ECE ascending
    # --------------------------------------------------------

    ranked = df.sort_values(
        by=[
            "Cross-Fitted OOF Log Loss",
            "Cross-Fitted OOF AUROC",
            "Cross-Fitted OOF Brier Score",
            "Cross-Fitted OOF ECE",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["Post-Calibration Log Loss Rank"] = (
        range(1, len(ranked) + 1)
    )

    # --------------------------------------------------------
    # Select top 8
    # --------------------------------------------------------

    shortlist = ranked.head(N_SHORTLIST).copy()

    shortlist.insert(
        0,
        "Shortlist ID",
        [f"P{i}" for i in range(1, N_SHORTLIST + 1)]
    )

    shortlist["Advance to Competition Validation"] = "YES"

    shortlist["Selection Stage"] = (
        "Post-calibration competition shortlist"
    )

    shortlist["Selection Reason"] = (
        "Top-8 by cross-fitted OOF Log Loss after selecting "
        "the best calibration method for each experiment"
    )

    shortlist["Primary Competition Metric"] = (
        "Cross-Fitted OOF Log Loss"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shortlist.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display useful columns
    # --------------------------------------------------------

    display_columns = [
        "Shortlist ID",
        "Post-Calibration Log Loss Rank",
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Cross-Fitted OOF AUROC",
        "Cross-Fitted OOF Brier Score",
        "Cross-Fitted OOF ECE",
    ]

    print("# POST-CALIBRATION COMPETITION SHORTLIST\n")

    print(
        shortlist[display_columns].to_string(
            index=False
        )
    )

    print()
    print(f"Saved : {OUTPUT_FILE}")
    print("Status: PASS")
    print()
    print(
        "Next stage: Log-Loss stability analysis and paired "
        "statistical comparison of P1-P8."
    )


if __name__ == "__main__":
    main()