from pathlib import Path
import sys

import pandas as pd

# ============================================================
# STEP 14C — COMPETITION-ALIGNED RANKING OF ALL 60 EXPERIMENTS
# ============================================================
# PRIMARY   : Global OOF Log Loss (LOWER is better)
# SECONDARY : Global OOF AUROC (HIGHER is better)
# Supporting: Brier score, AUPRC, ECE


def model_number(value: str) -> int:
    return int(value.replace("E", ""))


def scenario_number(value: str) -> int:
    return int(value.replace("S", ""))


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_file = (
        project_root / "data" / "experiment_selection_data" /
        "aggregate_experiment_results_data" / "all_60_experiment_results.csv"
    )
    output_dir = (
        project_root / "data" / "experiment_selection_data" / "rank_experiment_results_data"
    )
    output_file = output_dir / "ranked_all_60_experiment_results.csv"

    print("\n" + "=" * 76)
    print("STEP 14C — RANK 60 EXPERIMENTS FOR THE COMPETITION")
    print("=" * 76)
    print("Primary metric   : Global OOF Log Loss (lower is better)")
    print("Secondary metric : Global OOF AUROC (higher is better)")
    print(f"Input            : {input_file}")
    print(f"Output           : {output_file}\n")

    if not input_file.is_file():
        print(f"ERROR: input not found: {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)
    if len(df) != 60:
        raise RuntimeError(f"Expected 60 experiments, found {len(df)}")

    required = [
        "Model", "Scenario ID", "Global OOF Log Loss", "Global OOF AUROC",
        "Global OOF AUPRC", "Global OOF Brier Score", "Global OOF ECE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Primary competition ranks.
    df["Overall Log Loss Rank"] = (
        df["Global OOF Log Loss"].rank(method="min", ascending=True).astype(int)
    )
    df["Within-Model Log Loss Rank"] = (
        df.groupby("Model")["Global OOF Log Loss"]
        .rank(method="min", ascending=True).astype(int)
    )

    # Secondary discrimination ranks retained for interpretation.
    df["Overall AUROC Rank"] = (
        df["Global OOF AUROC"].rank(method="min", ascending=False).astype(int)
    )
    df["Within-Model AUROC Rank"] = (
        df.groupby("Model")["Global OOF AUROC"]
        .rank(method="min", ascending=False).astype(int)
    )

    if not (df.groupby("Model").size() == 6).all():
        raise RuntimeError("Every model must contain six scenarios")

    preferred = [
        "Model", "Model Used", "Type", "Scratch / pretrained", "File Name",
        "Scenario ID", "Scenario",
        "Overall Log Loss Rank", "Within-Model Log Loss Rank",
        "Overall AUROC Rank", "Within-Model AUROC Rank",
        "Global OOF Log Loss", "Global OOF AUROC", "Global OOF AUPRC",
        "Global OOF Brier Score", "Global OOF ECE",
        "Global OOF Balanced Accuracy @ 0.5", "OOF Subjects",
    ]
    remaining = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + remaining]

    # Exact competition order: Log Loss primary, AUROC secondary tie-breaker,
    # then model/scenario for deterministic display.
    df["_model_order"] = df["Model"].map(model_number)
    df["_scenario_order"] = df["Scenario ID"].map(scenario_number)
    df = df.sort_values(
        ["Global OOF Log Loss", "Global OOF AUROC", "_model_order", "_scenario_order"],
        ascending=[True, False, True, True],
    ).drop(columns=["_model_order", "_scenario_order"])

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format="%.9f")

    print("TOP 10 BY COMPETITION PRIMARY METRIC\n")
    print(df[[
        "Overall Log Loss Rank", "Model", "Scenario ID", "Scenario",
        "Global OOF Log Loss", "Global OOF AUROC", "Global OOF Brier Score",
    ]].head(10).to_string(index=False))

    print("\nBEST UNCALIBRATED SCENARIO WITHIN EACH MODEL\n")
    best = df[df["Within-Model Log Loss Rank"] == 1].copy()
    best["_model_order"] = best["Model"].map(model_number)
    best = best.sort_values("_model_order")
    print(best[[
        "Model", "Scenario ID", "Scenario", "Global OOF Log Loss",
        "Global OOF AUROC", "Global OOF Brier Score",
    ]].to_string(index=False))

    print("\nSTATUS: PASS")
    print("IMPORTANT: This is the UNCALIBRATED ranking. Final competition selection occurs after cross-fitted calibration.\n")


if __name__ == "__main__":
    main()
