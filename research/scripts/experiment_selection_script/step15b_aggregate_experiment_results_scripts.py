from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

# ============================================================
# STEP 14B — AGGREGATE COMPETITION METRICS FROM 60 OOF FILES
# ============================================================
#
# Competition requirement:
#   PRIMARY   = Log Loss (lower is better)
#   SECONDARY = AUROC (higher is better)
#
# Metrics are independently recomputed from each root-level
# oof_predictions.csv, not parsed from Slurm stdout.

MODEL_INFO = {
    "model01_simple3d_scratch": ("E1", "Simple3D (~3.6M)", "CNN", "Scratch"),
    "model02_resnet18_3d_scratch": ("E2", "MONAI 3D ResNet-18", "CNN", "Scratch"),
    "model03_resnet18_3d_medicalnet_pretrained": (
        "E3", "MedicalNet 3D ResNet-18", "CNN",
        "Pretrained: MedicalNet / Med3D, resnet_18_23dataset.pth",
    ),
    "model04_resnet18_2p5d_attention_imagenet_pretrained": (
        "E4", "2D ResNet-18 + slice attention", "CNN", "Pretrained: ImageNet-1K",
    ),
    "model05_r3d18_scratch": ("E5", "Torchvision R3D-18", "CNN", "Scratch"),
    "model06_r3d18_kinetics400_pretrained": (
        "E6", "Torchvision R3D-18", "CNN",
        "Pretrained: Kinetics-400, R3D_18_Weights.KINETICS400_V1",
    ),
    "model07_resnet18_2p5d_attention_scratch": (
        "E7", "2D ResNet-18 + slice attention", "CNN", "Scratch",
    ),
    "model08_densenet121_3d_scratch": (
        "E8", "MONAI 3D DenseNet-121", "CNN", "Scratch",
    ),
    "model09_swin3d_t_scratch": (
        "E9", "Torchvision Swin3D-Tiny", "Transformer", "Scratch",
    ),
    "model10_swin3d_t_kinetics400_pretrained": (
        "E10", "Torchvision Swin3D-Tiny", "Transformer",
        "Pretrained: Kinetics-400, Swin3D_T_Weights.KINETICS400_V1",
    ),
}

SCENARIO_INFO = {
    "scenario01_whole_noaug": ("S1", "Whole, no aug"),
    "scenario02_roi_noaug": ("S2", "ROI, no aug"),
    "scenario03_whole_roi_noaug": ("S3", "Whole + ROI, no aug"),
    "scenario04_whole_aug": ("S4", "Whole + aug"),
    "scenario05_roi_aug": ("S5", "ROI + aug"),
    "scenario06_whole_roi_aug": ("S6", "Whole + ROI + aug"),
}

EXPECTED_SUBJECTS = 1362
ECE_BINS = 10
METRIC_TOL = 5e-6


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def model_number(model: str) -> int:
    return int(model.replace("E", ""))


def scenario_number(scenario: str) -> int:
    return int(scenario.replace("S", ""))


def load_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"uid", "is_pathologic", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(df) != EXPECTED_SUBJECTS:
        raise ValueError(f"{path}: expected {EXPECTED_SUBJECTS} rows, found {len(df)}")
    if df["uid"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate UIDs detected")
    y = pd.to_numeric(df["is_pathologic"], errors="raise").to_numpy(dtype=np.int64)
    p = pd.to_numeric(df["probability"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path}: labels must be binary")
    if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{path}: probabilities must be finite and within [0,1]")
    return df


def metric_value(summary: dict, key: str):
    try:
        return float(summary["global_oof_metrics"][key])
    except Exception:
        return None


def check_close(name: str, calculated: float, stored: float | None, source: Path):
    if stored is None:
        return
    if not math.isfinite(stored) or abs(calculated - stored) > METRIC_TOL:
        raise ValueError(
            f"Metric integrity failure for {source}\n"
            f"  metric     : {name}\n"
            f"  recalculated: {calculated:.12f}\n"
            f"  cv_summary  : {stored:.12f}"
        )


def find_source_out(experiment_dir: Path) -> str:
    outs = sorted(experiment_dir.glob("*.out"))
    if len(outs) == 1:
        return str(outs[0].resolve())
    if len(outs) == 0:
        return ""
    return ";".join(str(x.resolve()) for x in outs)


def parse_experiment(model_key: str, scenario_key: str, experiment_dir: Path) -> dict:
    oof_path = experiment_dir / "oof_predictions.csv"
    summary_path = experiment_dir / "cv_summary.json"
    definition_path = experiment_dir / "experiment_definition.json"
    fold_metrics_path = experiment_dir / "fold_metrics.csv"

    required_files = [oof_path, summary_path, definition_path, fold_metrics_path]
    missing = [p for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts for {model_key}/{scenario_key}:\n" +
            "\n".join(f"  - {p}" for p in missing)
        )

    df = load_oof(oof_path)
    y = df["is_pathologic"].to_numpy(dtype=np.int64)
    p = df["probability"].to_numpy(dtype=np.float64)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    threshold = float(summary.get("threshold", 0.5))
    pred = (p >= threshold).astype(np.int64)
    precision_curve, recall_curve, _ = precision_recall_curve(y, p)

    metrics = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(auc(recall_curve, precision_curve)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, ECE_BINS),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }

    for key in ("log_loss", "auroc", "auprc", "brier_score", "ece", "balanced_accuracy"):
        check_close(key, metrics[key], metric_value(summary, key), summary_path)

    model, model_used, model_type, pretraining = MODEL_INFO[model_key]
    scenario_id, scenario = SCENARIO_INFO[scenario_key]

    return {
        "Model": model,
        "Model Used": model_used,
        "Type": model_type,
        "Scratch / pretrained": pretraining,
        "File Name": model_key,
        "Scenario ID": scenario_id,
        "Scenario": scenario,
        "Global OOF Log Loss": metrics["log_loss"],
        "Global OOF AUROC": metrics["auroc"],
        "Global OOF AUPRC": metrics["auprc"],
        "Global OOF Average Precision": metrics["average_precision"],
        "Global OOF Brier Score": metrics["brier_score"],
        "Global OOF ECE": metrics["ece"],
        "Global OOF Balanced Accuracy @ 0.5": metrics["balanced_accuracy"],
        "OOF Subjects": int(len(df)),
        "OOF Prediction File": str(oof_path.resolve()),
        "CV Summary File": str(summary_path.resolve()),
        "Experiment Definition File": str(definition_path.resolve()),
        "Fold Metrics File": str(fold_metrics_path.resolve()),
        "Source OUT File": find_source_out(experiment_dir),
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    experiments_root = project_root / "data" / "experiments_data"
    output_dir = (
        project_root / "data" / "experiment_selection_data" / "aggregate_experiment_results_data"
    )
    output_file = output_dir / "all_60_experiment_results.csv"

    print("\n" + "=" * 78)
    print("STEP 14B — AGGREGATE 60 EXPERIMENTS USING OOF COMPETITION METRICS")
    print("=" * 78)
    print(f"Project root : {project_root}")
    print(f"Experiments  : {experiments_root}")
    print(f"Output       : {output_file}\n")

    if not experiments_root.is_dir():
        print(f"ERROR: experiments root not found: {experiments_root}")
        sys.exit(1)

    rows = []
    for model_key, info in MODEL_INFO.items():
        model_dir = experiments_root / f"{model_key}_exp_data"
        for scenario_key in SCENARIO_INFO:
            scenario_dir = model_dir / f"{scenario_key}_exp_data"
            try:
                row = parse_experiment(model_key, scenario_key, scenario_dir)
            except Exception as exc:
                print(f"\nERROR while processing:\n{scenario_dir}\n\n{exc}")
                sys.exit(1)
            rows.append(row)
            print(
                f"[{len(rows):02d}/60] {row['Model']}-{row['Scenario ID']} | "
                f"LogLoss={row['Global OOF Log Loss']:.6f} | "
                f"AUROC={row['Global OOF AUROC']:.6f}"
            )

    df = pd.DataFrame(rows)
    if len(df) != 60:
        raise RuntimeError(f"Expected 60 experiments, found {len(df)}")
    if df.duplicated(["Model", "Scenario ID"]).any():
        raise RuntimeError("Duplicate model/scenario combinations detected")
    if not (df.groupby("Model").size() == 6).all():
        raise RuntimeError("Every model must have exactly six scenarios")
    if set(df["Model"]) != {f"E{i}" for i in range(1, 11)}:
        raise RuntimeError("Expected models E1-E10")
    if not (df["OOF Subjects"] == EXPECTED_SUBJECTS).all():
        raise RuntimeError("Not all experiments have 1362 OOF subjects")

    numeric_metric_cols = [
        "Global OOF Log Loss", "Global OOF AUROC", "Global OOF AUPRC",
        "Global OOF Brier Score", "Global OOF ECE",
        "Global OOF Balanced Accuracy @ 0.5",
    ]
    for col in numeric_metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy()).all():
            raise RuntimeError(f"Invalid values in {col}")

    df["_model_order"] = df["Model"].map(model_number)
    df["_scenario_order"] = df["Scenario ID"].map(scenario_number)
    df = df.sort_values(["_model_order", "_scenario_order"]).drop(
        columns=["_model_order", "_scenario_order"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format="%.9f")

    print("\nVALIDATION STATUS: PASS")
    print("All 60 experiments contain 1362 unique OOF predictions.")
    print("Log Loss/AUROC/AUPRC/Brier/ECE were independently recomputed from OOF probabilities.")
    print("Available cv_summary.json metrics matched the recalculated values.")
    print(f"Saved: {output_file}\n")


if __name__ == "__main__":
    main()
