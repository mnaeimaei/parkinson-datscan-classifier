#!/usr/bin/env python3
"""
STEP 31A — FREEZE EXPERIMENTAL ENS328R DEPLOYMENT PREDICTOR

ENS328R:
    P1 + R_E6 + P2 + P3 + P7 + P8
    equal-weight logit_mean
    final calibration = temperature

This step intentionally leaves data/final_predictor_data/ENS328 untouched.

What is frozen here
-------------------
1. Copy the five already-frozen ENS328 member calibrators unchanged.
2. Select R_E6 member calibration by five-fold cross-fitted Log Loss among:
       none / temperature / platt / isotonic
3. Fit the selected R_E6 calibrator on all Step-29 OOF rows for deployment.
4. Construct a six-member cross-fitted OOF ensemble:
       original P1/P2/P3/P7/P8 Step-16 calibrated_probability
       + R_E6 newly cross-fitted calibrated probability
5. Fit one final deployment temperature on all six-member raw ensemble OOF.
6. Save an independent ENS328R final_predictor_config.json.

IMPORTANT
---------
- Step 30 is post-hoc evidence, not a fresh unbiased winner estimate.
- All-OOF calibrator fits are deployment-only.
- Do not overwrite ENS328.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-6
ORIGINAL_MEMBERS = ("P1", "P2", "P3", "P7", "P8")
ROBUST_MEMBER = "R_E6"
FROZEN_MEMBERS = ("P1", "R_E6", "P2", "P3", "P7", "P8")
METHODS = ("none", "temperature", "platt", "isotonic")

DEFAULT_SHORTLIST = Path(
    "data/calibration_validation_data/"
    "competition_shortlist_data/"
    "competition_shortlist_8.csv"
)
DEFAULT_ROBUST_OOF = Path(
    "data/generalization_validation_data/"
    "step29_e6s5_robust_step11_confirmation/"
    "step29_robust_oof_predictions.csv"
)
DEFAULT_STEP30_SUMMARY = Path(
    "data/generalization_validation_data/"
    "step30_robust_e6_ensemble_integration/"
    "step30_summary.json"
)
DEFAULT_ENS328_DIR = Path("data/final_predictor_data/ENS328")
DEFAULT_OUTPUT = Path("data/final_predictor_data/ENS328R")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze independent six-member ENS328R deployment predictor."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--robust-oof", type=Path, default=None)
    p.add_argument("--step30-summary", type=Path, default=None)
    p.add_argument("--ens328-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--max-calibration-iter", type=int, default=150)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def resolve_stored_path(root: Path, value: Any) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate
    normalized = raw.replace("\\", "/")
    if "/data/" in normalized:
        rel = normalized.split("/data/", 1)[1]
        candidate = (root / "data" / rel).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve stored path: {raw}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip(p)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit_mean(matrix: np.ndarray) -> np.ndarray:
    return sigmoid(np.mean(logit(matrix), axis=1))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip(p)
    y = np.asarray(y, dtype=np.int64)
    return float((-(y * np.log(p) + (1 - y) * np.log1p(-p))).mean())


def fit_calibrator(ProbabilityCalibrator, method, y, p, max_iter):
    if method == "none":
        return None
    cal = ProbabilityCalibrator(method=method, output_epsilon=EPS)
    cal.fit(
        labels=np.asarray(y, dtype=np.int64),
        probabilities=np.asarray(p, dtype=np.float64),
        max_iter=max_iter,
    )
    return cal


def apply_calibrator(cal, method, p):
    if method == "none":
        return np.asarray(p, dtype=np.float64).copy()
    return np.asarray(cal.predict_proba(p), dtype=np.float64)


def cross_fit(
    ProbabilityCalibrator,
    *,
    y: np.ndarray,
    p: np.ndarray,
    folds: np.ndarray,
    method: str,
    max_iter: int,
) -> np.ndarray:
    if method == "none":
        return np.asarray(p, dtype=np.float64).copy()

    out = np.full(len(y), np.nan, dtype=np.float64)
    for heldout in sorted(np.unique(folds).tolist()):
        fit_mask = folds != heldout
        val_mask = folds == heldout
        cal = fit_calibrator(
            ProbabilityCalibrator,
            method,
            y[fit_mask],
            p[fit_mask],
            max_iter,
        )
        out[val_mask] = apply_calibrator(cal, method, p[val_mask])

    if not np.isfinite(out).all():
        raise RuntimeError(f"Cross-fit failed for method={method}")
    return out


def load_original_member_crossfit(
    *,
    project_root: Path,
    shortlist_path: Path,
    expected_subjects: int,
    expected_folds: int,
):
    shortlist = pd.read_csv(shortlist_path)
    required = {
        "Shortlist ID",
        "Calibration Method",
        "Cross-Fitted Prediction File",
    }
    missing = required - set(shortlist.columns)
    if missing:
        raise RuntimeError(f"Shortlist missing {sorted(missing)}")

    result = {}
    canonical_uid = canonical_y = canonical_fold = None

    for member in ORIGINAL_MEMBERS:
        rows = shortlist.loc[shortlist["Shortlist ID"].astype(str) == member]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one shortlist row for {member}")
        row = rows.iloc[0]
        path = resolve_stored_path(project_root, row["Cross-Fitted Prediction File"])
        df = pd.read_csv(path)

        req = {
            "uid",
            "fold",
            "is_pathologic",
            "raw_probability",
            "calibrated_probability",
        }
        miss = req - set(df.columns)
        if miss:
            raise RuntimeError(f"{member}: missing {sorted(miss)} in {path}")

        df = df.copy()
        df["uid"] = df["uid"].astype(str)
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
        df["is_pathologic"] = pd.to_numeric(
            df["is_pathologic"], errors="raise"
        ).astype(np.int64)
        df["raw_probability"] = pd.to_numeric(
            df["raw_probability"], errors="raise"
        ).astype(np.float64)
        df["calibrated_probability"] = pd.to_numeric(
            df["calibrated_probability"], errors="raise"
        ).astype(np.float64)
        df = df.sort_values("uid", kind="stable").reset_index(drop=True)

        if len(df) != expected_subjects or df["uid"].duplicated().any():
            raise RuntimeError(f"{member}: invalid subject coverage")
        if set(df["fold"].unique()) != set(range(expected_folds)):
            raise RuntimeError(f"{member}: invalid fold coverage")

        uid = df["uid"].to_numpy(dtype=str)
        y = df["is_pathologic"].to_numpy(dtype=np.int64)
        fold = df["fold"].to_numpy(dtype=np.int64)

        if canonical_uid is None:
            canonical_uid, canonical_y, canonical_fold = uid, y, fold
        else:
            if not np.array_equal(canonical_uid, uid):
                raise RuntimeError(f"{member}: UID mismatch")
            if not np.array_equal(canonical_y, y):
                raise RuntimeError(f"{member}: label mismatch")
            if not np.array_equal(canonical_fold, fold):
                raise RuntimeError(f"{member}: fold mismatch")

        result[member] = {
            "method": str(row["Calibration Method"]).strip().lower(),
            "path": path,
            "raw": df["raw_probability"].to_numpy(dtype=np.float64),
            "crossfit_calibrated":
                df["calibrated_probability"].to_numpy(dtype=np.float64),
            "shortlist_row": row,
        }

    return canonical_uid, canonical_y, canonical_fold, result


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else script_path.parents[2]
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.calibration.probability_calibration import ProbabilityCalibrator
    from src.inference.final_ensemble import FinalEnsemblePredictor

    shortlist_path = resolve(project_root, args.shortlist or DEFAULT_SHORTLIST)
    robust_oof_path = resolve(project_root, args.robust_oof or DEFAULT_ROBUST_OOF)
    step30_summary_path = resolve(
        project_root, args.step30_summary or DEFAULT_STEP30_SUMMARY
    )
    ens328_dir = resolve(project_root, args.ens328_dir or DEFAULT_ENS328_DIR)
    output_dir = resolve(project_root, args.output_dir or DEFAULT_OUTPUT)

    if output_dir == ens328_dir:
        raise RuntimeError("ENS328R output must not overwrite ENS328.")

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} is non-empty. Use --overwrite for intentional rebuild."
            )

    for path in (
        shortlist_path,
        robust_oof_path,
        step30_summary_path,
        ens328_dir / "final_predictor_config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    step30 = json.loads(step30_summary_path.read_text(encoding="utf-8"))
    if step30.get("descriptive_decision") != "ROBUST_ADD_PROMISING":
        raise RuntimeError(
            "Step30 did not identify ROBUST_ADD_PROMISING; refusing ENS328R freeze."
        )
    step30_structures = step30.get("fixed_structures", {})
    expected = ["P1", "R_E6", "P2", "P3", "P7", "P8"]
    if step30_structures.get("ROBUST_ADD") != expected:
        raise RuntimeError("Step30 ROBUST_ADD membership differs from ENS328R.")

    uid, y, folds, original = load_original_member_crossfit(
        project_root=project_root,
        shortlist_path=shortlist_path,
        expected_subjects=args.expected_subjects,
        expected_folds=args.expected_folds,
    )

    robust = pd.read_csv(robust_oof_path)
    required_robust = {"uid", "fold", "is_pathologic", "robust_probability"}
    missing = required_robust - set(robust.columns)
    if missing:
        raise RuntimeError(f"Robust OOF missing {sorted(missing)}")

    robust = robust.copy()
    robust["uid"] = robust["uid"].astype(str)
    robust["fold"] = pd.to_numeric(robust["fold"], errors="raise").astype(np.int64)
    robust["is_pathologic"] = pd.to_numeric(
        robust["is_pathologic"], errors="raise"
    ).astype(np.int64)
    robust["robust_probability"] = pd.to_numeric(
        robust["robust_probability"], errors="raise"
    ).astype(np.float64)
    robust = robust.sort_values("uid", kind="stable").reset_index(drop=True)

    if len(robust) != args.expected_subjects or robust["uid"].duplicated().any():
        raise RuntimeError("Invalid robust OOF coverage")
    if not np.array_equal(uid, robust["uid"].to_numpy(dtype=str)):
        raise RuntimeError("Robust OOF UID alignment mismatch")
    if not np.array_equal(y, robust["is_pathologic"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Robust OOF labels mismatch")
    if not np.array_equal(folds, robust["fold"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Robust OOF fold mismatch")

    robust_raw = robust["robust_probability"].to_numpy(dtype=np.float64)

    # Select robust member calibration using the same original five folds.
    rows = []
    robust_cf_by_method = {}
    for method in METHODS:
        p_cf = cross_fit(
            ProbabilityCalibrator,
            y=y,
            p=robust_raw,
            folds=folds,
            method=method,
            max_iter=args.max_calibration_iter,
        )
        robust_cf_by_method[method] = p_cf
        rows.append(
            {
                "method": method,
                "cross_fitted_log_loss": log_loss(y, p_cf),
            }
        )

    robust_calibration_results = pd.DataFrame(rows).sort_values(
        ["cross_fitted_log_loss", "method"], kind="stable"
    ).reset_index(drop=True)

    # Explicit deterministic tie preference matching earlier stages.
    method_order = {m: i for i, m in enumerate(METHODS)}
    robust_calibration_results["_order"] = robust_calibration_results["method"].map(
        method_order
    )
    robust_calibration_results = robust_calibration_results.sort_values(
        ["cross_fitted_log_loss", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)

    robust_method = str(robust_calibration_results.iloc[0]["method"])
    robust_cf = robust_cf_by_method[robust_method]

    output_dir.mkdir(parents=True, exist_ok=True)
    member_cal_dir = output_dir / "member_calibrators"
    member_cal_dir.mkdir(parents=True, exist_ok=True)

    robust_calibration_results["selected"] = (
        robust_calibration_results["method"] == robust_method
    )
    robust_calibration_results.to_csv(
        output_dir / "robust_member_calibration_selection.csv",
        index=False,
        float_format="%.9f",
    )

    # Copy original five deployment calibrators exactly.
    old_config = json.loads(
        (ens328_dir / "final_predictor_config.json").read_text(encoding="utf-8")
    )
    old_members = old_config.get("members", {})
    if tuple(old_config.get("ensemble", {}).get("members", [])) != ORIGINAL_MEMBERS:
        raise RuntimeError("Existing ENS328 config member order/content changed.")

    member_config = {}
    member_manifest_rows = []

    for member in ORIGINAL_MEMBERS:
        old_meta = dict(old_members[member])
        old_cal_rel = Path(old_meta["final_calibrator_path"])
        old_cal = ens328_dir / old_cal_rel
        if not old_cal.is_file():
            raise FileNotFoundError(old_cal)

        target_rel = Path("member_calibrators") / f"{member}.json"
        target = output_dir / target_rel
        shutil.copy2(old_cal, target)

        old_meta["final_calibrator_path"] = str(target_rel)
        old_meta["copied_unchanged_from"] = str(old_cal)
        member_config[member] = old_meta

        cal_state = json.loads(target.read_text(encoding="utf-8"))
        member_manifest_rows.append(
            {
                "Member ID": member,
                "Role": "existing ENS328 member",
                "Calibration Method":
                    old_meta["selected_member_calibration_method"],
                "Calibrator Path": str(target),
                "Temperature": cal_state.get("temperature", 1.0),
                "Source": str(old_cal),
                "SHA256": sha256_file(target),
            }
        )

    # Fit robust member deployment calibrator on all robust OOF rows.
    robust_final_cal = fit_calibrator(
        ProbabilityCalibrator,
        robust_method,
        y,
        robust_raw,
        args.max_calibration_iter,
    )

    robust_rel = Path("member_calibrators") / f"{ROBUST_MEMBER}.json"
    robust_path = output_dir / robust_rel

    if robust_method == "none":
        identity = ProbabilityCalibrator(method="none", output_epsilon=EPS)
        identity.fit(labels=y, probabilities=robust_raw)
        identity.save(robust_path)
        robust_state = identity.to_dict()
    else:
        robust_final_cal.save(robust_path)
        robust_state = robust_final_cal.to_dict()

    member_config[ROBUST_MEMBER] = {
        "calibration_candidate_id": "STEP31A_R_E6",
        "model": "model06_r3d18_kinetics400_pretrained",
        "scenario_id": "S5",
        "scenario": "roi_aug_plus_acquisition_robust",
        "selected_member_calibration_method": robust_method,
        "final_calibrator_path": str(robust_rel),
        "source_oof_prediction_file": str(robust_oof_path),
        "experiment_directory": str(robust_oof_path.parent),
        "model_used": "E6 / R3D-18 Kinetics-400",
        "model_type": "3D CNN",
        "initialization": "Kinetics-400 pretrained",
        "model_file_name": "model06_r3d18_kinetics400_pretrained.py",
        "step28_augmentation": True,
    }

    member_manifest_rows.append(
        {
            "Member ID": ROBUST_MEMBER,
            "Role": "new robust member",
            "Calibration Method": robust_method,
            "Calibrator Path": str(robust_path),
            "Temperature": robust_state.get("temperature", 1.0),
            "Source": str(robust_oof_path),
            "SHA256": sha256_file(robust_path),
        }
    )

    # Six-member cross-fitted ensemble for deployment-temperature fitting.
    cf_map = {
        member: original[member]["crossfit_calibrated"]
        for member in ORIGINAL_MEMBERS
    }
    cf_map[ROBUST_MEMBER] = robust_cf

    member_matrix = np.column_stack([cf_map[m] for m in FROZEN_MEMBERS])
    raw_ensemble = logit_mean(member_matrix)

    ensemble_oof_path = output_dir / "final_ensemble_oof_for_calibration.csv"
    pd.DataFrame(
        {
            "uid": uid,
            "fold": folds,
            "is_pathologic": y,
            "probability": raw_ensemble,
        }
    ).to_csv(ensemble_oof_path, index=False, float_format="%.12f")

    # Cross-fitted final-temperature diagnostic first. This is still post-hoc
    # because ENS328R was motivated by Step30, but it does not fit the final
    # temperature on the row it evaluates.
    final_temperature_cf = cross_fit(
        ProbabilityCalibrator,
        y=y,
        p=raw_ensemble,
        folds=folds,
        method="temperature",
        max_iter=args.max_calibration_iter,
    )

    # Deployment-only fit on all available OOF rows.
    final_cal = ProbabilityCalibrator(method="temperature", output_epsilon=EPS)
    final_cal.fit(
        labels=y,
        probabilities=raw_ensemble,
        max_iter=args.max_calibration_iter,
    )
    final_rel = Path("final_ensemble_temperature_calibrator.json")
    final_path = output_dir / final_rel
    final_cal.save(final_path)
    final_state = final_cal.to_dict()
    final_temp = float(final_state["temperature"])

    # Save cross-fitted six-member diagnostic table.
    selected_cf = apply_calibrator(final_cal, "temperature", raw_ensemble)
    diag = pd.DataFrame(
        {
            "uid": uid,
            "fold": folds,
            "is_pathologic": y,
            **{f"{m}_selected_probability": cf_map[m] for m in FROZEN_MEMBERS},
            "raw_ensemble_probability": raw_ensemble,
            "crossfitted_final_temperature_probability": final_temperature_cf,
            "deployment_temperature_probability": selected_cf,
        }
    )
    diag.to_csv(
        output_dir / "ens328r_crossfitted_member_diagnostic.csv",
        index=False,
        float_format="%.12f",
    )

    config = {
        "schema_version": 1,
        "status": "EXPERIMENTAL_FROZEN",
        "purpose": (
            "Independent ENS328R deployment candidate. Keep ENS328 untouched. "
            "Never refit using competition/test labels."
        ),
        "selection_evidence": {
            "source_step": 30,
            "candidate": "ROBUST_ADD",
            "posthoc_outer_fold_log_loss":
                float(step30["current_vs_robust"]["ROBUST_ADD_log_loss"]),
            "current_ens328_outer_fold_log_loss":
                float(step30["current_vs_robust"]["CURRENT_ENS328_log_loss"]),
            "delta_vs_current":
                float(step30["current_vs_robust"]["add_minus_current_log_loss"]),
            "methodological_status": step30.get("methodological_status"),
            "note": (
                "Step30 is post-hoc integration evidence. Final all-OOF "
                "calibrator fits below are deployment-only."
            ),
        },
        "ensemble": {
            "id": "ENS328R",
            "members": list(FROZEN_MEMBERS),
            "number_of_members": len(FROZEN_MEMBERS),
            "member_weights": {
                member: 1.0 / len(FROZEN_MEMBERS)
                for member in FROZEN_MEMBERS
            },
            "aggregation_rule": "logit_mean",
            "final_calibration_method": "temperature",
            "final_calibrator_path": str(final_rel),
            "final_temperature": final_temp,
            "final_calibrator_training_source": str(ensemble_oof_path),
        },
        "members": member_config,
        "numerics": {
            "probability_epsilon": EPS,
            "final_output_range": "[1e-6, 1-1e-6]",
        },
        "inference_contract": {
            "input_per_member": (
                "raw probability after averaging that member's five CV fold "
                "checkpoints, before post-hoc member calibration"
            ),
            "member_step": "apply frozen member-specific deployment calibrator",
            "ensemble_step": (
                "equal-weight mean of logit(selected_member_probability)"
            ),
            "final_step": "apply frozen final temperature calibrator",
            "output": "continuous probability of is_pathologic",
            "thresholding": "DO NOT threshold for competition submission",
        },
    }

    config_path = output_dir / "final_predictor_config.json"
    write_json(config_path, config)

    pd.DataFrame(member_manifest_rows).to_csv(
        output_dir / "frozen_member_manifest.csv",
        index=False,
        float_format="%.9f",
    )

    # Production-class contract smoke test.
    predictor = FinalEnsemblePredictor(config_path=config_path)
    if tuple(predictor.members) != FROZEN_MEMBERS:
        raise RuntimeError(
            f"FinalEnsemblePredictor member mismatch: {predictor.members}"
        )

    grid = {
        member: np.asarray([0.02, 0.15, 0.35, 0.50, 0.72, 0.91], dtype=np.float64)
        for member in FROZEN_MEMBERS
    }
    smoke = predictor.predict_with_intermediates(grid)
    final_p = np.asarray(smoke["final_probability"], dtype=np.float64)
    if not np.isfinite(final_p).all() or np.any(final_p <= 0) or np.any(final_p >= 1):
        raise RuntimeError("ENS328R predictor smoke test failed.")

    summary = {
        "status": "PASS",
        "predictor": "ENS328R",
        "members": list(FROZEN_MEMBERS),
        "robust_member_calibration_method": robust_method,
        "robust_member_cross_fitted_log_loss":
            float(robust_calibration_results.iloc[0]["cross_fitted_log_loss"]),
        "six_member_raw_cross_fitted_log_loss": log_loss(y, raw_ensemble),
        "six_member_crossfitted_final_temperature_log_loss":
            log_loss(y, final_temperature_cf),
        "six_member_deployment_temperature_applied_oof_log_loss":
            log_loss(y, selected_cf),
        "final_temperature": final_temp,
        "config": str(config_path),
        "ens328_untouched": True,
    }
    write_json(output_dir / "step31a_freeze_summary.json", summary)

    print("=" * 108)
    print("STEP 31A — ENS328R EXPERIMENTAL PREDICTOR FROZEN")
    print("=" * 108)
    print(f"Members                    : {'+'.join(FROZEN_MEMBERS)}")
    print(f"Robust member calibration  : {robust_method}")
    print(f"Final temperature          : {final_temp:.9f}")
    print(f"Output                     : {output_dir}")
    print("Existing ENS328            : UNTOUCHED")
    print("Status                     : PASS")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
