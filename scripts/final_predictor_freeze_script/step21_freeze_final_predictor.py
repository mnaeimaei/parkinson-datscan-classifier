#!/usr/bin/env python3
"""
Phase 12 — Freeze Final Competition Predictor

Frozen choice from Phase 11:
    R4 / ENS328
    P1 + P2 + P3 + P7 + P8
    equal-weight logit_mean
    final ensemble calibration = temperature

This script:
1. Verifies the Phase-11 winner has not changed.
2. Fits one FINAL member-level calibrator on all OOF rows for every selected
   member, using the member's already-selected calibration method.
3. Fits one FINAL ensemble temperature calibrator on all 1,362 raw ENS328 OOF
   ensemble probabilities.
4. Saves a portable final_predictor_config.json.
5. Loads the saved configuration through FinalEnsemblePredictor and validates
   the complete deployment transformation contract.

The 0.255582 cross-fitted score remains the unbiased calibration-selection
estimate. Any all-OOF calibrator fit performed here is for deployment only and
must not be re-used as a new model-selection score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN_ENSEMBLE_ID = "ENS328"
FROZEN_REPRESENTATIVE_ID = "R4"
FROZEN_MEMBERS = ("P1", "P2", "P3", "P7", "P8")
FROZEN_AGGREGATION = "logit_mean"
FROZEN_FINAL_CALIBRATION = "temperature"
FROZEN_CROSS_FITTED_LOG_LOSS = 0.255582


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze ENS328 and fit final deployment calibrators."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--winner", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--max-calibration-iter", type=int, default=150)
    parser.add_argument(
        "--winner-match-tolerance",
        type=float,
        default=2e-6,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_oof(
    path: Path,
    *,
    expected_subjects: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OOF file not found: {path}")

    df = pd.read_csv(path)

    required = {"uid", "is_pathologic", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing OOF columns {sorted(missing)}"
        )

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} rows, "
            f"found {len(df)}."
        )

    if df["uid"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate UIDs detected.")

    labels = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).to_numpy(dtype=np.int64)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: labels must be binary.")

    probability = pd.to_numeric(
        df["probability"], errors="raise"
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(probability).all():
        raise ValueError(f"{path}: probability contains NaN/Inf.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"{path}: probability outside [0,1].")

    df = df.copy()
    df["uid"] = df["uid"].astype(str)
    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def validate_phase11_ensemble_oof(
    path: Path,
    *,
    expected_subjects: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-11 ensemble prediction file not found: {path}"
        )

    df = pd.read_csv(path)
    required = {
        "uid",
        "fold",
        "is_pathologic",
        "raw_ensemble_probability",
        "calibrated_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing Phase-11 columns {sorted(missing)}"
        )

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} rows, "
            f"found {len(df)}."
        )

    if df["uid"].astype(str).duplicated().any():
        raise ValueError(
            f"{path}: duplicate UIDs in ensemble OOF predictions."
        )

    raw = pd.to_numeric(
        df["raw_ensemble_probability"], errors="raise"
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(raw).all():
        raise ValueError(
            f"{path}: raw ensemble probabilities contain NaN/Inf."
        )
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError(
            f"{path}: raw ensemble probabilities outside [0,1]."
        )

    return df


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()

    shortlist_path = (
        args.shortlist
        or project_root
        / "data"
        / "calibration_validation_data"
        / "competition_shortlist_data"
        / "competition_shortlist_8.csv"
    ).expanduser().resolve()

    winner_path = (
        args.winner
        or project_root
        / "data"
        / "ensemble_calibration_statistics_data"
        / "representative_ensembles"
        / "best_ensemble_after_cross_fitted_calibration.csv"
    ).expanduser().resolve()

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "final_predictor_data"
        / FROZEN_ENSEMBLE_ID
    ).expanduser().resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.calibration.probability_calibration import (
        ProbabilityCalibrator,
        fit_calibrator_from_oof_csv,
    )
    from src.inference.final_ensemble import FinalEnsemblePredictor

    print("\n" + "=" * 104)
    print("PHASE 12 — FREEZE FINAL COMPETITION PREDICTOR")
    print("=" * 104)
    print(f"Project root             : {project_root}")
    print(f"P1-P8 shortlist          : {shortlist_path}")
    print(f"Phase-11 winner          : {winner_path}")
    print(f"Output                   : {output_dir}")
    print(f"Frozen ensemble          : {FROZEN_ENSEMBLE_ID}")
    print(f"Frozen members           : {'+'.join(FROZEN_MEMBERS)}")
    print(f"Aggregation              : {FROZEN_AGGREGATION}")
    print(f"Final calibration        : {FROZEN_FINAL_CALIBRATION}")
    print("Calibration fit          : all available OOF rows, deployment only")
    print("GPU                      : NOT USED")
    print()

    if not shortlist_path.is_file():
        raise FileNotFoundError(
            f"Shortlist not found: {shortlist_path}"
        )
    if not winner_path.is_file():
        raise FileNotFoundError(
            f"Winner file not found: {winner_path}"
        )

    winner_df = pd.read_csv(winner_path)
    if len(winner_df) != 1:
        raise RuntimeError(
            f"Expected exactly one Phase-11 winner row, "
            f"found {len(winner_df)}."
        )

    winner = winner_df.iloc[0]

    required_winner = {
        "Representative ID",
        "Ensemble ID",
        "Members",
        "Aggregation Rule",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Cross-Fitted Prediction File",
    }
    missing = required_winner - set(winner_df.columns)
    if missing:
        raise RuntimeError(
            f"Winner file missing columns: {sorted(missing)}"
        )

    winner_members = tuple(
        value.strip()
        for value in str(winner["Members"]).split("+")
        if value.strip()
    )

    checks = {
        "Representative ID": (
            str(winner["Representative ID"]),
            FROZEN_REPRESENTATIVE_ID,
        ),
        "Ensemble ID": (
            str(winner["Ensemble ID"]),
            FROZEN_ENSEMBLE_ID,
        ),
        "Members": (
            winner_members,
            FROZEN_MEMBERS,
        ),
        "Aggregation Rule": (
            str(winner["Aggregation Rule"]),
            FROZEN_AGGREGATION,
        ),
        "Calibration Method": (
            str(winner["Calibration Method"]),
            FROZEN_FINAL_CALIBRATION,
        ),
    }

    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise RuntimeError(
                f"FINAL FREEZE ABORTED: {name} changed. "
                f"Expected {expected!r}, observed {observed!r}."
            )

    winner_ll = float(winner["Cross-Fitted OOF Log Loss"])
    if abs(winner_ll - FROZEN_CROSS_FITTED_LOG_LOSS) > max(
        args.winner_match_tolerance, 5e-7
    ):
        raise RuntimeError(
            "FINAL FREEZE ABORTED: Phase-11 winner Log Loss changed. "
            f"Expected ~{FROZEN_CROSS_FITTED_LOG_LOSS:.6f}, "
            f"observed {winner_ll:.9f}."
        )

    shortlist = pd.read_csv(shortlist_path)

    required_shortlist = {
        "Shortlist ID",
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "Scenario",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "OOF Prediction File",
    }
    missing = required_shortlist - set(shortlist.columns)
    if missing:
        raise RuntimeError(
            f"Shortlist missing columns: {sorted(missing)}"
        )

    shortlist["Shortlist ID"] = shortlist[
        "Shortlist ID"
    ].astype(str)

    frozen_rows = shortlist.loc[
        shortlist["Shortlist ID"].isin(FROZEN_MEMBERS)
    ].copy()

    if set(frozen_rows["Shortlist ID"]) != set(FROZEN_MEMBERS):
        raise RuntimeError(
            "Could not find all frozen members in competition shortlist."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    member_cal_dir = output_dir / "member_calibrators"
    member_cal_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fit final member-specific deployment calibrators.
    # ------------------------------------------------------------------
    member_config = {}
    member_manifest_rows = []

    print("Fitting final member-level calibrators:")

    for member_id in FROZEN_MEMBERS:
        row = frozen_rows.loc[
            frozen_rows["Shortlist ID"] == member_id
        ].iloc[0]

        method = str(row["Calibration Method"])
        oof_path = Path(
            str(row["OOF Prediction File"])
        ).expanduser().resolve()

        raw_oof = validate_raw_oof(
            oof_path,
            expected_subjects=args.expected_subjects,
        )

        calibrator = fit_calibrator_from_oof_csv(
            oof_path,
            method=method,
            output_epsilon=1e-6,
            max_iter=args.max_calibration_iter,
        )

        calibrator_relative = (
            Path("member_calibrators") / f"{member_id}.json"
        )
        calibrator_path = output_dir / calibrator_relative
        calibrator.save(calibrator_path)

        state = calibrator.to_dict()

        experiment_dir = oof_path.parent

        metadata = {
            "calibration_candidate_id": str(
                row["Calibration Candidate ID"]
            ),
            "model": str(row["Model"]),
            "scenario_id": str(row["Scenario ID"]),
            "scenario": str(row["Scenario"]),
            "selected_member_calibration_method": method,
            "final_calibrator_path": str(calibrator_relative),
            "source_oof_prediction_file": str(oof_path),
            "experiment_directory": str(experiment_dir),
        }

        # Keep optional columns when present.
        for source_col, target_key in [
            ("Model Used", "model_used"),
            ("Type", "model_type"),
            ("Scratch / pretrained", "initialization"),
            ("File Name", "model_file_name"),
        ]:
            if source_col in row.index and pd.notna(row[source_col]):
                metadata[target_key] = str(row[source_col])

        member_config[member_id] = metadata

        member_manifest_rows.append(
            {
                "Member ID": member_id,
                "Calibration Candidate ID":
                    metadata["calibration_candidate_id"],
                "Model": metadata["model"],
                "Scenario ID": metadata["scenario_id"],
                "Scenario": metadata["scenario"],
                "Final Member Calibration Method": method,
                "Final Calibrator Path": str(calibrator_path),
                "Final Temperature":
                    state.get("temperature", 1.0),
                "Source OOF Prediction File": str(oof_path),
                "Experiment Directory": str(experiment_dir),
                "OOF Subjects": len(raw_oof),
            }
        )

        extra = ""
        if method == "temperature":
            extra = f" T={state['temperature']:.6f}"

        print(
            f"  {member_id}: {metadata['model']} "
            f"{metadata['scenario_id']}  method={method}{extra}"
        )

    member_manifest = pd.DataFrame(member_manifest_rows)
    member_manifest_path = output_dir / "frozen_member_manifest.csv"
    member_manifest.to_csv(
        member_manifest_path,
        index=False,
        float_format="%.9f",
    )

    # ------------------------------------------------------------------
    # Fit final ensemble calibrator on all raw ENS328 OOF probabilities.
    # ------------------------------------------------------------------
    selected_crossfit_path = Path(
        str(winner["Cross-Fitted Prediction File"])
    ).expanduser().resolve()

    ensemble_crossfit_df = validate_phase11_ensemble_oof(
        selected_crossfit_path,
        expected_subjects=args.expected_subjects,
    )

    final_ensemble_oof_path = (
        output_dir / "final_ensemble_oof_for_calibration.csv"
    )

    pd.DataFrame(
        {
            "uid": ensemble_crossfit_df["uid"].astype(str),
            "is_pathologic": pd.to_numeric(
                ensemble_crossfit_df["is_pathologic"],
                errors="raise",
            ).astype(int),
            "probability": pd.to_numeric(
                ensemble_crossfit_df["raw_ensemble_probability"],
                errors="raise",
            ),
        }
    ).to_csv(
        final_ensemble_oof_path,
        index=False,
        float_format="%.12f",
    )

    final_calibrator = fit_calibrator_from_oof_csv(
        final_ensemble_oof_path,
        method=FROZEN_FINAL_CALIBRATION,
        output_epsilon=1e-6,
        max_iter=args.max_calibration_iter,
    )

    final_calibrator_relative = Path(
        "final_ensemble_temperature_calibrator.json"
    )
    final_calibrator_path = (
        output_dir / final_calibrator_relative
    )
    final_calibrator.save(final_calibrator_path)

    final_temperature = float(
        final_calibrator.to_dict()["temperature"]
    )

    print()
    print(
        "Final ensemble deployment temperature "
        f"(fit on all OOF): T={final_temperature:.9f}"
    )

    # ------------------------------------------------------------------
    # Save frozen deployment config.
    # ------------------------------------------------------------------
    config_path = output_dir / "final_predictor_config.json"

    config = {
        "schema_version": 1,
        "status": "FROZEN",
        "purpose": (
            "Final deployment configuration selected by cross-fitted "
            "OOF Log Loss. Never refit using competition/test labels."
        ),
        "selection_evidence": {
            "phase_11_representative_id":
                FROZEN_REPRESENTATIVE_ID,
            "phase_11_ensemble_id": FROZEN_ENSEMBLE_ID,
            "cross_fitted_oof_log_loss": winner_ll,
            "cross_fitted_oof_auroc":
                float(winner.get(
                    "Cross-Fitted OOF AUROC",
                    np.nan,
                )),
            "delta_log_loss_vs_p1":
                float(winner.get(
                    "Delta Log Loss vs P1",
                    np.nan,
                )),
            "phase_11_winner_file": str(winner_path),
            "note": (
                "These cross-fitted metrics remain the model-selection "
                "estimate. Final all-OOF calibrator fitting is deployment-only."
            ),
        },
        "ensemble": {
            "id": FROZEN_ENSEMBLE_ID,
            "members": list(FROZEN_MEMBERS),
            "number_of_members": len(FROZEN_MEMBERS),
            "member_weights": {
                member_id: 1.0 / len(FROZEN_MEMBERS)
                for member_id in FROZEN_MEMBERS
            },
            "aggregation_rule": FROZEN_AGGREGATION,
            "final_calibration_method":
                FROZEN_FINAL_CALIBRATION,
            "final_calibrator_path":
                str(final_calibrator_relative),
            "final_temperature": final_temperature,
            "final_calibrator_training_source":
                str(final_ensemble_oof_path),
        },
        "members": member_config,
        "numerics": {
            "probability_epsilon": 1e-6,
            "final_output_range": "[1e-6, 1-1e-6]",
        },
        "inference_contract": {
            "input_per_member": (
                "raw probability after averaging that member's five "
                "CV fold checkpoints, before post-hoc calibration"
            ),
            "member_step": (
                "apply member-specific final calibrator fitted on all "
                "member OOF predictions"
            ),
            "ensemble_step": (
                "equal-weight mean of logit(selected_member_probability)"
            ),
            "final_step": (
                "apply frozen final temperature calibrator"
            ),
            "output": (
                "continuous probability of is_pathologic / abnormal scan"
            ),
            "thresholding": "DO NOT threshold for competition submission",
        },
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Load saved objects through production aggregation class and run
    # a deterministic contract smoke test.
    # ------------------------------------------------------------------
    predictor = FinalEnsemblePredictor(
        config_path=config_path
    )

    test_grid = {
        member_id: np.asarray(
            [0.01, 0.10, 0.35, 0.50, 0.70, 0.90, 0.99],
            dtype=np.float64,
        )
        for member_id in FROZEN_MEMBERS
    }

    test_result = predictor.predict_with_intermediates(test_grid)
    final_test_probability = np.asarray(
        test_result["final_probability"],
        dtype=np.float64,
    )

    if not np.isfinite(final_test_probability).all():
        raise RuntimeError(
            "FinalEnsemblePredictor smoke test produced NaN/Inf."
        )
    if np.any(final_test_probability <= 0.0) or np.any(
        final_test_probability >= 1.0
    ):
        raise RuntimeError(
            "FinalEnsemblePredictor smoke test produced invalid probability."
        )

    # Reload each calibrator and verify exact serialized behavior.
    reload_checks = {}
    for member_id in FROZEN_MEMBERS:
        calibrator_path = (
            output_dir
            / member_config[member_id]["final_calibrator_path"]
        )
        loaded = ProbabilityCalibrator.load(calibrator_path)
        reference = loaded.predict_proba(
            np.asarray([0.2, 0.5, 0.8])
        )
        reload_checks[member_id] = [
            float(v) for v in reference
        ]

    # ------------------------------------------------------------------
    # Provenance hashes.
    # ------------------------------------------------------------------
    hash_targets = [
        config_path,
        member_manifest_path,
        final_ensemble_oof_path,
        final_calibrator_path,
    ] + [
        output_dir
        / member_config[member_id]["final_calibrator_path"]
        for member_id in FROZEN_MEMBERS
    ]

    hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in hash_targets
    }

    hashes_path = output_dir / "frozen_artifact_sha256.json"
    with hashes_path.open("w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=2, sort_keys=True)

    summary = {
        "status": "PASS",
        "frozen_predictor": FROZEN_ENSEMBLE_ID,
        "members": list(FROZEN_MEMBERS),
        "aggregation_rule": FROZEN_AGGREGATION,
        "final_calibration_method":
            FROZEN_FINAL_CALIBRATION,
        "final_deployment_temperature": final_temperature,
        "phase_11_cross_fitted_oof_log_loss": winner_ll,
        "expected_subjects": args.expected_subjects,
        "member_calibrator_reload_checks": reload_checks,
        "contract_smoke_test_final_probabilities": [
            float(v) for v in final_test_probability
        ],
        "artifacts": {
            "config": str(config_path),
            "member_manifest": str(member_manifest_path),
            "final_ensemble_oof_for_calibration":
                str(final_ensemble_oof_path),
            "final_ensemble_calibrator":
                str(final_calibrator_path),
            "sha256": str(hashes_path),
        },
    }

    summary_path = output_dir / "final_predictor_freeze_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 104)
    print("FINAL PREDICTOR FROZEN")
    print("=" * 104)
    print(f"Predictor                : {FROZEN_ENSEMBLE_ID}")
    print(f"Members                  : {'+'.join(FROZEN_MEMBERS)}")
    print(f"Aggregation              : {FROZEN_AGGREGATION}")
    print(
        f"Final calibration        : "
        f"{FROZEN_FINAL_CALIBRATION}"
    )
    print(
        f"Final deployment T       : {final_temperature:.9f}"
    )
    print(
        f"Selection CF OOF LL      : {winner_ll:.6f}"
    )
    print()
    print(f"Saved: {config_path}")
    print(f"Saved: {member_manifest_path}")
    print(f"Saved: {final_ensemble_oof_path}")
    print(f"Saved: {final_calibrator_path}")
    print(f"Saved: {hashes_path}")
    print(f"Saved: {summary_path}")
    print()
    print("STATUS: PASS")
    print(
        "Next: connect the five member checkpoint inference runners "
        "to FinalEnsemblePredictor and generate submission.csv."
    )


if __name__ == "__main__":
    main()
