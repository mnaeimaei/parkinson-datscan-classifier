#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ======================================================================
# STEP 11 — CREATE & FREEZE 5-FOLD CV SPLITS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STEP10_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessing_supervised_data"
    / "step10_supervised_dataset_manifest_data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessing_supervised_data"
    / "step11_create_freeze_cv_splits_data"
)

N_SPLITS = 5
RANDOM_SEED = 42
SHUFFLE = True

EXPECTED_SUBJECTS = 1362
EXPECTED_NORMAL = 615
EXPECTED_PATHOLOGIC = 747

UID_COLUMN = "uid"
LABEL_COLUMN = "is_pathologic"

SCENARIOS = ("A", "B", "C")


# ======================================================================
# HELPERS
# ======================================================================


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def find_csv_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*.csv")
        if p.is_file()
    )


def read_csv_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read CSV:\n{path}\nReason: {exc}")


def identify_supervised_manifest(csv_files: list[Path]) -> Path:
    """
    Find the most appropriate Step-10 CSV containing at minimum:
        uid
        is_pathologic

    Preference:
        - filename containing 'supervised'
        - filename containing 'manifest'
        - file containing all three scenario path definitions if applicable
    """

    candidates: list[tuple[int, Path]] = []

    for path in csv_files:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        columns = set(df.columns)

        if UID_COLUMN not in columns or LABEL_COLUMN not in columns:
            continue

        name = path.name.lower()
        score = 0

        if "supervised" in name:
            score += 20

        if "manifest" in name:
            score += 10

        if "frozen" in name:
            score += 5

        if "scenario" in name:
            score += 2

        candidates.append((score, path))

    if not candidates:
        fail(
            "Could not find a Step-10 CSV containing both "
            f"'{UID_COLUMN}' and '{LABEL_COLUMN}' in:\n{STEP10_DIR}"
        )

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    return candidates[0][1]


def identify_scenario_manifests(
    csv_files: list[Path],
) -> dict[str, Path]:
    """
    Try to locate one manifest for Scenario A/B/C.

    This is used only for UID-universe consistency checks.

    If Step 10 stores all scenarios in one combined manifest, this function
    may return fewer than three files; the combined supervised manifest
    still remains authoritative.
    """

    result: dict[str, Path] = {}

    for scenario in SCENARIOS:
        scenario_lower = scenario.lower()

        scored: list[tuple[int, Path]] = []

        for path in csv_files:
            name = path.name.lower()

            try:
                df = pd.read_csv(path, nrows=5)
            except Exception:
                continue

            if UID_COLUMN not in df.columns:
                continue

            score = 0

            patterns = [
                f"scenario_{scenario_lower}",
                f"scenario-{scenario_lower}",
                f"scenario{scenario_lower}",
                f"scenario {scenario_lower}",
            ]

            if any(pattern in name for pattern in patterns):
                score += 20

            if "manifest" in name:
                score += 5

            if "frozen" in name:
                score += 2

            if score > 0:
                scored.append((score, path))

        if scored:
            scored.sort(key=lambda x: (-x[0], str(x[1])))
            result[scenario] = scored[0][1]

    return result


def validate_base_manifest(df: pd.DataFrame) -> dict:
    required = {UID_COLUMN, LABEL_COLUMN}

    missing_columns = required - set(df.columns)

    if missing_columns:
        fail(
            "Step-10 manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows = len(df)
    missing_uid = int(df[UID_COLUMN].isna().sum())
    missing_labels = int(df[LABEL_COLUMN].isna().sum())
    duplicate_uids = int(df[UID_COLUMN].duplicated().sum())

    labels_numeric = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="coerce",
    )

    invalid_numeric = int(labels_numeric.isna().sum())

    if invalid_numeric > 0:
        fail(
            f"{invalid_numeric} label values could not be converted "
            "to numeric 0/1."
        )

    unique_labels = sorted(labels_numeric.astype(int).unique().tolist())
    invalid_labels = [
        label for label in unique_labels
        if label not in (0, 1)
    ]

    if invalid_labels:
        fail(
            "Invalid label values detected: "
            f"{invalid_labels}. Expected only 0 and 1."
        )

    df[LABEL_COLUMN] = labels_numeric.astype(np.int64)

    normal = int((df[LABEL_COLUMN] == 0).sum())
    pathologic = int((df[LABEL_COLUMN] == 1).sum())
    unique_uids = int(df[UID_COLUMN].nunique())

    problems = []

    if missing_uid:
        problems.append(f"missing UID values = {missing_uid}")

    if missing_labels:
        problems.append(f"missing labels = {missing_labels}")

    if duplicate_uids:
        problems.append(f"duplicate UIDs = {duplicate_uids}")

    if rows != EXPECTED_SUBJECTS:
        problems.append(
            f"rows = {rows}, expected {EXPECTED_SUBJECTS}"
        )

    if unique_uids != EXPECTED_SUBJECTS:
        problems.append(
            f"unique UIDs = {unique_uids}, "
            f"expected {EXPECTED_SUBJECTS}"
        )

    if normal != EXPECTED_NORMAL:
        problems.append(
            f"normal subjects = {normal}, expected {EXPECTED_NORMAL}"
        )

    if pathologic != EXPECTED_PATHOLOGIC:
        problems.append(
            f"pathologic subjects = {pathologic}, "
            f"expected {EXPECTED_PATHOLOGIC}"
        )

    if problems:
        fail(
            "Base supervised-manifest validation failed:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "rows": rows,
        "unique_uids": unique_uids,
        "missing_uids": missing_uid,
        "missing_labels": missing_labels,
        "duplicate_uids": duplicate_uids,
        "normal": normal,
        "pathologic": pathologic,
        "unique_labels": unique_labels,
    }


def create_fold_assignments(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df[[UID_COLUMN, LABEL_COLUMN]].copy()

    # Stable input ordering makes the frozen output easier to audit.
    work = work.sort_values(UID_COLUMN).reset_index(drop=True)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=SHUFFLE,
        random_state=RANDOM_SEED,
    )

    folds = np.full(len(work), -1, dtype=np.int64)

    X_dummy = np.zeros(len(work), dtype=np.uint8)
    y = work[LABEL_COLUMN].to_numpy()

    for fold_index, (_, val_idx) in enumerate(
        splitter.split(X_dummy, y)
    ):
        folds[val_idx] = fold_index

    if np.any(folds < 0):
        fail("Some subjects did not receive a fold assignment.")

    work["fold"] = folds

    return work


def build_fold_statistics(
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    overall_normal_pct = (
        (assignments[LABEL_COLUMN] == 0).mean() * 100.0
    )

    overall_pathologic_pct = (
        (assignments[LABEL_COLUMN] == 1).mean() * 100.0
    )

    for fold in range(N_SPLITS):
        fold_df = assignments[assignments["fold"] == fold]

        total = len(fold_df)
        normal = int((fold_df[LABEL_COLUMN] == 0).sum())
        pathologic = int((fold_df[LABEL_COLUMN] == 1).sum())

        normal_pct = normal / total * 100.0
        pathologic_pct = pathologic / total * 100.0

        rows.append(
            {
                "fold": fold,
                "total": total,
                "normal": normal,
                "pathologic": pathologic,
                "normal_percent": round(normal_pct, 6),
                "pathologic_percent": round(
                    pathologic_pct,
                    6,
                ),
                "normal_percent_delta_from_overall": round(
                    normal_pct - overall_normal_pct,
                    6,
                ),
                "pathologic_percent_delta_from_overall": round(
                    pathologic_pct - overall_pathologic_pct,
                    6,
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_fold_assignments(
    assignments: pd.DataFrame,
) -> dict:

    problems = []

    rows = len(assignments)
    unique_uids = assignments[UID_COLUMN].nunique()
    duplicate_uids = int(
        assignments[UID_COLUMN].duplicated().sum()
    )
    missing_folds = int(assignments["fold"].isna().sum())

    folds_present = sorted(
        assignments["fold"].astype(int).unique().tolist()
    )

    expected_folds = list(range(N_SPLITS))

    if rows != EXPECTED_SUBJECTS:
        problems.append(
            f"assignment rows = {rows}, expected {EXPECTED_SUBJECTS}"
        )

    if unique_uids != EXPECTED_SUBJECTS:
        problems.append(
            f"unique assignment UIDs = {unique_uids}, "
            f"expected {EXPECTED_SUBJECTS}"
        )

    if duplicate_uids:
        problems.append(
            f"duplicate UID assignments = {duplicate_uids}"
        )

    if missing_folds:
        problems.append(
            f"missing fold assignments = {missing_folds}"
        )

    if folds_present != expected_folds:
        problems.append(
            f"fold values = {folds_present}, "
            f"expected {expected_folds}"
        )

    leakage_results = {}

    all_uids = set(assignments[UID_COLUMN].astype(str))

    validation_seen = set()

    for fold in range(N_SPLITS):
        train_df = assignments[assignments["fold"] != fold]
        val_df = assignments[assignments["fold"] == fold]

        train_uids = set(train_df[UID_COLUMN].astype(str))
        val_uids = set(val_df[UID_COLUMN].astype(str))

        overlap = train_uids & val_uids
        union = train_uids | val_uids

        duplicate_validation = validation_seen & val_uids
        validation_seen |= val_uids

        leakage_results[str(fold)] = {
            "training_subjects": len(train_uids),
            "validation_subjects": len(val_uids),
            "train_validation_uid_overlap": len(overlap),
            "train_validation_union": len(union),
            "duplicate_validation_assignment": len(
                duplicate_validation
            ),
        }

        if overlap:
            problems.append(
                f"Fold {fold}: train/validation UID overlap "
                f"= {len(overlap)}"
            )

        if union != all_uids:
            problems.append(
                f"Fold {fold}: train + validation does not "
                "cover complete UID universe."
            )

        if duplicate_validation:
            problems.append(
                f"Fold {fold}: some validation UIDs were already "
                "assigned to another validation fold."
            )

    if validation_seen != all_uids:
        problems.append(
            "Not every UID appears exactly once as validation subject."
        )

    if problems:
        fail(
            "Fold validation failed:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "rows": rows,
        "unique_uids": int(unique_uids),
        "duplicate_uid_assignments": duplicate_uids,
        "missing_fold_assignments": missing_folds,
        "folds_present": folds_present,
        "leakage_checks": leakage_results,
    }


def validate_scenario_consistency(
    authoritative_uids: set[str],
    scenario_manifests: dict[str, Path],
) -> dict:

    result = {}

    for scenario in SCENARIOS:

        path = scenario_manifests.get(scenario)

        if path is None:
            result[scenario] = {
                "status": "NOT_SEPARATELY_FOUND",
                "note": (
                    "No standalone scenario manifest was automatically "
                    "identified in Step-10 output."
                ),
            }
            continue

        df = read_csv_checked(path)

        if UID_COLUMN not in df.columns:
            fail(
                f"Scenario {scenario} manifest does not contain "
                f"'{UID_COLUMN}': {path}"
            )

        scenario_uids = set(
            df[UID_COLUMN].astype(str)
        )

        missing = authoritative_uids - scenario_uids
        extra = scenario_uids - authoritative_uids

        duplicate_uids = int(
            df[UID_COLUMN].duplicated().sum()
        )

        status = (
            "PASS"
            if not missing
            and not extra
            and duplicate_uids == 0
            else "FAIL"
        )

        result[scenario] = {
            "status": status,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "rows": len(df),
            "unique_uids": len(scenario_uids),
            "duplicate_uids": duplicate_uids,
            "missing_uids": len(missing),
            "extra_uids": len(extra),
        }

        if status != "PASS":
            fail(
                f"Scenario {scenario} UID consistency failed. "
                f"Missing={len(missing)}, "
                f"extra={len(extra)}, "
                f"duplicates={duplicate_uids}"
            )

    return result


def reproducibility_check(
    base_df: pd.DataFrame,
    first_assignments: pd.DataFrame,
) -> bool:

    second_assignments = create_fold_assignments(base_df)

    first = first_assignments.sort_values(
        UID_COLUMN
    ).reset_index(drop=True)

    second = second_assignments.sort_values(
        UID_COLUMN
    ).reset_index(drop=True)

    return first.equals(second)


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:

    print()
    print("=" * 72)
    print("STEP 11 — CREATE & FREEZE 5-FOLD CV SPLITS")
    print("=" * 72)

    if not STEP10_DIR.exists():
        fail(
            "Step-10 input directory does not exist:\n"
            f"{STEP10_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 11A
    # ------------------------------------------------------------------

    print("\n# STEP 11A — READ FROZEN STEP-10 DATASET")

    csv_files = find_csv_files(STEP10_DIR)

    if not csv_files:
        fail(
            f"No CSV files found under Step-10 directory:\n{STEP10_DIR}"
        )

    supervised_manifest = identify_supervised_manifest(csv_files)

    df = read_csv_checked(supervised_manifest)

    validation = validate_base_manifest(df)

    print(
        f"\nSupervised manifest      : "
        f"{supervised_manifest.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Subjects found           : {validation['rows']}"
    )
    print(
        f"Unique UIDs              : {validation['unique_uids']}"
    )
    print(
        f"Normal                   : {validation['normal']}"
    )
    print(
        f"Pathologic               : {validation['pathologic']}"
    )
    print(
        f"Missing labels           : {validation['missing_labels']}"
    )
    print(
        f"Duplicate UIDs           : {validation['duplicate_uids']}"
    )

    # ------------------------------------------------------------------
    # STEP 11B
    # ------------------------------------------------------------------

    print("\n# STEP 11B — VALIDATE SPLITTING UNIT")

    patient_id_candidates = [
        "patient_id",
        "patient",
        "subject_id",
        "participant_id",
    ]

    patient_column = next(
        (
            column
            for column in patient_id_candidates
            if column in df.columns
        ),
        None,
    )

    if patient_column is not None:
        patient_counts = (
            df.groupby(patient_column)[UID_COLUMN]
            .nunique()
        )

        repeated_patients = int(
            (patient_counts > 1).sum()
        )

        if repeated_patients > 0:
            fail(
                f"Detected {repeated_patients} patient groups with "
                f"multiple UIDs using column '{patient_column}'.\n"
                "This dataset requires StratifiedGroupKFold rather than "
                "StratifiedKFold. Do not continue with UID-level splitting."
            )

        splitting_unit = patient_column

        print(
            f"Patient identifier        : {patient_column}"
        )
        print(
            "Repeated patients         : 0"
        )
        print(
            "One patient per UID       : PASS"
        )

    else:
        splitting_unit = UID_COLUMN

        print(
            "Patient identifier        : not available in Step-10 manifest"
        )
        print(
            "Splitting unit            : uid"
        )
        print(
            "Assumption                : one UID = one independent subject"
        )

    split_method = "StratifiedKFold"

    print(
        f"Split method             : {split_method}"
    )

    # ------------------------------------------------------------------
    # STEP 11C
    # ------------------------------------------------------------------

    print("\n# STEP 11C — CREATE 5 FOLDS")

    print(f"Number of folds          : {N_SPLITS}")
    print(f"Shuffle                  : {SHUFFLE}")
    print(f"Random seed              : {RANDOM_SEED}")
    print(f"Stratification target    : {LABEL_COLUMN}")

    assignments = create_fold_assignments(df)

    # ------------------------------------------------------------------
    # STEP 11D + 11E
    # ------------------------------------------------------------------

    print("\n# STEP 11D / 11E — ASSIGNMENTS & CLASS BALANCE")

    statistics = build_fold_statistics(assignments)

    print()
    print(
        statistics[
            [
                "fold",
                "total",
                "normal",
                "pathologic",
                "normal_percent",
                "pathologic_percent",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # STEP 11F
    # ------------------------------------------------------------------

    print("\n# STEP 11F — LEAKAGE CHECKS")

    fold_validation = validate_fold_assignments(
        assignments
    )

    for fold in range(N_SPLITS):
        info = fold_validation["leakage_checks"][str(fold)]

        print(
            f"Fold {fold}: "
            f"train={info['training_subjects']}, "
            f"validation={info['validation_subjects']}, "
            f"UID overlap="
            f"{info['train_validation_uid_overlap']}"
        )

    print("UID leakage             : PASS")

    # ------------------------------------------------------------------
    # STEP 11G
    # ------------------------------------------------------------------

    print("\n# STEP 11G — SCENARIO CONSISTENCY")

    scenario_manifests = identify_scenario_manifests(
        csv_files
    )

    authoritative_uids = set(
        assignments[UID_COLUMN].astype(str)
    )

    scenario_validation = validate_scenario_consistency(
        authoritative_uids,
        scenario_manifests,
    )

    for scenario in SCENARIOS:
        info = scenario_validation[scenario]

        print(
            f"Scenario {scenario:<15}: {info['status']}"
        )

    # ------------------------------------------------------------------
    # STEP 11H
    # ------------------------------------------------------------------

    print("\n# STEP 11H — REPRODUCIBILITY CHECK")

    reproducible = reproducibility_check(
        df,
        assignments,
    )

    if not reproducible:
        fail(
            "Repeated fold generation produced a different assignment."
        )

    print("Repeated split identical : PASS")

    # ------------------------------------------------------------------
    # STEP 11I / 11J
    # ------------------------------------------------------------------

    print("\n# STEP 11I — FREEZE OUTPUTS")

    fold_assignments_path = (
        OUTPUT_DIR / "fold_assignments.csv"
    )

    fold_statistics_path = (
        OUTPUT_DIR / "fold_statistics.csv"
    )

    cv_config_path = (
        OUTPUT_DIR / "cv_config.json"
    )

    validation_report_path = (
        OUTPUT_DIR / "step11_validation_report.json"
    )

    hashes_path = (
        OUTPUT_DIR / "step11_hashes.json"
    )

    assignments.to_csv(
        fold_assignments_path,
        index=False,
    )

    statistics.to_csv(
        fold_statistics_path,
        index=False,
    )

    cv_config = {
        "step": 11,
        "name": "create_and_freeze_5_fold_cv_splits",
        "input_manifest": str(
            supervised_manifest.relative_to(PROJECT_ROOT)
        ),
        "splitting_unit": splitting_unit,
        "patient_identifier_column": patient_column,
        "split_method": split_method,
        "n_splits": N_SPLITS,
        "shuffle": SHUFFLE,
        "random_state": RANDOM_SEED,
        "stratification_column": LABEL_COLUMN,
        "uid_column": UID_COLUMN,
        "label_column": LABEL_COLUMN,
        "total_subjects": int(len(assignments)),
        "normal_count": int(
            (assignments[LABEL_COLUMN] == 0).sum()
        ),
        "pathologic_count": int(
            (assignments[LABEL_COLUMN] == 1).sum()
        ),
        "folds": list(range(N_SPLITS)),
        "frozen_rule": (
            "Do not regenerate folds based on model performance."
        ),
    }

    save_json(cv_config, cv_config_path)

    validation_report = {
        "base_manifest_validation": validation,
        "fold_assignment_validation": fold_validation,
        "scenario_consistency": scenario_validation,
        "reproducibility_check": {
            "repeated_split_identical": reproducible
        },
        "final_validation_passed": True,
    }

    save_json(
        validation_report,
        validation_report_path,
    )

    # Hash everything except the hashes file itself.
    hash_targets = [
        fold_assignments_path,
        fold_statistics_path,
        cv_config_path,
        validation_report_path,
    ]

    hashes = {
        path.name: sha256_file(path)
        for path in hash_targets
    }

    save_json(hashes, hashes_path)

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print(
        f"\nFold assignments         : "
        f"{fold_assignments_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Fold statistics          : "
        f"{fold_statistics_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"CV configuration         : "
        f"{cv_config_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Validation report        : "
        f"{validation_report_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"SHA256 hashes            : "
        f"{hashes_path.relative_to(PROJECT_ROOT)}"
    )

    print("\n# STEP 11 — FINAL VALIDATION")

    print(
        f"Subjects                 : {len(assignments)}"
    )
    print(
        f"Assigned subjects        : "
        f"{assignments[UID_COLUMN].nunique()}"
    )
    print(
        "Missing folds            : 0"
    )
    print(
        "Duplicate assignments    : 0"
    )
    print(
        "UID leakage              : 0"
    )
    print(
        "Class stratification     : PASS"
    )
    print(
        "Reproducibility          : PASS"
    )
    print(
        "SHA256 generated         : PASS"
    )

    print()
    print("Final validation passed  : True")
    print()


if __name__ == "__main__":
    main()
