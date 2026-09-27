from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import pandas as pd

# Canonical internal Step-12 names. The frozen Step-10 manifest itself is not modified.
REQUIRED_MANIFEST_COLUMNS = {"uid", "is_pathologic", "whole_volume_path", "striatal_path"}
STEP10_COLUMN_RENAME_MAP = {
    "whole_path": "whole_volume_path",
    "roi_path": "striatal_path",
}
FOLD_COLUMN_CANDIDATES = ["fold", "fold_id", "cv_fold", "split_fold"]
SPLIT_COLUMN_CANDIDATES = ["split", "set", "subset", "partition"]
VAL_TOKENS = {"val", "valid", "validation", "dev"}


@dataclass(frozen=True)
class FoldSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    fold_column: str


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _normalize_uid(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("Encountered missing UID value.")
    out = series.astype(str).str.strip()
    if out.eq("").any():
        raise ValueError("Encountered empty UID after normalization.")
    return out


def _detect_fold_column(df: pd.DataFrame) -> str:
    found = [c for c in FOLD_COLUMN_CANDIDATES if c in df.columns]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"Ambiguous fold columns: {found}")
    raise ValueError(
        "Could not identify fold column. Expected one of: "
        + ", ".join(FOLD_COLUMN_CANDIDATES)
    )


def load_supervised_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load the frozen Step-10 supervised manifest.

    Frozen Step-10 schema:
        uid
        is_pathologic
        whole_path
        roi_path

    Canonical Step-12 in-memory schema:
        uid
        is_pathologic
        whole_volume_path
        striatal_path

    The Step-10 CSV on disk is never rewritten.
    """
    df = _read_csv(manifest_path).copy()

    if df.empty:
        raise ValueError("Step-10 supervised manifest is empty.")

    # Avoid failures caused by accidental whitespace in CSV headers.
    df.columns = [str(c).strip() for c in df.columns]

    # Refuse ambiguous input rather than silently overwriting a column.
    for source, target in STEP10_COLUMN_RENAME_MAP.items():
        if source in df.columns and target in df.columns:
            raise ValueError(
                f"Manifest contains both '{source}' and '{target}'. "
                "Cannot determine which column is authoritative."
            )

    rename_map = {
        source: target
        for source, target in STEP10_COLUMN_RENAME_MAP.items()
        if source in df.columns
    }
    if rename_map:
        print("Step-10 manifest column mapping:")
        for source, target in rename_map.items():
            print(f"  {source} -> {target}")
        df = df.rename(columns=rename_map)

    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Manifest missing required Step-12 columns after schema mapping: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["uid"] = _normalize_uid(df["uid"])
    if df["uid"].duplicated().any():
        dupes = df.loc[df["uid"].duplicated(keep=False), "uid"].tolist()[:20]
        raise ValueError(f"Duplicate UIDs in supervised manifest, examples: {dupes}")

    numeric_labels = pd.to_numeric(df["is_pathologic"], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("Missing/non-numeric labels found in is_pathologic.")
    labels = set(numeric_labels.unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Invalid labels found. Expected only 0/1, observed: {sorted(labels)}"
        )
    df["is_pathologic"] = numeric_labels.astype(int)

    for path_column in ["whole_volume_path", "striatal_path"]:
        if df[path_column].isna().any():
            raise ValueError(f"Missing values found in {path_column}.")
        df[path_column] = df[path_column].astype(str).str.strip()
        if df[path_column].eq("").any():
            raise ValueError(f"Empty paths found in {path_column}.")

    return df


def _load_combined_fold_csv(path: Path) -> tuple[pd.DataFrame, str]:
    df = _read_csv(path).copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "uid" not in df.columns:
        raise ValueError("Fold assignment CSV must contain a 'uid' column.")
    df["uid"] = _normalize_uid(df["uid"])
    if df["uid"].duplicated().any():
        dupes = df.loc[df["uid"].duplicated(keep=False), "uid"].tolist()[:20]
        raise ValueError(f"Duplicate UIDs in fold assignments, examples: {dupes}")
    fold_col = _detect_fold_column(df)
    df[fold_col] = pd.to_numeric(df[fold_col], errors="raise").astype(int)
    return df[["uid", fold_col]], fold_col


def _fold_index_from_filename(path: Path) -> int | None:
    m = re.search(r"fold[_-]?(\d+)", path.stem.lower())
    return int(m.group(1)) if m else None


def _load_separate_fold_csvs(directory: Path) -> tuple[pd.DataFrame, str]:
    files = sorted(
        p for p in directory.glob("*.csv") if _fold_index_from_filename(p) is not None
    )
    if not files:
        raise FileNotFoundError(f"No fold_*.csv files found in: {directory}")

    assignments: list[pd.DataFrame] = []
    for path in files:
        fold_idx = _fold_index_from_filename(path)
        assert fold_idx is not None
        df = _read_csv(path).copy()
        df.columns = [str(c).strip() for c in df.columns]
        if "uid" not in df.columns:
            raise ValueError(f"Separate fold file lacks uid column: {path}")
        df["uid"] = _normalize_uid(df["uid"])

        # Case 1: file itself contains an authoritative fold column.
        fold_cols = [c for c in FOLD_COLUMN_CANDIDATES if c in df.columns]
        if fold_cols:
            col = fold_cols[0]
            tmp = df[["uid", col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="raise").astype(int)
            # If this file already contains all fold assignments, return it.
            if tmp["uid"].is_unique and len(tmp[col].unique()) > 1:
                tmp = tmp.rename(columns={col: "fold"})
                return tmp, "fold"

        # Case 2: fold_k.csv contains train/validation rows. Keep validation UIDs only.
        split_cols = [c for c in SPLIT_COLUMN_CANDIDATES if c in df.columns]
        if split_cols:
            split_col = split_cols[0]
            values = df[split_col].astype(str).str.strip().str.lower()
            val_mask = values.isin(VAL_TOKENS)
            if not val_mask.any():
                raise ValueError(
                    f"{path} has split column '{split_col}' but no validation token "
                    f"among {sorted(VAL_TOKENS)}"
                )
            uids = df.loc[val_mask, ["uid"]].copy()
        else:
            # Case 3: fold_k.csv contains only validation members.
            uids = df[["uid"]].copy()

        uids["fold"] = fold_idx
        assignments.append(uids)

    combined = pd.concat(assignments, ignore_index=True)
    if combined["uid"].duplicated().any():
        dupes = combined.loc[
            combined["uid"].duplicated(keep=False), "uid"
        ].tolist()[:20]
        raise ValueError(
            "A UID was assigned as validation in multiple separate fold files, examples: "
            + str(dupes)
        )
    return combined, "fold"


def load_fold_assignments(folds_path: Path) -> tuple[pd.DataFrame, str]:
    folds_path = Path(folds_path).expanduser().resolve()
    if folds_path.is_dir():
        return _load_separate_fold_csvs(folds_path)
    return _load_combined_fold_csv(folds_path)


def join_manifest_and_folds(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    fold_col: str,
) -> pd.DataFrame:
    manifest_uids = set(manifest["uid"])
    fold_uids = set(folds["uid"])
    missing_in_folds = sorted(manifest_uids - fold_uids)
    extra_in_folds = sorted(fold_uids - manifest_uids)
    if missing_in_folds or extra_in_folds:
        raise ValueError(
            "UID mismatch between manifest and folds. "
            f"missing_in_folds={len(missing_in_folds)}, "
            f"extra_in_folds={len(extra_in_folds)}; "
            f"missing examples={missing_in_folds[:10]}, "
            f"extra examples={extra_in_folds[:10]}"
        )

    return manifest.merge(folds, on="uid", how="inner", validate="one_to_one")


def split_for_validation_fold(
    merged: pd.DataFrame,
    fold_col: str,
    val_fold: int,
) -> FoldSplit:
    available = sorted(merged[fold_col].unique().tolist())
    if val_fold not in available:
        raise ValueError(
            f"Validation fold {val_fold} is absent. Available folds: {available}"
        )

    val = merged.loc[merged[fold_col] == val_fold].reset_index(drop=True)
    train = merged.loc[merged[fold_col] != val_fold].reset_index(drop=True)
    overlap = set(train["uid"]) & set(val["uid"])
    if overlap:
        raise AssertionError(
            f"Train/validation UID overlap detected: {list(overlap)[:10]}"
        )
    return FoldSplit(train=train, val=val, fold_column=fold_col)


def resolve_existing_path(
    explicit: str | None,
    candidates: Iterable[Path],
    description: str,
    allow_directory: bool = False,
) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        valid = p.exists() if allow_directory else p.is_file()
        if not valid:
            raise FileNotFoundError(f"Explicit {description} path does not exist: {p}")
        return p

    existing = []
    for p in candidates:
        p = Path(p)
        if p.is_file() or (allow_directory and p.is_dir()):
            existing.append(p.resolve())

    # Prefer a concrete file over a directory when both exist.
    file_hits = [p for p in existing if p.is_file()]
    if len(file_hits) == 1:
        return file_hits[0]
    if len(file_hits) > 1:
        raise RuntimeError(
            f"Multiple candidate {description} files found: {file_hits}. "
            "Pass one explicitly."
        )

    dir_hits = [p for p in existing if p.is_dir()]
    if len(dir_hits) == 1:
        return dir_hits[0]
    if not existing:
        raise FileNotFoundError(
            f"Could not auto-resolve {description}. "
            "Pass its path explicitly on the command line."
        )
    raise RuntimeError(
        f"Multiple candidate {description} paths found: {existing}. "
        "Pass one explicitly."
    )
