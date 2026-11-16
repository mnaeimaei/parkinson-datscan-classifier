from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANDIDATES = {
    'foreground_median': 'foreground_median',
    'foreground_middle_25_75_mean': 'foreground_middle_25_75_mean',
    'foreground_trimmed_mean_10_90': 'foreground_trimmed_mean_10_90',
}


def safe_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def describe(values: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(values, errors='coerce').dropna()
    x = x[np.isfinite(x) & (x > 0)]
    if x.empty:
        return {'count': 0}
    mean = float(x.mean()); std = float(x.std(ddof=0)); median = float(x.median())
    return {
        'count': int(len(x)),
        'min': safe_float(x.min()),
        'p05': safe_float(x.quantile(.05)),
        'p25': safe_float(x.quantile(.25)),
        'median': safe_float(median),
        'mean': safe_float(mean),
        'p75': safe_float(x.quantile(.75)),
        'p95': safe_float(x.quantile(.95)),
        'max': safe_float(x.max()),
        'std': safe_float(std),
        'coefficient_of_variation': safe_float(std / mean) if mean else None,
        'max_to_min_ratio': safe_float(x.max() / x.min()) if x.min() > 0 else None,
        'p95_to_p05_ratio': safe_float(x.quantile(.95) / x.quantile(.05)) if x.quantile(.05) > 0 else None,
        'log10_std': safe_float(np.log10(x).std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Step 5A2: evaluate whole-scan intensity statistics as normalization-reference candidates.')
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--histogram-bins', type=int, default=50)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    missing = [col for col in CANDIDATES.values() if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    candidate_stats = {name: describe(df[col]) for name, col in CANDIDATES.items()}

    # Rank only by between-scan relative variability. This is descriptive evidence,
    # not sufficient by itself to declare a biologically valid normalization reference.
    ranking = sorted(
        [
            {'candidate': name, **stats}
            for name, stats in candidate_stats.items()
            if stats.get('coefficient_of_variation') is not None
        ],
        key=lambda x: x['coefficient_of_variation'],
    )

    summary = {
        'analysis_type': 'global_normalization_candidate_evaluation',
        'purpose': 'Evaluate whether a single whole-scan foreground statistic could serve as a normalization reference.',
        'input_csv': str(args.input_csv),
        'number_of_scans': int(len(df)),
        'candidate_statistics': candidate_stats,
        'ranking_by_between_scan_coefficient_of_variation': ranking,
        'interpretation': {
            'lower_variability_is_better_for_scale_stability': True,
            'important_limitation': 'Low variability alone does not prove biological suitability. Global statistics can still depend on field of view, acquisition, reconstruction, and disease-related uptake.',
            'decision': 'This step evaluates global candidates only; anatomical-reference analyses are still required before selecting the final normalization strategy.'
        }
    }
    with (args.output_dir / 'global_normalization_candidate_statistics.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame(ranking).to_csv(args.output_dir / 'global_normalization_candidate_ranking.csv', index=False)

    for name, col in CANDIDATES.items():
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.empty:
            continue
        plt.figure(figsize=(8,5)); plt.hist(vals, bins=args.histogram_bins); plt.xlabel('Candidate normalization scale'); plt.ylabel('Number of scans'); plt.title(name.replace('_',' ').title()); plt.tight_layout(); plt.savefig(args.output_dir / f'{name}_histogram.png', dpi=160); plt.close()

    print('\nSTEP 5A2 COMPLETED')
    for row in ranking:
        print(f"{row['candidate']}: CV={row['coefficient_of_variation']:.6f}, p95/p05={row['p95_to_p05_ratio']:.6f}")
    print('NOTE: ranking by numerical stability does not by itself select the final biological reference.')


if __name__ == '__main__':
    main()
