from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


def find_nifti_files(input_dir: Path) -> list[Path]:
    files = list(input_dir.rglob('*.nii')) + list(input_dir.rglob('*.nii.gz'))
    return sorted(set(files))


def safe_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little', signed=False)


def sample_values(values: np.ndarray, max_samples: int, seed_text: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= max_samples:
        return values
    rng = np.random.default_rng(stable_seed(seed_text))
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def describe(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors='coerce').dropna()
    if values.empty:
        return {k: None for k in ['min','p05','p25','median','mean','p75','p95','max','std','coefficient_of_variation']} | {'count': 0}
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        'count': int(len(values)),
        'min': safe_float(values.min()),
        'p05': safe_float(values.quantile(0.05)),
        'p25': safe_float(values.quantile(0.25)),
        'median': safe_float(values.median()),
        'mean': safe_float(mean),
        'p75': safe_float(values.quantile(0.75)),
        'p95': safe_float(values.quantile(0.95)),
        'max': safe_float(values.max()),
        'std': safe_float(std),
        'coefficient_of_variation': safe_float(std / mean) if mean != 0 else None,
    }


def analyze_scan(path: Path, threshold: float, max_samples: int) -> dict[str, Any]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f'Expected 3D NIfTI, got {image.shape}: {path}')

    data = image.get_fdata(dtype=np.float32)
    finite = np.isfinite(data)
    foreground = finite & (data > threshold)
    values = data[foreground]

    row: dict[str, Any] = {
        'file_name': path.name,
        'input_path': str(path),
        'shape_x': int(image.shape[0]),
        'shape_y': int(image.shape[1]),
        'shape_z': int(image.shape[2]),
        'spacing_x': float(image.header.get_zooms()[0]),
        'spacing_y': float(image.header.get_zooms()[1]),
        'spacing_z': float(image.header.get_zooms()[2]),
        'orientation': ''.join(nib.aff2axcodes(image.affine)),
        'total_voxels': int(data.size),
        'finite_voxels': int(finite.sum()),
        'nan_voxels': int(np.isnan(data).sum()),
        'positive_inf_voxels': int(np.isposinf(data).sum()),
        'negative_inf_voxels': int(np.isneginf(data).sum()),
        'negative_voxels': int(np.count_nonzero(finite & (data < 0))),
        'zero_voxels': int(np.count_nonzero(finite & (data == 0))),
        'foreground_voxels': int(foreground.sum()),
        'foreground_fraction': float(foreground.mean()),
        'background_fraction': float(1.0 - foreground.mean()),
        'valid_foreground': bool(values.size > 0),
    }

    if values.size == 0:
        return row

    sampled = sample_values(values, max_samples, path.name)
    p01,p05,p10,p25,p50,p75,p90,p95,p99 = [float(v) for v in np.percentile(sampled,[1,5,10,25,50,75,90,95,99])]
    middle = sampled[(sampled >= p25) & (sampled <= p75)]
    trimmed = sampled[(sampled >= p10) & (sampled <= p90)]

    row.update({
        'foreground_min': float(values.min()),
        'foreground_p01': p01,
        'foreground_p05': p05,
        'foreground_p10': p10,
        'foreground_p25': p25,
        'foreground_median': p50,
        'foreground_mean': float(values.mean(dtype=np.float64)),
        'foreground_p75': p75,
        'foreground_p90': p90,
        'foreground_p95': p95,
        'foreground_p99': p99,
        'foreground_max': float(values.max()),
        'foreground_std': float(values.std(dtype=np.float64)),
        'foreground_middle_25_75_mean': float(middle.mean()) if middle.size else p50,
        'foreground_trimmed_mean_10_90': float(trimmed.mean()) if trimmed.size else p50,
        'percentile_sample_size': int(sampled.size),
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description='Step 5A1: quantify between-scan intensity-scale variation.')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--foreground-threshold', type=float, default=0.0)
    parser.add_argument('--max-percentile-samples', type=int, default=1_000_000)
    parser.add_argument('--histogram-bins', type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = find_nifti_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f'No NIfTI files found in {args.input_dir}')

    rows, failures = [], []
    for i, path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] Analyzing {path.name}')
        try:
            rows.append(analyze_scan(path, args.foreground_threshold, args.max_percentile_samples))
        except Exception as exc:
            failures.append({'file_name': path.name, 'input_path': str(path), 'error': str(exc)})

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / 'intensity_scale_per_scan.csv'
    df.to_csv(csv_path, index=False)

    columns = [
        'foreground_mean','foreground_median','foreground_std','foreground_p95','foreground_p99','foreground_max',
        'foreground_middle_25_75_mean','foreground_trimmed_mean_10_90','foreground_fraction','background_fraction'
    ]
    stats = {c: describe(df[c]) for c in columns if c in df.columns}
    summary = {
        'analysis_type': 'between_scan_intensity_scale_variation',
        'purpose': 'Quantify how much overall foreground intensity scale varies across scans.',
        'number_of_files': len(files),
        'successful': len(rows),
        'failed': len(failures),
        'foreground_definition': {'rule': 'finite voxel intensity > foreground_threshold', 'foreground_threshold': args.foreground_threshold},
        'intensity_statistics_across_scans': stats,
        'failures': failures,
    }
    with (args.output_dir / 'intensity_scale_variation_statistics.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    for c in ['foreground_mean','foreground_median','foreground_p95','foreground_p99','foreground_trimmed_mean_10_90']:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors='coerce').dropna()
        if vals.empty:
            continue
        plt.figure(figsize=(8,5)); plt.hist(vals, bins=args.histogram_bins); plt.xlabel(c); plt.ylabel('Number of scans'); plt.title(f'{c} across scans'); plt.tight_layout(); plt.savefig(args.output_dir / f'{c}_histogram.png', dpi=160); plt.close()

    print('\nSTEP 5A1 COMPLETED')
    print(f'Scans: {len(files)} | successful: {len(rows)} | failed: {len(failures)}')
    print(f'Per-scan output: {csv_path}')


if __name__ == '__main__':
    main()
