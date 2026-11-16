"""Checks that the CaliVox configs of this repository load and run.

No patient data is used: a tiny fake step-10/step-11 tree is written with synthetic volumes
in exactly the formats of the research pipeline (uid, is_pathologic, whole_path, roi_path;
uid, fold), with the whole volume and the ROI crop sharing world coordinates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins"))

from datscan import localized_window  # noqa: E402

from calivox.config import ExperimentConfig  # noqa: E402

WHOLE = (32, 32, 48)      # small stand-in for 192 x 192 x 160
ROI = (12, 12, 10)        # small stand-in for 44 x 44 x 36


def _affine(spacing=2.46):
    a = np.diag([spacing, spacing, spacing, 1.0])
    a[:3, 3] = [-40.0, -40.0, -60.0]
    return a


def roi_affine(whole_affine, offset):
    a = whole_affine.copy()
    a[:3, 3] = whole_affine[:3, :3] @ np.asarray(offset, float) + whole_affine[:3, 3]
    return a


@pytest.fixture(scope="module")
def fake_tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("research")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        uid = f"U{i:03d}"
        label = int(i % 2)
        whole = rng.random(WHOLE).astype(np.float32)
        off = (10, 10, int(rng.integers(14, 24)))
        whole[off[0]:off[0] + ROI[0], off[1]:off[1] + ROI[1], off[2]:off[2] + ROI[2]] += 3.0 - 1.5 * label
        aff = _affine()
        wp, rp = Path("data/frozen/whole") / f"{uid}.nii.gz", Path("data/frozen/roi") / f"{uid}.nii.gz"
        (root / wp).parent.mkdir(parents=True, exist_ok=True)
        (root / rp).parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(whole, aff), str(root / wp))
        crop = whole[off[0]:off[0] + ROI[0], off[1]:off[1] + ROI[1], off[2]:off[2] + ROI[2]]
        nib.save(nib.Nifti1Image(np.ascontiguousarray(crop), roi_affine(aff, off)), str(root / rp))
        rows.append({"uid": uid, "is_pathologic": label, "whole_path": str(wp), "roi_path": str(rp), "roi_z0": off[2]})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame({"uid": [r["uid"] for r in rows], "fold": [i % 5 for i in range(20)]}).to_csv(root / "folds.csv", index=False)
    return root


def config(name: str, tree: Path, tmp: Path) -> ExperimentConfig:
    raw = yaml.safe_load((ROOT / "configs" / name).read_text())
    raw["data"].update(manifest=str(tree / "manifest.csv"), root=str(tree))
    raw["cv"]["folds_file"] = str(tree / "folds.csv")
    raw["training"].update(epochs=1, num_workers=0, amp=False, device="cpu", num_threads=1)
    raw["output_dir"], raw["cache_dir"] = str(tmp / "runs"), str(tmp / "cache")
    raw["plugins"] = [str(ROOT / "plugins" / "datscan.py")] if raw.get("plugins") else []
    for m in raw["models"].values():
        m.pop("pretrained", None)             # offline: no weight downloads in tests
        m.pop("batch_size", None)             # tiny fake volumes: batch 1 would break BatchNorm
    raw["training"]["batch_size"] = 4
    path = tmp / name
    path.write_text(yaml.safe_dump(raw))
    return ExperimentConfig.from_yaml(path)


# ---- the 2.5D window rule ---------------------------------------------------------

def test_window_centred_on_roi_and_shifted_at_edges():
    a = _affine()
    assert localized_window(a, (192, 192, 160), roi_affine(a, (74, 74, 62)), (44, 44, 36)) == (60, 100)
    assert localized_window(a, (192, 192, 160), roi_affine(a, (74, 74, 0)), (44, 44, 36)) == (0, 40)
    assert localized_window(a, (192, 192, 160), roi_affine(a, (74, 74, 124)), (44, 44, 36)) == (120, 160)


def test_window_fails_loudly():
    a = _affine()
    with pytest.raises(ValueError, match="outside"):
        localized_window(a, (192, 192, 160), roi_affine(a, (74, 74, 140)), (44, 44, 36))
    with pytest.raises(ValueError, match="wider"):
        localized_window(a, (192, 192, 160), roi_affine(a, (74, 74, 10)), (44, 44, 50))


# ---- configs ----------------------------------------------------------------------

def test_configs_load_with_original_settings():
    c3 = ExperimentConfig.from_yaml(ROOT / "configs" / "datscan_3d.yaml")
    c25 = ExperimentConfig.from_yaml(ROOT / "configs" / "datscan_2p5d.yaml")
    assert set(c3.models) == {"E1", "E2", "E3", "E5", "E6", "E8", "E9", "E10"} and set(c25.models) == {"E4", "E7"}
    assert set(c3.scenarios) == set(c25.scenarios) == {f"S{i}" for i in range(1, 7)}
    e6 = c3.for_run("E6", "S5")
    assert e6.model.pretrained == "kinetics400" and e6.training.batch_size == 2 and e6.scenario.augment
    assert c3.for_run("E1", "S2").training.batch_size == 8 and c3.for_run("E1", "S3").training.batch_size == 1
    assert c3.cv.validation == "heldout" and c3.training.learning_rate == 1e-4 and c3.training.epochs == 100
    assert c25.augment.spatial_mode == "inplane_2d" and c3.augment.spatial_mode == "3d"
    spec = yaml.safe_load((ROOT / "configs" / "ens328.yaml").read_text())
    assert [m["run"] for m in spec["members"]] == ["E6__S5", "E7__S4", "E6__S2", "E4__S2", "E2__S2"]


def test_2p5d_whole_input_is_the_localised_window(fake_tree, tmp_path):
    from calivox.data import load_array, load_manifest, preprocess_inputs

    cfg = config("datscan_2p5d.yaml", fake_tree, tmp_path)
    man = load_manifest(cfg)
    paths = preprocess_inputs(cfg, man, ["whole", "roi"])
    df = pd.read_csv(fake_tree / "manifest.csv")
    for i in (0, 7):
        w = load_array(paths["whole"][i], "z")
        assert w.shape == (40, 32, 32)                        # 40 slices, (D, H, W)
        z0 = int(df.loc[i, "roi_z0"])
        start, _ = localized_window(_affine(), WHOLE, roi_affine(_affine(), (10, 10, z0)), ROI, 40)
        full = nib.load(str(fake_tree / df.loc[i, "whole_path"])).get_fdata()
        assert np.allclose(w, np.transpose(full[:, :, start:start + 40], (2, 1, 0)))
    assert load_array(paths["roi"][0], "z").shape == (10, 12, 12)  # ROI used as frozen


@pytest.mark.parametrize("name,model,scenario", [("datscan_3d.yaml", "E1", "S2"),
                                                 ("datscan_3d.yaml", "E2", "S6"),
                                                 ("datscan_2p5d.yaml", "E7", "S4")])
def test_one_epoch_cv_uses_frozen_folds(fake_tree, tmp_path, name, model, scenario):
    from calivox.training import cross_validate

    cfg = config(name, fake_tree, tmp_path).for_run(model, scenario)
    res = cross_validate(cfg, tmp_path / "runs" / f"{model}__{scenario}")
    assert len(res.oof) == 20 and res.oof["logit"].notna().all()
    folds = pd.read_csv(fake_tree / "folds.csv")
    assert (res.oof.set_index("id")["fold"] == folds.set_index("uid")["fold"]).all()
    assert res.metrics["validation"] == "heldout"


def test_2p5d_bundle_carries_the_plugin(fake_tree, tmp_path):
    from calivox.bundle import BundlePredictor, freeze
    from calivox.training import cross_validate

    cfg = config("datscan_2p5d.yaml", fake_tree, tmp_path).for_run("E7", "S1")
    run = cross_validate(cfg, tmp_path / "runs" / "E7__S1").run_dir
    bundle = freeze(run, tmp_path / "bundle")
    assert (bundle / "assets" / "datscan.py").exists()
    df = pd.read_csv(fake_tree / "manifest.csv")
    for c in ("whole_path", "roi_path"):
        df[c] = [str(fake_tree / p) for p in df[c]]
    df.to_csv(tmp_path / "absolute.csv", index=False)
    assert BundlePredictor(bundle).verify(tmp_path / "absolute.csv")["all_equivalent"]
