# DaT-SPECT Classification for Parkinsonian Disorder Assessment

**A [CaliVox](https://github.com/mnaeimaei/calivox) use case:** classifying DaT-SPECT brain scans as **normal or pathologic** with calibrated probabilities. The study covers deterministic NIfTI preprocessing, a bilateral striatal ROI, a benchmark of 60 CNN and transformer experiments, cross-fitted calibration and a final five-member ensemble (**ENS328**).

The repository has two layers:

| Layer | What it contains |
| --- | --- |
| `research/` | The original study pipeline, unchanged in behaviour: image preprocessing (metadata, RAS, 2.46 mm resampling, registration-assisted occipital normalisation, striatal localisation and crop), the frozen manifest and CV split, and the original training, calibration, ensemble, validation and runtime scripts. |
| `configs/` + `plugins/` | The experiment layer, expressed as **CaliVox** configs: all 60 experiments (10 models × 6 scenarios), cross-fitted calibration, the ENS328 ensemble and a frozen runtime bundle, each one command. |

CaliVox takes over at the frozen step-10 manifest. Everything DaT-specific before that point (registration, template-based reference region, striatal localisation) stays in `research/`. The one DaT-specific rule needed after it, the localised 2.5D slice window of E4/E7, is a 70-line CaliVox plugin (`plugins/datscan.py`).

---

## Reported results

The figures below come from the original study runs (`research/`, 1,362 subjects, frozen stratified 5-fold CV, out-of-fold predictions). They were **not re-run as part of this restructuring**; see [Reproducing with CaliVox](#reproducing-with-calivox).

| Class | Subjects |
| --- | ---: |
| Normal | 615 |
| Pathologic | 747 |
| **Total** | **1,362** |

**Best single experiment: E6-S5**, Kinetics-400 pretrained R3D-18 on the striatal ROI with augmentation:

| Metric (OOF) | Result |
| --- | ---: |
| Log loss | **0.297122** |
| AUROC | 0.944693 |
| AUPRC | 0.958617 |
| Brier score | 0.087979 |
| ECE | 0.028650 |

Post-hoc calibration did not improve E6-S5. Its uncalibrated probabilities were already the best single-model configuration.

**Final ensemble: ENS328**, member calibration → equal-weight logit mean → final temperature (T = 0.817997185):

| Member | Model | Scenario | Member calibration |
| --- | --- | --- | --- |
| P1 | E6: R3D-18, Kinetics-400 | S5: ROI + aug. | none |
| P2 | E7: 2.5D ResNet-18 + attention, scratch | S4: whole + aug. | temperature |
| P3 | E6: R3D-18, Kinetics-400 | S2: ROI | temperature |
| P7 | E4: 2.5D ResNet-18 + attention, ImageNet | S2: ROI | temperature |
| P8 | E2: 3D ResNet-18, scratch | S2: ROI | temperature |

**Cross-fitted OOF log loss: 0.255582**, with 25 fold checkpoints in the frozen runtime.

Observations from the study, specific to this dataset and setup:
- The striatal ROI was the most valuable input.
- Larger or more complex architectures were not automatically better.
- Whole + ROI late fusion generally underperformed the best single input.
- Swin3D was weak at this data size.
- MedicalNet pretraining helped little.

> **Validation note.** These are cross-fitted / OOF estimates used for model development and **selection** on the same 1,362 labelled subjects. Repeated selection over the same labels is optimistic (winner's curse). ENS328's 0.255582 is a model-selection estimate, not independent external clinical validation. CaliVox's `ensemble` command adds a *nested* estimate (selection repeated inside outer folds), which is the more honest figure to report alongside it.

---

## How it works

```text
research/  (DaT-specific, unchanged)                CaliVox  (configs/)
───────────────────────────────────                 ─────────────────────────────────────────────
raw NIfTI ─▶ metadata ─▶ RAS ─▶ 2.46 mm             calivox benchmark   10 models × 6 scenarios
  ─▶ rigid registration / FP-CIT template             ─▶ 5-fold CV on the frozen folds, OOF logits
  ─▶ occipital-reference normalisation              calivox calibrate   none / temperature /
  ─▶ bilateral striatal localisation + crop           platt / isotonic, 5-fold cross-fitted
  ─▶ step 10 manifest  (uid, is_pathologic,         calivox ensemble    ENS328 (fixed members) or
                        whole_path, roi_path)  ───▶   greedy search + nested estimate
  ─▶ step 11 frozen folds (uid, fold)               calivox freeze      runtime bundle + SHA-256
                                                    calivox verify      bundle == training OOF?
                                                    calivox predict     probabilities
```

| Scenario | Input | Augmentation |
| --- | --- | --- |
| S1 | Whole volume (192 × 192 × 160) | no |
| S2 | Striatal ROI (44 × 44 × 36) | no |
| S3 | Whole + ROI (late fusion) | no |
| S4 | Whole volume | yes |
| S5 | Striatal ROI | yes |
| S6 | Whole + ROI | yes |

| Experiment | Model | Initialisation | Config |
| --- | --- | --- | --- |
| E1 | Simple3D (~3.6 M parameters) | scratch | `datscan_3d.yaml` |
| E2 | MONAI 3D ResNet-18 | scratch | `datscan_3d.yaml` |
| E3 | MONAI 3D ResNet-18 | MedicalNet / Med3D | `datscan_3d.yaml` |
| E4 | 2D ResNet-18 + slice attention (2.5D) | ImageNet | `datscan_2p5d.yaml` |
| E5 | torchvision R3D-18 | scratch | `datscan_3d.yaml` |
| E6 | torchvision R3D-18 | Kinetics-400 | `datscan_3d.yaml` |
| E7 | 2D ResNet-18 + slice attention (2.5D) | scratch | `datscan_2p5d.yaml` |
| E8 | MONAI 3D DenseNet-121 | scratch | `datscan_3d.yaml` |
| E9 | torchvision Swin3D-Tiny | scratch | `datscan_3d.yaml` |
| E10 | torchvision Swin3D-Tiny | Kinetics-400 | `datscan_3d.yaml` |

The configs carry over the original settings from `research/src/configs/training_config.py` and `research/src/augmentation/augmentations.py`:
- **Optimisation:** AdamW, lr 1e-4, weight decay 1e-4; ReduceLROnPlateau 0.5 / 5, min lr 1e-7; up to 100 epochs.
- **Stopping and clipping:** early stopping after 15 epochs with a 5-epoch warm-up; gradient clipping 1.0; seed 2026.
- **Loss:** BCE with logits on a single logit.
- **Batch sizes:** the original per-model, per-input values.
- **Network details:** ResNet shortcut type A; head dropout 0.3 for E1/E4/E7; fusion head dropout 0.0.
- **Augmentation:** 3D for the 3D models, in-plane 2D for E4/E7, training only.
- **Validation:** best checkpoint chosen on the held-out fold, as in the original protocol.

---

## Getting started

```bash
git clone https://github.com/mnaeimaei/parkinson-datscan-classifier
cd parkinson-datscan-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CaliVox + the research pipeline's dependencies
make test                              # use-case tests (no patient data needed)
```

Patient data is **not** part of this repository: the `research/data/` folders contain only placeholders. Place the raw scans and labels where `research/README.md` describes, then run preprocessing steps 1–11 of the research pipeline. This produces the step-10 manifest and the step-11 frozen folds that the CaliVox configs read.

For E3, download the MedicalNet `resnet_18_23dataset.pth` into `weights/`. The Kinetics-400 and ImageNet weights are fetched by torchvision on first use.

## Reproducing with CaliVox

```bash
# one experiment (writes runs/E6__S5/: config, folds, 5 checkpoints, oof.csv, metrics.json)
calivox cv configs/datscan_3d.yaml --model E6 --scenario S5 -v

# the full 60-experiment grid
calivox benchmark configs/datscan_3d.yaml   --skip-existing --write runs/benchmark_3d.md
calivox benchmark configs/datscan_2p5d.yaml --skip-existing --write runs/benchmark_2p5d.md

# 240 calibration configurations (60 runs × {none, temperature, platt, isotonic}), cross-fitted
calivox calibrate runs --write runs/calibrated.md

# the ENS328 ensemble from its fixed member list (+ nested estimate) ...
calivox ensemble runs --members configs/ens328.yaml --out runs/ENS328.json
# ... or a fresh greedy search over the shortlist
calivox ensemble runs --out runs/greedy.json

# frozen runtime bundle, equivalence check, inference
calivox freeze runs/ENS328.json --out bundles/ENS328
calivox verify bundles/ENS328 --manifest <step-10 manifest with absolute paths>
calivox predict bundles/ENS328 --manifest new_scans.csv --out submission.csv
```

`make` targets wrap these commands (`make benchmark`, `make calibrate`, `make ens328`, `make bundle`).

`new_scans.csv` lists `uid, whole_path, roi_path` for scans already preprocessed by the research pipeline (steps 1–9; the original runtime in `research/scripts/final_runtime_bundle_script/` does this for raw NIfTI). The bundle then applies the slice window, the 25 fold models, member calibration and the final temperature.

**What to expect.** Results from these configs should be close to the reported ones, not bit-identical. The training loop is re-implemented in CaliVox, and GPU non-determinism differs between runs. To reproduce the reported numbers exactly, use the original scripts in `research/`.

### Original scripts → CaliVox

| Original (research/scripts/…) | CaliVox |
| --- | --- |
| `experiments_script/model0X_*/scenario0Y_*.py` (60 files) + `common_cv_experiment.py` | `calivox benchmark configs/datscan_*.yaml` |
| `experiments_script/two_point_five_d_input.py` | `plugins/datscan.py` → step `roi_slice_window` |
| `calibration_validation_script/` (cross-fitted calibration, 240 configurations) | `calivox calibrate runs` |
| `competition_ensemble_analysis_script/` | `calivox ensemble runs` (greedy + nested) |
| `final_predictor_freeze_script/`, `final_runtime_bundle_script/` | `calivox freeze` |
| `generalization_validation_script/step25a_*`, `step31d_*` (runtime equivalence) | `calivox verify` |
| `generalization_validation_script/` (nested ensemble validation) | `calivox ensemble` nested estimate |

Steps 1–13, the domain-stress and robust-augmentation experiments, and the statistics scripts have no CaliVox equivalent and remain in `research/`.

---

## Project structure

```text
parkinson-datscan-classifier/
├── configs/
│   ├── datscan_3d.yaml        E1–E3, E5, E6, E8–E10 × S1–S6
│   ├── datscan_2p5d.yaml      E4, E7 × S1–S6 (localised slice window, in-plane augmentation)
│   └── ens328.yaml            the final ensemble's fixed member list
├── plugins/
│   └── datscan.py             roi_slice_window: E4/E7 whole-volume window around the striatum
├── research/                  the original study pipeline (see research/README.md)
│   ├── data/                  placeholders only — no patient data in the repository
│   ├── scripts/               steps 1–35: preprocessing, experiments, calibration, ensembles, runtime
│   └── src/                   models, training, registration, calibration, evaluation, inference
├── tests/                     use-case tests on synthetic volumes in the step-10/11 formats
├── weights/                   (not tracked) MedicalNet weights for E3
├── runs/  cache/  bundles/    (not tracked) CaliVox outputs
├── Makefile  requirements.txt  LICENSE
└── README.md
```

## Tests

```bash
make test
```

The tests write a tiny fake step-10/step-11 tree of synthetic volumes in exactly the research pipeline's formats; no patient data is used. They check:
- Both configs load with the original settings (models, scenarios, batch sizes, held-out validation, in-plane augmentation for 2.5D).
- `roi_slice_window` reproduces the original rule: centred, shifted at the edges, failing loudly when the ROI doesn't fit.
- A 1-epoch CV run of a 3D, a fused and a 2.5D experiment uses the frozen folds.
- A 2.5D bundle carries the plugin and reproduces its training-time OOF logits.

## Notes on the restructuring

- **Nothing is removed.** `research/` holds the original `src/`, `scripts/`, `data/` placeholders and README.
- **One fix in `research/`.** `step16b_create_competition_shortlist_scripts.py` had a hard-coded cluster path (`/cluster/home/…`). It now uses `$DATSCAN_PROJECT_ROOT`, or the repository root derived from the script's location like the other steps.
- **Stale docstring.** The scenario scripts of E4 (`model04_*/scenario0*.py`) describe the whole-volume input as "40 uniformly spaced axial slices", but the shared `two_point_five_d_input.py` uses 40 consecutive slices centred on the striatal ROI. The plugin follows the shared module.
- **Checkpoint selection.** The original protocol chooses the best epoch on the held-out fold (`cv.validation: heldout`), which makes OOF metrics slightly optimistic. Setting `cv.validation: inner` selects on a split of the training folds instead and leaves the held-out fold fully unseen.

## Tech stack

Python · PyTorch · torchvision · MONAI · nibabel · SimpleITK · scikit-learn · SciPy · pandas · CaliVox · SLURM / HPC (original runs)

## License

MIT; see [LICENSE](LICENSE). The DaT-SPECT data, MedicalNet, Kinetics-400 and ImageNet weights are subject to their own terms.
