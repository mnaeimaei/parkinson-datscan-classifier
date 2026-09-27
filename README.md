# DaT-SPECT Classification for Parkinsonian Disorder Assessment

An end-to-end deep-learning pipeline for classifying **DaT-SPECT brain scans as normal or pathologic**, with a focus on robust preprocessing, striatal-region analysis, CNN and transformer benchmarking, probability calibration, and ensemble inference.

The project was developed as a competition-oriented medical imaging workflow and evaluates multiple image representations, training strategies, pretrained models, and calibration methods under a fixed cross-validation framework.

## Project Highlights

- **1,362 DaT-SPECT subjects**
  - 615 normal
  - 747 pathologic
- Deterministic preprocessing of NIfTI volumes
- Registration-assisted intensity normalization using an occipital reference region
- Automatic bilateral striatal ROI localization and cropping
- Whole-volume, ROI-only, and dual-input classification scenarios
- **60 controlled deep-learning experiments**
- 2D, 2.5D, and 3D CNN architectures
- 3D vision transformer experiments
- Scratch and pretrained model comparisons
- Stratified **5-fold cross-validation**
- Out-of-fold (OOF) evaluation
- Cross-fitted probability calibration
- Ensemble model selection and frozen inference pipeline
- Final selected ensemble: **ENS328**

---

## Classification Task

The pipeline performs binary classification of DaT-SPECT scans:

```text
DaT-SPECT scan
      |
      v
Deep-learning pipeline
      |
      v
Probability of pathologic scan
```

The model produces a **continuous probability** rather than only a hard class label. This is important because the competition objective is based primarily on **Log Loss**, which evaluates both classification correctness and probability quality.

---

## Dataset

The frozen supervised dataset contains **1,362 unique subjects**:

| Class | Subjects |
|---|---:|
| Normal | 615 |
| Pathologic | 747 |
| **Total** | **1,362** |

The initial training configuration uses:

- no oversampling
- no class weighting
- no artificial balancing

All classification scenarios use the same frozen subject identities and the same cross-validation assignments.

---

## End-to-End Pipeline

```text
Raw DaT-SPECT NIfTI
        |
        v
Read metadata
        |
        v
Reorient to RAS
        |
        v
Resample to 2.46 x 2.46 x 2.46 mm
        |
        v
Rigid registration / reference localization
        |
        v
Occipital-reference intensity normalization
        |
        v
Bilateral striatal localization
        |
        +-----------------------+
        |                       |
        v                       v
Whole volume               Striatal ROI
192 x 192 x 160            44 x 44 x 36
        |                       |
        +-----------+-----------+
                    |
                    v
          Classification scenarios
                    |
                    v
            Deep-learning models
                    |
                    v
            5-fold OOF prediction
                    |
                    v
          Probability calibration
                    |
                    v
              Ensemble model
                    |
                    v
          Pathologic probability
```

---

## Image Preprocessing

### 1. Metadata Analysis

For every NIfTI scan, the preprocessing pipeline examines:

- image dimensions
- voxel spacing
- anatomical orientation
- datatype
- affine information

### 2. Orientation Standardization

All scans are converted to a consistent **RAS** anatomical orientation:

- **R** — Right
- **A** — Anterior
- **S** — Superior

### 3. Voxel Resampling

The scans are resampled to a common isotropic spacing:

```text
2.46 x 2.46 x 2.46 mm
```

using trilinear interpolation.

### 4. Reference-Region Normalization

Several normalization strategies were investigated, including global image statistics and anatomical reference regions.

The final pipeline uses:

- rigid registration
- FP-CIT template localization
- occipital cortex as the reference region
- mean occipital intensity as the primary normalization statistic

The normalized image is calculated as:

```text
Normalized image = Resampled image / Final reference value
```

Final normalization processing succeeded for all **1,362 scans**.

### 5. Bilateral Striatal ROI

The striatal region is localized using the frozen registration/template-based localization pipeline.

The final ROI size is:

```text
44 x 44 x 36 voxels
```

Crops were generated and automatically quality-checked for all 1,362 scans.

### 6. Fixed Whole-Volume Input

Whole scans are converted to a fixed model input using center cropping and zero padding.

Final whole-volume shape:

```text
192 x 192 x 160 voxels
```

---

## Classification Scenarios

Three input representations are evaluated both **without** and **with** data augmentation.

| Scenario | Input | Augmentation |
|---|---|---|
| S1 | Whole volume | No |
| S2 | Striatal ROI | No |
| S3 | Whole volume + ROI | No |
| S4 | Whole volume | Yes |
| S5 | Striatal ROI | Yes |
| S6 | Whole volume + ROI | Yes |

This produces six scenarios for each model configuration.

---

## Model Benchmark

Ten model configurations are evaluated across CNN and transformer families.

| Experiment | Model | Family | Initialization |
|---|---|---|---|
| E1 | Simple3D (~3.6M parameters) | 3D CNN | Scratch |
| E2 | MONAI 3D ResNet-18 | 3D CNN | Scratch |
| E3 | MedicalNet 3D ResNet-18 | 3D CNN | MedicalNet / Med3D pretrained |
| E4 | 2D ResNet-18 + slice attention | 2.5D CNN | ImageNet pretrained |
| E5 | Torchvision R3D-18 | 3D CNN | Scratch |
| E6 | Torchvision R3D-18 | 3D CNN | Kinetics-400 pretrained |
| E7 | 2D ResNet-18 + slice attention | 2.5D CNN | Scratch |
| E8 | MONAI 3D DenseNet-121 | 3D CNN | Scratch |
| E9 | Torchvision Swin3D-Tiny | Transformer | Scratch |
| E10 | Torchvision Swin3D-Tiny | Transformer | Kinetics-400 pretrained |

With:

```text
10 model configurations
x 3 input representations
x augmentation OFF / ON
= 60 experiments
```

---

## Data Augmentation

Augmentation is applied **only to training data**.

The investigated transformations include:

- small rotations
- small translations
- small scaling
- slight intensity scaling
- Gaussian noise
- minor smoothing

Aggressive transformations that may distort striatal morphology are avoided.

The augmented and non-augmented configurations are evaluated independently.

---

## Cross-Validation

The project uses a frozen **stratified 5-fold cross-validation** split.

```text
1,362 subjects
      |
      +-- Fold 0
      +-- Fold 1
      +-- Fold 2
      +-- Fold 3
      +-- Fold 4
```

For every experiment, five independent models are trained.

Each model:

1. trains on four folds,
2. validates on the remaining fold,
3. starts from a fresh initialization appropriate to that experiment,
4. saves the best checkpoint,
5. generates predictions for subjects not used to train that fold model.

The same fold assignments are used across all classification scenarios to enable controlled comparison.

---

## Training

The binary classification models use a single output logit.

### Loss

```python
torch.nn.BCEWithLogitsLoss()
```

During training:

```text
Model -> raw logit -> BCEWithLogitsLoss
```

During inference:

```text
Model -> raw logit -> sigmoid -> probability
```

### Training Procedure

Each fold includes:

- epoch-based training
- validation after every epoch
- metric tracking
- checkpointing
- early stopping
- reloading of the best checkpoint
- generation of held-out predictions

The best checkpoint is selected using validation loss.

---

## Out-of-Fold Evaluation

After training all five folds, the held-out predictions are concatenated.

```text
Fold 0 validation predictions
Fold 1 validation predictions
Fold 2 validation predictions
Fold 3 validation predictions
Fold 4 validation predictions
              |
              v
   1,362 unique OOF predictions
```

Each subject therefore receives a prediction from a model that was **not trained on that subject**.

### Evaluation Metrics

The primary competition-oriented metric is:

- **Log Loss**

Supporting metrics include:

- AUROC
- AUPRC / Average Precision
- Brier Score
- Expected Calibration Error (ECE)
- Balanced Accuracy
- Sensitivity
- Specificity
- Precision
- F1
- Confusion matrix

Log Loss is emphasized because the final output is a probability and overconfident incorrect predictions receive a large penalty.

---

## Results Across 60 Experiments

The strongest uncalibrated single experiment was:

### E6-S5

**Kinetics-400 pretrained R3D-18 + striatal ROI + augmentation**

| Metric | OOF Result |
|---|---:|
| Log Loss | **0.297122** |
| AUROC | **0.944693** |
| AUPRC | **0.958617** |
| Brier Score | **0.087979** |
| ECE | **0.028650** |

The experiment analysis also found, for this dataset and evaluation setup, that:

- striatal ROI localization was highly valuable;
- larger or more complex architectures were not automatically better;
- whole-volume + ROI late fusion generally underperformed the strongest single-input configurations;
- Swin3D models were weak under the evaluated dataset size and training regime;
- MedicalNet pretraining provided limited benefit;
- Kinetics-400 pretrained R3D-18 with ROI augmentation produced the strongest single-model competition-oriented result.

These findings are specific to the experiments performed in this project and should not be interpreted as general conclusions about the architectures.

---

## Probability Calibration

Because the system outputs probabilities, calibration is evaluated explicitly.

For each of the 60 experiments, four alternatives are compared:

1. no calibration
2. temperature scaling
3. Platt scaling
4. isotonic regression

The calibration evaluation is performed using **5-fold cross-fitting** so that a calibrator is not evaluated on the same subjects used to fit it.

```text
60 experiments
x 3 fitted calibration methods
= 180 fitted configurations

+ 60 uncalibrated baselines
= 240 evaluated configurations
```

The primary calibration-selection metric is again **cross-fitted OOF Log Loss**.

The best E6-S5 model did not benefit from additional post-hoc calibration; its uncalibrated probabilities remained the strongest single-model configuration.

---

## Final Ensemble — ENS328

After model shortlisting, calibration analysis, ensemble search, and cross-fitted ensemble evaluation, the final selected predictor is:

```text
ENS328
```

It combines five complementary members:

| Member | Model | Scenario | Member Calibration |
|---|---|---|---|
| P1 | E6 — R3D-18, Kinetics-400 pretrained | S5 — ROI + augmentation | None |
| P2 | E7 — 2.5D ResNet-18 + attention, scratch | S4 — Whole + augmentation | Temperature |
| P3 | E6 — R3D-18, Kinetics-400 pretrained | S2 — ROI, no augmentation | Temperature |
| P7 | E4 — 2.5D ResNet-18 + attention, ImageNet pretrained | S2 — ROI, no augmentation | Temperature |
| P8 | E2 — 3D ResNet-18, scratch | S2 — ROI, no augmentation | Temperature |

The ensemble uses:

```text
Member predictions
       |
       v
Member-level calibration
       |
       v
Equal-weight logit mean
       |
       v
Final temperature scaling
       |
       v
Final probability
```

The selected ENS328 configuration achieved:

```text
Cross-Fitted OOF Log Loss: 0.255582
```

The frozen deployment ensemble uses a final temperature of:

```text
T = 0.817997185
```

---

## Final Inference Pipeline

The frozen deployment path reproduces training-time preprocessing before inference:

```text
Unseen NIfTI scan
      |
      v
Orientation standardization
      |
      v
2.46-mm isotropic resampling
      |
      v
Reference-region localization
      |
      v
Intensity normalization
      |
      v
Striatal ROI localization
      |
      +-----------------------+
      |                       |
      v                       v
Whole-volume input         ROI input
      |                       |
      +-----------+-----------+
                  |
                  v
         ENS328 model members
                  |
                  v
       5 fold checkpoints/member
                  |
                  v
        Fold averaging/member
                  |
                  v
       Member-level calibration
                  |
                  v
        Equal-weight logit mean
                  |
                  v
     Final temperature calibration
                  |
                  v
      Continuous probability
                  |
                  v
          submission.csv
```

ENS328 contains five members with five fold-specific checkpoints each, resulting in **25 selected fold checkpoints** in the frozen runtime.

---

## Project Structure

The repository is organized into three main layers:

- `data/` — raw data, preprocessing outputs, experiment results, validation artifacts, and templates
- `scripts/` — executable pipeline steps for preprocessing, training, calibration, ensemble analysis, validation, and deployment
- `src/` — reusable Python modules for models, training, registration, calibration, evaluation, and inference

A simplified view of the project structure is shown below:

```text
.
├── data/
│   ├── raw_images_data/
│   ├── raw_label_data/
│   ├── smoke_test_images_data/
│   ├── smoke_test_label_data/
│   │
│   ├── preprocessing_image_data/
│   │   ├── step1_metadata_reader_data/
│   │   ├── step2_orientation_standardizer_data/
│   │   ├── step3_voxel_spacing_analyzer_data/
│   │   ├── step4_voxel_resampler_data/
│   │   ├── step5*_reference_and_normalization_data/
│   │   ├── step6*_registration_and_normalization_data/
│   │   ├── step7*_striatal_localization_data/
│   │   ├── step8_bilateral_striatal_crop_data/
│   │   └── step9_classification_inputs_data/
│   │
│   ├── preprocessing_supervised_data/
│   │   ├── step10_supervised_dataset_manifest_data/
│   │   ├── step11_create_freeze_cv_splits_data/
│   │   ├── step12_build_model_input_pipeline_data/
│   │   └── step13_final_pretraining_validation_data/
│   │
│   ├── experiments_data/
│   ├── experiment_selection_data/
│   ├── experiment_selection_validation_data/
│   ├── generalization_validation_data/
│   ├── inference_data/
│   ├── splits/
│   │
│   └── template/
│       ├── aal/
│       └── dat_spect/
│
├── data-demo/
│
├── scripts/
│   ├── preprocessing_image_script/
│   │   ├── step1_metadata_reader_script/
│   │   ├── step2_orientation_standardizer_script/
│   │   ├── step3_voxel_spacing_analyzer_script/
│   │   ├── step4_voxel_resampler_script/
│   │   ├── step5*_reference_and_normalization_scripts/
│   │   ├── step6*_registration_and_normalization_scripts/
│   │   ├── step7*_striatal_localization_scripts/
│   │   ├── step8_bilateral_striatal_crop_script/
│   │   └── step9_classification_inputs_script/
│   │
│   ├── preprocessing_supervised_script/
│   │   ├── step10_supervised_dataset_manifest_script/
│   │   ├── step11_create_freeze_cv_splits_script/
│   │   ├── step12_build_model_input_pipeline_script/
│   │   └── step13_final_pretraining_validation_script/
│   │
│   ├── experiments_script/
│   │   ├── common_cv_experiment.py
│   │   ├── model01_simple3d_scratch_exp_script/
│   │   ├── model02_resnet18_3d_scratch_exp_script/
│   │   ├── model03_resnet18_3d_medicalnet_pretrained_exp_script/
│   │   ├── model04_resnet18_2p5d_attention_imagenet_pretrained_exp_script/
│   │   ├── model05_r3d18_scratch_exp_script/
│   │   ├── model06_r3d18_kinetics400_pretrained_exp_script/
│   │   ├── model07_resnet18_2p5d_attention_scratch_exp_script/
│   │   ├── model08_densenet121_3d_scratch_exp_script/
│   │   ├── model09_swin3d_t_scratch_exp_script/
│   │   └── model10_swin3d_t_kinetics400_pretrained_exp_script/
│   │
│   ├── experiment_selection_script/
│   ├── calibration_validation_script/
│   ├── post_calibration_stability_script/
│   ├── post_calibration_statistics_script/
│   ├── competition_ensemble_analysis_script/
│   ├── ensemble_calibration_statistics_script/
│   │
│   ├── final_predictor_freeze_script/
│   ├── final_submission_preflight_script/
│   ├── final_runtime_bundle_script/
│   │
│   └── generalization_validation_script/
│       ├── domain-stress validation
│       ├── nested ensemble validation
│       ├── robust augmentation experiments
│       ├── runtime-equivalence checks
│       └── ENS328R validation / packaging
│
└── src/
    ├── augmentation/
    │   └── augmentations.py
    │
    ├── calibration/
    │   └── probability_calibration.py
    │
    ├── configs/
    │   └── training_config.py
    │
    ├── evaluation/
    │   ├── aggregate_cv_results.py
    │   ├── evaluate.py
    │   └── metrics.py
    │
    ├── inference/
    │   ├── inference.py
    │   └── final_ensemble.py
    │
    ├── models/
    │   ├── dual_input_fusion.py
    │   ├── model01_simple3d_scratch.py
    │   ├── model02_resnet18_3d_scratch.py
    │   ├── model03_resnet18_3d_medicalnet_pretrained.py
    │   ├── model04_resnet18_2p5d_attention_imagenet_pretrained.py
    │   ├── model05_r3d18_scratch.py
    │   ├── model06_r3d18_kinetics400_pretrained.py
    │   ├── model07_resnet18_2p5d_attention_scratch.py
    │   ├── model08_densenet121_3d_scratch.py
    │   ├── model09_swin3d_t_scratch.py
    │   └── model10_swin3d_t_kinetics400_pretrained.py
    │
    ├── registration/
    │   ├── affine_registrar.py
    │   ├── constrained_registrar.py
    │   ├── reference_region_mapper.py
    │   ├── registration_policy.py
    │   └── robust_affine_registrar.py
    │
    └── training/
        ├── checkpointing.py
        ├── early_stopping.py
        ├── factory.py
        ├── losses.py
        ├── optimizer.py
        ├── scheduler.py
        └── trainer.py
```

The full repository contains additional generated outputs, QC artifacts, experiment-specific scripts, runtime bundles, checkpoints, and validation files that are intentionally omitted here for readability.


## Main Technologies

- Python
- PyTorch
- Torchvision
- MONAI
- SimpleITK
- Deep Learning
- Medical Imaging
- 2D / 2.5D / 3D CNNs
- Vision Transformers
- NIfTI imaging
- DaT-SPECT
- Cross-Validation
- Probability Calibration
- Ensemble Learning
- SLURM / HPC experimentation

---

## Experimental Design Summary

```text
1,362 DaT-SPECT subjects
        |
        v
Deterministic preprocessing
        |
        v
3 image-input strategies
        |
        x
10 model configurations
        |
        x
augmentation OFF / ON
        |
        v
60 experiments
        |
        v
5-fold OOF evaluation
        |
        v
240 calibration configurations
        |
        v
competition shortlist
        |
        v
ensemble search
        |
        v
ENS328
        |
        v
frozen offline inference runtime
```

---

## Validation Note

The reported metrics are **cross-fitted / out-of-fold estimates used during model development and model selection**.

The project evaluates many model, scenario, calibration, and ensemble configurations using the same set of 1,362 labeled subjects. Repeated selection over the same OOF labels can introduce **selection optimism** or a **winner's-curse effect**.

Therefore:

```text
Best OOF configuration != guaranteed best unseen-data configuration
```

The ENS328 OOF Log Loss of **0.255582** should be interpreted as a model-selection estimate from this experimental pipeline, not as independent external clinical validation.

---

## Project Scope

This repository focuses on the engineering and experimental pipeline for:

- DaT-SPECT preprocessing
- striatal ROI extraction
- medical image classification
- CNN and transformer benchmarking
- pretrained vs. scratch learning
- probability-quality evaluation
- calibration
- model ensembling
- reproducible offline inference

The central research question is not only **which model separates normal and pathologic scans**, but also **which complete pipeline produces reliable probabilities under a controlled evaluation framework**.
