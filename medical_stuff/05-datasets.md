---
title: "05: Candidate Datasets: Medical and Tabular"
tags: [datasets, medical, tabular, benchmarks, verification, icml-2027]
status: living
prerequisites: ["[[02-typed-concept-schema]]", "[[04-hurdle-test-and-audit]]"]
related: ["[[06-gates-and-baselines]]", "[[08-open-questions]]"]
---

# 05: Candidate Datasets: Medical and Tabular

This document catalogs and audits candidate real-world datasets and controlled synthetic testbeds for evaluating Typed Concept Bottleneck Models for ICML 2027.

> [!WARNING]
> Every dataset must pass the [[04-hurdle-test-and-audit]] before experimental claims are recorded in the paper. Do NOT assume that an existing CBM benchmark has valid extent annotations without verification.

---

## 1. Medical Imaging Candidates

### A. VinDr-CXR / CheXmask / MIMIC-CXR (Chest Radiographs)
- **Clinical Task ($Y$):** Detection of Acute Heart Failure, Pneumonia, or 30-day readmission.
- **Candidate Typed Concepts ($C_j = (S_j, E_j)$):**
  1. **Cardiomegaly:**
     - *State ($S$):* Normal heart size ($S=0$) vs. Cardiomegaly ($S=1$).
     - *Extent ($E$):* **Cardiothoracic Ratio (CTR)**: ratio of transverse cardiac diameter to transverse thoracic diameter ($E \in (0.50, 1.0]$ when enlarged).
     - *Provenance:* Can be computed directly from CheXmask / VinDr-CXR segmented masks.
  2. **Pleural Effusion:**
     - *State ($S$):* Absent ($S=0$) vs. Present ($S=1$).
     - *Extent ($E$):* Segmented effusion area as a percentage of the hemithorax area ($E \in (0, 100\%]$) or fluid meniscus height in mm.
  3. **Pulmonary Edema:**
     - *State ($S$):* Absent ($S=0$) vs. Present ($S=1$).
     - *Extent ($E$):* Radiographic Assessment of Lung Edema (RALE) score ($E \in \{1, \dots, 48\}$ or normalized continuous opacity density).
- **Verification Status:**
  - `VERIFY:` CheXmask provides pixel-level segmentations for 60,000+ MIMIC-CXR images. Run CTR extraction script to verify continuous unimodal distribution on $S=1$.
  - `VERIFY:` Confirm whether VinDr-CXR bounding boxes are sufficient proxies for effusion volume if segmentations have high inter-observer variability.

---

### B. Derm7pt / ISIC (Dermatology)
- **Clinical Task ($Y$):** Melanoma vs. Benign Nevus classification.
- **Candidate Typed Concepts:**
  1. **Atypical Pigment Network:**
     - *State ($S$):* Absent ($S=0$) vs. Present ($S=1$).
     - *Extent ($E$):* Percentage of lesion surface exhibiting atypical reticulation ($E \in (0, 100\%]$).
  2. **Irregular Streaks / Globules:**
     - *State ($S$):* Absent vs. Present.
     - *Extent ($E$):* Radial dispersion angle or count density per unit circumference.
  3. **Lesion Diameter & Breslow Thickness:**
     - *State ($S$):* Lesion presence / biopsy confirmation.
     - *Extent ($E$):* Maximum diameter in mm ($E \in [2.0, 35.0]$) and Breslow depth in mm.
- **Verification Status:**
  - `VERIFY:` Derm7pt dataset contains 7-point checklist annotations for 1,011 clinical cases with calibrated dermoscopic criteria. Check if extent labels are continuous or discrete 3-tier scales (Absent / Mild / Severe).

---

## 2. Clinical Tabular & EHR Candidates

### A. MIMIC-IV Clinical Tabular (ICU / Emergency)
- **Clinical Task ($Y$):** In-hospital mortality, Sepsis onset, or Need for Renal Replacement Therapy (RRT).
- **Candidate Typed Concepts:**
  1. **Acute Kidney Injury (AKI):**
     - *State ($S$):* No AKI ($S=0$) vs. AKI present ($S=1$).
     - *Extent ($E$):* Peak Serum Creatinine delta above baseline ($\Delta \text{Cr} = \text{Cr}_{\text{peak}} - \text{Cr}_{\text{baseline}}$ in mg/dL, with $E > 0.3$).
  2. **Myocardial Injury (Troponin Elevation):**
     - *State ($S$):* Troponin within normal limits ($S=0$) vs. Elevated ($S=1$).
     - *Extent ($E$):* High-sensitivity Troponin T level ($\text{ng/L}$) above the 99th percentile upper reference limit.
  3. **Lactic Acidosis:**
     - *State ($S$):* Normal lactate ($S=0$) vs. Hyperlactatemia ($S=1$, lactate $> 2.0\text{ mmol/L}$).
     - *Extent ($E$):* Excess lactate above threshold: $E = \text{Lactate} - 2.0 \in (0, 20]\text{ mmol/L}$.
- **Verification Status:**
  - Tabular EHR data naturally satisfies the hurdle test: baseline physiology is not "zero kidney injury with 0.1 severity"; the hurdle corresponds to clinical disease thresholds.
  - `TODO:` Extract a cohort of 15,000 MIMIC-IV ICU stays with complete baseline and 48-hour trajectory labels.

---

## 3. Controlled Synthetic Benchmarks

### A. Synthetic Medical Finding Generator (`code/typed_cbm/synthetic.py`)
To isolate algebraic properties, exact counterfactual validity, and type-I error control, we implement a dedicated synthetic generator:
- **Input:** $X \sim \mathcal{N}(0, \Sigma) \in \mathbb{R}^{d_x}$.
- **Ground Truth Hurdle Generation:**
  - Logit for state: $z_{s, j} = \mathbf{w}_{s, j}^\top X + b_{s, j}$.
  - State gate: $S_j \sim \operatorname{Bernoulli}(\sigma(z_{s, j}))$.
  - Continuous extent (latent index): $z_{e, j} = \mathbf{w}_{e, j}^\top X + b_{e, j}$.
  - Observed extent:
    $$E_j = \begin{cases} \emptyset & \text{if } S_j = 0 \\ \exp(z_{e, j} + \epsilon_e), \; \epsilon_e \sim \mathcal{N}(0, \sigma_0^2) & \text{if } S_j = 1 \end{cases}$$
- **Task Label:**
  $$Y = \operatorname{Bernoulli}\left( \sigma \left( \sum_{j=1}^K \left[ \alpha_j S_j + \beta_j (S_j \cdot E_j) \right] + \gamma^\top X_{\text{nuisance}} \right) \right)$$
- **Properties Checked:**
  - Absence invariance under $do(S_j = 0)$.
  - Extent sensitivity under $do(S_j = 1, E_j = e)$.
  - Exact Type-I error control under spurious correlations.

---

### B. MedMNIST v3 Benchmarks (`code/typed_cbm/medmnist_loader.py`)
Standardized lightweight biomedical image benchmarks implemented with the standard preprocessing pipeline:
- **Preprocessing Transform:**
  ```python
  data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[.5], std=[.5])
  ])
  ```
- **Evaluated Subsets:**
  1. **RetinaMNIST (Diabetic Retinopathy Grading):**
     - Modality: Fundus photography (1,600 images, $28 \times 28 \times 3$).
     - Natural Typed Concept Structure:
       - *State ($S$):* No DR ($S=0$, label 0) vs. DR present ($S=1$, labels 1–4).
       - *Extent ($E$):* Retinopathy severity grade $E \in \{1, 2, 3, 4\}$ (Mild, Moderate, Severe, Proliferative).
  2. **PathMNIST (Colon Pathology):**
     - Modality: Histopathology tissue slides (107,180 images, $28 \times 28 \times 3$, 9 tissue types).
  3. **BloodMNIST (Blood Cell Microscopy):**
     - Modality: Peripheral blood cell microscope images (17,092 images, $28 \times 28 \times 3$, 8 cell types).

---

## Benchmark Audit Summary Table

| Benchmark | Data Modality | Concept / Class Count | State Type | Extent Type | Hurdle Test Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Generator** | Vector ($D=32$) | $K=4$ concepts | Binary $\{0, 1\}$ | LogNormal $(0, \infty)$ | ✅ Passed (Synthetic) |
| **RetinaMNIST** | Fundus ($28 \times 28 \times 3$) | 5 severity grades | Binary $\{0, 1\}$ | Ordinal grade $1 \dots 4$ | ✅ Loaders verified; typed experiments not yet run |
| **PathMNIST** | Histology ($28 \times 28 \times 3$) | 9 tissue classes | Categorical | Area / Cellularity | ✅ Loaders verified; typed experiments not yet run |
| **BloodMNIST** | Microscopy ($28 \times 28 \times 3$) | 8 cell types | Multi-class | Size / Granularity | ✅ Loaders verified; typed experiments not yet run |
| **ChestMNIST** | Radiograph ($28 \times 28$) | 14 findings | 3-state (unknown/absent/present) | Synthetic (image-correlated) | ✅ 2026-10-23: effusion (idx 2) chosen via G0 gate (AUROC 0.750); E0 passes at 8000/3000 caps; seeded E1 (10 seeds) + seeded 6-arm E2 complete — see [[06-gates-and-baselines]] |
| **PUMS (ACS 2023 1-YR)** | Tabular census (~1% sample, WY+DC) | 1 (WKHP hours worked) | Employed / non-employed | WKHP hours 1–99 | ✅ Hurdle-structure audit PASSED 2026-10-23; ESR-gate temporal misalignment documented ([[04-hurdle-test-and-audit]]) |
| **PAD-UFES-20** | Smartphone clinical images | 6 lesion types | Present (58% biopsy-proven) | Diameter mm 1–100 | ✅ Extent hurdle PASSED + patient-level refusal structure (99.5% joint missingness) 2026-10-23; blank-vs-UNK encoding audit pending raw CSV (CC BY 4.0, Mendeley zr7vgbcyr2.1) |
| **MIMIC-CXR (CheXmask)**| 2D Radiograph | $K=3$ (Effusion, Cardiomegaly, Edema) | Binary $\{0, 1\}$ | Area fraction / CTR $[0, 1]$ | 🔄 Pending Extraction |
| **MIMIC-IV Tabular** | Tabular EHR ($D=45$) | $K=4$ (AKI, Troponin, Lactate, Shock) | Binary $\{0, 1\}$ | Positive physical units | 🔄 Cohort Defined |
| **Derm7pt** | Clinical Dermoscopy | $K=5$ (Network, Streaks, Pigment) | Binary / 3-tier | Area / Intensity | ⏳ Pending Check |

---

## 4. Candidate Concept Usability Audit (`code/typed_cbm/concept_registry.py`)

Every candidate concept is audited against three mandatory criteria via `usable_as_typed(c)`:
1. **Independent Magnitude:** Is magnitude an independent physical/semantic measurement, or is it circular/collinear with the discrete state?
2. **$\ge 2$ Extent States:** Does the concept schema possess at least 2 distinct states carrying magnitude (required for Cohen's $\kappa$ inter-state extent agreement; standard hurdle requires $\ge 1$)?
3. **$\ge 30$ Rows per State:** Are there sufficient samples across all discrete states after missingness injection ($p_{\text{unknown}} \in [0.1, 0.5]$)?

### Candidate Concept Registry Table

| Concept Key | Native States | Native Magnitude | Circularity Check | $\ge 2$ Extent States? | $\ge 30$ Rows? | Hurdle Usable? | Diagnostic Assessment |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`chestmnist:mass`** | `["present", "absent"]` | None (Injected LogNormal) | **No** (Independent continuous lesion diameter) | No ($1$ active state) | **Yes** ($N=78,468$) | **USABLE** | Ideal testbed for single-hurdle typed CBMs under controlled missingness. |
| **`retinamnist:grade`** | `["G0", "G1", "G2", "G3", "G4"]` | `"grade 1-4"` | **YES (Circular)** ($M \equiv \operatorname{int}(S)$) | Yes ($4$ active states) | Borderline ($G_4$ has $\sim 35$ train rows) | **REJECTED** | Fails independent measurement test. Severity is identical to state integer. Requires native continuous lesion area. |
| **`chestmnist:source_bbox`**| `["present", "absent"]` | Bounding box area ($w \times h$) | **No** (Radiologist spatial annotation) | No ($1$ active state) | **Yes** ($\sim 85$ mass bboxes) | **USABLE** | High-fidelity physical ground-truth extent for curated subset evaluation. |

---

## 5. Type-Aware Concept Representations in PyC (`torch_concepts`)

Typed concepts map directly into `pyc.Annotations` using type-aware cardinality specification and `MixConceptEmbeddingToConcept`:

```python
import torch
import torch_concepts as pyc
from torch_concepts.nn import MixConceptEmbeddingToConcept

# Standard multi-type concept panel
annotations = pyc.Annotations(
    labels=["smoking", "genotype", "tar", "cancer"],
    cardinalities=[1, 3, 1, 1],
    types=["binary", "categorical", "continuous", "binary"],
)

predictor = MixConceptEmbeddingToConcept(
    in_concepts=annotations,   # type-aware: uses cardinalities and types
    in_embeddings=16,
    out_concepts=1,
)

concepts   = torch.randn(32, 6)        # (batch, sum(cardinalities))
embeddings = torch.randn(32, 6, 16)    # (batch, sum(cardinalities), emb_size)
tasks = predictor(concepts=concepts, embeddings=embeddings)  # Shape: (32, 1)
```

For a clinical panel of typed findings (e.g. Mass and Cardiomegaly):
```python
typed_annotations = pyc.Annotations(
    labels=["mass_state", "mass_magnitude", "cardiomegaly_state", "cardiomegaly_magnitude"],
    cardinalities=[3, 1, 3, 1],
    types=["categorical", "continuous", "categorical", "continuous"],
)
```

---

## Action Items & Next Steps

- [x] Implement candidate concept audit registry in `code/typed_cbm/concept_registry.py`.
- [x] Verify circularity failure mode on `retinamnist:grade`.
- [x] Wire `pyc.Annotations` with `MixConceptEmbeddingToConcept` in reference pipeline.
- `TODO:` Extract native bounding box areas from NIH ChestX-ray14 `BBox_List_2017.csv` for ChestMNIST test set.
- `TODO:` Finalize CheXmask CTR extraction pipeline in `code/data/extract_chexmask.py`.

