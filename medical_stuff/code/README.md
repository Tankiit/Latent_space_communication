# Typed Concept Bottleneck Models (Typed CBMs) — Reference Codebase

Reference implementation for the **Typed Concepts (State + Extent)** project, targeted for **ICML 2027**.
Built on and extending **`torch_concepts` (`pyc`)** with MedMNIST benchmarks.

---

## 🏗️ Architecture & PyC (`torch_concepts`) Integration

The codebase adheres strictly to the `torch_concepts` (`pyc`) API and calling conventions:

```python
import torch
import torch_concepts as pyc
from typed_cbm.layers import TypedBottleneckLayer

# 1. Standard PyC Linear CBM Style:
encoder = pyc.nn.LinearEmbeddingToConcept(in_embeddings=64, out_concepts=5)
predictor = pyc.nn.LinearConceptToConcept(in_concepts=5, out_concepts=1)

x = torch.randn(32, 64)
concepts = encoder(embeddings=x)
tasks = predictor(concepts=concepts)

# 2. Typed CBM Extension (Tagged-Union Hurdle Bottleneck):
from typed_cbm.models import TypedCBM

typed_model = TypedCBM(in_embeddings=64, num_concepts=5, num_classes=2)
tasks, concepts, batch, params = typed_model(embeddings=x)
```

---

## 🧭 Repository Structure

```text
code/
├── typed_cbm/
│   ├── __init__.py           # Package exports
│   ├── schema.py             # TaggedUnionBatch, TypedConceptSpec, and type invariants
│   ├── layers.py             # HurdleEmbeddingToConcept (BaseConceptLayer), TypedBottleneckLayer, LogNormalHurdleLoss
│   ├── models.py             # TypedCBM, StandardCBM, UncoupledTwoHeadCBM, CEMBaseline (using pyc.nn)
│   ├── interventions.py      # Typed do() operators and Absence Invariance leakage evaluator
│   ├── hurdle_audit.py       # 5-step Hurdle Audit Battery (Vuong Tobit vs Hurdle, decoupling, calibration)
│   ├── medmnist_loader.py    # Standardized MedMNIST loaders (PathMNIST, BloodMNIST, RetinaMNIST)
│   └── synthetic.py          # Synthetic clinical hurdle data generator
├── tests/
│   ├── test_schema.py        # Spec validation and TaggedUnion batching tests
│   ├── test_layers.py        # Hurdle forward pass, masked gradient flow tests
│   ├── test_models.py        # Forward passes and intervention leakage tests (including PyC style check)
│   ├── test_audit.py         # HurdleAuditor tests (Vuong, CV, ECE, coverage)
│   └── test_medmnist_loader.py # Tests for PathMNIST, BloodMNIST, and RetinaMNIST dataloaders
├── experiments_e0_e2.py      # Flagship MedMNIST Typed Concepts experiments (E0 hurdle audit, E1 bias sweep, E2 model)
├── gate_and_diagnostics.py   # Pre-E2 gates: state-signal AUROC, MAR strength, generator circularity, MNAR decomposition
├── e2_arms.py                # E2 arm registry (incl. pooled_masked, regression_impute, two_part), matched tuning, decision rule, gain attribution, legal_queries
├── run_e2_seeds.py           # Seeded 10-seed honest-arms E2 run -> results/e2_effusion_seeds.json
├── try_medmnist.py           # Verification script for PathMNIST, BloodMNIST, RetinaMNIST dataloaders
├── run_gate1_audit.py        # Executable Gate 1 benchmark script comparing TypedCBM vs baselines
└── README.md
```

---

## ⚡ Quick Start

Ensure your environment is active (`conda activate torch-multimodal`).

### 1. Run Experiments E0 - E2 on MedMNIST (ChestMNIST)
```bash
PYTHONPATH=code conda run -n torch-multimodal python code/experiments_e0_e2.py
```
- **E0 (Hurdle Property Audit):** Confirms 0.0000 non-PRESENT rows contain magnitude and exactly 1 state carries magnitude.
- **E1 (Estimator Bias Sweep):** Evaluates Zero-Fill vs. Drop-Rows under MCAR, MAR, and MNAR missingness.
- **E2 (Typed Model Head):** Evaluates state decodability, magnitude RMSE, and absence masking on undefined rows.

### 2. Run Pre-E2 Gates on the Target Finding (effusion, idx 2)
```bash
PYTHONPATH=code conda run -n torch-multimodal python code/gate_and_diagnostics.py
```
- **G0:** held-out AUROC (present vs absent) + majority baseline; requires >= 0.65 to wire E2.
- **D1:** corr(feats[:,0], log-magnitude | present) — MAR re-keying strength.
- **D2:** feature-explained share of magnitude vs generator noise (circularity check).
- **D3:** MNAR mask decomposition into feature part vs noise part.

### 3. Verify MedMNIST DataLoaders (PathMNIST, BloodMNIST, RetinaMNIST)
```bash
PYTHONPATH=code conda run -n torch-multimodal python code/try_medmnist.py
```

### 4. Run Unit Tests (16 tests, all passing)
```bash
PYTHONPATH=code conda run -n torch-multimodal python -m unittest discover -s code/tests -v
```

---

## 🔗 Connection to Vault Notes

- **Mathematical Schema & Hurdle Layers:** See [[02-typed-concept-schema]].
- **Audit Battery Specifications:** See [[04-hurdle-test-and-audit]].
- **Dataset Catalog & MedMNIST Benchmarks:** See [[05-datasets]].
- **Baseline Models and Reviewer Gates:** See [[06-gates-and-baselines]].
- **Vocabulary and CEM Flaws:** See [[07-terminology]].
