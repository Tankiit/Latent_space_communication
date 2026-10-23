"""
Candidate Concept Usability Audit & Type-Aware PyC Annotations Demonstration.
"""

import json
from pathlib import Path
import torch
import torch_concepts as pyc
from torch_concepts.nn import MixConceptEmbeddingToConcept

from typed_cbm.concept_registry import (
    CONCEPTS,
    usable_as_typed,
    audit_summary,
    get_example_annotations,
    get_typed_medical_annotations,
)


def run_audit():
    print("=" * 80)
    print("CANDIDATE CONCEPT USABILITY AUDIT (MedMNIST / ChestX-ray14)")
    print("=" * 80)

    print("\nAuditing candidate concepts against typed concept criteria:")
    print("  1. Independent Magnitude: Is magnitude an independent physical measurement (not circular)?")
    print("  2. Two Extent States: Does schema contain >= 2 states with magnitude (kappa requirement)?")
    print("  3. Enough Rows: Are there >= ~30 rows per state after missingness/injection?")

    audit_records = {}
    print("\n" + "=" * 95)
    print(f"{'Concept Key':<26} | {'Indep Mag?':<12} | {'>=2 Ext States?':<16} | {'>=30 Rows?':<12} | {'Hurdle Usable?':<16}")
    print("-" * 95)

    for key, c in CONCEPTS.items():
        summary = audit_summary(c)
        audit_records[key] = summary
        ind_str = "YES" if summary["independent_magnitude"] else "NO (circular)"
        two_str = "YES" if summary["two_extent_states"] else "NO (single-hurdle)"
        row_str = "YES" if summary["enough_rows"] else "NO (underpowered)"
        usable_str = "USABLE" if summary["hurdle_usable"] else "REJECTED"

        print(f"{key:<26} | {ind_str:<12} | {two_str:<16} | {row_str:<12} | {usable_str:<16}")
    print("-" * 95)

    # Detailed Audit Explanations
    print("\n[Detailed Diagnostic Notes]")
    print("  * 'chestmnist:mass':")
    print("      - Magnitude: Injected LogNormal lesion diameter (mm) conditioned strictly on PRESENT.")
    print("      - Label noise: Text-mined NLP annotations (~10-15% label noise).")
    print("      - Verdict: USABLE for single-hurdle typed CBMs; sample size (N=78,468) is vast.")
    print("  * 'retinamnist:grade':")
    print("      - Magnitude: 'grade 1-4' is identical to the discrete state integer (M == int(S)).")
    print("      - Circularity: FAILS independent measurement test. Severity is completely collinear with state.")
    print("      - Sample size: N=1,600 total; G4 has only ~35 samples in train split (<30 after injection).")
    print("      - Verdict: REJECTED as a typed concept unless native continuous biomarker (e.g. lesion area) is added.")
    print("  * 'chestmnist:source_bbox':")
    print("      - Magnitude: Bounding box area (w * h) from NIH ChestX-ray14 BBox_List_2017.csv.")
    print("      - Verification: Gold-standard radiologist bboxes for 984 images (~85 mass bboxes).")
    print("      - Verdict: USABLE for native physical extent verification on curated subset.")

    # Save to disk
    Path("results").mkdir(exist_ok=True)
    with open("results/candidate_concepts_audit.json", "w") as f:
        json.dump(audit_records, f, indent=2)
    print("\nSaved audit report to 'results/candidate_concepts_audit.json'.")

    # ------------------------------------------------------------------ PyC Integration Demo
    print("\n" + "=" * 80)
    print("TYPE-AWARE PYC ANNOTATIONS & MIXCONCEPTEMBEDDINGTOCONCEPT")
    print("=" * 80)

    # 1. User-specified example annotations
    print("\n[1. User Reference Annotations: Smoking, Genotype, Tar, Cancer]")
    annotations = pyc.Annotations(
        labels=["smoking", "genotype", "tar", "cancer"],
        cardinalities=[1, 3, 1, 1],
        types=["binary", "categorical", "continuous", "binary"],
    )
    print("Annotations Object:", annotations)
    print("  * Labels:", annotations.labels)
    print("  * Cardinalities:", annotations.cardinalities, f"(Total Concept Dim = {sum(annotations.cardinalities)})")
    print("  * Types:", annotations.types)

    predictor = MixConceptEmbeddingToConcept(
        in_concepts=annotations,   # type-aware: uses cardinalities and types
        in_embeddings=16,
        out_concepts=1,
    )
    print("\nInstantiated MixConceptEmbeddingToConcept:")
    print(predictor)

    batch_size = 32
    total_concepts = sum(annotations.cardinalities)  # 6
    emb_size = 16

    concepts = torch.randn(batch_size, total_concepts)
    embeddings = torch.randn(batch_size, total_concepts, emb_size)

    tasks = predictor(concepts=concepts, embeddings=embeddings)
    print(f"\nForward Pass Successful:")
    print(f"  Input Concepts:   {concepts.shape}    (batch, sum(cardinalities))")
    print(f"  Input Embeddings: {embeddings.shape} (batch, sum(cardinalities), emb_size)")
    print(f"  Output Tasks:     {tasks.shape}      (batch, out_concepts)")

    # 2. Medical Typed Concept Annotations
    print("\n[2. Medical Panel Annotations: Mass & Cardiomegaly (State + Magnitude)]")
    med_annotations = get_typed_medical_annotations()
    print("Medical Annotations Object:", med_annotations)
    med_predictor = MixConceptEmbeddingToConcept(
        in_concepts=med_annotations,
        in_embeddings=16,
        out_concepts=1,
    )
    med_dim = sum(med_annotations.cardinalities)
    med_concepts = torch.randn(batch_size, med_dim)
    med_embeddings = torch.randn(batch_size, med_dim, emb_size)
    med_tasks = med_predictor(concepts=med_concepts, embeddings=med_embeddings)
    print(f"  Medical Concept Inputs:     {med_concepts.shape}")
    print(f"  Medical Task Prediction:    {med_tasks.shape}")

    print("\n" + "=" * 80)
    print("All audits and type-aware PyC demonstrations completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
