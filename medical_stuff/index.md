---
title: Typed concepts vault, index
tags: [index, icml-2027]
status: living
---

# Typed concepts: map of contents

Working vault for the typed (state + extent) uncertainty project. **Near-term submission: CLeaR 2027, deadline Nov 16, 2026 (verified; see [[09-venue-icml]])**; ICML 2027 remains the fallback venue. The ICLR and AISTATS plans were dropped.

**Status of the framing: RESOLVED 2026-10-23 (see [[01-claim-and-story]] revision).** Option A (supervised hurdle) is the anchor. Verified algebra shows typed ≡ two-part hurdle models *in estimation* (indicator recovers α and β; readouts [state, gated E] ≡ [mask, zero-filled E] pointwise), so the paper is **not a method paper**: the contribution is (i) selection bias with known truth (E1 MNAR/MAR), (ii) the hurdle audit battery with two real-data passes (PUMS WKHP, PAD-UFES-20 diameter), (iii) the typed schema's interface guarantees (ill-typed interventions unrepresentable; phantom-read elimination). G0–D3 gates passed on ChestMNIST effusion; seeded E0–E2 complete.

## Notes
- [[01-claim-and-story]]: candidate claim, five-beat story, and the 2026-10-23 revision (estimation ties; contribution relocated)
- [[02-typed-concept-schema]]: tagged-union support, distribution layer, worked example, verified readout identity
- [[03-option-a-vs-b]]: typing in the data vs typing in the model (resolved: Option A for this paper)
- [[04-hurdle-test-and-audit]]: audit battery spec + 2026-10-23 results (PUMS, PAD-UFES-20, ChestMNIST)
- [[05-datasets]]: medical and tabular candidates, sources, licenses, audit status
- [[06-gates-and-baselines]]: pre-E2 gates (passed) and the 6-arm seeded E2 results
- [[07-terminology]]: typed vs mixed vs mixture (CEM vocabulary trap)
- [[08-open-questions]]: open items and task register (updated 2026-10-23)
- [[09-venue-icml]]: CLeaR 2027 near-term (verified dates), ICML 2027 fallback, timeline

## Conventions
- `TODO:` marks a hole to fill. Nothing marked TODO has been decided.
- `VERIFY:` marks a fact stated from memory that must be checked against the source.
- Statements in the ICLR-era specs are copied only where still relevant, and tagged `(ICLR-era)`.
- Experiment numbers live in `results/*.json` under `code/`'s working directory; every claim in this vault cites its file.
