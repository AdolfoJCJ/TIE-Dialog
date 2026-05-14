

## TIE–Dialog — 📈 Conversational Dynamics Lab 📉 (CNøde)

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
(https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog)

<img width="2500" height="1875" alt="Presentación sin título (1)" src="https://github.com/user-attachments/assets/2bdb0244-5b1d-49b3-bef4-09be5c9574dd" />


---

# TIE–Dialog

## Multi-signal modeling of conversational transitions, breakdowns, and repair dynamics

TIE–Dialog is a **Streamlit-based research framework** for analyzing dialogue as a **dynamic informational system evolving over time**.

Rather than treating conversation as a sequence of isolated utterances, TIE–Dialog models dialogue as a structured trajectory where:

* coherence emerges,
* destabilizes,
* reorganizes,
* and recovers.

The system combines:

* contextual coherence dynamics,
* structural graph invariants,
* geometric trajectory analysis,
* transition-pressure modeling,
* and multi-signal event detection.

This enables detection of:

* breakdown–repair dynamics,
* semantic drift,
* structural reorganization,
* transition zones,
* participant-level trajectories,
* and temporally extended conversational phases.

> **Important:**
> TIE–Dialog is conceptually inspired by the Theory of Informational Emergence (TIE), but the software itself is intentionally **theory-agnostic**.
> It operates entirely on measurable conversational structure without ontological assumptions.

---

# 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 🧩 What does TIE–Dialog do?

TIE–Dialog models conversation as a **multi-layer dynamic system** composed of interacting signals:

| Layer               | Function                            |
| ------------------- | ----------------------------------- |
| IC–II               | Contextual coherence dynamics       |
| C_inv               | Structural/topological stability    |
| IC–III              | Geometric conversational trajectory |
| Transition Pressure | Multi-signal transition field       |

These layers interact to model how conversations:

* maintain coherence,
* destabilize,
* transition,
* and reorganize over time.

---

# 🔹 IC–II — Contextual Coherence Dynamics

## Cₜ — contextual coherence

TIE–Dialog models coherence as an evolving trajectory rather than a static similarity score.

C_t=f(E_t,\mathcal{C}_{t-1})

Where:

* (E_t) = current turn embedding,
* (\mathcal{C}_{t-1}) = evolving contextual field.

Cₜ integrates:

* contextual alignment,
* temporal continuity,
* local displacement penalties,
* trajectory persistence.

This produces a **continuous coherence field** rather than binary coherence labels.

---

## Emergent thresholds (Φ)

Thresholds are derived directly from the empirical coherence distribution.

* **Φ_low** → breakdown boundary
* **Φ_high** → stable coherence boundary

These thresholds are:

* adaptive,
* dialogue-dependent,
* and data-driven.

---

## Conversational regimes (S–B–R)

Using coherence dynamics, TIE–Dialog segments conversation into:

| Regime | Meaning             |
| ------ | ------------------- |
| S      | Stable              |
| B      | Breakdown           |
| R      | Repair / transition |

This enables extraction of breakdown–repair structures over time.

---

# 🔹 Structural Coherence (C_inv)

TIE–Dialog models conversational structure independently of semantic similarity.

## C_inv — invariant structural coherence

Structural stability is computed from rolling similarity graphs using:

* k-NN graph construction,
* normalized Laplacians,
* spectral graph invariants,
* eigenvalue dynamics.

C_{inv}(t)=1-\Delta G_t

---

## Interpretation

| Signal | Interpretation                  |
| ------ | ------------------------------- |
| Cₜ     | contextual / semantic alignment |
| C_inv  | structural stability            |
| ΔC_inv | structural reorganization       |

This allows detection of structural changes even when local semantic similarity remains high.

---

# 🔹 IC–III — Geometric Conversational Layer

The conversation is also modeled as a trajectory in embedding space.

## Core geometric quantities

### Local displacement

d_i=|E_i-E_{i-1}|

Measures semantic movement between consecutive turns.

---

### Curvature

(\kappa_i) measures directional change in the conversational trajectory.

High curvature indicates conversational reorientation.

---

## Geometric transition driver (Dₜ)

TIE–Dialog integrates geometric signals into a dynamic transition driver.

D_t=f(d_i,\kappa_i,\rho_t)

Where:

* (d_i) = displacement,
* (\kappa_i) = curvature,
* (\rho_t) = semantic compactness.

Dₜ captures geometric transition pressure within the dialogue trajectory.

---

# 🔹 Transition Pressure Landscape

One of the central components of TIE–Dialog is the modeling of conversational instability as a continuous pressure field.

## Transition pressure

P_t^{transition}=w_s\Delta C_t+w_i\Delta C_{inv}+w_dD_t

This combines:

* semantic disruption,
* structural reconfiguration,
* geometric transition pressure.

---

## Transition zones

Instead of modeling events as isolated points, TIE–Dialog extracts:

* transition regions,
* breakdown windows,
* repair phases,
* structural reorganization zones.

Each zone includes:

* onset,
* duration,
* peak pressure,
* recovery dynamics.

This enables region-based analysis of conversational transitions.

---

# 🔹 Multi-signal Event Detection

TIE–Dialog detects events using coupled signals rather than isolated thresholds.

## Event signals

* semantic drop ((\Delta C_t))
* structural drop ((\Delta C_{inv}))
* geometric transition driver (Dₜ)

---

## Event types

| Event          | Meaning                        |
| -------------- | ------------------------------ |
| RUPTURE_STRONG | semantic + structural collapse |
| RUPTURE_SEM    | semantic drift                 |
| RUPTURE_STRUCT | structural reorganization      |
| STABLE         | stable continuation            |

This makes event detection:

* interpretable,
* multi-layered,
* and robust against local noise.

---

# 🔹 Participant Trajectories (Cᵢ)

TIE–Dialog models speaker-level coherence dynamics.

## Cᵢ — participant coherence trajectories

Tracks how each participant aligns with the evolving conversational structure.

Enables detection of:

* stabilizing participants,
* divergence initiators,
* repair agents,
* alignment asymmetries.

---

# 🔹 Continuous Conversational State Modeling

Beyond discrete turns, TIE–Dialog models continuous participant states using:

* inertia-based trajectories,
* contextual diffusion,
* state continuity dynamics.

This reveals latent conversational organization beyond local turn structure.

---

# 🔹 Potentiality (℘ₜ)

TIE–Dialog includes a metric for conversational openness and exploratory structure.

## ℘ₜ — structural potentiality

Computed from:

* questions,
* conditionals,
* modal expressions,
* exploratory formulations.

This models movement toward:

* uncertainty,
* openness,
* proto-coherent conversational states.

---

# 🔹 Validation Framework

TIE–Dialog includes multiple validation layers.

---

## Baseline comparison

The system compares itself against simpler approaches:

* turn-to-turn cosine disruption,
* moving-context similarity,
* geometric displacement baselines.

---

## Randomized controls

Includes shuffled-order dialogue baselines to test whether detected structures depend on temporal organization rather than utterance content alone.

---

## Ablation diagnostics

The framework supports component ablations:

* no structural layer,
* no geometric layer,
* semantic-only configurations.

This tests the contribution of each layer to event reconstruction.

---

## Added Structural Value (ASV)

TIE–Dialog computes Added Structural Value metrics to estimate how much detected structure cannot be reconstructed by simpler baselines.

This includes:

* global reducibility,
* event-level reconstruction,
* shuffled-event localization comparisons.

---

# 🔹 Human Annotation Alignment

TIE–Dialog supports comparison between:

* human-annotated breakdown/repair regions,
* automatically extracted transition zones.

The system compares:

* temporal overlap,
* event localization,
* transition persistence,
* recovery structure.

Importantly, comparison is region-based rather than point-based.

---

# 📁 Dataset Format

Upload a `.csv` or `.xlsx` containing:

## Required columns

```text
turn
participant
text
```

## Optional

```text
timestamp
```

If `turn` is missing, it is generated automatically.

---

# 📤 Outputs

## Full dialogue analysis

### `tie_dialog_full_results.csv`

Includes:

* Cₜ
* C_inv
* S/B/R regimes
* rupture classifications
* IC–III metrics
* participant trajectories
* potentiality metrics
* transition windows

---

## IC–II dynamics

### `tie_dialog_ic2_dynamics.csv`

Includes:

* contextual coherence,
* resonance,
* informational change.

---

## IC–III geometry

### `tie_dialog_ic3_geometry.csv`

Includes:

* displacement,
* curvature,
* compactness,
* geometric drivers.

---

## PDF report

### `tie_dialog_report.pdf`

Automatically generated report including:

* plots,
* transition zones,
* detected events,
* summary metrics,
* configuration parameters.

---

# 🎯 Use Cases

TIE–Dialog can be applied to:

* conversation analysis,
* computational linguistics,
* dialogue systems,
* human–AI interaction,
* team communication analysis,
* repair dynamics research,
* discourse instability analysis,
* conversational transition modeling.

---

# 🔎 Representation Modes

Supported embedding systems:

* SBERT
* E5
* BGE
* INSTRUCTOR
* TF-IDF fallback

The active embedding mode is displayed in the interface.

---

# 🧠 Conceptual Position

TIE–Dialog treats conversation as a structured dynamic system where:

* coherence is temporal,
* transitions are measurable,
* breakdown is structured rather than noise,
* repair is traceable,
* and conversational organization emerges from interacting informational layers.

The framework focuses not only on:

* what conversations mean,

but also on:

* how they evolve structurally over time.

---

# ⚠️ Notes & Limitations

* Φ thresholds are dialogue-dependent operational estimates.
* Event detection is parameter-sensitive by design.
* Structural signals depend on embedding quality.
* Results are best interpreted within-dialogue rather than as universal constants.
* Transition zones represent operational computational structures, not ground-truth psychological states.

---

# 📌 License

TIE–Dialog is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

This means that modified versions deployed as network services must also provide access to their corresponding source code under the same license.

Commercial licensing is available upon request.

Copyright (C) 2026 Adolfo J. Céspedes Jiménez

---

# 📚 Citation

If you use TIE–Dialog in academic work, please cite:

```text
CITATION.cff
```

You may also cite the corresponding:

* Zenodo releases,
* preprints,
* and ResearchGate publications.


