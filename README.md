

## TIE–Dialog — 📈 Conversational Dynamics Lab 📉 (CNøde)

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
(https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog)

<img width="2500" height="1875" alt="Presentación sin título (1)" src="https://github.com/user-attachments/assets/2bdb0244-5b1d-49b3-bef4-09be5c9574dd" />

# TIE–Dialog

**Turn-by-turn Conversational Coherence Analysis**

TIE–Dialog is a **Streamlit-based research tool** for analyzing conversational coherence as a **dynamic informational signal evolving over time**.

It models dialogue as a structured process where coherence is **maintained, disrupted, and reorganized**, enabling the detection of:

* breakdown–repair dynamics (S–B–R)
* emergent coherence thresholds (Φ)
* structural and geometric transitions
* participant-level trajectories

> **Important:**
> TIE–Dialog is inspired by the Theory of Informational Emergence (TIE), but the software itself is **theory-agnostic**.
> It operates purely on **measurable conversational structure**, without ontological assumptions.

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧩 What does TIE–Dialog do?

TIE–Dialog models dialogue as a **time-evolving informational system** and analyzes it across three coupled layers:

* **IC–II:** Coherence dynamics
* **C_inv:** Structural invariants
* **IC–III:** Geometric trajectory

These layers interact to explain **how conversations evolve structurally and semantically**.

---

## 🔹 Coherence Dynamics (IC–II)

### Cₜ — contextual coherence

Cₜ is not just similarity. It is computed as an **identity-over-trajectory signal**:

* integrates **context alignment**
* penalizes **local displacement**
* accumulates **trajectory consistency over time**

This produces a **continuous coherence field**, not a binary measure.



---

### Emergent thresholds (Φ)

From the empirical distribution of Cₜ:

* **Φ_low** → breakdown boundary
* **Φ_high** → stable coherence boundary

These thresholds are **data-driven**, not fixed.

---

### Conversational regimes (S–B–R)

Using Cₜ:

* **S (Stable)** → coherent continuation
* **B (Break)** → loss of alignment
* **R (Repair)** → recovery phase

This enables extraction of **breakdown–repair structures**.

---

## 🔹 Structural Coherence (C_inv)

TIE–Dialog models structure independently of semantics:

* **C_inv — invariant structural coherence**

Computed from **rolling similarity graphs** over turns using:

* k-NN graph construction
* normalized Laplacian
* spectral invariants

### Interpretation

| Signal | Meaning                         |
| ------ | ------------------------------- |
| Cₜ     | semantic / contextual alignment |
| C_inv  | structural stability            |

---

## 🔹 IC–III Geometric Layer

The conversation is also modeled as a **trajectory in embedding space**.

### Core quantities

* **dᵢ** — semantic displacement (cosine-based)
* **κᵢ** — curvature (trajectory change)

These capture **how the dialogue moves**, not just what it means.

---

## 🔹 Structural Drivers (IC–III → IC–II)

TIE–Dialog explicitly models **what drives coherence changes**.

### Additional signals

* **ρₜ — semantic compactness**
  → how tightly clustered local meaning is

* **Dₜ — structural driver**
  → combines displacement, curvature, and compactness

These signals quantify **reconfiguration pressure** in the dialogue.

---

## 🔹 Multi-signal Event Detection

Instead of relying only on thresholds, TIE–Dialog uses **multi-channel scoring**:

* semantic drop (ΔCₜ)
* structural drop (ΔC_inv)
* structural driver (Dₜ)

### Event types

* **RUPTURE_STRONG** → semantic + structural collapse
* **RUPTURE_SEM** → semantic drift
* **RUPTURE_STRUCT** → structural reframe
* **STABLE**

This makes event detection **robust and interpretable**.

---

## 🔹 Participant Trajectories

Speaker-level dynamics are modeled via:

* **Cᵢ — participant coherence trajectories**

These track how each participant aligns with the evolving context.

### Enables detection of:

* stabilizing agents
* divergence initiators
* repair agents

---

## 🔹 Continuous State Trajectories

Beyond discrete turns, TIE–Dialog models **continuous participant states**:

* inertia-based trajectories
* diffusion of coherence across speakers

This reveals **latent conversational structure**.

---

## 🔹 Potentiality (℘ₜ)

TIE–Dialog includes a metric for **structural openness**:

* **℘ₜ — potentiality**

Based on:

* questions
* conditionals
* modal expressions

This captures movement toward **proto-coherent states** (exploration, uncertainty).

---

## 📁 Dataset Format

Upload a `.csv` or `.xlsx` with:

### Required

```
turn (int)
participant (str)
text (str)
```

### Optional

```
timestamp
```

If `turn` is missing, it is automatically generated.

---

## 📤 Outputs

### Full dataset

**tie_dialog_full_results.csv**

Includes:

* Cₜ, C_inv
* S/B/R regimes
* rupture classifications
* IC–III metrics (dᵢ, κᵢ)
* ρₜ, Dₜ
* participant trajectories (Cᵢ)
* ℘ₜ (potentiality)

---

### IC–II dynamics

**tie_dialog_ic2_dynamics.csv**

* Cₜ
* resonance
* informational change

---

### IC–III geometry

**tie_dialog_ic3_geometry.csv**

* dᵢ
* κᵢ
* τ(t)
* ρₜ
* Dₜ

---

### PDF report

**tie_dialog_report.pdf**

Includes:

* summary metrics
* plots (Cₜ, IC–III, S–B–R)
* detected events
* configuration

---

## 🎯 Use Cases

TIE–Dialog can be used for:

* conversation analysis
* computational linguistics
* dialogue system evaluation
* team communication diagnostics
* breakdown–repair studies
* human–AI interaction analysis

---

## 🔎 Representation Modes

* **SBERT / E5 / BGE embeddings (recommended)**
* **TF-IDF fallback**

The active mode is shown in the UI.

---

## 🧠 Conceptual Position

TIE–Dialog treats conversation as a **dynamic informational system** where:

* coherence is **not static**
* breakdown is **structural, not noise**
* repair is **measurable and traceable**

It provides a framework to analyze:

* when conversations hold together
* when they break
* how they recover
* and what structural forces drive those transitions

---


# ⚠️ Notes & limitations

* Φ thresholds are **dataset-dependent operational estimates**.
* Event detection is **parameter-sensitive by design** to maintain interpretability.
* Results are best interpreted **within-dialogue**, not as universal constants.
* Structural signals depend on embedding quality.

---

# 📌 License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

# 📚 Citation

If you use **TIE–Dialog** in academic work, please cite it using the metadata provided in:

```
CITATION.cff
```

You may also cite the corresponding **software releases and preprints** hosted on Zenodo and ResearchGate.

