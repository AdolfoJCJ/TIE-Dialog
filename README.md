

## TIE–Dialog — 📈 Conversational Dynamics Lab 📉 (CNøde)

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
(https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog)

<img width="849" height="560" alt="newplot - 2026-02-23T202009 710" src="https://github.com/user-attachments/assets/720ad64e-c886-48bd-9737-02501cb91fb7" />

TIE–Dialog is a Streamlit-based research tool for **turn-by-turn conversational coherence analysis**.
It models coherence as a dynamic signal (**Cₜ**) and supports the detection of **breakdown–repair dynamics (S–B–R)**, emergent coherence thresholds (**Φ**), participant-level trajectories (**Cᵢ**), and a geometric layer (**IC–III**) over an induced semantic trajectory.

> **Important note:** TIE–Dialog is inspired by the Theory of Informational Emergence (TIE), but the software itself is **theory-agnostic**. It makes no ontological assumptions and operates purely on measurable conversational structure.


🚀 Quickstart

bash
pip install -r requirements.txt
streamlit run app.py


# 🧩 What does TIE–Dialog do?

**TIE–Dialog** models dialogue as a **time-evolving informational system** and provides complementary analyses of conversational dynamics at three levels:

* semantic coherence
* structural stability
* geometric evolution

The framework combines **IC-II (informational coherence dynamics)** and **IC-III (geometric structure of dialogue trajectories)**.

---

# 🔹 Coherence dynamics (IC-II)

The system computes a **turn-by-turn coherence signal**:

**Ct — contextual coherence**

Ct models how each turn aligns with the evolving conversational context using the IC-II formulation.

From this signal the system automatically derives **emergent coherence thresholds**:

* **Φ_low** — lower coherence boundary
* **Φ_high** — upper coherence boundary

These thresholds are estimated from the empirical Ct distribution.

---

## Conversational regimes

Using Ct and the detected events, TIE–Dialog identifies three regimes:

* **S — Stable**
  Coherent continuation of the current conversational frame.

* **B — Break**
  Rupture candidate (loss of alignment with the context).

* **R — Repair**
  Recovery or re-alignment after a rupture.

These regimes allow the extraction of **breakdown–repair structures** and **S–B–R triadic units**.

---

# 🔹 Structural coherence (C_inv)

In addition to semantic coherence, TIE–Dialog computes:

**C_inv — invariant structural coherence**

C_inv is derived from **rolling similarity graphs** built over recent turns.

The graph structure is summarized through **spectral invariants of the normalized Laplacian**.

Interpretation:

| Signal |           Meaning                    |
| ------ | ------------------------------------ |
| Ct     | semantic/contextual alignment        |
| C_inv  | structural stability of the dialogue |

This allows distinguishing different rupture types:

| Pattern       |       Interpretation                   |
| ------------- | -------------------------------------- |
| Ct ↓, C_inv ↓ | strong rupture (semantic + structural) |
| Ct ↓, C_inv ~ | semantic drift                         |
| Ct ~, C_inv ↓ | structural reframe                     |

---

# 🔹 Participant trajectories

TIE–Dialog also models **speaker-level dynamics**.

Computed measures include:

**Cᵢ — participant coherence trajectories**

These track how each speaker's contributions align with the evolving context.

Two representations are provided:

* embedding-based Ci trajectories
* continuous speaker state trajectories

This allows identifying roles such as:

* stabilizing participants
* divergence initiators
* repair agents

---

# 🔹 IC–III geometric layer

The IC-III layer models the **geometry of the dialogue trajectory in embedding space**.

Key quantities:

* **dᵢ** — semantic displacement
* **κᵢ** — curvature (trajectory bending)
* **τ(t)** — cumulative informational deformation

These quantities capture structural properties of conversational evolution.

---

# 🔹 IC-III → IC-II bridge

The framework also estimates **structural drivers of coherence dynamics**.

Derived signals include:

* **ρ(t)** — semantic compactness
* **Dₜ** — structural stress
* **Δ*** — estimated structural lag between geometry and coherence

These metrics help study **how structural changes precede or follow coherence shifts**.

---

# 🔹 Automatic rupture typing

TIE–Dialog also classifies events by combining Ct and C_inv:

* **RUPTURE_STRONG**
* **RUPTURE_SEM**
* **RUPTURE_STRUCT**
* **STABLE**

This classification helps interpret conversational transitions.

---

# 📁 Dataset format

Upload a `.csv` or `.xlsx` file containing:

### Required

```
turn (int)        — turn index
participant (str) — speaker label
text (str)        — utterance content
```

### Optional

```
timestamp
```

If `turn` is missing, the application automatically generates it.

---

# 📤 Outputs

TIE–Dialog supports exporting:

**tie_dialog_full_results.csv**

Full per-turn dataset including:

* coherence signals
* S/B/R regimes
* rupture classifications
* IC-III metrics
* participant trajectories

---

**tie_dialog_ic2_dynamics.csv**

IC-II coherence dynamics:

* resonance
* informational change
* Ct

---

**tie_dialog_ic3_geometry.csv**

IC-III structural metrics:

* dᵢ
* κᵢ
* τ
* ρ
* Dₜ
* Ĉₜ

---

**tie_dialog_main_plot.html**

Interactive export of the main coherence plot.

---

**tie_dialog_report.pdf**

Automatically generated report containing:

* summary metrics
* key plots
* event tables
* parameter configuration

---

# 🎯 Intended Use Cases

TIE–Dialog can be used for:

* conversation analysis research
* computational linguistics experiments
* dialogue system evaluation
* team communication diagnostics
* breakdown–repair asymmetry studies
* conversational structure analysis

---

# 🔎 Representation modes

The system supports two semantic representation modes:

**SBERT embeddings (recommended)**
High-quality semantic representations.

**TF-IDF fallback**
Lightweight option for environments without transformer models.

The active representation mode is displayed in the UI.

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

