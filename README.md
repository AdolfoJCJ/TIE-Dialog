

 TIE–Dialog — 📈 Conversational Dynamics Lab 📉 (CNøde)

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
(https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog)

<img width="849" height="560" alt="newplot - 2026-02-14T151706 958" src="https://github.com/user-attachments/assets/2e48078b-533f-4149-ac64-bad2ecf57ed2" />

TIE–Dialog is a Streamlit-based research tool for **turn-by-turn conversational coherence analysis**.
It models coherence as a dynamic signal (**Cₜ**) and supports the detection of **breakdown–repair dynamics (S–B–R)**, emergent coherence thresholds (**Φ**), participant-level trajectories (**Cᵢ**), and a geometric layer (**IC–III**) over an induced semantic trajectory.

> **Important note:** TIE–Dialog is inspired by the Theory of Informational Emergence (TIE), but the software itself is **theory-agnostic**. It makes no ontological assumptions and operates purely on measurable conversational structure.


🚀 Quickstart

bash
pip install -r requirements.txt
streamlit run app.py


🧩 What does TIE–Dialog do?

TIE–Dialog models dialogue as a time-evolving coherence system and provides complementary signal- and structure-level analyses.

🔹 Coherence signals

Computes turn-by-turn coherence measures:

Ct_new — context-aware coherence

IC-IIa — sigma alignment coherence

Ct_old — adjacent-turn baseline

Automatically estimates emergent thresholds:

Φ_low, Φ_high (percentile-based)

Detects dynamic regimes:

S (stable)

B (breakdown)

R (repair)

Extracts minimal S–B–R triadic units and supports breakdown–repair asymmetry analysis.

🔹 Invariant structural coherence (C_inv)

Optionally computes C_inv, a structure-level coherence signal derived from rolling similarity graphs over recent turns.

Ct measures semantic alignment.

C_inv measures structural stability.

C_inv is invariant under orthonormal transformations of normalized embedding spaces and reflects changes in the dialogue’s local semantic geometry rather than coordinate representation.

🔹 Participant trajectories

Cᵢ — individual embedding-based coherence

Continuous speaker state trajectories

🔹 IC–III geometric layer

Models structural evolution of the dialogue:

dᵢ — semantic displacement

κᵢ — curvature

τ(t) — informational time

🔹 IC–III → IC–II bridge

Structural–phenomenological coupling metrics:

ρ(t) — semantic compactness

Dₜ — structural stress

Ĉₜ — coherence proxy

Δ* — structural lag

📁 Dataset format

Upload a `.csv` or `.xlsx` file with the following columns:

Required

* `turn` (int) — turn index
* `participant` (str) — speaker label
* `text` (str) — utterance content

Optional

* `timestamp`

> If `turn` is missing, the app generates it automatically.

---

📤 Outputs (downloads)

TIE–Dialog supports exporting:

* `tie_dialog_full_results.csv`
  Full per-turn table including coherence signals, regimes, rupture/repair markers, IC-II and IC-III metrics, and participant trajectories.

* `tie_dialog_ic2_dynamics.csv`
  IC-II dynamics and transition metrics (e.g., residuals, asynchrony, normalized informational change).

* `tie_dialog_ic3_geometry.csv`
  IC–III geometric layer (dᵢ, κᵢ, τ) and the structural driver Dₜ.

* `tie_dialog_main_plot.html`
  Interactive export of the main coherence plot.

* `tie_dialog_consultancy_report.pdf`
  A readable, narrative PDF report designed for interpretation and applied use cases.

🎯 Intended Use Cases

Conversation analysis research

Computational linguistics experiments

Team communication diagnostics

Dialogue system evaluation

Breakdown–repair asymmetry studies


🔎 Representation modes

TIE–Dialog supports semantic representations via:

* **SBERT embeddings** (recommended when available)
* **TF-IDF / bag-of-words fallback** (for lightweight environments)

The selected mode is displayed in the UI during analysis.


⚠️ Notes & limitations

* Φ thresholds are **operational** and depend on the chosen coherence mode and dataset.
* Event detection is parameter-sensitive by design (the goal is interpretability, not black-box classification).
* Results are best interpreted comparatively (within-dialogue dynamics), rather than as absolute universal constants.


📌 License

This project is released under the MIT License.
See the LICENSE file for details.

📚 Citation

If you use TIE–Dialog in academic work, please cite it using the metadata provided in CITATION.cff.

You can also cite the corresponding preprints and software releases hosted on Zenodo and ResearchGate.
