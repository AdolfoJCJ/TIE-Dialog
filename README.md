# TIE–Dialog — 📈 Conversational Dynamics Lab 📉

## A multivariate computational framework for conversational dynamics

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog

<img width="2500" height="1875" alt="TIE–Dialog interface" src="https://github.com/user-attachments/assets/2bdb0244-5b1d-49b3-bef4-09be5c9574dd" />

---

## Overview

**TIE–Dialog** is a Streamlit-based research framework for representing and analyzing dialogue as a **continuous, evolving multivariate system**.

Rather than treating conversation as a sequence of isolated utterances, or assuming that conversational change can be captured by a single transition score, TIE–Dialog constructs several complementary continuous dimensions and studies how they evolve and interact over time.

The current framework centers on three computational dimensions:

- **$S_t$ — contextual / semantic drift**
- **$R_t$ — structural reconfiguration**
- **$D_t$ — geometry driver**

Together they define the turn-level computational state

$$
\mathbf{z}_t = \left(S_t, R_t, D_t\right)
$$

TIE–Dialog then analyzes not only the values of these dimensions, but also:

- their turn-to-turn changes,
- the magnitude and direction of joint state movement,
- local dependence among dimensions,
- changes in that dependence structure,
- lead–lag associations,
- phase-space trajectories,
- operational event regions,
- participant-level coherence trajectories,
- and robustness across representations and parameter settings.

> **Scientific position**  
> TIE–Dialog was originally motivated by the **Theory of Informational Emergence (TIE)**, but the software is designed to remain empirically testable independently of the broader theory. Its computational outputs are operational measurements derived from dialogue representations; they do not require accepting any ontological assumptions associated with TIE.

---

# 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app provides two main usage styles:

- **Canonical mode** — uses fixed defaults for cleaner and more reproducible runs.
- **Explore mode** — exposes parameters for sensitivity analysis, robustness testing, and methodological exploration.

---

# 📁 Dataset Format

TIE–Dialog accepts `.csv` or `.xlsx` dialogue files.

## Required columns

```text
turn
participant
text
```

## Optional column

```text
timestamp
```

If `turn` is missing, the app can generate turn indices automatically.

---

# 🧩 Computational Architecture

TIE–Dialog combines contextual, structural, and geometric representations of the same conversation.

| Component | Main role |
|---|---|
| **Embeddings** | Represent turn content in a vector space |
| **IC–II / $C_t$** | Estimate contextual continuity over time |
| **$C_{\mathrm{inv}}$** | Estimate persistence of rolling graph structure |
| **IC–III** | Characterize local geometric movement in embedding space |
| **$S_t$, $R_t$, $D_t$** | Define the continuous multivariate conversational state |
| **Cross-dimensional analysis** | Measure joint movement, dependence, reorganization, and temporal association |
| **Event diagnostics** | Produce operational semantic, structural, and breakdown-like regions |
| **Robustness layer** | Test sensitivity to embeddings, baselines, shuffled order, ablations, and parameters |

The framework deliberately separates **continuous state variables** from **event-oriented diagnostic scores**. This distinction is central to the current version of TIE–Dialog.

---

# 🔎 Semantic Representation

Each turn is first mapped to an embedding representation.

Supported modes include:

- **MiniLM / SBERT**
- **E5**
- **BGE**
- **INSTRUCTOR**
- **TF-IDF fallback**

The active representation model can be changed while keeping the rest of the computational pipeline fixed, allowing the user to test whether observed dynamics are specific to one embedding space or remain partially stable across several representations.

---

# 🔹 IC–II — Contextual Coherence Dynamics

## Cₜ — contextual coherence

TIE–Dialog models coherence as a **temporally evolving contextual trajectory**, rather than as a static pairwise similarity score.

At each turn, the current utterance embedding is evaluated relative to an evolving contextual representation containing both:

- longer-term conversational memory,
- and recent local context.

The IC–II layer uses semantic resonance, local displacement, and temporal persistence to construct a contextual coherence signal:

$$
C_t \in [0,1]
$$

Interpretation:

- **higher $C_t$** → stronger continuity with the evolving conversational context,
- **lower $C_t$** → greater contextual divergence.

$C_t$ is the main contextual continuity variable used by the current framework.

---

# 🔹 Sₜ — Contextual / Semantic Drift

The first dimension of the continuous state is defined as

$$
S_t = 1 - C_t
$$

Therefore:

- **low $S_t$** → strong contextual continuity,
- **high $S_t$** → stronger contextual / semantic drift.

$S_t$ is a **continuous descriptive dimension**. It is not, by itself, a transition probability or an event label.

---

# 🔹 Structural Persistence — C_inv

TIE–Dialog also characterizes the organization of the conversation through **rolling similarity graphs**.

Within each valid rolling window:

1. turns are represented as nodes,
2. cosine similarity defines weighted relations between turns,
3. the graph is sparsified using k-nearest-neighbour connectivity,
4. a normalized graph Laplacian is constructed,
5. spectral features and weighted-degree statistics summarize graph organization.

This produces a structural feature vector

$$
\Pi(G_t)
$$

The framework compares consecutive graph representations and converts their distance into a structural persistence measure:

$$
C_{\mathrm{inv}}(t)
$$

Interpretation:

- **high $C_{\mathrm{inv}}$** → relatively persistent rolling graph organization,
- **low $C_{\mathrm{inv}}$** → stronger change in that organization.

$C_{\mathrm{inv}}$ should not be described as independent of semantics: its graphs are constructed from embedding-based similarity. Its role is instead to provide a **structural description of the representation that differs from direct contextual similarity measures such as $C_t$**.

Because $C_{\mathrm{inv}}$ requires a complete rolling window, early turns may be undefined (`NaN`) until enough observations are available.

---

# 🔹 Rₜ — Structural Reconfiguration

The second continuous state dimension is defined as

$$
R_t = 1 - C_{\mathrm{inv}}(t)
$$

Therefore:

- **low $R_t$** → relatively persistent local graph structure,
- **high $R_t$** → stronger structural reconfiguration.

$R_t$ describes change in local relational organization, rather than simple turn-to-turn semantic displacement.

---

# 🔹 IC–III — Geometric Conversational Dynamics

The IC–III layer treats the sequence of turn embeddings as a trajectory through representation space.

It extracts several local geometric descriptors.

## Local displacement — dᵢ

For consecutive normalized turn embeddings, the current implementation uses cosine-based displacement:

$$
d_i(t)
=
\frac{1-\cos\!\left(E_t,E_{t-1}\right)}{2}
$$

This quantity is subsequently smoothed and normalized within the dialogue.

Interpretation:

- **low $d_i$** → little semantic movement between consecutive turns,
- **high $d_i$** → stronger local displacement.

## Curvature proxy — κᵢ

The current implementation uses the local change in displacement as a discrete curvature proxy:

$$
\kappa_i(t)
\approx
\left|d_i(t)-d_i(t-1)\right|
$$

This should be interpreted as a **proxy for local reorientation**, not as literal differential-geometric curvature of the original embedding manifold.

## Semantic compactness — ρₜ

TIE–Dialog also estimates local semantic compactness:

$$
\rho_t \in [0,1]
$$

It summarizes how tightly neighboring utterance embeddings cluster within a local window.

- **higher $\rho_t$** → greater local semantic concentration,
- **lower $\rho_t$** → more dispersed local semantic organization.

---

# 🔹 Dₜ — Geometry Driver

The third continuous state dimension combines the geometric channels into a single bounded descriptor:

$$
D_t = f\!\left(d_i,\kappa_i,\rho_t\right)
$$

The current implementation gives greatest weight to displacement and curvature, with inverse compactness contributing additional information and a compactness-dependent gating term damping the driver in highly compact regions.

Interpretation:

- **low $D_t$** → relatively limited local geometric reconfiguration,
- **high $D_t$** → stronger local geometric change in the conversational trajectory.

$D_t$ is a continuous descriptive quantity, not an event probability.

---

# 🔷 Continuous Multivariate State

The central representation of the current framework is

$$
\mathbf{z}_t = \left(S_t, R_t, D_t\right)
$$

where:

- **$S_t$** captures contextual / semantic drift,
- **$R_t$** captures structural reconfiguration,
- **$D_t$** captures geometric trajectory change.

The three dimensions are deliberately kept separate.

This allows TIE–Dialog to investigate whether conversational transitions are better characterized by **multivariate organization** than by a single composite score.

---

# 🔷 Cross-Dimensional Dynamics

The current version of TIE–Dialog includes a dedicated continuous cross-dimensional analysis layer.

Importantly, this module uses only **$S_t$, $R_t$, and $D_t$**. Event-oriented variables such as semantic drop, structural drop, strong-event score, or transition pressure are excluded from this analysis.

## First differences

Turn-to-turn changes are defined as

$$
\Delta S_t = S_t - S_{t-1}
$$

$$
\Delta R_t = R_t - R_{t-1}
$$

$$
\Delta D_t = D_t - D_{t-1}
$$

A difference is only defined when both consecutive observations are available.

These signals describe **movement along each individual dimension**.

---

## Multivariate change magnitude — Jₜ

TIE–Dialog measures the total one-turn movement of the complete 3D state as

$$
J_t
=
\frac{
\sqrt{
(\Delta S_t)^2+
(\Delta R_t)^2+
(\Delta D_t)^2
}
}{
\sqrt{3}
}
$$

Since each state dimension is bounded to $[0,1]$, division by $\sqrt{3}$ normalizes the maximum possible raw step to 1.

Interpretation:

- **$J_t$ ≈ 0** → little joint state movement,
- **larger $J_t$** → stronger multivariate change.

$J_t$ measures **magnitude**, not direction.

It is not a transition probability.

---

## Direction of state movement

For valid non-zero movements, TIE–Dialog also computes the unit direction components

$$
\widehat{\Delta \mathbf{z}}_t
=
\frac{\Delta \mathbf{z}_t}{\left\|\Delta \mathbf{z}_t\right\|_2}
=
\left(
\frac{\Delta S_t}{\left\|\Delta \mathbf{z}_t\right\|_2},
\frac{\Delta R_t}{\left\|\Delta \mathbf{z}_t\right\|_2},
\frac{\Delta D_t}{\left\|\Delta \mathbf{z}_t\right\|_2}
\right)
$$

These values describe **which dimensions contribute to the direction of the current multivariate movement**, independently of its overall magnitude.

---

# 🔷 Local Dependence Among State Levels

The framework estimates trailing-window Pearson correlations among the continuous dimensions:

$$
r_{SR}(t),\qquad r_{SD}(t),\qquad r_{RD}(t)
$$

The window is **trailing / causal**: the value at turn t uses only observations available up to that turn.

Interpretation within a local window:

- **$r > 0$** → the two dimensions tend to vary in the same direction,
- **$r < 0$** → they tend to vary in opposite directions,
- **$r \approx 0$** → weak local linear association.

These curves represent a **time-varying local dependence structure** rather than one global correlation for the entire conversation.

Missing observations are ignored pairwise, and a minimum number of finite pairs is required before a rolling correlation is reported.

---

# 🔷 Local Dependence Among Changes

The same rolling analysis is applied to first differences:

$$
r(\Delta S,\Delta R)
$$

$$
r(\Delta S,\Delta D)
$$

$$
r(\Delta R,\Delta D)
$$

This asks a different question from correlation among levels:

> when one dimension changes from one turn to the next, do changes in another dimension tend to occur in the same direction, in the opposite direction, or independently within the recent local window?

These are descriptive associations and do not imply causal coupling.

---

# 🔷 Dependency-Structure Reorganization — Qₜ

At each valid turn, the three rolling level correlations define a local correlation structure:

$$
\mathbf{C}_t
=
\begin{bmatrix}
1 & r_{SR}(t) & r_{SD}(t) \\
r_{SR}(t) & 1 & r_{RD}(t) \\
r_{SD}(t) & r_{RD}(t) & 1
\end{bmatrix}
$$

TIE–Dialog then measures how much this structure changes between consecutive turns using a normalized Frobenius-distance formulation.

With

$$
\Delta \mathbf{r}_t
=
\left(
\Delta r_{SR}(t),
\Delta r_{SD}(t),
\Delta r_{RD}(t)
\right)
$$

the current implementation computes

$$
Q_t
=
\frac{
\sqrt{
2\left[
(\Delta r_{SR})^2+
(\Delta r_{SD})^2+
(\Delta r_{RD})^2
\right]
}
}{
\sqrt{24}
}
$$

Interpretation:

- **$Q_t$ ≈ 0** → the local dependency structure changed very little,
- **larger $Q_t$** → stronger reorganization of the relationships among S, R, and D.

$Q_t$ therefore measures **change in multivariate organization**, rather than the value or change of any single dimension.

It is not an event probability.

---

# 🔷 Lead–Lag Analysis

TIE–Dialog computes lagged Pearson associations for each pair of dimensions across a user-defined range of turn offsets.

The convention is:

- **$\mathrm{lag} > 0$** → the first named dimension leads the second,
- **$\mathrm{lag} < 0$** → the second leads the first,
- **$\mathrm{lag} = 0$** → synchronous association.

Lead–lag analysis is computed for both:

- continuous state levels,
- and first differences.

The strongest lag is summarized by absolute correlation together with the number of valid paired observations.

These results are **descriptive temporal associations**. They do not establish causal influence.

---

# 🔷 Phase-Space Views

The continuous state can also be visualized in pairwise phase spaces:

$$
(S_t,R_t),\qquad (S_t,D_t),\qquad (R_t,D_t)
$$

Successive turns form trajectories through these spaces.

These plots are useful for inspecting:

- recurrent configurations,
- excursions from locally stable regions,
- large joint movements,
- trajectory loops,
- and possible shifts between conversational regimes.

Unlike the correlation plots, phase-space views preserve the **actual joint state trajectory** rather than reducing it to a dependence coefficient.

---

# 🔹 Operational Event Diagnostics

TIE–Dialog retains an event-oriented layer alongside the continuous-state analysis.

Derived event signals include:

- semantic disruption,
- structural disruption,
- geometric contribution,
- composite strong-event scores,
- and transition-pressure-style diagnostics.

Operational labels currently include:

| Label | Computational interpretation |
|---|---|
| **STABLE** | no current event rule is met |
| **SEM_DRIFT** | semantic disruption dominates locally |
| **STRUCT_RECONFIG** | structural disruption with sufficient geometric contribution |
| **BREAKDOWN** | strong multi-signal disruption |

These regions are **derived diagnostics**. They are not identical to the continuous variables $S_t$ or $R_t$, and they should not be treated as ground-truth psychological states.

The main interface can overlay these operational regions on the coherence trajectory for exploratory inspection.

---

# 🔹 Φ Thresholds and Simplified Regime Views

The app also retains percentile-based Φ thresholds and simplified public regime visualizations derived from coherence.

These are useful for compact visualization of relatively stable and disrupted regions, but they are not the central representation used by the current cross-dimensional analysis.

Thresholds are dialogue-dependent operational estimates rather than universal constants.

---

# 🔹 Participant Trajectories — Cᵢ

TIE–Dialog can estimate participant-specific coherence trajectories.

For each participant, the app tracks how their turns align with an evolving conversational context while applying configurable temporal inertia.

These views support exploratory analysis of:

- participant-specific continuity,
- divergence,
- differential responsiveness,
- alignment asymmetries,
- and possible stabilizing or destabilizing patterns.

Participant trajectories are descriptive computational representations and should not be interpreted as direct psychological measurements.

---

# 🧪 Validation and Robustness Framework

TIE–Dialog contains several diagnostic layers for testing how dependent its outputs are on particular modeling choices.

## Embedding comparison

The same dialogue can be processed using multiple embedding systems while keeping the remaining parameters fixed.

The comparison includes:

- per-embedding summary statistics,
- overlaid coherence trajectories,
- pairwise trajectory correlations,
- DTW-based trajectory similarity,
- event-score similarity,
- event-region alignment,
- shuffled-order comparison,
- and variance decomposition across representation model and dialogue structure.

Agreement across embeddings should be interpreted as **representational robustness**, not as proof of human validity.

---

## Baseline comparison / Added Structural Value

The app compares richer TIE–Dialog outputs against simpler baselines such as:

- turn-to-turn cosine disruption,
- moving-context cosine disruption,
- geometric displacement,
- and randomized shuffled-order controls.

These diagnostics ask whether the richer representation is reducible to simpler local similarity measures, both globally and at the level of event reconstruction.

Low reducibility can indicate added trajectory-dependent structure, but it does not by itself establish that this structure corresponds to human-perceived conversational transitions.

---

## Shuffled-order controls

Randomized controls disrupt the temporal order of the dialogue while preserving the utterance set.

They are used to test whether observed coherence or event organization depends on sequential structure rather than on utterance content alone.

The framework can compare event-location overlap and displacement between the original and shuffled dialogue.

---

## Ablation diagnostics

TIE–Dialog supports component ablations such as:

- removal of the structural channel,
- removal of the geometric channel,
- contextual-coherence-only configurations.

These tests examine how much each computational layer contributes to composite event reconstruction.

---

## Parameter robustness

The app can perturb key parameters while keeping the embedding representation fixed.

The robustness analysis summarizes changes in:

- coherence trajectories,
- DTW similarity,
- event masks,
- and parameter movement across repeated perturbed runs.

This provides a sensitivity analysis of the framework rather than assuming that one parameter configuration is uniquely correct.

---

# 🧪 Batch Validation Modes

The current interface supports:

- **single-dialogue analysis**,
- **embedding batch validation**,
- **hyperparameter batch robustness**.

These modes make it possible to distinguish three different questions:

1. What dynamics appear in one conversation?
2. Which patterns survive changes in semantic representation?
3. Which patterns survive reasonable changes in model parameters?

---

# 📤 Outputs

TIE–Dialog can export turn-level results containing core and derived variables such as:

- `Ct`
- `$C_{\mathrm{inv}}$`
- `S_t`
- `R_t`
- `D_t`
- `dS_t`
- `dR_t`
- `dD_t`
- `J_t`
- movement-direction components
- rolling level correlations
- rolling change correlations
- `Q_t`
- IC–II auxiliary signals
- IC–III geometric descriptors
- participant trajectories
- operational event labels
- and additional diagnostics where enabled.

The app also provides downloadable CSV outputs and PDF reports.

---

# 📄 Reports

TIE–Dialog can generate analysis reports containing items such as:

- run configuration,
- coherence dynamics,
- core computational summaries,
- embedding robustness tables,
- baseline comparisons,
- shuffled-order diagnostics,
- ablation results,
- and parameter robustness summaries.

The report system is intended to make individual runs easier to inspect, archive, and compare.

---

# 🔬 Current Research Direction

The current research direction of TIE–Dialog is broader than simple event detection.

The framework is being used to investigate whether **human-perceived conversational transitions are associated with reproducible multivariate dynamical signatures**.

A central question is:

> **Do the temporal relationships among contextual drift, structural reconfiguration, and conversational geometry reorganize systematically around transitions perceived by human observers?**

This reframes the problem from

> “Which single score detects a transition?”

into

> “Does the conversational system exhibit a reproducible change in multivariate organization around perceived transitions?”

Candidate signatures may involve:

- coordinated movement across S, R, and D,
- unusually large multivariate steps,
- characteristic movement directions,
- changing correlation structure,
- local dependency reorganization,
- lead–lag sequences,
- or recurrent trajectories through state space.

These are **empirical hypotheses to be tested**, not assumptions built into the framework.

---

# 🎯 Research Use Cases

TIE–Dialog is designed for exploratory and methodological research involving:

- computational discourse analysis,
- conversational dynamics,
- computational linguistics,
- dialogue systems,
- human–AI interaction,
- semantic drift,
- structural reorganization,
- conversational transition analysis,
- interaction dynamics,
- representational robustness,
- and multivariate temporal organization.

At its current stage, TIE–Dialog should be understood as **research software**, not as a validated production system for diagnosing conversational states.

---

# 🧠 Conceptual Position

TIE–Dialog treats conversational organization as fundamentally temporal and multivariate.

Its current computational progression is:

```text
utterances
→ embeddings
→ contextual / structural / geometric signals
→ z_t = (S_t, R_t, D_t)
→ multivariate dynamics
→ empirical validation
```

The framework therefore focuses not only on **what utterances represent**, but also on **how conversational organization changes through time**.

A key methodological principle of the current version is that conversational transitions should not be assumed in advance to correspond to one scalar variable. Instead, TIE–Dialog makes several candidate dimensions measurable and allows their relationships to be examined empirically.

---

# ⚠️ Scientific Status and Limitations

TIE–Dialog is an evolving research framework.

Current limitations include:

- $S_t$, $R_t$, and $D_t$ are computational operationalizations rather than established psychological constructs.
- All embedding-derived channels depend to some extent on the chosen representation model.
- $C_{\mathrm{inv}}$ cannot be estimated until a complete rolling structural window is available.
- Rolling correlations can be unstable in short windows and should be interpreted locally.
- Strong rolling correlations do not imply causal interaction.
- Lead–lag associations are descriptive and do not establish temporal causality.
- Dialogue-level normalization can limit direct interpretation of absolute values across datasets.
- Event labels depend on operational rules and thresholds.
- Parameter choices can affect event-oriented outputs and some continuous trajectories.
- Agreement across embeddings demonstrates representational stability, not necessarily human validity.
- Shuffled controls, baselines, and ablations provide diagnostic evidence rather than definitive validation.
- Operational transition or breakdown regions should not be treated as ground-truth psychological states.

The purpose of the framework is therefore not to assume that a particular theory of conversational dynamics is correct, but to make candidate structures **measurable, inspectable, falsifiable, and empirically comparable**.

---

# 📌 License

TIE–Dialog is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

Modified versions deployed as network services must provide access to their corresponding source code under the terms of the license.

Commercial licensing is available upon request.

Copyright (C) 2026 Adolfo J. Céspedes Jiménez

---

# 📚 Citation

If you use TIE–Dialog in academic work, please cite the repository metadata provided in:

```text
CITATION.cff
```

Associated Zenodo releases, preprints, and related research outputs may also be cited where appropriate.

---
