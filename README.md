# TIE–Dialog — 📈 Conversational Dynamics Lab 📉

## A multicomponent computational framework for conversational dynamics

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog

<img width="2500" height="1875" alt="TIE–Dialog interface" src="https://github.com/user-attachments/assets/2bdb0244-5b1d-49b3-bef4-09be5c9574dd" />

---

## Overview

**TIE–Dialog** is a Streamlit-based research framework for representing and analyzing dialogue as a **continuous, evolving multicomponent system**.

Rather than treating conversation as a sequence of isolated utterances, or assuming that conversational change can be captured by one transition score, TIE–Dialog keeps several computational observables separate and examines how they evolve, dissociate, and interact over time.

The current primary state is five-dimensional:

$$
\mathbf{z}_t =
\left(
S_t,
R_t,
d_t,
\kappa_t,
u_t
\right)
$$

where:

- **$S_t$** — contextual discontinuity
- **$R_t$** — structural instability / low persistence
- **$d_t$** — local angular displacement in embedding space
- **$\kappa_t$** — turning-angle curvature of the embedding trajectory
- **$u_t$** — local semantic dispersion

A central methodological principle of the current version is:

> **Do not collapse distinct computational signals into a single transition score before their relationships have been empirically established.**

This is why the former composite **Geometry Driver $D_t$** is no longer part of the primary Research Tutorial state. It is retained only as an **exploratory / backwards-compatible diagnostic summary**.

TIE–Dialog analyzes:

- the five continuous state coordinates,
- turn-to-turn changes in each coordinate,
- total multivariate movement,
- the contribution of each coordinate to that movement,
- time-varying dependence among dimensions,
- change in the full dependency structure,
- lead–lag associations among first differences,
- operational event diagnostics,
- participant-level coherence trajectories,
- and robustness across representations and parameter settings.

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

TIE–Dialog combines contextual, structural, and geometric descriptions of the same conversation.

| Component | Main role |
|---|---|
| **Embeddings** | Represent turn content in a vector space |
| **IC–II / $C_t$** | Estimate contextual continuity over time |
| **$C_{\mathrm{inv}}$** | Estimate persistence of rolling graph structure |
| **$S_t$** | Contextual discontinuity: $1-C_t$ |
| **$R_t$** | Structural instability: $1-C_{\mathrm{inv}}$ |
| **IC–III / $d_t$** | Local angular displacement |
| **IC–III / $\kappa_t$** | Turning-angle curvature |
| **$u_t$** | Local semantic dispersion: $1-\rho_t$ |
| **Cross-dimensional analysis** | Measure joint movement, dependence, reorganization, and temporal association |
| **Event diagnostics** | Retain operational semantic, structural, and breakdown-like regions as secondary diagnostics |
| **Robustness layer** | Test sensitivity to embeddings, baselines, shuffled order, ablations, and parameters |

The framework deliberately separates:

1. **primary continuous observables**, and
2. **secondary composite / event-oriented diagnostics**.

This distinction is central to the current version.

---

# 🔎 Semantic Representation

Each conversational turn is first mapped to an embedding representation.

Supported modes include:

- **MiniLM / SBERT**
- **E5**
- **BGE**
- **INSTRUCTOR**
- **TF-IDF fallback**

The representation model can be changed while keeping the remaining computational pipeline fixed.

This makes it possible to ask whether an observed dynamic pattern is:

- specific to one embedding space,
- partially robust across representations,
- or unstable under representational changes.

Agreement across embeddings should be interpreted as **representational robustness**, not as proof of psychological or human validity.

---

# 🔹 IC–II — Contextual Coherence Dynamics

## $C_t$ — contextual continuity

TIE–Dialog models coherence as a **temporally evolving contextual trajectory**, rather than as a static pairwise similarity score.

At each turn, the current utterance embedding is evaluated relative to an evolving contextual representation containing both:

- longer-term conversational memory,
- and recent local context.

The IC–II layer uses:

- semantic resonance,
- local displacement,
- temporal persistence,
- and a recovery term

to construct:

$$
C_t \in [0,1]
$$

Interpretation:

- **higher $C_t$** → stronger continuity with the evolving conversational context
- **lower $C_t$** → greater contextual divergence

$C_t$ remains an **operational computational measure**, not a direct psychological measurement of coherence.

---

# 🔹 $S_t$ — Contextual Discontinuity

The first coordinate of the primary state is:

$$
S_t = 1-C_t
$$

Therefore:

- **low $S_t$** → stronger contextual continuity
- **high $S_t$** → greater contextual discontinuity

The current version uses the term **Contextual Discontinuity** rather than treating $S_t$ itself as a directional event called “semantic drift”.

A directional process is better represented by its change:

$$
\Delta S_t = S_t-S_{t-1}
$$

$S_t$ is a continuous state coordinate, not a transition probability.

---

# 🔹 Structural Persistence — $C_{\mathrm{inv}}$

TIE–Dialog characterizes local conversational organization through **rolling similarity graphs**.

Within each valid rolling window:

1. turns are represented as nodes,
2. cosine similarity defines weighted relations,
3. the graph is sparsified using k-nearest-neighbour connectivity,
4. a normalized graph Laplacian is constructed,
5. spectral features and weighted-degree statistics summarize graph organization.

This produces a structural feature vector:

$$
\Pi(G_t)
$$

TIE–Dialog compares consecutive graph summaries and converts their distance into:

$$
C_{\mathrm{inv}}(t)
$$

Interpretation:

- **high $C_{\mathrm{inv}}$** → relatively persistent rolling graph organization
- **low $C_{\mathrm{inv}}$** → stronger change in that organization

$C_{\mathrm{inv}}$ is not independent of semantics: the graph itself is built from embedding similarities. Its role is to provide a **relational / structural description** that differs from direct contextual continuity.

Because $C_{\mathrm{inv}}$ requires two consecutive complete rolling graph summaries, early turns are undefined until enough observations are available.

For a graph window of size $W$, the first finite $C_{\mathrm{inv}}$ can appear at zero-indexed turn $t=W$.

---

# 🔹 $R_t$ — Structural Instability

The second primary coordinate is:

$$
R_t = 1-C_{\mathrm{inv}}(t)
$$

Therefore:

- **low $R_t$** → relatively persistent local graph organization
- **high $R_t$** → greater structural instability / lower persistence

As with $S_t$, $R_t$ is a **state level**, not an event label.

Directional structural change is represented more directly by:

$$
\Delta R_t = R_t-R_{t-1}
$$

---

# 🔹 IC–III — Causal Geometric Conversational Dynamics

The IC–III layer treats the sequence of turn embeddings as a trajectory through representation space.

The current primary geometric observables are deliberately kept separate:

$$
d_t,\qquad \kappa_t,\qquad u_t
$$

They are computed causally: the value at turn $t$ uses only information available at or before $t$.

No centered future-turn window, future-derived threshold, or dialogue-wise calibration is applied to these primary geometric coordinates.

---

## Local angular displacement — $d_t$

Let $E_t$ and $E_{t-1}$ be unit-normalized consecutive turn embeddings.

TIE–Dialog computes:

$$
d_t =
\frac{
\arccos\left(
\frac{
E_t\cdot E_{t-1}
}{
\|E_t\|\|E_{t-1}\|
}
\right)
}{\pi}
$$

so that:

$$
d_t \in [0,1]
$$

Interpretation:

- **$d_t\approx0$** → consecutive embeddings point in very similar directions
- **larger $d_t$** → stronger angular displacement between consecutive turns
- **$d_t=1$** → maximally opposite directions in the normalized representation space

Angular displacement is a monotonic re-expression of cosine similarity as a distance-like quantity.

It does **not** introduce new information beyond cosine similarity; it expresses that relation in a form where:

$$
0 = \text{minimal angular change}
$$

and larger values correspond to greater local displacement.

---

## Turning-angle curvature — $\kappa_t$

Local displacement alone does not describe whether the conversational trajectory **changes direction**.

Define consecutive displacement vectors:

$$
v_{t-1}=E_{t-1}-E_{t-2}
$$

and

$$
v_t=E_t-E_{t-1}
$$

TIE–Dialog computes the normalized turning angle:

$$
\kappa_t =
\frac{
\arccos\left(
\frac{
v_{t-1}\cdot v_t
}{
\|v_{t-1}\|\|v_t\|
}
\right)
}{\pi}
$$

with:

$$
\kappa_t \in [0,1]
$$

Interpretation:

- **low $\kappa_t$** → the trajectory continues in a relatively similar direction
- **high $\kappa_t$** → the trajectory makes a stronger local turn

This is a **discrete turning-angle curvature measure** in the chosen embedding space.

It should not be interpreted as literal differential-geometric curvature of an underlying psychological or semantic manifold.

Importantly, $\kappa_t$ measures **change in trajectory direction**, not merely change in step size.

---

## Semantic compactness — $\rho_t$

TIE–Dialog also estimates local semantic compactness:

$$
\rho_t \in [0,1]
$$

Using the default configuration, $\rho_t$ is computed over a trailing local neighbourhood:

$$
[t-2,\ldots,t]
$$

The turn embeddings inside the local window are unit-normalized, and their dispersion around the local centroid is converted into a bounded compactness score.

Interpretation:

- **high $\rho_t$** → greater local semantic concentration
- **low $\rho_t$** → greater local semantic dispersion

Short conversational contributions are retained by default rather than filtered out.

---

# 🔹 $u_t$ — Local Semantic Dispersion

The fifth primary coordinate is:

$$
u_t = 1-\rho_t
$$

Therefore:

- **low $u_t$** → locally compact semantic organization
- **high $u_t$** → greater local semantic dispersion

Using $u_t$ aligns its direction with the other primary coordinates: larger values correspond to greater discontinuity, instability, displacement, turning, or dispersion.

---

# 🔷 Primary Five-Dimensional State

The primary computational representation is now:

$$
\boxed{
\mathbf{z}_t=
\left(
S_t,
R_t,
d_t,
\kappa_t,
u_t
\right)
}
$$

The five coordinates are intentionally **not collapsed into a single transition score**.

This allows TIE–Dialog to test whether conversational transitions are associated with:

- one coordinate,
- several coordinates acting together,
- characteristic dissociations,
- recurrent temporal sequences,
- or multiple families of multivariate organization.

The current framework therefore does not assume that all dimensions should rise together during a transition.

A transition may, for example, involve:

$$
d_t\uparrow,\quad
\kappa_t\uparrow
$$

without a comparable rise in $R_t$, or it may primarily involve structural instability with little local angular displacement.

Such dissociations are treated as potentially informative rather than as failures of a composite score.

---

# 🔷 First Differences

Turn-to-turn changes are computed separately for all five primary coordinates:

$$
\Delta S_t=S_t-S_{t-1}
$$

$$
\Delta R_t=R_t-R_{t-1}
$$

$$
\Delta d_t=d_t-d_{t-1}
$$

$$
\Delta \kappa_t=\kappa_t-\kappa_{t-1}
$$

$$
\Delta u_t=u_t-u_{t-1}
$$

A difference is defined only when both consecutive observations are finite.

These signals describe **directional movement** along each coordinate and are central to the current Research Tutorial analysis.

---

# 🔷 Multivariate Movement Magnitude — $J_t$

TIE–Dialog also computes a secondary summary of total one-turn movement in the complete five-dimensional state:

$$
J_t=
\frac{
\sqrt{
(\Delta S_t)^2+
(\Delta R_t)^2+
(\Delta d_t)^2+
(\Delta \kappa_t)^2+
(\Delta u_t)^2
}
}{
\sqrt{5}
}
$$

Since each coordinate is bounded to $[0,1]$, division by $\sqrt{5}$ bounds the maximum possible raw step to 1.

Interpretation:

- **$J_t\approx0$** → little complete-state movement
- **larger $J_t$** → stronger total multivariate movement

However:

> **$J_t$ is a secondary movement descriptor, not a transition detector.**

Equal numerical scaling does not establish that all five coordinates have equal reliability, equal psychological meaning, or membership in one latent construct.

The primary interpretation should therefore remain component-wise.

---

# 🔷 Direction and Composition of Multivariate Movement

For every valid non-zero step, TIE–Dialog computes the unit direction:

$$
\widehat{\Delta\mathbf{z}}_t
=
\frac{
\Delta\mathbf{z}_t
}{
\|\Delta\mathbf{z}_t\|_2
}
$$

This describes **where the five-dimensional state moved**, independently of how far it moved.

TIE–Dialog also computes squared-change contribution shares:

$$
C_j(t)=
\frac{
(\Delta z_{j,t})^2
}{
\sum_k(\Delta z_{k,t})^2
}
$$

for:

$$
j\in\{S,R,d,\kappa,u\}
$$

For a non-zero valid step:

$$
C_S+C_R+C_d+C_\kappa+C_u=1
$$

These shares prevent a large $J_t$ value from hiding **which coordinate actually produced the movement**.

---

# 🔷 Dynamic Coupling

With five primary coordinates, there are:

$$
\binom{5}{2}=10
$$

pairwise relationships.

TIE–Dialog estimates trailing-window Pearson correlations both among:

- state levels,
- and first differences.

For example:

$$
r(\Delta S,\Delta R)
$$

$$
r(\Delta S,\Delta d)
$$

$$
r(\Delta d,\Delta\kappa)
$$

$$
r(\Delta\kappa,\Delta u)
$$

Interpretation inside a local trailing window:

- **$r>0$** → the pair tends to move in the same direction
- **$r<0$** → the pair tends to move in opposite directions
- **$r\approx0$** → little local linear association

These correlations are descriptive and do not imply causal coupling.

---

# 🔷 Dependency-Structure Reorganization — $Q_t$

The ten rolling level correlations define the off-diagonal structure of a local:

$$
5\times5
$$

correlation matrix.

TIE–Dialog measures how much that complete dependency structure changes between adjacent turns using a normalized Frobenius-distance formulation.

For five dimensions:

$$
Q_t=
\frac{
\sqrt{
2\sum_{i<j}
\left[
r_{ij}(t)-r_{ij}(t-1)
\right]^2
}
}{
\sqrt{80}
}
$$

Interpretation:

- **$Q_t\approx0$** → little change in the local dependency structure
- **larger $Q_t$** → stronger reorganization of relationships among the five coordinates

$Q_t$ is a **second-order exploratory descriptor**.

It is not a primary transition metric or an event probability.

---

# 🔷 Lead–Lag Analysis

The current lead–lag analysis is performed on **first differences**.

For every one of the ten dimension pairs, TIE–Dialog computes lagged Pearson associations across a user-defined range.

Convention:

- **lag $>0$** → the first named change leads the second
- **lag $<0$** → the second leads the first
- **lag $=0$** → synchronous association

Because selecting the strongest lag retrospectively is statistically optimistic, TIE–Dialog evaluates the observed:

$$
\max_{\ell}|r_\ell|
$$

against a **circular-shift max-over-lags null**.

The resulting `p_max_over_lags` therefore controls the search across candidate lags **within each pair**.

Because ten pairwise lead–lag tests are performed, the app additionally reports a **Holm-adjusted**:

```text
p_holm
```

to control family-wise error across the ten dimension pairs.

Even with these corrections, lead–lag remains a **descriptive temporal association**, not a causal estimator.

Different coordinates also have different intrinsic temporal response properties, so proposed temporal sequences should be checked against synthetic latency calibration before being interpreted mechanistically.

---

# 🔹 Exploratory Geometry Summary — $D_t$

The current version retains a scalar geometry summary only for legacy / exploratory diagnostics:

$$
D_t^{\mathrm{exploratory}}
=
\sqrt{
\frac{
d_t^2+\kappa_t^2+u_t^2
}{3}
}
$$

This equal-weight RMS summary is **not** part of the primary state:

$$
\mathbf{z}_t\neq(S_t,R_t,D_t)
$$

and is **not passed into the primary cross-dimensional analysis**.

Its purpose is backwards compatibility and exploratory inspection.

The framework does not currently claim that angular displacement, curvature, and semantic dispersion form a validated one-dimensional latent construct.

---

# 🔹 Operational Event Diagnostics

TIE–Dialog retains a legacy / operational event-oriented layer alongside the primary continuous analysis.

Derived event signals include:

- semantic disruption,
- structural disruption,
- exploratory geometric contribution,
- composite strong-event scores,
- and transition-pressure-style diagnostics.

Operational labels include:

| Label | Computational interpretation |
|---|---|
| **STABLE** | no current event rule is met |
| **SEM_DRIFT** | semantic disruption dominates locally |
| **STRUCT_RECONFIG** | structural disruption with sufficient geometric contribution |
| **BREAKDOWN** | strong multi-signal disruption |

These labels are **secondary computational diagnostics**.

They are not identical to the primary coordinates $S_t$, $R_t$, $d_t$, $\kappa_t$, or $u_t$, and they should not be treated as ground-truth psychological states.

The Research Tutorial analysis does **not** use these legacy event labels as its primary definition of conversational transition.

---

# 🔹 Φ Thresholds and Simplified Regime Views

The app retains percentile-based $\Phi$ thresholds and simplified regime visualizations derived from coherence.

These remain useful for exploratory visualization of relatively stable and disrupted regions, but they are not the central representation used in the current multicomponent analysis.

Thresholds are dialogue-dependent operational estimates rather than universal constants.

---

# 🔹 Participant Trajectories — $C_i$

TIE–Dialog can estimate participant-specific coherence trajectories.

For each participant, the app tracks how their turns align with an evolving conversational context while applying configurable temporal inertia.

These views support exploratory analysis of:

- participant-specific continuity,
- divergence,
- differential responsiveness,
- alignment asymmetries,
- and possible stabilizing or destabilizing patterns.

Participant trajectories are computational descriptions, not direct psychological measurements.

---

# 🧪 Validation and Robustness Framework

TIE–Dialog includes several diagnostic layers for testing how dependent its outputs are on modeling choices.

## Embedding comparison

The same dialogue can be processed using multiple representation systems while keeping the remaining pipeline fixed.

The comparison can include:

- per-embedding summary statistics,
- trajectory overlays,
- pairwise trajectory correlations,
- DTW-based similarity,
- event-score similarity,
- event-region alignment,
- shuffled-order comparison,
- and variance decomposition across representation model and dialogue structure.

Agreement across embeddings demonstrates **representational stability**, not human validity.

---

## Baseline comparison

The app compares richer TIE–Dialog outputs with simpler baselines such as:

- turn-to-turn cosine disruption,
- moving-context cosine disruption,
- geometric displacement,
- and randomized shuffled-order controls.

These diagnostics ask whether richer computational structure adds information beyond simpler similarity measures.

Low reducibility does not by itself demonstrate correspondence with human-perceived transitions.

---

## Shuffled-order controls

Randomized controls disrupt conversational order while preserving the utterance set.

They are used to test whether observed organization depends on temporal sequencing rather than only on the collection of utterance contents.

---

## Ablation diagnostics

TIE–Dialog retains component-ablation diagnostics for the legacy event layer and related robustness analyses.

These tests can examine how contextual, structural, and geometric components affect operational event reconstruction.

They should not be confused with the primary five-dimensional state, whose coordinates are deliberately analyzed separately.

---

## Parameter robustness

The app can perturb key parameters while keeping the embedding representation fixed.

Robustness analyses summarize changes in:

- coherence trajectories,
- DTW similarity,
- event masks,
- and outputs across repeated perturbed runs.

This provides sensitivity analysis rather than assuming that one parameter configuration is uniquely correct.

---

# 🧪 Batch Validation Modes

The interface supports:

- **single-dialogue analysis**
- **embedding batch validation**
- **hyperparameter batch robustness**

These modes address different questions:

1. What dynamics appear in one conversation?
2. Which patterns survive changes in semantic representation?
3. Which patterns survive reasonable parameter perturbations?

---

# 📤 Outputs

Turn-level results can include:

- `Ct`
- `C_inv`
- `S_t`
- `R_t`
- `d_t`
- `kappa_t`
- `rho_t`
- `u_t`
- `dS_t`
- `dR_t`
- `dd_t`
- `dkappa_t`
- `du_t`
- `J_t`
- `J_share_S`
- `J_share_R`
- `J_share_d`
- `J_share_kappa`
- `J_share_u`
- movement-direction components
- rolling pairwise correlations
- `Q_t`
- lead–lag summaries
- `p_max_over_lags`
- `p_holm`
- IC–II auxiliary signals
- participant trajectories
- operational event labels
- `D_t_exploratory`
- and additional diagnostics where enabled

The app also provides downloadable CSV outputs, HTML plots, and PDF reports.

---

# 📄 Reports

TIE–Dialog can generate analysis reports containing items such as:

- run configuration,
- coherence dynamics,
- the five-dimensional primary state,
- geometric observables,
- multivariate movement summaries,
- embedding robustness tables,
- baseline comparisons,
- shuffled-order diagnostics,
- ablation results,
- and parameter robustness summaries.

The report system is intended to make individual runs easier to inspect, archive, and compare.

---

# 🔬 Current Research Direction

The current Research Tutorial direction is **not** to build another scalar transition detector.

The primary question is whether human-perceived conversational transitions are associated with **reproducible multicomponent temporal structure**.

A central formulation is:

> **Do regions of high collective human transition judgment exhibit recurrent multivariate temporal signatures across contextual discontinuity, structural instability, angular displacement, trajectory curvature, and local semantic dispersion?**

Equivalently:

$$
H_t
\quad\text{vs.}\quad
\mathbf{z}_t=
(S_t,R_t,d_t,\kappa_t,u_t)
$$

where the human signal remains an **external validation target** rather than being used to construct the computational coordinates.

Candidate signatures may involve:

- coordinate-specific changes,
- recurrent combinations of changes,
- dissociations among dimensions,
- unusually large multivariate movement,
- characteristic movement composition,
- changes in local coupling,
- reorganization of dependency structure,
- lead–lag sequences,
- or multiple recurrent families of transition dynamics.

These are **empirical hypotheses**, not assumptions built into the framework.

A pattern only becomes scientifically informative if it is distinguishable from appropriate control regions and generalizes beyond the particular dialogue in which it was discovered.

---

# 🎯 Research Use Cases

TIE–Dialog is designed for exploratory and methodological research involving:

- computational discourse analysis,
- conversational dynamics,
- computational linguistics,
- dialogue systems,
- human–AI interaction,
- conversational transition analysis,
- semantic trajectory analysis,
- structural reorganization,
- representation-space geometry,
- interaction dynamics,
- representational robustness,
- and multivariate temporal organization.

At its current stage, TIE–Dialog should be understood as **research software**, not as a validated production system for diagnosing conversational states.

---

# 🧠 Conceptual Position

The current computational progression is:

```text
utterances
→ embeddings
→ contextual continuity C_t
→ structural persistence C_inv
→ geometric observables d_t, κ_t, ρ_t
→ primary state z_t = (S_t, R_t, d_t, κ_t, u_t)
→ component-wise change and multivariate dynamics
→ external human validation
```

A key methodological principle is:

> **Measure candidate dimensions first; establish their empirical relationships second; construct composite variables only if the evidence justifies doing so.**

This is deliberately different from assuming in advance that every kind of conversational transition should correspond to one high scalar score.

---

# ⚠️ Scientific Status and Limitations

TIE–Dialog is an evolving research framework.

Current limitations include:

- $S_t$, $R_t$, $d_t$, $\kappa_t$, and $u_t$ are computational operationalizations rather than established psychological constructs.
- All embedding-derived channels depend on the chosen representation model.
- $S_t$ and $R_t$ are currently best interpreted as **offline within-dialogue descriptors**, because $C_t$ and $C_{\mathrm{inv}}$ use dialogue-level scaling.
- By contrast, the current $d_t$, $\kappa_t$, and $u_t$ implementations are temporally causal.
- $C_{\mathrm{inv}}$ cannot be estimated until sufficient rolling structural context is available.
- The five coordinates may have different intrinsic temporal response functions.
- Apparent lead–lag structure may therefore partly reflect measurement architecture and should be checked with synthetic latency calibration.
- Rolling correlations can be unstable in short windows.
- Correlation and lead–lag do not establish causal interaction.
- $J_t$ is a descriptive state-movement magnitude, not evidence that the five coordinates form one latent variable.
- $Q_t$ is a second-order exploratory descriptor.
- The exploratory $D_t$ summary is not a validated geometry construct.
- Event labels depend on operational rules and thresholds and are retained as secondary diagnostics.
- Parameter choices can affect event-oriented outputs and some continuous trajectories.
- Agreement across embeddings demonstrates representational stability, not necessarily human validity.
- Shuffled controls, baselines, ablations, and null models provide diagnostic evidence rather than definitive validation.
- Human transition judgments are themselves temporally uncertain and should not be treated as perfectly precise ground truth.

The purpose of the framework is not to assume that a particular theory of conversational dynamics is correct.

Its purpose is to make candidate computational structures:

**measurable, inspectable, falsifiable, comparable, and empirically testable.**

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
