# TIE–Dialog — 📈 Conversational Dynamics Lab 📉

## A multicomponent computational framework for conversational dynamics

## 🌐 Live Demo (Hugging Face Spaces)

👉 **Run TIE–Dialog in your browser:**  
https://huggingface.co/spaces/AdolfoJCJ/TIE-Dialog

<img width="1448" height="1086" alt="TIE–Dialog interface" src="https://github.com/user-attachments/assets/78915c18-727d-424e-9289-f2c255cb201b" />

---

## Overview

**TIE–Dialog** is a Streamlit-based research framework for representing and analyzing dialogue as a **continuous, evolving multicomponent system**.

Rather than treating conversation as a sequence of isolated utterances, or assuming that conversational change can be captured by one transition score, TIE–Dialog keeps several computational observables separate and examines how they evolve, dissociate, and interact over time.

The current primary state is five-dimensional:

$$
\mathbf{z}_t =\left(S_t,R_t,d_t,\kappa_t,u_t\right)
$$

where:

- **$S_t$** — contextual discontinuity
- **$R_t$** — structural instability / low persistence
- **$d_t$** — local angular displacement in embedding space
- **$\kappa_t$** — turning-angle curvature of the embedding trajectory
- **$u_t$** — local semantic dispersion

TIE–Dialog analyzes:

- the five continuous state coordinates,
- turn-to-turn changes in each coordinate,
- raw and variability-adjusted multivariate movement,
- the relative contribution of each coordinate to that movement,
- reorganization in the composition of multivariate change,
- time-varying dependence among dimensions,
- change in the full dependency structure,
- operational event diagnostics,
- participant-level coherence trajectories,
- and robustness across representations and parameter settings.

---

# 🚀 Quickstart

bash
pip install -r requirements.txt
streamlit run app.py


The app provides two main usage styles:

* **Canonical mode** — uses fixed defaults for cleaner and more reproducible runs.
* **Explore mode** — exposes parameters for sensitivity analysis, robustness testing, and methodological exploration.

---

# 📁 Dataset Format

TIE–Dialog accepts `.csv` or `.xlsx` dialogue files.

## Required columns


turn-participant-text

## Optional column

timestamp

If `turn` is missing, the app can generate turn indices automatically.

---

# 🧩 Computational Architecture

TIE–Dialog combines contextual, structural, and geometric descriptions of the same conversation.

| Component                        | Main role                                                                                    |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| **Embeddings**                   | Represent turn content in a vector space                                                     |
| **IC–II / $C_t$**                | Estimate contextual continuity over time                                                     |
| **$C_{\mathrm{inv}}$**           | Estimate persistence of rolling graph structure                                              |
| **$S_t$**                        | Contextual discontinuity: $1-C_t$                                                            |
| **$R_t$**                        | Structural instability: $1-C_{\mathrm{inv}}$                                                 |
| **IC–III / $d_t$**               | Local angular displacement                                                                   |
| **IC–III / $\kappa_t$**          | Turning-angle curvature                                                                      |
| **$u_t$**                        | Local semantic dispersion: $1-\rho_t$                                                        |
| **Multivariate movement**        | Measure raw and variability-adjusted movement through the five-dimensional state             |
| **Movement composition**         | Represent how multivariate change is distributed across the five coordinates                 |
| **Compositional reorganization** | Measure changes in the relative organization of multivariate movement                        |
| **Cross-dimensional analysis**   | Measure dependence, reorganization, and temporal association among dimensions                |
| **Event diagnostics**            | Retain operational semantic, structural, and breakdown-like regions as secondary diagnostics |
| **Robustness layer**             | Test sensitivity to embeddings, baselines, shuffled order, ablations, and parameters         |

The framework deliberately separates:

1. **primary continuous observables**, and
2. **secondary composite / event-oriented diagnostics**.

This distinction is central to the current version.

---

# 🔎 Semantic Representation

Each conversational turn is first mapped to an embedding representation.

Supported modes include:

* **MiniLM / SBERT**
* **E5**
* **BGE**
* **INSTRUCTOR**
* **TF-IDF fallback**

The representation model can be changed while keeping the remaining computational pipeline fixed.

This makes it possible to ask whether an observed dynamic pattern is:

* specific to one embedding space,
* partially robust across representations,
* or unstable under representational changes.

Agreement across embeddings should be interpreted as **representational robustness**, not as proof of psychological or human validity.

---

# 🔹 IC–II — Contextual Coherence Dynamics

## $C_t$ — contextual continuity

TIE–Dialog models coherence as a **temporally evolving contextual trajectory**, rather than as a static pairwise similarity score.

At each turn, the current utterance embedding is evaluated relative to an evolving contextual representation containing both:

* longer-term conversational memory,
* and recent local context.

The IC–II layer uses:

* semantic resonance,
* local displacement,
* temporal persistence,
* and a recovery term

to construct:

$$
C_t \in [0,1]
$$

Interpretation:

* **higher $C_t$** → stronger continuity with the evolving conversational context
* **lower $C_t$** → greater contextual divergence

$C_t$ remains an **operational computational measure**, not a direct psychological measurement of coherence.

---

# 🔹 $S_t$ — Contextual Discontinuity

The first coordinate of the primary state is:

$$
S_t = 1-C_t
$$

Therefore:

* **low $S_t$** → stronger contextual continuity
* **high $S_t$** → greater contextual discontinuity

The current version uses the term **Contextual Discontinuity** rather than treating $S_t$ itself as a directional event called “semantic drift”.

A directional process is represented more directly by its change:

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

* **high $C_{\mathrm{inv}}$** → relatively persistent rolling graph organization
* **low $C_{\mathrm{inv}}$** → stronger change in that organization

$C_{\mathrm{inv}}$ is not independent of semantics: the graph itself is built from embedding similarities.

Its role is to provide a **relational / structural description** that differs from direct contextual continuity.

Because $C_{\mathrm{inv}}$ requires two consecutive complete rolling graph summaries, early turns are undefined until enough observations are available.

For a graph window of size $W$, the first finite $C_{\mathrm{inv}}$ can appear at zero-indexed turn $t=W$.

---

# 🔹 $R_t$ — Structural Instability

The second primary coordinate is:

$$
R_t = 1-C_{\mathrm{inv}}(t)
$$

Therefore:

* **low $R_t$** → relatively persistent local graph organization
* **high $R_t$** → greater structural instability / lower persistence

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

---

## Local angular displacement — $d_t$

Let $E_t$ and $E_{t-1}$ be unit-normalized consecutive turn embeddings.

TIE–Dialog computes:

$$
d_t =\frac{\arccos\left(\frac{E_t\cdot E_{t-1}}{\|E_t\|\|E_{t-1}\|}\right)}{\pi}
$$

so that:

$$
d_t \in [0,1]
$$

Interpretation:

* **$d_t\approx0$** → consecutive embeddings point in very similar directions
* **larger $d_t$** → stronger angular displacement between consecutive turns
* **$d_t=1$** → maximally opposite directions in the normalized representation space

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

and:

$$
v_t=E_t-E_{t-1}
$$

TIE–Dialog computes the normalized turning angle:

$$
\kappa_t =\frac{\arccos\left(\frac{v_{t-1}\cdot v_t}{\|v_{t-1}\|\|v_t\|}\right)}{\pi}
$$

with:

$$
\kappa_t \in [0,1]
$$

Interpretation:

* **low $\kappa_t$** → the trajectory continues in a relatively similar direction
* **high $\kappa_t$** → the trajectory makes a stronger local turn

This is a **discrete turning-angle curvature measure** in the chosen embedding space.

It should not be interpreted as literal differential-geometric curvature of an underlying psychological or semantic manifold.

Importantly, $\kappa_t$ measures **change in trajectory direction**, not merely change in step size.

---

# 🔹 $u_t$ — Local Semantic Dispersion

The fifth primary coordinate is:

$$
u_t = 1-\rho_t
$$

Therefore:

* **low $u_t$** → locally compact semantic organization
* **high $u_t$** → greater local semantic dispersion

Using $u_t$ aligns its direction with the other primary coordinates: larger values correspond to greater discontinuity, instability, displacement, turning, or dispersion.

---

# 🔷 Primary Five-Dimensional State

The primary computational representation is:

$$
\boxed{\mathbf{z}_t=\left(S_t,R_t,d_t,\kappa_t,u_t\right)}
$$

The five coordinates are intentionally **not collapsed into a single transition score**.

This allows TIE–Dialog to test whether conversational transitions are associated with:

* one coordinate,
* several coordinates acting together,
* characteristic dissociations,
* recurrent temporal sequences,
* changes in relative movement composition,
* changes in dependency structure,
* or multiple families of multivariate organization.

The current framework therefore does not assume that all dimensions should rise together during a transition.

A transition may, for example, involve:

$$
d_t\uparrow,\qquad \kappa_t\uparrow
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

These signals describe **directional movement** along each coordinate.

---

# 🔷 Raw Multivariate Movement Magnitude — $J_t$

TIE–Dialog computes a secondary summary of total one-turn movement in the complete five-dimensional state:

$$
J_t=\frac{\sqrt{(\Delta S_t)^2+(\Delta R_t)^2+(\Delta d_t)^2+(\Delta \kappa_t)^2+(\Delta u_t)^2}}{\sqrt{5}}
$$

Since each coordinate is bounded to $[0,1]$, division by $\sqrt{5}$ bounds the maximum possible raw step to $1$.

Interpretation:

* **$J_t\approx0$** → little complete-state movement
* **larger $J_t$** → stronger total multivariate movement

However:

> **$J_t$ is a secondary movement descriptor, not a transition detector.**

Equal numerical scaling does not establish that all five coordinates have equal reliability, equal psychological meaning, or membership in one latent construct.

---

# 🔷 Variability-Adjusted Multivariate Movement — $v_{\mathrm{rel},t}$

Raw first differences can have substantially different empirical variability across coordinates.

The current multivariate analysis therefore also expresses each coordinate change relative to a robust coordinate-specific variability scale:

$$
r_{S,t}=\frac{\Delta S_t}{s_S}
$$

$$
r_{R,t}=\frac{\Delta R_t}{s_R}
$$

$$
r_{d,t}=\frac{\Delta d_t}{s_d}
$$

$$
r_{\kappa,t}=\frac{\Delta \kappa_t}{s_\kappa}
$$

$$
r_{u,t}=\frac{\Delta u_t}{s_u}
$$

where $s_j$ denotes the corresponding robust variability estimate.

The variability-adjusted movement vector is:

$$
\mathbf{r}_t=\left(r_{S,t},r_{R,t},r_{d,t},r_{\kappa,t},r_{u,t}\right)
$$

Total variability-adjusted movement is then:

$$
v_{\mathrm{rel},t}=\sqrt{\frac{1}{5}\sum_jr_{j,t}^2}
$$

or equivalently:

$$
v_{\mathrm{rel},t}=\sqrt{\frac{r_{S,t}^2+r_{R,t}^2+r_{d,t}^2+r_{\kappa,t}^2+r_{u,t}^2}{5}}
$$

Interpretation:

* **small $v_{\mathrm{rel},t}$** → relatively little multivariate movement after accounting for ordinary coordinate-specific variability
* **large $v_{\mathrm{rel},t}$** → stronger total movement relative to those variability scales

$v_{\mathrm{rel},t}$ describes **how much multivariate movement occurred**.

It does not describe how that movement is distributed across dimensions.

---

# 🔷 Composition of Variability-Adjusted Movement

For each valid non-zero variability-adjusted step, the relative contribution of each coordinate is:

$$
p_{j,t}=\frac{r_{j,t}^2}{\sum_k r_{k,t}^2}
$$

for:

$$
j\in\{S,R,d,\kappa,u\}
$$

The movement composition is therefore:

$$
\mathbf{p}_t=\left(p_{S,t},p_{R,t},p_{d,t},p_{\kappa,t},p_{u,t}\right)
$$

with:

$$
\sum_jp_{j,t}=1
$$

Whereas $v_{\mathrm{rel},t}$ describes **movement magnitude**, $\mathbf{p}_t$ describes **movement composition**.

Two turns can therefore have similar values of $v_{\mathrm{rel},t}$ while exhibiting very different internal distributions of change across the five coordinates.

---

# 🔷 Compositional Reorganization — $TV_t$ and $JSD_t$

The current Research Tutorial distinguishes between:

$$
\text{magnitude of multivariate change}
$$

and:

$$
\text{organization of multivariate change}
$$

Consecutive movement compositions are compared using Total Variation and Jensen–Shannon Divergence.

## Total Variation

$$
TV_t=\frac{1}{2}\sum_j\left|p_{j,t}-p_{j,t-1}\right|
$$

Interpretation:

* **low $TV_t$** → the relative contribution of the five coordinates remains similar
* **high $TV_t$** → the composition of multivariate movement changes strongly

## Jensen–Shannon Divergence

$$
JSD_t=JSD\left(\mathbf{p}_{t-1},\mathbf{p}_t\right)
$$

Using base-$2$ logarithms, Jensen–Shannon Divergence is symmetric and bounded.

Both $TV_t$ and $JSD_t$ describe **reorganization of the relative movement composition**.

They are distinct from total movement magnitude.

It is therefore possible to observe:

$$
TV_t\uparrow,\qquadJSD_t\uparrow
$$

without a comparably large increase in:

$$
v_{\mathrm{rel},t}
$$

This distinction is central to the current Research Tutorial.

---

# 🔷 Alternative Composition Robustness

The primary composition uses squared variability-adjusted changes:

$$
p_{j,t}^{(L2)}=\frac{r_{j,t}^2}{\sum_k r_{k,t}^2}
$$

A natural robustness alternative uses absolute magnitudes:

$$
p_{j,t}^{(L1)}=\frac{|r_{j,t}|}{\sum_k|r_{k,t}|}
$$

The $L1$ formulation reduces the influence of disproportionately large coordinate changes relative to the squared formulation.

Agreement across the $L1$ and $L2$ definitions should be interpreted as **robustness to the composition rule**, not as independent confirmation of the phenomenon.

---

# 🔷 Direction of Multivariate Movement

For every valid non-zero raw step, TIE–Dialog can compute the unit movement direction:

$$
\widehat{\Delta\mathbf{z}}_t=\frac{\Delta\mathbf{z}_t}{\|\Delta\mathbf{z}_t\|_2}
$$

This describes **where the five-dimensional state moved**, independently of how far it moved.

The framework also computes raw squared-change contribution shares:

$$
C_j(t)=\frac{(\Delta z_{j,t})^2}{\sum_k(\Delta z_{k,t})^2}
$$

for:

$$
j\in\{S,R,d,\kappa,u\}
$$

For a valid non-zero raw step:

$$
C_S+C_R+C_d+C_\kappa+C_u=1
$$

These raw shares remain useful descriptive summaries.

The current human-boundary analysis, however, focuses on the variability-adjusted composition $\mathbf{p}_t$.

---

# 🔷 Dynamic Coupling

With five primary coordinates, there are:

$$
\binom{5}{2}=10
$$

pairwise relationships.

TIE–Dialog estimates trailing-window Pearson correlations both among:

* state levels,
* and first differences.

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

* **$r>0$** → the pair tends to move in the same direction
* **$r<0$** → the pair tends to move in opposite directions
* **$r\approx0$** → little local linear association

These correlations are descriptive and do not imply causal coupling.

---

# 🔷 Dependency-Structure Reorganization — $Q_t$

The ten pairwise rolling correlations define the off-diagonal structure of a local $5\times5$ correlation matrix.

TIE–Dialog measures how much this dependency structure changes between adjacent turns using a normalized Frobenius-distance formulation:

$$
Q_t=\frac{\sqrt{2\sum_{i<j}\left[r_{ij}(t)-r_{ij}(t-1)\right]^2}}{\sqrt{80}}
$$

The denominator $\sqrt{80}$ provides a fixed theoretical normalization based on the element-wise range of the ten unique pairwise correlations.

Interpretation:

* **$Q_t\approx0$** → little change in the local dependency structure
* **larger $Q_t$** → stronger reorganization of relationships among the five coordinates

$Q_t$ is a **second-order exploratory descriptor**.

It is not a primary transition metric or an event probability.

---

# 🔹 Exploratory Geometry Summary — $D_t$

The current version retains a scalar geometry summary only for legacy / exploratory diagnostics:

$$
D_t^{\mathrm{exploratory}}=\sqrt{\frac{d_t^2+\kappa_t^2+u_t^2}{3}}
$$

This equal-weight RMS summary is **not** part of the primary state:

$$
\mathbf{z}_t\neq(S_t,R_t,D_t)
$$

and is **not passed into the primary cross-dimensional analysis**.

Its purpose is backwards compatibility and exploratory inspection.

The framework does not currently claim that angular displacement, curvature, and semantic dispersion form a validated one-dimensional latent construct.

---

# 🔹 Legacy / Secondary Event Diagnostics

TIE–Dialog retains an operational event-oriented layer alongside the primary continuous analysis.

Derived event signals can include:

* semantic disruption,
* structural disruption,
* exploratory geometric contribution,
* composite strong-event scores,
* and transition-pressure-style diagnostics.

Operational labels include:

| Label               | Computational interpretation                                 |
| ------------------- | ------------------------------------------------------------ |
| **STABLE**          | no current event rule is met                                 |
| **SEM_DRIFT**       | semantic disruption dominates locally                        |
| **STRUCT_RECONFIG** | structural disruption with sufficient geometric contribution |
| **BREAKDOWN**       | strong multi-signal disruption                               |

These labels are **secondary computational diagnostics**.

They are not identical to the primary coordinates $S_t$, $R_t$, $d_t$, $\kappa_t$, or $u_t$, and they should not be treated as ground-truth psychological states.

The current Research Tutorial does **not** use these legacy event labels as its primary definition of conversational transition.

---

## $\Phi$ Thresholds and Simplified Regime Views

The app retains percentile-based $\Phi$ thresholds and simplified regime visualizations derived from coherence.

These remain useful for exploratory visualization of relatively stable and disrupted regions, but they are not the central representation used in the current multicomponent analysis.

Thresholds are dialogue-dependent operational estimates rather than universal constants.

---

# 🔹 Participant Trajectories — $C_i$

TIE–Dialog can estimate participant-specific coherence trajectories.

For each participant, the app tracks how their turns align with an evolving conversational context while applying configurable temporal inertia.

These views support exploratory analysis of:

* participant-specific continuity,
* divergence,
* differential responsiveness,
* alignment asymmetries,
* and possible stabilizing or destabilizing patterns.

Participant trajectories are computational descriptions, not direct psychological measurements.

---

# 🧪 Validation and Robustness Framework

TIE–Dialog includes several diagnostic layers for testing how dependent its outputs are on modeling choices.

## Embedding comparison

The same dialogue can be processed using multiple representation systems while keeping the remaining pipeline fixed.

The comparison can include:

* per-embedding summary statistics,
* trajectory overlays,
* pairwise trajectory correlations,
* DTW-based similarity,
* event-score similarity,
* event-region alignment,
* shuffled-order comparison,
* and variance decomposition across representation model and dialogue structure.

Agreement across embeddings demonstrates **representational stability**, not human validity.

---

## Baseline comparison

The framework can compare richer TIE–Dialog outputs with simpler semantic and lexical baselines such as:

* SBERT turn-to-turn distance,
* SBERT moving-context distance,
* TF-IDF turn-to-turn distance,
* TF-IDF moving-context distance,
* geometric displacement,
* and randomized temporal controls.

These diagnostics ask whether richer computational structure contains information beyond simpler similarity measures.

Low reducibility does not by itself demonstrate correspondence with human-perceived transitions.

---

## Shuffled-order and temporal null controls

Randomized and temporally shifted controls can disrupt conversational alignment while preserving relevant properties of the original dialogue.

These controls ask whether observed computational structure depends on:

* actual temporal sequencing,
* actual alignment with external human events,
* or merely the distribution of values occurring within a dialogue.

For human-boundary analyses, dialogue-wise circular shifts can preserve the number and relative spacing of human events while changing their temporal alignment with computational dynamics.

---

## Ablation diagnostics

TIE–Dialog retains component-ablation diagnostics for the legacy event layer and related robustness analyses.

These tests can examine how contextual, structural, and geometric components affect operational event reconstruction.

They should not be confused with the primary five-dimensional state, whose coordinates are deliberately analyzed separately.

---

## Parameter robustness

The app can perturb key parameters while keeping the embedding representation fixed.

Robustness analyses summarize changes in:

* coherence trajectories,
* DTW similarity,
* event masks,
* and outputs across repeated perturbed runs.

This provides sensitivity analysis rather than assuming that one parameter configuration is uniquely correct.

---

# 🧪 Batch Validation Modes

The interface supports:

* **single-dialogue analysis**
* **embedding batch validation**
* **hyperparameter batch robustness**

These modes address different questions:

1. What dynamics appear in one conversation?
2. Which patterns survive changes in semantic representation?
3. Which patterns survive reasonable parameter perturbations?

---

# 📤 Outputs

Turn-level results can include:

* `Ct`
* `Ct_im`
* `C_inv`
* `S_t`
* `R_t`
* `d_t`
* `kappa_t`
* `rho_t`
* `u_t`
* `dS_t`
* `dR_t`
* `dd_t`
* `dkappa_t`
* `du_t`
* `J_t`
* `J_share_S`
* `J_share_R`
* `J_share_d`
* `J_share_kappa`
* `J_share_u`
* `v_rel_t`
* `Vrel_share_S`
* `Vrel_share_R`
* `Vrel_share_d`
* `Vrel_share_kappa`
* `Vrel_share_u`
* movement-direction components
* rolling pairwise correlations
* `Q_t`
* lead–lag summaries
* `p_max_over_lags`
* `p_holm`
* IC–II auxiliary signals
* participant trajectories
* operational event labels
* `D_t_exploratory`
* and additional diagnostics where enabled

The current Research Tutorial additionally derives:

* `TV_t`
* `JSD_t`

from the variability-adjusted movement composition.

If these quantities are not generated directly by the Streamlit application, they should be understood as **analysis-level derived variables rather than canonical app outputs**.

The app also provides downloadable CSV outputs, HTML plots, and PDF reports.

---

# 📄 Reports

TIE–Dialog can generate analysis reports containing items such as:

* run configuration,
* coherence dynamics,
* the five-dimensional primary state,
* geometric observables,
* multivariate movement summaries,
* embedding robustness tables,
* baseline comparisons,
* shuffled-order diagnostics,
* ablation results,
* and parameter robustness summaries.

The report system is intended to make individual runs easier to inspect, archive, and compare.

---

# 🔬 Current Research Direction

The current Research Tutorial does **not** aim to construct another scalar transition detector.

Its central exploratory question is:

> **Are human-consensus conversational boundaries associated with temporally localized reorganization in the relative composition of multivariate conversational change, rather than simply with increased total movement?**

The relevant movement composition is:

$$
\mathbf{p}_t=\left(p_{S,t},p_{R,t},p_{d,t},p_{\kappa,t},p_{u,t}\right)
$$

Reorganization between consecutive turns is quantified primarily through:

$$
TV_t=\frac{1}{2}\sum_j\left|p_{j,t}-p_{j,t-1}\right|
$$

and:

$$
JSD_t=JSD\left(\mathbf{p}_{t-1},\mathbf{p}_t\right)
$$

The corresponding exploratory prediction is:

$$
TV(t_0)>\text{temporally matched null expectation}
$$

and:

$$
JSD(t_0)>\text{temporally matched null expectation}
$$

without requiring a comparably strong increase in:

$$
v_{\mathrm{rel}}(t_0)
$$

Here, $t_0$ is the temporal anchor of a human-consensus transition region.

---

## Human-Consensus Criterion

Human events are defined independently of TIE–Dialog.

Candidate transition regions are localized using peaks in the aggregated human annotation field.

A candidate qualifies as a high-consensus event when at least:

$$
4\text{ of }6
$$

annotators mark a boundary within:

$$
t_0\pm3
$$

turns.

A broader:

$$
t_0\pm5
$$

criterion is used as a sensitivity analysis.

Transitions are therefore treated as **regions of collective human agreement**, rather than perfectly precise point events.

---

## Current Exploratory Evidence

In the current $12$-dialogue discovery dataset, the primary $\pm3$ criterion identifies $50$ high-consensus human events, of which $33$ are evaluable at the exact temporal anchor under the common-validity requirements of the primary multivariate measures.

Using dialogue-wise circular-shift null alignments, the current exploratory pattern is:

$$
TV(t_0)\approx20.5\%\text{ above null expectation}
$$

and:

$$
JSD(t_0)\approx29.9\%\text{ above null expectation}
$$

whereas:

$$
v_{\mathrm{rel}}(t_0)\approx7.0\%\text{ above null expectation}
$$

with substantially weaker and less consistent cross-dialogue evidence.

The central distinction is therefore:

$$
\boxed{\text{magnitude of change}\neq\text{organization of change}}
$$

A conversational boundary may coincide with a strong redistribution of multivariate change even when total movement is not exceptionally large.

---

## Temporal Localization

In the subset of events with complete data from $t_0-2$ through $t_0+2$, both $TV_t$ and $JSD_t$ show a localized maximum around $t_0$.

The same temporal concentration is not comparably clear for $v_{\mathrm{rel},t}$.

This supports describing the current effect as **temporally aligned compositional reorganization**, rather than simply increased movement somewhere in a broad transition window.

---

## Cross-Dialogue Consistency

The primary exploratory effect is distributed across dialogues rather than being attributable to a single conversation.

Under the original primary analysis, positive dialogue-level effects are observed in approximately:

$$
10/12
$$

dialogues for $TV_t$, and:

$$
10/12
$$

dialogues for $JSD_t$.

After adjustment for total movement, speaker change, and the four external semantic / lexical baselines, the corresponding directional consistency is approximately:

$$
11/12
$$

for $TV_t$, and:

$$
10/12
$$

for $JSD_t$.

A leave-one-dialogue-out diagnostic refitting the full external-control model shows positive effects after omission of each individual dialogue:

$$
12/12
$$

for both primary compositional-reorganization measures.

These leave-one-dialogue-out analyses are **influence diagnostics**, not independent replications.

---

## External Semantic and Lexical Baselines

The same human-boundary analysis has also been applied to four simpler baseline signals:

* SBERT turn-to-turn distance
* SBERT moving-context distance
* TF-IDF turn-to-turn distance
* TF-IDF moving-context distance

In the current discovery dataset, their boundary-aligned deviations from the circular-shift null are small relative to the $TV_t$ and $JSD_t$ effects.

The approximate deviations are:

$$
+0.28\%
$$

for SBERT turn-to-turn distance,

$$
-1.10\%
$$

for SBERT moving-context distance,

$$
+0.68\%
$$

for TF-IDF turn-to-turn distance,

and:

$$
+1.23\%
$$

for TF-IDF moving-context distance.

This suggests that the current compositional effect is not trivially reproduced by these simpler scalar semantic or lexical distance measures.

It does **not** establish independence from semantic representation more generally.

---

## Exploratory Mechanism

The current data do not show a stable universal direction in which one particular movement component consistently gains or loses share.

That is, there is no strong recurrent pattern of the form:

$$
\Delta p_S>0
$$

or:

$$
\Delta p_R>0
$$

or any equivalent fixed direction across all five coordinates.

Instead, the higher-order shape of the composition changes.

At human-consensus boundaries, exploratory analyses show:

$$
\Delta H(\mathbf{p})>0
$$

where $H(\mathbf{p})$ is normalized compositional entropy, together with:

$$
\Delta\max_jp_j<0
$$

relative to temporally shifted null expectations.

This is consistent with:

$$
\boxed{\text{more concentrated movement}\rightarrow\text{more distributed movement}}
$$

around the human-consensus boundary.

In other words, the boundary-aligned effect appears to involve **reduced dominance of a single movement component and greater distribution across dimensions**, rather than a fixed transfer toward one specific coordinate.

Because entropy, maximum share, $TV_t$, and $JSD_t$ are mathematically related summaries of the same composition, these results should be interpreted as **mechanistic characterization of one phenomenon**, not as independent confirmations.

---

## $L1$ / $L2$ Composition Robustness

The primary composition uses squared variability-adjusted changes:

$$
p_{j,t}^{(L2)}=\frac{r_{j,t}^2}{\sum_k r_{k,t}^2}
$$

To test whether the exploratory effect depends specifically on squaring the coordinate changes, the analysis was repeated using:

$$
p_{j,t}^{(L1)}=\frac{|r_{j,t}|}{\sum_k|r_{k,t}|}
$$

The resulting boundary-aligned deviations remain similar:

$$
TV^{(L1)}\approx20.7\%\text{ above null}
$$

and:

$$
JSD^{(L1)}\approx33.7\%\text{ above null}
$$

in the current discovery dataset.

This weakens the possibility that the phenomenon is produced solely by the squared-contribution formulation.

It does not establish representational independence more generally.

---

## Single-Scalar Reconstruction Diagnostic

A diagnostic reconstruction asks whether the apparent multivariate reorganization can be recreated from a single scalar movement trajectory.

The tested scalar source is:

$$
v_{\mathrm{rel},t}
$$

Multiple causal temporal filters are applied to this single source, allowing reconstructed component magnitudes to have different temporal response profiles.

Those reconstructed components are then converted into a synthetic movement composition, from which synthetic $TV_t$ and $JSD_t$ are computed.

In the current discovery dataset, the reconstructed boundary-aligned deviations are approximately:

$$
TV_{\mathrm{scalar}}\approx2.1\%\text{ above null}
$$

and:

$$
JSD_{\mathrm{scalar}}\approx1.4\%\text{ above null}
$$

compared with the much larger deviations in the observed multivariate composition.

An incremental diagnostic based on:

$$
TV_{\mathrm{excess},t}=TV_{\mathrm{real},t}-TV_{\mathrm{scalar},t}
$$

and:

$$
JSD_{\mathrm{excess},t}=JSD_{\mathrm{real},t}-JSD_{\mathrm{scalar},t}
$$

shows that most of the boundary-specific reorganization remains after subtracting the tested scalar reconstruction.

This provides a bounded result:

> the observed boundary-aligned compositional reorganization is not easily reconstructed from total movement alone using the tested family of simple linear causal temporal filters.

This does **not** establish that the five computational coordinates correspond to independent causal processes.

---

## Scientific Status of the Current Result

All human-boundary results described above remain **exploratory and post hoc**.

The same discovery dataset contributed to the development of the hypothesis itself.

Therefore:

* within-sample robustness is not independent confirmation,
* $TV_t$, $JSD_t$, entropy, and dominant-share analyses are mathematically related through the same movement composition,
* robustness to one alternative composition rule does not establish representation independence,
* external semantic baselines do not exhaust possible simpler explanations,
* scalar reconstruction diagnostics rule out only the tested model class,
* and stronger causal or theoretical claims require independent data.

The current Research Tutorial is therefore not intended to establish that TIE–Dialog has already identified a validated universal property of conversational transitions.

Its purpose is to determine whether the discovery dataset supports a phenomenon that is:

* specific,
* interpretable,
* internally robust,
* falsifiable,
* and sufficiently well defined to justify independent replication.

The current candidate finding can be summarized as:

> **Human-consensus conversational boundaries appear to coincide with a temporally localized redistribution in the relative structure of multivariate conversational change, characterized by reduced dominance of a single component and increased distribution across dimensions, without a comparably strong and consistent increase in total movement.**

The critical independent test is:

> **Does this compositional-reorganization effect reappear in new conversations and new human annotations under an analysis protocol fixed in advance?**

---

# 🎯 Research Use Cases

TIE–Dialog is designed for exploratory and methodological research involving:

* computational discourse analysis,
* conversational dynamics,
* computational linguistics,
* dialogue systems,
* human–AI interaction,
* conversational transition analysis,
* semantic trajectory analysis,
* structural reorganization,
* representation-space geometry,
* interaction dynamics,
* representational robustness,
* multivariate temporal organization,
* and compositional reorganization of conversational change.

At its current stage, TIE–Dialog should be understood as **research software**, not as a validated production system for diagnosing conversational states.

---

# 🧠 Conceptual Position

The current computational progression is:

utterances
→ embeddings
→ contextual continuity C_t
→ structural persistence C_inv
→ geometric observables d_t, κ_t, ρ_t
→ primary state z_t = (S_t, R_t, d_t, κ_t, u_t)
→ component-wise first differences
→ variability-adjusted multivariate movement
→ movement composition p_t
→ compositional reorganization
→ external human validation


A key methodological principle is:

> **Measure candidate dimensions first; establish their empirical relationships second; construct composite variables only if the evidence justifies doing so.**

A second principle motivated by the current Research Tutorial is:

> **The magnitude of multivariate change and the organization of multivariate change should be treated as distinct computational properties.**

This differs from assuming in advance that every kind of conversational transition should correspond to one high scalar score.

---

# ⚠️ Scientific Status and Limitations

TIE–Dialog is an evolving research framework.

Current limitations include:

* $S_t$, $R_t$, $d_t$, $\kappa_t$, and $u_t$ are computational operationalizations rather than established psychological constructs.
* All embedding-derived channels depend on the chosen representation model.
* $S_t$ and $R_t$ are currently best interpreted as **offline within-dialogue descriptors**, because $C_t$ and $C_{\mathrm{inv}}$ use dialogue-level scaling.
* By contrast, the current $d_t$, $\kappa_t$, and $u_t$ implementations are temporally causal.
* $C_{\mathrm{inv}}$ cannot be estimated until sufficient rolling structural context is available.
* The five coordinates may have different intrinsic temporal response functions.
* Apparent lead–lag structure may therefore partly reflect measurement architecture and should be checked with synthetic latency calibration.
* Rolling correlations can be unstable in short windows.
* Correlation and lead–lag do not establish causal interaction.
* $J_t$ is a descriptive state-movement magnitude, not evidence that the five coordinates form one latent variable.
* $v_{\mathrm{rel},t}$ is a variability-adjusted movement descriptor, not a transition probability.
* $\mathbf{p}_t$ is a relative composition and therefore describes contribution shares rather than absolute component strength.
* Because compositional shares sum to $1$, changes in one component's relative contribution necessarily affect the relative contributions of the others.
* $TV_t$, $JSD_t$, entropy, and maximum-share statistics are related summaries of the same underlying composition and should not be interpreted as independent confirmations.
* $Q_t$ is a second-order exploratory descriptor.
* The exploratory $D_t$ summary is not a validated geometry construct.
* Event labels depend on operational rules and thresholds and are retained as secondary diagnostics.
* Parameter choices can affect event-oriented outputs and some continuous trajectories.
* Agreement across embeddings demonstrates representational stability, not necessarily human validity.
* Shuffled controls, baselines, ablations, circular-shift nulls, reconstruction tests, and robustness analyses provide diagnostic evidence rather than definitive validation.
* Human transition judgments are themselves temporally uncertain and should not be treated as perfectly precise ground truth.
* The current human-boundary compositional-reorganization result was discovered exploratorily in a $12$-dialogue discovery sample.
* Within-sample robustness, including leave-one-dialogue-out analyses, does not constitute independent replication.
* The current scalar reconstruction diagnostic rules out only the tested family of simple temporal-filter explanations and does not establish causal independence among the five dimensions.
* Independent confirmation requires new conversational data, new human annotations, and an analysis protocol fixed before examining the replication outcomes.

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

CITATION.cff


Associated Zenodo releases, preprints, and related research outputs may also be cited where appropriate.
