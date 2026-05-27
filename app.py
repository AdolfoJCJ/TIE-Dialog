"""
TIE–Dialog Core Mechanics
Minimal Operational Framework for Conversational Coherence Dynamics

Developed by CNøde Lab
Author: Adolfo J. Cespedes Jimenez

This module implements the minimal computational core of TIE–Dialog:

1. Contextual coherence dynamics (IC-II)
2. Structural coherence via graph invariants (C_inv)
3. Geometric conversational displacement (IC-III)
4. S / T / B conversational regime detection
"""

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple


# =========================================================
# Utility Functions
# =========================================================

def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalizes embedding rows."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def cosine_similarity_matrix(E: np.ndarray) -> np.ndarray:
    """Computes pairwise cosine similarity matrix."""
    En = normalize_rows(E)
    return En @ En.T


def ema(x: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Exponential moving average."""
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return x

    y = np.empty_like(x)
    y[0] = x[0]

    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]

    return y


# =========================================================
# Structural Coherence Layer (C_inv)
# =========================================================

def build_knn_adjacency(
    E_window: np.ndarray,
    k_nn: int = 5,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Builds symmetric weighted k-NN graph.
    """

    S = cosine_similarity_matrix(E_window)

    np.fill_diagonal(S, 0.0)

    S = np.clip(S, 0.0, 1.0)

    n = S.shape[0]

    if n <= 1:
        return np.zeros((n, n), dtype=float)

    A = np.zeros_like(S)

    k = max(1, min(k_nn, n - 1))

    bias = 1e-12 * np.arange(n)

    for i in range(n):

        row = S[i] + bias

        idx = np.argpartition(row, -k)[-k:]

        for j in idx:

            if S[i, j] >= threshold:
                A[i, j] = S[i, j]

    return np.maximum(A, A.T)


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Computes normalized graph Laplacian."""

    degree = A.sum(axis=1)

    D_inv_sqrt = np.diag(
        1.0 / np.sqrt(degree + 1e-12)
    )

    return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt


def laplacian_signature(
    A: np.ndarray,
    k_eigs: int = 6,
) -> np.ndarray:
    """
    Extracts graph invariant signature from Laplacian spectrum.
    """

    if A.shape[0] < 3:
        return np.zeros(k_eigs)

    L = normalized_laplacian(A)

    eigs = np.sort(
        np.real(np.linalg.eigvalsh(L))
    )

    non_trivial = eigs[1:]

    signature = non_trivial[:k_eigs]

    if signature.size < k_eigs:

        signature = np.pad(
            signature,
            (0, k_eigs - signature.size)
        )

    degree = A.sum(axis=1)

    return np.concatenate([
        signature,
        np.array([
            degree.mean(),
            degree.std()
        ])
    ])


def compute_C_inv_series(
    E: np.ndarray,
    window: int = 8,
    k_nn: int = 5,
    threshold: float = 0.10,
    k_eigs: int = 6,
) -> np.ndarray:
    """
    Structural coherence trajectory.
    """

    E = np.asarray(E, dtype=float)

    T = E.shape[0]

    if T == 0:
        return np.zeros(0)

    signatures: List[Optional[np.ndarray]] = [None] * T

    distances = np.full(T, np.nan)

    C_inv = np.full(T, np.nan)

    for t in range(T):

        if t < window - 1:
            continue

        E_window = E[t - window + 1:t + 1]

        A = build_knn_adjacency(
            E_window,
            k_nn=k_nn,
            threshold=threshold,
        )

        signatures[t] = laplacian_signature(
            A,
            k_eigs=k_eigs,
        )

    for t in range(1, T):

        if signatures[t] is None:
            continue

        if signatures[t - 1] is None:
            continue

        distances[t] = np.linalg.norm(
            signatures[t] - signatures[t - 1]
        )

    valid = np.isfinite(distances)

    if not np.any(valid):
        return C_inv

    scale = np.nanquantile(
        distances[valid],
        0.95
    ) + 1e-12

    C_inv[valid] = (
        1.0 - np.clip(
            distances[valid] / scale,
            0.0,
            1.0
        )
    )

    return C_inv


# =========================================================
# IC-II Contextual Coherence Dynamics
# =========================================================

def compute_ic2_dynamics(
    E: np.ndarray,
    alpha_context: float = 0.84,
    baseline: float = 0.40,
) -> Dict[str, np.ndarray]:
    """
    Computes contextual coherence trajectories.
    """

    E = normalize_rows(E)

    n = E.shape[0]

    if n == 0:

        z = np.zeros(0)

        return {
            "Ct": z,
            "resonance": z,
            "identity": z,
        }

    context = np.zeros_like(E)

    context[0] = E[0]

    local_window = 4

    for t in range(1, n):

        global_context = (
            alpha_context * context[t - 1]
            + (1 - alpha_context) * E[t - 1]
        )

        local_context = np.mean(
            E[max(0, t - local_window):t],
            axis=0
        )

        context[t] = (
            0.3 * global_context
            + 0.7 * local_context
        )

    resonance = np.array([
        np.dot(E[t], context[t])
        for t in range(n)
    ])

    displacement = np.zeros(n)

    for t in range(1, n):

        displacement[t] = np.linalg.norm(
            E[t] - E[t - 1]
        )

    scale = max(
        np.quantile(displacement[1:], 0.95),
        1e-6
    )

    displacement = np.clip(
        displacement / scale,
        0.0,
        1.0
    )

    identity = np.zeros(n)

    identity[0] = 1.0

    gamma = 0.7

    for t in range(1, n):

        local_identity = (
            0.5 * resonance[t]
            + 0.5 * (1.0 - displacement[t])
        )

        recovery = (
            0.4
            * resonance[t]
            * (1.0 - identity[t - 1])
        )

        identity[t] = (
            gamma * identity[t - 1]
            + (1 - gamma) * local_identity
            + recovery
        )

    p5, p95 = np.percentile(identity, [5, 95])

    identity = np.clip(
        (identity - p5) / (p95 - p5 + 1e-9),
        0.0,
        1.0
    )

    Ct = 1.0 / (
        1.0 + np.exp(
            -3.0 * (identity - baseline)
        )
    )

    return {
        "Ct": np.clip(Ct, 0.0, 1.0),
        "resonance": resonance,
        "identity": identity,
    }


# =========================================================
# IC-III Geometric Layer
# =========================================================

def compute_ic3_geometry(
    E: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Computes conversational displacement and curvature.
    """

    E = normalize_rows(E)

    n = E.shape[0]

    if n == 0:

        z = np.zeros(0)

        return {
            "d_i": z,
            "kappa_i": z,
        }

    d_i = np.zeros(n)

    for t in range(1, n):

        similarity = np.dot(
            E[t],
            E[t - 1]
        )

        similarity = np.clip(
            similarity,
            -1.0,
            1.0
        )

        d_i[t] = 0.5 * (1.0 - similarity)

    d_i = ema(d_i, alpha=0.35)

    if d_i.size > 3:

        low, high = np.percentile(
            d_i,
            [10, 90]
        )

        if high - low > 1e-9:

            d_i = (
                d_i - low
            ) / (high - low)

            d_i = np.clip(
                d_i,
                0.0,
                1.0
            )

    kappa_i = np.zeros(n)

    kappa_i[1:] = (
        d_i[1:] - d_i[:-1]
    )

    return {
        "d_i": d_i,
        "kappa_i": kappa_i,
    }


# =========================================================
# Conversational Regimes
# =========================================================

def assign_regimes(
    Ct: np.ndarray,
    q_low: float = 0.20,
    q_high: float = 0.80,
) -> Tuple[np.ndarray, float, float]:
    """
    Assigns:
    S = Stable
    T = Transitional
    B = Break
    """

    Ct = np.asarray(Ct, dtype=float)

    Ct_valid = Ct[2:] if Ct.size > 2 else Ct

    phi_low = np.quantile(
        Ct_valid,
        q_low
    )

    phi_high = np.quantile(
        Ct_valid,
        q_high
    )

    labels = np.array(
        ["S"] * Ct.size,
        dtype=object
    )

    for t in range(Ct.size):

        if t < 2:
            labels[t] = "W"

        elif Ct[t] <= phi_low:
            labels[t] = "B"

        elif Ct[t] >= phi_high:
            labels[t] = "S"

        else:
            labels[t] = "T"

    return labels, phi_low, phi_high


# =========================================================
# Example Execution
# =========================================================

if __name__ == "__main__":

    print(
        "Running TIE–Dialog minimal operational pipeline..."
    )

    np.random.seed(42)

    mock_embeddings = np.random.randn(
        15,
        128
    )

    ic2 = compute_ic2_dynamics(
        mock_embeddings
    )

    ic3 = compute_ic3_geometry(
        mock_embeddings
    )

    C_inv = compute_C_inv_series(
        mock_embeddings,
        window=6,
        k_nn=3,
    )

    regimes, phi_low, phi_high = assign_regimes(
        ic2["Ct"]
    )

    results = pd.DataFrame({
        "turn": np.arange(1, 16),
        "Ct": ic2["Ct"],
        "C_inv": C_inv,
        "d_i": ic3["d_i"],
        "regime": regimes,
    })

    print(
        f"\nΦ_low={phi_low:.3f} | Φ_high={phi_high:.3f}"
    )

    print("\nDialogue dynamics:\n")

    print(results.to_string(index=False))
