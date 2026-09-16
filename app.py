# TIE-Dialog — A Computational Framework for Representing
# the Temporal Organization of Conversation
#
# Copyright (C) 2026 Adolfo J. Céspedes
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License,
# version 3, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Affero General Public License for more details.


# ============================
# app.py - (imports + labels + helpers + Public View + IC-II/IC-III core)
# ============================

import textwrap
import streamlit as st
st.set_page_config(page_title="Conversational Dynamics Lab — CNøde", layout="wide")

import math
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Sequence

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # imported (ok if unused)
from datetime import datetime

# =========================================================
# OPTIONAL LIBRARIES
# =========================================================
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    TfidfVectorizer = None
    sk_normalize = None

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False

try:
    from scipy.signal import savgol_filter, find_peaks
    _SCIPY_AVAILABLE = True
    _HAS_FIND_PEAKS = True
except Exception:
    try:
        from scipy.signal import savgol_filter
        _SCIPY_AVAILABLE = True
    except Exception:
        _SCIPY_AVAILABLE = False
    _HAS_FIND_PEAKS = False


# =========================================================
# CACHED / UTILITY FUNCTIONS (cache-safe)
# =========================================================
@st.cache_data(show_spinner=False)
def embed_texts(
    texts: List[str],
    mode: str = "auto",
    sbert_model: Optional[str] = None,
) -> Tuple[np.ndarray, str, str]:
    """
    Returns: (E, used_mode, status_msg)
    NO Streamlit calls inside (cache-safe).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    mode = (mode or "auto").lower().strip()

    # --- E5 path ---
    if mode == "e5" and _SBERT_AVAILABLE:
        try:
            model_name = (
                sbert_model.strip()
                if sbert_model and sbert_model.strip()
                else "intfloat/e5-base-v2"
            )
            model = SentenceTransformer(model_name)

            texts_e5 = [f"passage: {t}" for t in texts]
            E = model.encode(texts_e5, normalize_embeddings=True)

            return (
                np.asarray(E, dtype=float),
                f"e5:{model_name}",
                f"✅ Using E5 embeddings: {model_name}",
            )
        except Exception as e:
            msg = f"⚠️ E5 failed ({type(e).__name__}): {e}. Falling back to TF-IDF."

    # --- BGE path ---
    if mode == "bge" and _SBERT_AVAILABLE:
        try:
            model_name = (
                sbert_model.strip()
                if sbert_model and sbert_model.strip()
                else "BAAI/bge-base-en-v1.5"
            )
            model = SentenceTransformer(model_name)

            E = model.encode(texts, normalize_embeddings=True)

            return (
                np.asarray(E, dtype=float),
                f"bge:{model_name}",
                f"✅ Using BGE embeddings: {model_name}",
            )
        except Exception as e:
            msg = f"⚠️ BGE failed ({type(e).__name__}): {e}. Falling back to TF-IDF."

    # --- INSTRUCTOR path ---
    if mode == "instructor" and _SBERT_AVAILABLE:
        try:
            model_name = (
                sbert_model.strip()
                if sbert_model and sbert_model.strip()
                else "hkunlp/instructor-base"
            )
            model = SentenceTransformer(model_name)

            instruction = "Represent the dialogue to capture coherence dynamics and structural transitions:"
            texts_inst = [(instruction, t) for t in texts]

            E = model.encode(texts_inst, normalize_embeddings=True)

            return (
                np.asarray(E, dtype=float),
                f"instructor:{model_name}",
                f"✅ Using Instructor embeddings: {model_name}",
            )
        except Exception as e:
            msg = f"⚠️ Instructor failed ({type(e).__name__}): {e}. Falling back to TF-IDF."

    # --- SBERT path ---
    if mode in ("auto", "sbert") and _SBERT_AVAILABLE:
        try:
            model_name = (
                sbert_model.strip()
                if sbert_model and sbert_model.strip()
                else "sentence-transformers/all-MiniLM-L6-v2"
            )
            model = SentenceTransformer(model_name)

            E = model.encode(texts, normalize_embeddings=True)

            return (
                np.asarray(E, dtype=float),
                f"sbert:{model_name}",
                f"✅ Using SBERT embeddings: {model_name}",
            )
        except Exception as e:
            msg = f"⚠️ SBERT failed ({type(e).__name__}): {e}. Falling back to TF-IDF."

    # --- availability fallback ---
    if mode in ("sbert", "e5", "bge", "instructor") and not _SBERT_AVAILABLE:
        msg = "⚠️ sentence-transformers is not installed. Falling back to TF-IDF."
    else:
        msg = "ℹ️ Using TF-IDF embeddings."

    # --- TF-IDF / fallback ---
    if TfidfVectorizer is None:
        vocab: Dict[str, int] = {}
        rows = []

        for t in texts:
            vec = {}
            for tok in t.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                vec[vocab[tok]] = 1.0
            rows.append(vec)

        dim = len(vocab) if vocab else 1
        E = np.zeros((len(rows), dim), float)

        for i, vec in enumerate(rows):
            for j, val in vec.items():
                E[i, j] = val

        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        E = E / norms

        return E, "onehot", "⚠️ TF-IDF unavailable => using one-hot fallback."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).astype(float)

    if sk_normalize is not None:
        X = sk_normalize(X, norm="l2", axis=1, copy=False)

    return X.toarray(), "tfidf", msg

# =====================
# Bilingual labels
# =====================
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "📈 Conversational Dynamics Lab 📉",
        "app_subtitle": "Developed by CNøde — Research in Informational Systems",
        "what_does": "What does TIE-Dialog do?",
        "params": "Parameters",
        "load_demo": "Load demo",
        "expected_cols": "Expected columns: turn, timestamp (optional), participant, text.",
        "upload": "Upload a dataset (.csv or .xlsx)",
        "sem_repr": "Semantic representation",
        "emb_mode": "Embeddings mode",
        "sbert_model": "SBERT model (optional)",
        "coh_mode": "Coherence mode",
        "coh_form": "Coherence formulation",
        "ic2": "IC-II coherence dynamics",
        "geom": "IC–III geometric layer",
        "smooth": "Cₜ smoothing",
        "phi": "Φ thresholds (percentiles by default)",
        "events": "Event detection",
        "quanto_legacy": "Quanto (legacy Qa)",
        "layout": "Layout",
        "compute": "Compute metrics",
        "preview": "Dialogue preview",
        "global_metrics": "Global metrics",
        "table_title": "Turn-Level Dialogue Analysis",
        "overview": "Overview plot (TIE–Dialog)",
        "geom_plot": "IC–III geometric layer: dᵢ, κᵢ and τ(t)",
        "legacy_q": "Legacy Quantum of Coherence (Qₐ)",
        "csv": "Download CSV results",
        "download_full_csv": "Download full results (CSV)",
        "download_ic2_csv": "Download IC–IIa dynamics (CSV)",
        "download_ic3_csv": "Download IC–III geometric layer (CSV)",
        "debug_header": "Debug",
        "lang": "Language",
        "public_header": "Public View (Smoothed S–T–B)",
        "public_span": "Smoothing span (EWMA)",
        "public_show_thresholds": "Show Φ thresholds in public plot",
        "public_title": "Public plot: Smoothed coherence + S–B–R regimes",
        "ci_header": "Participant trajectories (Ci)",
        "ci_alpha": "Ci context inertia α (per-participant)",
        "ci_method": "Ci method",
        "ci_title": "Per-participant coherence trajectories (Ci)",
        "state_alpha": "State trajectory inertia α (continuous lines)",
        "state_title": "Per-participant continuous trajectories (state) + Φ thresholds",
    },
    "es": {
        "app_title": "📈 Conversational Dynamics Lab 📉",
        "app_subtitle": "Developed by CNøde — Informational Systems Lab",
        "what_does": "¿Qué hace esta app?",
        "params": "Parámetros",
        "load_demo": "Cargar demo",
        "expected_cols": "Columnas esperadas: turn, timestamp (opcional), participant, text.",
        "upload": "Sube un dataset (.csv o .xlsx)",
        "sem_repr": "Representación semántica",
        "emb_mode": "Modo de embeddings",
        "sbert_model": "Modelo SBERT (opcional)",
        "coh_mode": "Modo de coherencia",
        "coh_form": "Formulación de coherencia",
        "ic2": "Dinámica de coherencia IC-II",
        "geom": "Capa geométrica IC–III",
        "smooth": "Suavizado de Cₜ",
        "phi": "Umbrales Φ (por defecto percentiles)",
        "events": "Detección de eventos",
        "quanto_legacy": "Quanto (Qa legacy)",
        "layout": "Layout",
        "compute": "Calcular métricas",
        "preview": "Vista previa del diálogo",
        "global_metrics": "Métricas globales",
        "table_title": "Análisis dinámico por turno",
        "overview": "Plot overview (TIE–Dialog)",
        "geom_plot": "Capa geométrica IC–III: dᵢ, κᵢ y τ(t)",
        "legacy_q": "Quantum of Coherence legacy (Qₐ)",
        "csv": "Descargar resultados CSV",
        "download_full_csv": "Descargar resultados completos (CSV)",
        "download_ic2_csv": "Descargar dinámica IC–IIa (CSV)",
        "download_ic3_csv": "Descargar capa geométrica IC–III (CSV)",
        "ctx_header": "Coherencia context-aware (Ct_new)",
        "debug_header": "Debug",
        "lang": "Idioma",
        "public_header": "Vista pública (S–B–R suavizado)",
        "public_span": "Span de suavizado (EWMA)",
        "public_show_thresholds": "Mostrar umbrales Φ en plot público",
        "public_title": "Plot público: coherencia suavizada + regímenes S–B–R",
        "ci_header": "Trayectorias por participante (Ci)",
        "ci_alpha": "Inercia de contexto α (Ci por participante)",
        "ci_method": "Método Ci",
        "ci_title": "Trayectorias de coherencia por participante (Ci) + umbrales Φ",
        "state_alpha": "Inercia de trayectoria α (líneas continuas)",
        "state_title": "Trayectorias continuas por participante (state) + umbrales Φ",
    }
}

# =====================
# Warm-up configuration
# =====================

WARMUP_TURNS = 2

# =========================================================
# Canonical event / transition configuration
# =========================================================

EVENT_STRONG_THR = 0.48
EVENT_SEM_THR = 0.32
EVENT_STRUCT_THR = 0.26
EVENT_D_THR = 0.16
EVENT_SEM_MARGIN = 0.04
EVENT_STRUCT_MARGIN = 0.04

EVENT_MIN_SEM_LEN = 1
EVENT_MIN_STRUCT_LEN = 1
EVENT_MIN_STRONG_LEN = 1

# =========================================================
# Primary geometric observables
# =========================================================
GEOM_COMPACTNESS_WINDOW = 2

def apply_warmup_ramp(Ct: np.ndarray, warm: int = WARMUP_TURNS, floor: float = 0.10) -> np.ndarray:
    """
    Deprecated no-op.

    Important: do NOT distort early Ct values.
    Early turns may contain real semantic/pragmatic ruptures.
    Structural uncertainty should be handled in C_inv, not by suppressing Ct.
    """
    return np.asarray(Ct, float).copy()

# -------------------------------
# Numeric helpers
# -------------------------------
def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _savgol(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    if not _SCIPY_AVAILABLE:
        return x
    win = max(3, int(win) | 1)
    poly = max(1, int(poly))
    if win <= poly:
        win = poly + (3 if (poly % 2) == 0 else 2)
    if (win % 2) == 0:
        win += 1
    if len(x) < win:
        return x
    return savgol_filter(x, window_length=win, polyorder=poly, mode="interp")

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
    
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Classic DTW distance between two 1D sequences.
    Lower = more similar.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        return np.nan

    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],      # insertion
                dp[i, j - 1],      # deletion
                dp[i - 1, j - 1],  # match
            )

    return float(dp[n, m])


def dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Converts DTW distance into a bounded similarity score in (0, 1].
    Higher = more similar.
    """
    d = dtw_distance(a, b)
    if not np.isfinite(d):
        return np.nan
    return float(1.0 / (1.0 + d))
        
def _normalize_ct(Ct: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    Ct = np.asarray(Ct, float)
    if Ct.size == 0:
        return Ct
    p5 = float(np.percentile(Ct, 5))
    p95 = float(np.percentile(Ct, 95))
    if p95 - p5 < 1e-6:
        norm = np.full_like(Ct, 0.5)
    else:
        norm = (Ct - p5) / (p95 - p5)
    norm = np.clip(norm, 0.0, 1.0)
    span = float(upper - lower)
    return lower + span * norm

def _detrend_ct(Ct: np.ndarray, alpha: float = 0.025) -> np.ndarray:
    Ct = np.asarray(Ct, float)
    if Ct.size == 0:
        return Ct
    trend = _ema(Ct, alpha=alpha)
    Ct = Ct - (trend - np.mean(trend))
    return Ct

def _center_ct(Ct: np.ndarray, target: float = 0.60) -> np.ndarray:
    Ct = np.asarray(Ct, float)
    if Ct.size == 0:
        return Ct
    mean_Ct = float(np.mean(Ct))
    Ct = Ct + (target - mean_Ct)
    return Ct

def _norm01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)

# -------------------------------
# Invariant-based coherence (C_inv)
# -------------------------------
def _cosine_sim_matrix(E: np.ndarray) -> np.ndarray:
    """E: (n, d) embeddings. Returns cosine similarity matrix (n, n)."""
    E = np.asarray(E, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms
    return En @ En.T

def _build_weighted_adj_from_embeddings(
    E_window: np.ndarray,
    k_nn: int = 5,
    thr: float = 0.0,
    tie_eps: float = 1e-12,   
) -> np.ndarray:
    """
    Build symmetric weighted adjacency from cosine similarities.
    Uses k-NN sparsification (stable & fast).
    Deterministic tie-break avoids jitter when similarities are near-equal.
    """
    S = _cosine_sim_matrix(E_window)
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, 1.0)

    n = S.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    k = int(max(1, min(int(k_nn), n - 1)))
    A = np.zeros_like(S)

    # deterministic tie-break: add tiny increasing noise by column index
    col_bias = tie_eps * (np.arange(n, dtype=float) / max(1, n - 1))

    for i in range(n):
        row = S[i] + col_bias
        idx = np.argpartition(row, -k)[-k:]
        for j in idx:
            if S[i, j] >= thr:
                A[i, j] = S[i, j]

    A = np.maximum(A, A.T)
    return A

def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return A
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))
    I = np.eye(n)
    return I - D_inv_sqrt @ A @ D_inv_sqrt

def _laplacian_spectrum_features(A: np.ndarray, k_eigs: int = 6) -> np.ndarray:
    """
    Fixed-length feature vector from normalized Laplacian spectrum.
    Take smallest non-trivial eigenvalues (exclude ~0).
    """
    n = A.shape[0]
    k = int(max(1, k_eigs))

    if n < 3:
        return np.zeros((k,), dtype=float)

    L = _normalized_laplacian(A)
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(np.real(evals))

    evals_nt = evals[1:] if evals.size > 1 else evals
    take = evals_nt[:k]
    if take.size < k:
        take = np.pad(take, (0, k - take.size), constant_values=0.0)

    return np.clip(take.astype(float), 0.0, 2.0)

def _degree_stats(A: np.ndarray) -> np.ndarray:
    """Mean and std of weighted degree (2 dims)."""
    if A.size == 0:
        return np.zeros((2,), dtype=float)
    deg = A.sum(axis=1)
    return np.array([float(deg.mean()), float(deg.std())], dtype=float)

def compute_Pi_graph_invariants(
    E_window: np.ndarray,
    k_nn: int = 5,
    thr: float = 0.0,
    k_eigs: int = 6,
) -> np.ndarray:
    A = _build_weighted_adj_from_embeddings(E_window, k_nn=k_nn, thr=thr)
    spec = _laplacian_spectrum_features(A, k_eigs=k_eigs)
    degs = _degree_stats(A)
    return np.concatenate([spec, degs], axis=0)

def compute_C_inv_series(
    E: np.ndarray,
    window: int = 8,
    k_nn: int = 5,
    thr: float = 0.10,
    k_eigs: int = 6,
    D_max: Optional[float] = None,
) -> np.ndarray:
    """
    C_inv(t) compares Pi(G_t) vs Pi(G_{t-1}) for rolling window graphs.
    The first graph summary Pi(G_t) exists at t = window-1 (0-indexed), but
    C_inv itself needs two consecutive graph summaries. Therefore the first
    finite C_inv value can appear at t = window (turn window+1 if turns start at 1).
    Earlier values remain NaN so they cannot pollute scaling or downstream logic.
    """
    E = np.asarray(E, dtype=float)
    T = int(E.shape[0])
    if T == 0:
        return np.zeros((0,), dtype=float)

    W = int(max(3, window))
    Pi_list: List[Optional[np.ndarray]] = [None] * T
    Cinv = np.full((T,), np.nan, dtype=float)
    dists = np.full((T,), np.nan, dtype=float)

    # compute Pi only once window is full
    for t in range(T):
        if t < W - 1:
            continue
        Ewin = E[(t - W + 1):(t + 1)]
        Pi_list[t] = compute_Pi_graph_invariants(Ewin, k_nn=k_nn, thr=thr, k_eigs=k_eigs)

    # distances between consecutive valid Pi vectors
    for t in range(1, T):
        if Pi_list[t] is None or Pi_list[t - 1] is None:
            continue
        dists[t] = float(np.linalg.norm(Pi_list[t] - Pi_list[t - 1], ord=2))

    valid = np.isfinite(dists)
    if not np.any(valid):
        # if dialogue too short, return all-NaN (caller can handle)
        return Cinv

    if (D_max is None) or (not np.isfinite(D_max)) or (float(D_max) <= 0):
        scale = float(np.nanquantile(dists[valid], 0.95)) + 1e-12
    else:
        scale = float(D_max)

    Cinv[valid] = 1.0 - np.clip(dists[valid] / scale, 0.0, 1.0)
    return Cinv

def merge_consecutive(indices, gap=1):
    if not indices:
        return []
    indices = sorted(indices)
    merged = []
    last = indices[0]
    merged.append(last)
    for v in indices[1:]:
        if v - last > gap:
            merged.append(v)
        last = v
    return merged

def smooth_coherence(
    y: np.ndarray,
    method: str = "ema",
    ema_alpha: float = 0.20,
    ewma_span: int = 9,
) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    method = (method or "ema").lower().strip()
    if method == "ewma":
        ewma_span = int(max(3, ewma_span))
        return pd.Series(y).ewm(span=ewma_span, adjust=False).mean().to_numpy()
    a = float(np.clip(ema_alpha, 0.0, 0.999))
    return _ema(y, alpha=a)

# -------------------------------
# IC-II operators & helpers
# -------------------------------
def _sigma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-x))

def resonance_op(a: np.ndarray, b: np.ndarray) -> float:
    return _cos(a, b)

def discrete_derivative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    d = np.zeros_like(x)
    d[0] = 0.0
    if x.size > 1:
        d[1:] = x[1:] - x[:-1]
    return d


# =========================================================
# Primary continuous state + cross-dimensional analysis
# =========================================================
def compute_primary_state(
    Ct: np.ndarray,
    C_inv: Optional[np.ndarray],
    d_i: np.ndarray,
    kappa_i: np.ndarray,
    rho_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Construct the primary state:
        z_t = (S_t, R_t, d_t, kappa_t, u_t)

    where:
      S_t = 1 - C_t       contextual discontinuity level
      R_t = 1 - C_inv     structural instability / low persistence
      d_t                 angular displacement in embedding space
      kappa_t             turning-angle curvature of the embedding trajectory
      u_t = 1 - rho_t     local semantic dispersion

    These five coordinates are kept separate. No geometry composite is required
    for the primary analysis. If C_inv is unavailable/not yet estimable, R_t
    remains NaN and any full-state summary that requires all five coordinates is
    likewise undefined at those turns.
    """
    Ct = np.asarray(Ct, float)
    d_i = np.asarray(d_i, float)
    kappa_i = np.asarray(kappa_i, float)
    rho_t = np.asarray(rho_t, float)

    lengths = [Ct.size, d_i.size, kappa_i.size, rho_t.size]
    if C_inv is not None:
        lengths.append(np.asarray(C_inv).size)
    n = min(lengths) if lengths else 0

    Ct = Ct[:n]
    d_i = d_i[:n]
    kappa_i = kappa_i[:n]
    rho_t = rho_t[:n]

    S = np.full(n, np.nan, dtype=float)
    valid_ct = np.isfinite(Ct)
    S[valid_ct] = 1.0 - np.clip(Ct[valid_ct], 0.0, 1.0)

    R = np.full(n, np.nan, dtype=float)
    if C_inv is not None:
        C_arr = np.asarray(C_inv, float)[:n]
        valid_r = np.isfinite(C_arr)
        R[valid_r] = 1.0 - np.clip(C_arr[valid_r], 0.0, 1.0)

    d = np.full(n, np.nan, dtype=float)
    valid_d = np.isfinite(d_i)
    d[valid_d] = np.clip(d_i[valid_d], 0.0, 1.0)

    kappa = np.full(n, np.nan, dtype=float)
    valid_k = np.isfinite(kappa_i)
    kappa[valid_k] = np.clip(kappa_i[valid_k], 0.0, 1.0)

    u = np.full(n, np.nan, dtype=float)
    valid_rho = np.isfinite(rho_t)
    u[valid_rho] = 1.0 - np.clip(rho_t[valid_rho], 0.0, 1.0)

    return {
        "S_t": S,
        "R_t": R,
        "d_t": d,
        "kappa_t": kappa,
        "u_t": u,
    }


def _finite_first_difference(x: np.ndarray) -> np.ndarray:
    """First difference preserving missingness."""
    x = np.asarray(x, float)
    d = np.full(x.shape, np.nan, dtype=float)
    if x.size <= 1:
        return d
    valid = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    idx = np.where(valid)[0] + 1
    d[idx] = x[idx] - x[idx - 1]
    return d


def _masked_corr(a: np.ndarray, b: np.ndarray, min_pairs: int = 4) -> float:
    """Pearson correlation using only finite paired observations."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0:
        return np.nan
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < int(max(2, min_pairs)):
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def rolling_corr_series(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 7,
    min_pairs: Optional[int] = None,
) -> np.ndarray:
    """Trailing-window Pearson correlation using only observations up to t."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(x.size, y.size)
    out = np.full(n, np.nan, dtype=float)
    w = int(max(3, window))
    if min_pairs is None:
        min_pairs = max(4, int(math.ceil(0.60 * w)))
    min_pairs = int(min(min_pairs, w))
    for t in range(n):
        start = max(0, t - w + 1)
        out[t] = _masked_corr(
            x[start:t + 1], y[start:t + 1], min_pairs=min_pairs
        )
    return out


def lagged_corr_table(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 6,
    min_pairs: int = 6,
) -> pd.DataFrame:
    """
    Lagged Pearson association.

    lag > 0: first named signal leads the second by `lag` turns.
    lag < 0: second named signal leads the first by abs(lag) turns.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    rows = []
    m = int(max(0, max_lag))
    for lag in range(-m, m + 1):
        if lag > 0:
            xa, ya = x[:-lag], y[lag:]
        elif lag < 0:
            k = abs(lag)
            xa, ya = x[k:], y[:-k]
        else:
            xa, ya = x, y

        mask = np.isfinite(xa) & np.isfinite(ya)
        n_pairs = int(mask.sum())
        r = _masked_corr(xa, ya, min_pairs=min_pairs) if n_pairs >= min_pairs else np.nan
        rows.append({
            "lag": int(lag),
            "corr": r,
            "abs_corr": abs(r) if np.isfinite(r) else np.nan,
            "n_pairs": n_pairs,
        })
    return pd.DataFrame(rows)


def best_lag_summary(
    lag_df: pd.DataFrame,
    pair_name: str,
    representation: str,
) -> Dict[str, object]:
    """Return the strongest observed lag by absolute correlation."""
    if lag_df is None or lag_df.empty:
        return {
            "pair": pair_name, "representation": representation,
            "best_lag": np.nan, "corr": np.nan, "abs_corr": np.nan,
            "n_pairs": 0,
        }
    valid = lag_df[np.isfinite(lag_df["corr"].to_numpy(dtype=float))].copy()
    if valid.empty:
        return {
            "pair": pair_name, "representation": representation,
            "best_lag": np.nan, "corr": np.nan, "abs_corr": np.nan,
            "n_pairs": 0,
        }
    idx = valid["abs_corr"].astype(float).idxmax()
    row = valid.loc[idx]
    return {
        "pair": pair_name,
        "representation": representation,
        "best_lag": int(row["lag"]),
        "corr": float(row["corr"]),
        "abs_corr": float(row["abs_corr"]),
        "n_pairs": int(row["n_pairs"]),
    }


def max_over_lags_circular_null(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pair_name: str,
    representation: str,
    max_lag: int = 6,
    min_pairs: int = 6,
) -> Dict[str, object]:
    """Search-corrected best-lag statistic using a circular-shift null."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    observed_df = lagged_corr_table(x, y, max_lag=max_lag, min_pairs=min_pairs)
    summary = best_lag_summary(
        observed_df, pair_name=pair_name, representation=representation
    )
    observed = float(summary["abs_corr"]) if np.isfinite(summary["abs_corr"]) else np.nan
    if (n < max(4, min_pairs + 1)) or (not np.isfinite(observed)):
        return {
            **summary,
            "null_method": "circular_shift_max_over_lags",
            "null_n": 0,
            "null_mean_max_abs_r": np.nan,
            "null_q95_max_abs_r": np.nan,
            "p_max_over_lags": np.nan,
        }

    null_max = []
    for shift in range(1, n):
        y_shift = np.roll(y, shift)
        null_df = lagged_corr_table(
            x, y_shift, max_lag=max_lag, min_pairs=min_pairs
        )
        vals = null_df["abs_corr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            null_max.append(float(np.max(vals)))

    null_arr = np.asarray(null_max, float)
    if null_arr.size == 0:
        p_value = null_mean = null_q95 = np.nan
    else:
        p_value = float((1.0 + np.sum(null_arr >= observed)) / (1.0 + null_arr.size))
        null_mean = float(np.mean(null_arr))
        null_q95 = float(np.quantile(null_arr, 0.95))

    return {
        **summary,
        "null_method": "circular_shift_max_over_lags",
        "null_n": int(null_arr.size),
        "null_mean_max_abs_r": null_mean,
        "null_q95_max_abs_r": null_q95,
        "p_max_over_lags": p_value,
    }


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjustment, preserving NaNs."""
    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]
    if valid_idx.size == 0:
        return out
    pv = p[valid_idx]
    order = np.argsort(pv)
    m = len(pv)
    adjusted_sorted = np.empty(m, float)
    running = 0.0
    for rank, pos in enumerate(order):
        candidate = (m - rank) * pv[pos]
        running = max(running, candidate)
        adjusted_sorted[pos] = min(1.0, running)
    out[valid_idx] = adjusted_sorted
    return out


def compute_cross_dimensional_analysis(
    S: np.ndarray,
    R: np.ndarray,
    d: np.ndarray,
    kappa: np.ndarray,
    u: np.ndarray,
    rolling_window: int = 7,
    max_lag: int = 6,
) -> Dict[str, object]:
    """
    Descriptive dynamics of the primary five-dimensional state
        z_t=(S_t,R_t,d_t,kappa_t,u_t).

    No composite Geometry Driver enters this analysis. J_t and Q_t are secondary
    summaries only; primary interpretation should retain the vector-valued
    trajectories and their component-specific changes.
    """
    dims = {
        "S": np.asarray(S, float),
        "R": np.asarray(R, float),
        "d": np.asarray(d, float),
        "kappa": np.asarray(kappa, float),
        "u": np.asarray(u, float),
    }
    n = min(arr.size for arr in dims.values()) if dims else 0
    dims = {k: v[:n] for k, v in dims.items()}
    names = list(dims.keys())

    changes = {name: _finite_first_difference(arr) for name, arr in dims.items()}

    # Five-dimensional step magnitude. This is deliberately a secondary summary:
    # equal coordinate scaling does not imply equal construct reliability.
    valid_step = np.ones(n, dtype=bool)
    for arr in changes.values():
        valid_step &= np.isfinite(arr)

    energy = np.full(n, np.nan, dtype=float)
    if n:
        stacked_sq = np.vstack([changes[name] ** 2 for name in names])
        energy[valid_step] = np.sum(stacked_sq[:, valid_step], axis=0)

    J_raw = np.full(n, np.nan, dtype=float)
    J_raw[valid_step] = np.sqrt(energy[valid_step])
    J_t = np.full(n, np.nan, dtype=float)
    J_t[valid_step] = np.clip(J_raw[valid_step] / np.sqrt(float(len(names))), 0.0, 1.0)

    directions = {name: np.full(n, np.nan, dtype=float) for name in names}
    shares = {name: np.full(n, np.nan, dtype=float) for name in names}
    moving = valid_step & (J_raw > 1e-12)
    stationary = valid_step & (J_raw <= 1e-12)
    for name in names:
        directions[name][moving] = changes[name][moving] / J_raw[moving]
        directions[name][stationary] = 0.0
        shares[name][moving] = (changes[name][moving] ** 2) / energy[moving]

    # All 10 pairwise rolling correlations for the 5D state.
    rolling_level_corr = {}
    rolling_change_corr = {}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}|{b}"
            pairs.append((a, b, key))
            rolling_level_corr[key] = rolling_corr_series(
                dims[a], dims[b], window=rolling_window
            )
            rolling_change_corr[key] = rolling_corr_series(
                changes[a], changes[b], window=rolling_window
            )

    # Q_t: normalized Frobenius change of the 5x5 rolling correlation matrix.
    # For p dimensions, the maximum possible off-diagonal change is
    # sqrt(4*p*(p-1)); p=5 -> sqrt(80).
    p_dim = len(names)
    frob_max = np.sqrt(4.0 * p_dim * (p_dim - 1.0))
    Q_t = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        prev = np.array([rolling_level_corr[key][t - 1] for _, _, key in pairs], float)
        curr = np.array([rolling_level_corr[key][t] for _, _, key in pairs], float)
        if np.all(np.isfinite(prev)) and np.all(np.isfinite(curr)):
            dr = curr - prev
            frob = np.sqrt(2.0 * np.sum(dr ** 2))
            Q_t[t] = float(np.clip(frob / frob_max, 0.0, 1.0))

    min_pairs = max(5, min(8, int(rolling_window)))
    pretty = {"S": "S", "R": "R", "d": "d", "kappa": "κ", "u": "u"}
    lag_changes = {}
    null_rows = []
    for a, b, key in pairs:
        label = f"Δ{pretty[a]}→Δ{pretty[b]}"
        lag_changes[label] = lagged_corr_table(
            changes[a], changes[b], max_lag=max_lag, min_pairs=min_pairs
        )
        null_rows.append(
            max_over_lags_circular_null(
                changes[a], changes[b],
                pair_name=label,
                representation="changes",
                max_lag=max_lag,
                min_pairs=min_pairs,
            )
        )

    lag_null_summary = pd.DataFrame(null_rows)
    if not lag_null_summary.empty and "p_max_over_lags" in lag_null_summary.columns:
        lag_null_summary["p_holm"] = _holm_adjust(
            lag_null_summary["p_max_over_lags"].to_numpy(dtype=float)
        )

    return {
        "dimensions": dims,
        "changes": changes,
        "J_t": J_t,
        "directions": directions,
        "shares": shares,
        "rolling_level_corr": rolling_level_corr,
        "rolling_change_corr": rolling_change_corr,
        "Q_t": Q_t,
        "lag_changes": lag_changes,
        "lag_null_summary": lag_null_summary,
    }


def compute_ic2_dynamics(
    E: np.ndarray,
    alpha_context: float = 0.84,
    b: float = 0.40,
    eps: float = 1e-12,
) -> dict:
    """
Canonical IC-II contextual coherence.

1. Unit-normalize turn embeddings.

2. Construct a hybrid semantic context:
   global_ctx(t) = a * I_m(t-1) + (1-a) * E(t-1)
   local_ctx(t)  = mean of up to the previous 4 turn embeddings
   I_m(t)        = 0.3 * global_ctx(t) + 0.7 * local_ctx(t)

3. Measure:
   resonance(t)    = cosine(E(t), I_m(t))
   displacement(t) = ||E(t) - E(t-1)||

4. Construct a temporally persistent identity trajectory from
   resonance and normalized displacement, with a recovery term.

5. Convert the normalized identity trajectory into contextual
   coherence C_t using:
   C_t = sigmoid(3 * (identity_traj - b))
"""
    E = np.asarray(E, float)
    n = E.shape[0]
    if n == 0:
        z = np.zeros(0, float)
        return {"I_s": E, "I_m": E, "res": z, "dI_norm": z, "C_t": z}

    # unit-normalize embeddings for stable cosine + geometry consistency
    I_s = E.copy()
    norms = np.linalg.norm(I_s, axis=1, keepdims=True) + eps
    I_s = I_s / norms

    # --- Hybrid context (GLOBAL + LOCAL) ---
    I_m = np.zeros_like(I_s)

    I_m[0] = I_s[0]
    a = float(np.clip(alpha_context, 0.0, 0.999))
    window = 4  # 🔑 clave: 3–6

    for t in range(1, n):
        # GLOBAL (long memory)
        global_ctx = a * I_m[t - 1] + (1.0 - a) * I_s[t - 1]

        # LOCAL (last turns)
        start = max(0, t - window)
        local_ctx = np.mean(I_s[start:t], axis=0)

        # COMBINATION
        I_m[t] = 0.3 * global_ctx + 0.7 * local_ctx

    # resonance r_t
    res = np.zeros(n, float)
    for t in range(n):
        res[t] = _cos(I_s[t], I_m[t], eps=eps)

    # local displacement ΔI_t
    dI = np.zeros(n, float)
    for t in range(1, n):
        dI[t] = float(np.linalg.norm(I_s[t] - I_s[t - 1]))

    # robust scaling by 95th percentile
    s = float(np.quantile(dI[1:], 0.95)) if n > 2 else float(np.max(dI) + eps)
    s = max(s, 1e-6)
    dI_s = np.clip(dI / s, 0.0, 1.0)

# =========================
# Identity over trajectory 
# =========================
    identity_traj = np.zeros(n, float)
    identity_traj[0] = 1.0

    gamma = 0.7

    for t in range(1, n):
        local_identity = 0.5 * res[t] + 0.5 * (1.0 - dI_s[t])

        recovery = 0.4 * res[t] * (1.0 - identity_traj[t - 1])

        identity_traj[t] = (
            gamma * identity_traj[t - 1]
            + (1.0 - gamma) * local_identity
            + recovery
        )


    p5, p95 = np.percentile(identity_traj, [5, 95])
    identity_traj = (identity_traj - p5) / (p95 - p5 + 1e-8)
    identity_traj = np.clip(identity_traj, 0.0, 1.0)

    z = 3.0 * (identity_traj - float(b))
    C_t = _sigma(z)

    return {
        "I_s": I_s,
        "I_m": I_m,
        "res": res,
        "dI_norm": dI_s,
        "identity_traj": identity_traj,
        "C_t": np.clip(C_t, 0.0, 1.0),
    }

def compute_ic3_geometry(
    E: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """
    Causal intrinsic geometry of the embedding trajectory.

    d_i(t): angular displacement between consecutive unit-normalized turn
    embeddings, divided by pi so that d_i in [0,1].

    kappa_i(t): turning angle between consecutive displacement vectors
    v_{t-1}=E_{t-1}-E_{t-2} and v_t=E_t-E_{t-1}, also divided by pi.
    It measures change in trajectory direction rather than change in step size.

    Every value at t depends only on turns <= t. No dialogue-wide rescaling,
    future-derived threshold, or centered smoother is used.
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0
    if n == 0:
        return {"d_i": np.zeros(0, float), "kappa_i": np.zeros(0, float)}

    En = E.copy()
    norms = np.linalg.norm(En, axis=1, keepdims=True) + eps
    En = En / norms

    d_i = np.zeros(n, float)
    step_vecs = np.zeros_like(En)
    step_norms = np.zeros(n, float)

    for t in range(1, n):
        cos_sim = float(np.dot(En[t], En[t - 1]))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        d_i[t] = float(np.arccos(cos_sim) / np.pi)

        v = En[t] - En[t - 1]
        step_vecs[t] = v
        step_norms[t] = float(np.linalg.norm(v))

    kappa_i = np.zeros(n, float)
    for t in range(2, n):
        a = step_vecs[t - 1]
        b = step_vecs[t]
        na = step_norms[t - 1]
        nb = step_norms[t]
        if na <= eps or nb <= eps:
            continue
        cos_turn = float(np.dot(a, b) / (na * nb + eps))
        cos_turn = float(np.clip(cos_turn, -1.0, 1.0))
        kappa_i[t] = float(np.arccos(cos_turn) / np.pi)

    return {
        "d_i": np.clip(d_i, 0.0, 1.0),
        "kappa_i": np.clip(kappa_i, 0.0, 1.0),
    }


def semantic_compactness_rho(
    E: np.ndarray,
    texts: list,
    w: int = 2,
    mode: str = "centroid",
    min_tokens: int = 0,
) -> np.ndarray:
    """
    Causal local semantic compactness rho_t in [0,1].

    At t the window is [t-w,...,t]. Embeddings are unit-normalized inside this
    function. Short turns are retained by default because brief contributions
    can be interactionally important. u_t=1-rho_t is the primary dispersion
    coordinate used by the Research Tutorial analysis.
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0
    if n == 0:
        return np.zeros(0, float)

    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms

    tok_ok = np.ones(n, dtype=bool)
    if texts is not None and min_tokens is not None and int(min_tokens) > 0:
        for i, t in enumerate(texts[:n]):
            s = t if isinstance(t, str) else ""
            tok_ok[i] = len(s.strip().split()) >= int(min_tokens)

    rho = np.zeros(n, float)
    w = int(max(0, w))
    for t in range(n):
        start = max(0, t - w)
        idx = np.arange(start, t + 1)
        idx = idx[tok_ok[idx]]
        if idx.size < 2:
            rho[t] = 1.0
            continue

        window_vecs = En[idx]
        if str(mode).lower().strip() == "pairwise":
            dists = []
            for i in range(len(window_vecs)):
                for j in range(i + 1, len(window_vecs)):
                    dists.append(float(np.linalg.norm(window_vecs[i] - window_vecs[j])))
            mean_d = float(np.mean(dists)) if dists else 0.0
        else:
            centroid = window_vecs.mean(axis=0)
            dists = np.linalg.norm(window_vecs - centroid, axis=1)
            mean_d = float(dists.mean())
        rho[t] = 1.0 / (1.0 + mean_d)

    return np.clip(rho, 0.0, 1.0)


def manifold_driver_D(
    *,
    di: np.ndarray,
    kappa: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """
    Exploratory geometry magnitude retained for legacy diagnostics only.

    This equal-weight RMS summary is deliberately NOT a primary Research
    Tutorial coordinate and is never passed to compute_cross_dimensional_analysis.
    Keeping it available preserves backwards compatibility without claiming that
    displacement, curvature and dispersion form a validated scalar construct.
    """
    di = np.clip(np.asarray(di, float), 0.0, 1.0)
    kappa = np.clip(np.asarray(kappa, float), 0.0, 1.0)
    rho = np.clip(np.asarray(rho, float), 0.0, 1.0)
    n = min(di.size, kappa.size, rho.size)
    if n == 0:
        return np.zeros(0, float)
    u = 1.0 - rho[:n]
    D = np.sqrt((di[:n] ** 2 + kappa[:n] ** 2 + u ** 2) / 3.0)
    return np.clip(D, 0.0, 1.0)


def compute_all_signals(E, texts, Ct_base, C_inv, ic3):
    """Assemble primary causal observables plus an exploratory geometry summary."""
    rho_t = semantic_compactness_rho(
        E,
        texts,
        w=GEOM_COMPACTNESS_WINDOW,
        mode="centroid",
        min_tokens=0,
    )
    d_i = np.clip(np.asarray(ic3["d_i"], float), 0.0, 1.0)
    kappa_i = np.clip(np.asarray(ic3["kappa_i"], float), 0.0, 1.0)
    u_t = 1.0 - np.clip(rho_t, 0.0, 1.0)

    D_exploratory = manifold_driver_D(di=d_i, kappa=kappa_i, rho=rho_t)

    return {
        "Ct": np.asarray(Ct_base, float),
        "C_inv": np.asarray(C_inv, float) if C_inv is not None else None,
        "rho_t": np.asarray(rho_t, float),
        "u_t": np.asarray(u_t, float),
        "d_i": d_i,
        "kappa_i": kappa_i,
        # Backward-compatible aliases used by older diagnostics/downloads.
        "d_i_driver": d_i,
        "kappa_i_driver": kappa_i,
        "D_t": np.asarray(D_exploratory, float),
    }


# -------------------------------------------------
# Break detectors on exploratory geometry magnitude
# -------------------------------------------------
def detect_geom_breaks(
    *,
    D: np.ndarray,
    kappa: np.ndarray,
    D_hi: float = 0.70,
    dD_hi: float = 0.12,
    refractory: int = 2,
) -> List[int]:
    D = np.asarray(D, float)
    kappa = np.asarray(kappa, float)
    n = min(D.size, kappa.size)
    if n == 0:
        return []
    dD = np.zeros(n, float)
    dD[1:] = D[1:] - D[:-1]
    out = []
    last = -10**9
    for t in range(n):
        if t - last < int(refractory):
            continue
        if float(D[t]) >= float(D_hi) and float(dD[t]) >= float(dD_hi):
            out.append(int(t))
            last = int(t)
    return out


def match_breaks(a: List[int], b: List[int], delta_max: int = 6) -> List[Dict[str, int]]:
    """Match indices in a to nearest indices in b within +/-delta_max."""
    A = [int(x) for x in (a or [])]
    B = [int(x) for x in (b or [])]
    if not A or not B:
        return []
    out = []
    for i in A:
        cand = [(j, abs(j - i)) for j in B if abs(j - i) <= int(delta_max)]
        if not cand:
            continue
        j = min(cand, key=lambda z: z[1])[0]
        out.append({"a": int(i), "b": int(j), "lag": int(j - i)})
    return out


def plot_ic3_geometry(
    d_i: np.ndarray,
    kappa_i: np.ndarray,
    u_t: np.ndarray,
    height: int = 560,
) -> go.Figure:
    """Plot the exact three geometric coordinates used in the primary 5D state."""
    d_plot = np.clip(np.asarray(d_i, float), 0.0, 1.0)
    kappa_plot = np.clip(np.asarray(kappa_i, float), 0.0, 1.0)
    u_plot = np.clip(np.asarray(u_t, float), 0.0, 1.0)
    n = min(len(d_plot), len(kappa_plot), len(u_plot))
    x = np.arange(1, n + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d_plot[:n], mode="lines", name="dₜ — angular displacement"))
    fig.add_trace(go.Scatter(x=x, y=kappa_plot[:n], mode="lines", name="κₜ — turning-angle curvature"))
    fig.add_trace(go.Scatter(x=x, y=u_plot[:n], mode="lines", name="uₜ — local semantic dispersion (1−ρₜ)"))
    fig.update_layout(
        title="Primary geometric observables",
        height=int(height),
        margin=dict(l=40, r=260, t=45, b=40),
        xaxis_title="Turn",
        yaxis_title="Value (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
    )
    return fig

# -------------------------------
# Data input
# -------------------------------
REQUIRED_COLS = ["turn", "participant", "text"]

def _clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    for c in missing:
        if c == "turn":
            df[c] = np.arange(1, len(df) + 1, dtype=int)
        else:
            df[c] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    try:
        df["turn"] = df["turn"].astype(int)
    except Exception:
        df["turn"] = pd.to_numeric(df["turn"], errors="coerce").fillna(0).astype(int)
    df["participant"] = df["participant"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["turn"] > 0].sort_values("turn").reset_index(drop=True)
    return df, missing

def plot_ci_lines(
    turns: np.ndarray,
    ci_df: pd.DataFrame,
    height: int = 520,
    title: str = "Per-participant coherence (Ci)",
) -> go.Figure:
    turns = np.asarray(turns, dtype=int)

    fig = go.Figure()

    if ci_df is None or ci_df.shape[1] == 0:
        fig.update_layout(
            title=title,
            height=int(height),
            margin=dict(l=40, r=200, t=40, b=40),
            xaxis_title="Turn",
            yaxis_title="Ci (0–1)",
            yaxis=dict(range=[0, 1]),
        )
        return fig

    # Plot every Ci_* column
    for col in ci_df.columns:
        if not str(col).startswith("Ci_"):
            continue
        y = np.asarray(ci_df[col].to_numpy(), dtype=float)
        n = min(turns.size, y.size)
        fig.add_trace(
            go.Scatter(
                x=turns[:n],
                y=np.clip(y[:n], 0.0, 1.0),
                mode="lines",
                name=str(col).replace("Ci_", ""),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=40, b=40),
        xaxis_title="Turn",
        yaxis_title="Ci (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig

# ============================
# app.py — PART 2/3
# (events + SBR + quantos + context-aware + Ci computation + plots)
# Ci plot and State plot are both available (defined elsewhere)
# plot_ct_main no longer references undefined friction/geom_* vars
# removed dead/unreachable code under _base_layout
# plot_ct_main now returns fig and includes markers/thresholds/annotations
# ============================

def parse_turn_string(x):

    if pd.isna(x):
        return []

    s = str(x).strip()

    if s == "":
        return []

    return [
        int(v.strip()) - 1
        for v in s.split(",")
        if v.strip() != ""
    ]

def points_to_mask(points: List[int], n: int, w: int) -> np.ndarray:
    """points are 0-indexed; returns boolean mask length n."""
    m = np.zeros(n, dtype=bool)
    if n <= 0:
        return m
    w = int(max(0, w))
    for p in points:
        p = int(p)
        a = max(0, p - w)
        b = min(n - 1, p + w)
        m[a:b + 1] = True
    return m


def mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous [start,end] segments where mask True (0-indexed)."""
    mask = np.asarray(mask, bool)
    segs: List[Tuple[int, int]] = []
    i, n = 0, mask.size
    while i < n:
        if mask[i]:
            s = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            segs.append((s, i))
        i += 1
    return segs

def enforce_min_persistence(mask, min_len=2):
    mask = np.asarray(mask, dtype=bool)
    segments = mask_to_segments(mask)
    clean = np.zeros_like(mask, dtype=bool)

    for s, e in segments:
        if (e - s + 1) >= int(min_len):
            clean[s:e+1] = True

    return clean
    
# -------------------------------
# Multi-signal event scoring
# -------------------------------
def compute_event_scores(
    Ct,
    C_inv,
    D_t,
    strong_weights=(0.25, 0.35, 0.40),
    transition_weights=(0.15, 0.25, 0.60),
):
    Ct = np.asarray(Ct, float)
    n = len(Ct)

    dCt = np.zeros(n, float)
    dCt[1:] = Ct[1:] - Ct[:-1]
    sem_drop = np.clip(-dCt, 0.0, 1.0)
    sem_drop = _ema(sem_drop, alpha=0.35)
    sem_drop[~np.isfinite(sem_drop)] = 0.0

    if C_inv is not None:
        C_inv = np.asarray(C_inv, float)
        dCinv = np.zeros(n, float)
        dCinv[1:] = C_inv[1:] - C_inv[:-1]

        struct_drop = np.clip(-dCinv, 0.0, 1.0)
        struct_drop[~np.isfinite(struct_drop)] = 0.0
        struct_drop = _ema(struct_drop, alpha=0.35)

        positive_mask = struct_drop > 0
        if np.any(positive_mask):
            noise_floor = np.quantile(struct_drop[positive_mask], 0.60)
            struct_drop[struct_drop < noise_floor] = 0.0

        if np.max(struct_drop) > 1e-6:
            struct_drop = struct_drop / np.max(struct_drop)
    else:
        struct_drop = np.zeros(n, float)

    D_t = np.asarray(D_t, float)
    D_t[~np.isfinite(D_t)] = 0.0

    sw_sem, sw_struct, sw_geom = [float(v) for v in strong_weights]
    sw_total = sw_sem + sw_struct + sw_geom
    if sw_total <= 0:
        sw_sem, sw_struct, sw_geom = 0.25, 0.35, 0.40
    else:
        sw_sem, sw_struct, sw_geom = (
            sw_sem / sw_total,
            sw_struct / sw_total,
            sw_geom / sw_total,
        )

    strong_score = (
        sw_sem * sem_drop +
        sw_struct * struct_drop +
        sw_geom * D_t
    )
    strong_score = np.clip(strong_score, 0.0, 1.0)

    semantic_score = 0.55 * sem_drop + 0.20 * (1.0 - struct_drop) + 0.25 * D_t
    structural_score = 0.70 * struct_drop + 0.15 * (1.0 - sem_drop) + 0.15 * D_t

    tw_sem, tw_struct, tw_geom = [float(v) for v in transition_weights]
    tw_total = tw_sem + tw_struct + tw_geom
    if tw_total <= 0:
        tw_sem, tw_struct, tw_geom = 0.15, 0.25, 0.60
    else:
        tw_sem, tw_struct, tw_geom = (
            tw_sem / tw_total,
            tw_struct / tw_total,
            tw_geom / tw_total,
        )

    transition_pressure = (
        tw_sem * sem_drop +
        tw_struct * struct_drop +
        tw_geom * np.asarray(D_t, float)
    )

    transition_pressure = np.clip(
        transition_pressure,
        0.0,
        1.0,
    )
    
    return {
        "sem_drop": np.clip(sem_drop, 0.0, 1.0),
        "struct_drop": np.clip(struct_drop, 0.0, 1.0),
        "strong_score": np.clip(strong_score, 0.0, 1.0),
        "semantic_score": np.clip(semantic_score, 0.0, 1.0),
        "structural_score": np.clip(structural_score, 0.0, 1.0),
        "transition_pressure": transition_pressure,
    }


# -------------------------------
# Baseline comparison helpers
# -------------------------------
def compute_baseline_dynamics(E: np.ndarray, moving_window: int = 5) -> Dict[str, np.ndarray]:
    """
    Simple baseline dynamics.

    Outputs are disruption/event-pressure signals in [0, 1]:
    higher values mean larger local/contextual disruption.
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0

    if n == 0:
        z = np.zeros(0, float)
        return {"cosine": z, "moving_avg": z}

    En = E.copy()
    En = En / (np.linalg.norm(En, axis=1, keepdims=True) + 1e-12)

    cosine_disruption = np.zeros(n, float)
    moving_avg_disruption = np.zeros(n, float)

    for t in range(1, n):
        cos_sim = float(np.dot(En[t], En[t - 1]))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        sim01 = 0.5 * (cos_sim + 1.0)
        cosine_disruption[t] = 1.0 - sim01

        a = max(0, t - int(moving_window))
        ctx = np.mean(En[a:t], axis=0)
        ctx = ctx / (np.linalg.norm(ctx) + 1e-12)
        ctx_sim = float(np.dot(En[t], ctx))
        ctx_sim = float(np.clip(ctx_sim, -1.0, 1.0))
        ctx_sim01 = 0.5 * (ctx_sim + 1.0)
        moving_avg_disruption[t] = 1.0 - ctx_sim01

    return {
        "cosine": np.clip(_norm01(cosine_disruption), 0.0, 1.0),
        "moving_avg": np.clip(_norm01(moving_avg_disruption), 0.0, 1.0),
    }


def extract_event_centers(signal, thr, gap=1):
    x = np.asarray(signal, float)
    mask = np.isfinite(x) & (x >= thr)
    points = np.where(mask)[0].tolist()

    if not points:
        return [], []

    groups = []
    current = [points[0]]

    for p in points[1:]:
        if p - current[-1] <= gap:
            current.append(p)
        else:
            groups.append(current)
            current = [p]

    groups.append(current)

    centers = []
    strengths = []

    for g in groups:
        center = int(round(np.mean(g)))
        centers.append(center)
        strengths.append(float(np.nanmean(x[g])))

    return centers, strengths


def event_overlap_rate(reference_events, comparison_events, tolerance: int = 2) -> float:
    """Share of reference events reconstructed by comparison events within ±tolerance turns."""
    if not reference_events:
        return np.nan

    matches = 0
    for e in reference_events:
        if any(abs(int(e) - int(b)) <= int(tolerance) for b in comparison_events):
            matches += 1

    return float(matches / len(reference_events))
    
def compute_iou(mask_a, mask_b):

    inter = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return np.nan

    return inter / union
    

def mean_event_displacement(reference_events, comparison_events) -> float:
    """
    Mean distance, in turns, from each real TIE event to the closest shuffled event.
    Higher values mean shuffled events occur in different temporal locations.
    """
    if not reference_events or not comparison_events:
        return np.nan

    distances = []
    for e in reference_events:
        closest = min(abs(int(e) - int(b)) for b in comparison_events)
        distances.append(closest)

    return float(np.mean(distances))

def compute_random_shuffled_baseline(
    *,
    E: np.ndarray,
    texts: List[str],
    participants: List[str],
    smooth_method: str,
    env_alpha: float,
    env_span: int,
    use_cinv: bool,
    cinv_window: int,
    cinv_knn: int,
    cinv_thr: float,
    cinv_keigs: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Random shuffled dialogue baseline.

    It preserves the same utterances but destroys the original temporal order.
    The returned signal is the shuffled dialogue's strong event score.
    """
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(texts))

    E_shuf = np.asarray(E, float)[idx]
    texts_shuf = [texts[i] for i in idx]
    participants_shuf = [participants[i] for i in idx]

    run_shuf = run_pipeline_from_embeddings(
        E=E_shuf,
        texts=texts_shuf,
        participants=participants_shuf,
        smooth_method=str(smooth_method),
        env_alpha=float(env_alpha),
        env_span=int(env_span),
        use_cinv=bool(use_cinv),
        cinv_window=int(cinv_window),
        cinv_knn=int(cinv_knn),
        cinv_thr=float(cinv_thr),
        cinv_keigs=int(cinv_keigs),
    )

    return np.asarray(run_shuf["event_scores"]["strong_score"], float)

def build_baseline_metrics_table(signals: Dict[str, np.ndarray], event_thr: float = 0.42) -> pd.DataFrame:
    rows = []
    for name, values in signals.items():
        x = np.asarray(values, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            rows.append({"Signal": name, "Mean": np.nan, "Std": np.nan, "Max": np.nan, "Event-like points": 0})
            continue
        rows.append({
            "Signal": name,
            "Mean": float(np.mean(x)),
            "Std": float(np.std(x)),
            "Max": float(np.max(x)),
            "Event-like points": int(np.sum(x >= float(event_thr))),
        })
    return pd.DataFrame(rows)


def plot_baseline_comparison(signals: Dict[str, np.ndarray], title: str = "Baseline comparison") -> go.Figure:
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()

    n = max((len(np.asarray(v)) for v in signals.values()), default=0)
    x = np.arange(1, n + 1)

    styles = {
        "TIE–Dialog composite": dict(width=4, dash="solid"),
        "Turn-to-turn cosine": dict(width=2, dash="solid"),
        "Moving-context cosine": dict(width=2, dash="dash"),
        "Geometric displacement": dict(width=2, dash="dashdot"),
        "Random shuffled baseline": dict(width=2, dash="dot"),
    }

    for name, values in signals.items():
        y = np.asarray(values, float)
        fig.add_trace(go.Scatter(
            x=x[:len(y)],
            y=y,
            mode="lines",
            name=name,
            line=styles.get(name, dict(width=2)),
        ))

    fig.update_layout(
        title=dict(
            text=title,
            y=0.97,              
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18)
        ),
        height=520,
        xaxis_title="Turn",
        yaxis_title="Disruption / event pressure (0–1)",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=40, t=90, b=40),  
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,             
            xanchor="center",
            x=0.5
        ),
    )

    return fig
    
def run_pipeline_from_embeddings(
    E: np.ndarray,
    texts: List[str],
    participants: List[str],
    smooth_method: str,
    env_alpha: float,
    env_span: int,
    use_cinv: bool,
    cinv_window: int,
    cinv_knn: int,
    cinv_thr: float,
    cinv_keigs: int,
    alpha_context: float = 0.84,
    b: float = 0.40,
    strong_weights=(0.25, 0.35, 0.40),
    transition_weights=(0.15, 0.25, 0.60),
):
    E = np.asarray(E, float)

    # C_inv
    C_inv_local = None
    if use_cinv:
        C_inv_local = compute_C_inv_series(
            E,
            window=int(cinv_window),
            k_nn=int(cinv_knn),
            thr=float(cinv_thr),
            k_eigs=int(cinv_keigs),
            D_max=None,
        )

    # IC-II
    ic2_local = compute_ic2_dynamics(
        E,
        alpha_context=float(alpha_context),
        b=float(b),
    )

    Ct_raw_local = np.asarray(ic2_local["C_t"], float)
    Ct_raw_local = np.clip(Ct_raw_local, 0.0, 1.0)

    # Canonical Ct: same definition as Single Dialogue
    Ct_base_local = np.clip(Ct_raw_local, 0.0, 1.0)
    Ct_base_local = apply_warmup_ramp(
        Ct_base_local,
        warm=WARMUP_TURNS,
        floor=0.10,
    )
    Ct_base_local = np.clip(Ct_base_local, 0.0, 1.0)

    Ct_smooth_local = smooth_coherence(
        Ct_base_local,
        method=smooth_method,
        ema_alpha=float(env_alpha),
        ewma_span=int(env_span),
    )
    Ct_smooth_local = np.clip(Ct_smooth_local, 0.0, 1.0)


    # IC-III + signals
    ic3_local = compute_ic3_geometry(E=E)

    signals_local = compute_all_signals(
        E=E,
        texts=texts,
        Ct_base=Ct_base_local,
        C_inv=C_inv_local,
        ic3=ic3_local,
    )

    event_scores_local = compute_event_scores(
        Ct=signals_local["Ct"],
        C_inv=signals_local["C_inv"],
        D_t=signals_local["D_t"],
        strong_weights=strong_weights,
        transition_weights=transition_weights,
    )

    # Primary five-dimensional state.
    continuous_local = compute_primary_state(
        Ct=signals_local["Ct"],
        C_inv=signals_local["C_inv"],
        d_i=signals_local["d_i"],
        kappa_i=signals_local["kappa_i"],
        rho_t=signals_local["rho_t"],
    )

    # Canonical event classification
    event_labels_local = classify_event_scores(
        event_scores_local,
        D_t=signals_local["D_t"],
    )

    event_masks_local = labels_to_event_masks(
        event_labels_local,
        min_sem_len=EVENT_MIN_SEM_LEN,
        min_struct_len=EVENT_MIN_STRUCT_LEN,
        min_strong_len=EVENT_MIN_STRONG_LEN,
    )

    # Same post-processing as Single Dialogue
    strong_mask_local = np.asarray(
        event_masks_local["strong"],
        dtype=bool,
    )
    semantic_mask_local = np.asarray(
        event_masks_local["semantic"],
        dtype=bool,
    )
    structural_mask_local = np.asarray(
        event_masks_local["structural"],
        dtype=bool,
    )

    if semantic_mask_local.any():
        semantic_mask_local = points_to_mask(
            np.where(semantic_mask_local)[0].tolist(),
            len(semantic_mask_local),
            w=0,
        )

    if structural_mask_local.any():
        structural_mask_local = points_to_mask(
            np.where(structural_mask_local)[0].tolist(),
            len(structural_mask_local),
            w=0,
        )

    if strong_mask_local.any():
        strong_mask_local = points_to_mask(
            np.where(strong_mask_local)[0].tolist(),
            len(strong_mask_local),
            w=1,
        )

    # Strong events have priority
    semantic_mask_local = (
        semantic_mask_local & (~strong_mask_local)
    )
    structural_mask_local = (
        structural_mask_local & (~strong_mask_local)
    )

    event_masks_local["strong"] = strong_mask_local
    event_masks_local["semantic"] = semantic_mask_local
    event_masks_local["structural"] = structural_mask_local

    event_labels_final_local = masks_to_display_labels(
        strong_mask=strong_mask_local,
        semantic_mask=semantic_mask_local,
        structural_mask=structural_mask_local,
    )

    return {
        "Ct": Ct_base_local,
        "Ct_smooth": Ct_smooth_local,
        "C_inv": C_inv_local,
        "rho_t": signals_local["rho_t"],
        "S_t": continuous_local["S_t"],
        "R_t": continuous_local["R_t"],
        "d_t": continuous_local["d_t"],
        "kappa_t": continuous_local["kappa_t"],
        "u_t": continuous_local["u_t"],
        "D_t": signals_local["D_t"],  # exploratory legacy summary only
        "event_scores": event_scores_local,
        "event_labels": event_labels_final_local,
        "event_masks": event_masks_local,
    }
    
def shuffle_texts_only(texts: List[str], seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    return [texts[i] for i in idx]


def perturb_parameters(base_params: Dict[str, float], pct: float, seed: int = 42) -> Dict[str, float]:
    """
    Multiplies each numeric parameter by a random factor in [1-pct, 1+pct].
    pct = 0.25 means ±25%
    """
    rng = np.random.default_rng(seed)
    out = dict(base_params)

    for k, v in out.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            factor = rng.uniform(1.0 - pct, 1.0 + pct)
            new_v = v * factor

            # keep sensible types/ranges for your parameters
            if k in {"cinv_window", "cinv_knn", "cinv_keigs", "env_span"}:
                new_v = max(1, int(round(new_v)))
            elif k in {"env_alpha", "cinv_thr"}:
                new_v = float(new_v)

            out[k] = new_v

    return out

def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0  # both empty => identical absence of events

    return float(inter / union)
    
def zscore_signal(x: np.ndarray) -> np.ndarray:
    """
    Standardizes a signal to mean 0 and std 1.
    Useful for correlation because it compares shape rather than raw scale.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return x

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if not np.isfinite(std) or std < 1e-9:
        return np.zeros_like(x)

    return (x - mean) / (std + 1e-9)


def minmax_signal(x: np.ndarray) -> np.ndarray:
    """
    Rescales a signal to [0, 1].
    Useful for DTW because it compares dynamic shape without scale bias.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return x

    lo = np.nanmin(x)
    hi = np.nanmax(x)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(x)

    return (x - lo) / (hi - lo + 1e-9)


def compare_embedding_runs(runs: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compares coherence/event dynamics across embedding models.

    Normalization logic:
    - Correlations use z-scored signals.
    - DTW similarities use min-max normalized signals.
    This makes the comparison focus on dynamic shape rather than raw scale.
    """

    summary_rows = []
    pair_rows = []

    names = list(runs.keys())

    # =========================
    # Per-embedding summary
    # =========================
    for name in names:
        run = runs[name]

        Ct = np.asarray(run["Ct"], float)
        strong_score = np.asarray(run["event_scores"]["strong_score"], float)

        strong_mask = np.asarray(run.get("strong_mask", []), dtype=bool)

        summary_rows.append({
            "embedding": name,
            "mean_Ct": float(np.nanmean(Ct)) if Ct.size else np.nan,
            "std_Ct": float(np.nanstd(Ct)) if Ct.size else np.nan,
            "min_Ct": float(np.nanmin(Ct)) if Ct.size else np.nan,
            "max_Ct": float(np.nanmax(Ct)) if Ct.size else np.nan,
            "mean_strong_score": float(np.nanmean(strong_score)) if strong_score.size else np.nan,
            "std_strong_score": float(np.nanstd(strong_score)) if strong_score.size else np.nan,
            "strong_events_n": int(np.sum(strong_mask)) if strong_mask.size else np.nan,
        })

    # =========================
    # Pairwise comparison
    # =========================
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            run_a = runs[a]
            run_b = runs[b]

            Ct_a_raw = np.asarray(run_a["Ct"], float)
            Ct_b_raw = np.asarray(run_b["Ct"], float)

            score_a_raw = np.asarray(run_a["event_scores"]["strong_score"], float)
            score_b_raw = np.asarray(run_b["event_scores"]["strong_score"], float)

            n_ct = min(len(Ct_a_raw), len(Ct_b_raw))
            n_score = min(len(score_a_raw), len(score_b_raw))

            Ct_a_raw = Ct_a_raw[:n_ct]
            Ct_b_raw = Ct_b_raw[:n_ct]

            score_a_raw = score_a_raw[:n_score]
            score_b_raw = score_b_raw[:n_score]

            # Normalized versions
            Ct_a_corr = zscore_signal(Ct_a_raw)
            Ct_b_corr = zscore_signal(Ct_b_raw)

            Ct_a_dtw = minmax_signal(Ct_a_raw)
            Ct_b_dtw = minmax_signal(Ct_b_raw)

            score_a_corr = zscore_signal(score_a_raw)
            score_b_corr = zscore_signal(score_b_raw)

            score_a_dtw = minmax_signal(score_a_raw)
            score_b_dtw = minmax_signal(score_b_raw)

            pair_rows.append({
                "pair": f"{a} vs {b}",

                # Raw descriptive differences
                "mean_Ct_diff_raw": float(abs(np.nanmean(Ct_a_raw) - np.nanmean(Ct_b_raw))),
                "std_Ct_diff_raw": float(abs(np.nanstd(Ct_a_raw) - np.nanstd(Ct_b_raw))),

                # Shape comparisons
                "Ct_corr_zscore": _safe_corr(Ct_a_corr, Ct_b_corr),
                "Ct_dtw_similarity_minmax": dtw_similarity(Ct_a_dtw, Ct_b_dtw),

                "strong_score_corr_zscore": _safe_corr(score_a_corr, score_b_corr),
                "strong_score_dtw_similarity_minmax": dtw_similarity(score_a_dtw, score_b_dtw),
            })

    summary_df = pd.DataFrame(summary_rows)
    pairs_df = pd.DataFrame(pair_rows)

    return summary_df, pairs_df
        
def event_indices_from_score(score: np.ndarray, q: float = 0.85) -> List[int]:
    x = np.asarray(score, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    thr = float(np.quantile(x, q))
    return list(np.where(score >= thr)[0])


def event_alignment_score(a: List[int], b: List[int], window: int = 3) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    matched = 0
    used_b = set()

    for i in a:
        candidates = [
            j for j in b
            if abs(int(i) - int(j)) <= int(window) and j not in used_b
        ]
        if candidates:
            best = min(candidates, key=lambda j: abs(int(i) - int(j)))
            used_b.add(best)
            matched += 1

    return float(matched / max(len(a), len(b), 1))


def build_event_alignment_matrix(
    runs: Dict[str, dict],
    window: int = 3,
    event_q: float = 0.85,
) -> pd.DataFrame:
    names = list(runs.keys())
    matrix = []

    for a in names:
        row = []
        events_a = event_indices_from_score(
            runs[a]["event_scores"]["strong_score"],
            q=event_q
        )

        for b in names:
            events_b = event_indices_from_score(
                runs[b]["event_scores"]["strong_score"],
                q=event_q
            )
            row.append(event_alignment_score(events_a, events_b, window=window))

        matrix.append(row)

    return pd.DataFrame(matrix, index=names, columns=names)


def build_real_vs_shuffled_embedding_table(
    runs: Dict[str, dict],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    names = list(runs.keys())
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            xa = np.asarray(runs[a]["event_scores"]["strong_score"], float)
            xb = np.asarray(runs[b]["event_scores"]["strong_score"], float)

            n = min(len(xa), len(xb))
            xa = xa[:n]
            xb = xb[:n]

            real_corr = _safe_corr(xa, xb)
            shuffled_corr = _safe_corr(rng.permutation(xa), rng.permutation(xb))

            rows.append({
                "pair": f"{a} vs {b}",
                "real_corr": float(real_corr) if np.isfinite(real_corr) else np.nan,
                "shuffled_corr": float(shuffled_corr) if np.isfinite(shuffled_corr) else np.nan,
                "delta_real_minus_shuffled": (
                    float(real_corr - shuffled_corr)
                    if np.isfinite(real_corr) and np.isfinite(shuffled_corr)
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def build_variance_decomposition_table(runs: Dict[str, dict]) -> pd.DataFrame:
    rows = []

    for emb_name, run in runs.items():
        x = np.asarray(run["event_scores"]["strong_score"], float)
        for t, value in enumerate(x):
            rows.append({
                "embedding": emb_name,
                "turn": int(t + 1),
                "value": float(value),
            })

    df_long = pd.DataFrame(rows)

    if df_long.empty:
        return pd.DataFrame(columns=["component", "variance", "share_of_total"])

    total_var = float(df_long["value"].var())

    emb_var = float(df_long.groupby("embedding")["value"].mean().var())
    dialogue_var = float(df_long.groupby("turn")["value"].mean().var())

    denom = total_var if total_var > 1e-12 else np.nan

    return pd.DataFrame([
        {
            "component": "embedding",
            "variance": emb_var,
            "share_of_total": emb_var / denom if np.isfinite(denom) else np.nan,
        },
        {
            "component": "dialogue_turn_structure",
            "variance": dialogue_var,
            "share_of_total": dialogue_var / denom if np.isfinite(denom) else np.nan,
        },
    ])

def _perturb_value(value, pct, lo=None, hi=None, is_int=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    v = float(value)
    delta = rng.uniform(-pct, pct)
    new_v = v * (1.0 + delta)

    if lo is not None:
        new_v = max(float(lo), new_v)
    if hi is not None:
        new_v = min(float(hi), new_v)

    if is_int:
        new_v = int(round(new_v))
        if lo is not None:
            new_v = max(int(lo), new_v)
        if hi is not None:
            new_v = min(int(hi), new_v)

    return new_v


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0
    return float(inter / union)


def label_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    a = np.asarray(list(a), dtype=object)
    b = np.asarray(list(b), dtype=object)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan

    return float(np.mean(a == b))


def run_parameter_robustness(
    *,
    E: np.ndarray,
    texts: List[str],
    participants: List[str],
    base_params: Dict[str, object],
    base_ct: np.ndarray,
    base_event_masks: Dict[str, np.ndarray],
    n_runs: int = 30,
    pct: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    rows = []

    for r in range(int(n_runs)):
        p = {}

        p["alpha_context"] = _perturb_value(base_params["alpha_context"], pct, lo=0.70, hi=0.95, rng=rng)
        p["b"] = _perturb_value(base_params["b"], pct, lo=0.20, hi=0.70, rng=rng)

        p["env_alpha"] = _perturb_value(base_params["env_alpha"], pct, lo=0.05, hi=0.60, rng=rng)
        p["env_span"] = _perturb_value(base_params["env_span"], pct, lo=3, hi=25, is_int=True, rng=rng)

        p["cinv_window"] = _perturb_value(base_params["cinv_window"], pct, lo=6, hi=24, is_int=True, rng=rng)
        p["cinv_knn"] = _perturb_value(base_params["cinv_knn"], pct, lo=2, hi=10, is_int=True, rng=rng)
        p["cinv_thr"] = _perturb_value(base_params["cinv_thr"], pct, lo=0.00, hi=0.30, rng=rng)
        p["cinv_keigs"] = _perturb_value(base_params["cinv_keigs"], pct, lo=3, hi=12, is_int=True, rng=rng)

        run = run_pipeline_from_embeddings(
            E=E,
            texts=texts,
            participants=participants,
            smooth_method=str(base_params["smooth_method"]),
            env_alpha=float(p["env_alpha"]),
            env_span=int(p["env_span"]),
            use_cinv=bool(base_params["use_cinv"]),
            cinv_window=int(p["cinv_window"]),
            cinv_knn=int(p["cinv_knn"]),
            cinv_thr=float(p["cinv_thr"]),
            cinv_keigs=int(p["cinv_keigs"]),
            alpha_context=float(p["alpha_context"]),
            b=float(p["b"]),
        )

        ct = np.asarray(run["Ct"], float)
        masks = run["event_masks"]

        rows.append({
            "run": r + 1,
            "Ct_corr": _safe_corr(base_ct, ct),
            "Ct_dtw_similarity": dtw_similarity(base_ct, ct),
            "strong_jaccard": jaccard_similarity(base_event_masks["strong"], masks["strong"]),
            "semantic_jaccard": jaccard_similarity(base_event_masks["semantic"], masks["semantic"]),
            "structural_jaccard": jaccard_similarity(base_event_masks["structural"], masks["structural"]),
            "env_alpha": p["env_alpha"],
            "env_span": p["env_span"],
            "cinv_window": p["cinv_window"],
            "cinv_knn": p["cinv_knn"],
            "cinv_thr": p["cinv_thr"],
            "cinv_keigs": p["cinv_keigs"],
            "alpha_context": p["alpha_context"],
            "b": p["b"],
        })

    df_runs = pd.DataFrame(rows)
    
    perturbed_param_names = [
        "alpha_context",
        "b",
        "env_alpha",
        "env_span",
        "cinv_window",
        "cinv_knn",
        "cinv_thr",
        "cinv_keigs",
    ]

    movement_rows = []

    for param in perturbed_param_names:
        if param not in df_runs.columns:
            continue

        base_value = base_params.get(param, np.nan)
        values = pd.to_numeric(df_runs[param], errors="coerce")

        mean_value = float(values.mean())
        min_value = float(values.min())
        max_value = float(values.max())
        mean_abs_delta = float((values - float(base_value)).abs().mean())

        if abs(float(base_value)) > 1e-9:
            mean_pct_delta = float(
                ((values - float(base_value)).abs() / abs(float(base_value))).mean()
            )
        else:
            mean_pct_delta = np.nan

        movement_rows.append({
            "parameter": param,
            "baseline": base_value,
            "mean_perturbed": mean_value,
            "min_perturbed": min_value,
            "max_perturbed": max_value,
            "mean_abs_delta": mean_abs_delta,
            "mean_pct_delta": mean_pct_delta,
        })

    df_movement = pd.DataFrame(movement_rows)

    df_summary = pd.DataFrame([{
        "Ct_corr_mean": float(df_runs["Ct_corr"].mean()),
        "Ct_corr_std": float(df_runs["Ct_corr"].std()),
        "Ct_dtw_similarity_mean": float(df_runs["Ct_dtw_similarity"].mean()),
        "strong_jaccard_mean": float(df_runs["strong_jaccard"].mean()),
        "semantic_jaccard_mean": float(df_runs["semantic_jaccard"].mean()),
        "structural_jaccard_mean": float(df_runs["structural_jaccard"].mean()),
    }])

    return df_runs, df_summary, df_movement
        
def classify_event_scores(
    event_scores,
    D_t=None,
    strong_thr=EVENT_STRONG_THR,
    sem_thr=EVENT_SEM_THR,
    struct_thr=EVENT_STRUCT_THR,
    d_thr=EVENT_D_THR,
    sem_margin=EVENT_SEM_MARGIN,
    struct_margin=EVENT_STRUCT_MARGIN,
):
    strong = np.asarray(event_scores["strong_score"], float)
    sem = np.asarray(event_scores["semantic_score"], float)
    struct = np.asarray(event_scores["structural_score"], float)

    if D_t is None:
        D = np.zeros_like(strong)
    else:
        D = np.asarray(D_t, float)

    labels = []

    for i in range(len(strong)):

        # 1) strong rupture only when really strong
        if strong[i] >= strong_thr:
            labels.append("BREAKDOWN")

        # 2) semantic rupture when semantic clearly dominates structural
        elif sem[i] >= sem_thr and sem[i] >= struct[i] + sem_margin:
            labels.append("SEM_DRIFT")

        # 3) structural rupture when structural clearly dominates semantic
        elif struct[i] >= struct_thr and D[i] >= d_thr:
            labels.append("STRUCT_RECONFIG")

        else:
            labels.append("STABLE")

    return labels
    
def strong_events_to_mask(strong_score, thr=0.35, min_len=1):
    strong = np.asarray(strong_score, float)
    mask = strong >= float(thr)

    segments = mask_to_segments(mask)
    clean_mask = np.zeros_like(mask, dtype=bool)

    for s, e in segments:
        if (e - s + 1) >= int(min_len):
            clean_mask[s:e+1] = True

    return clean_mask
    
def labels_to_event_masks(labels, min_sem_len=4, min_struct_len=3, min_strong_len=2):
    labels = np.asarray(labels, dtype=object)

    strong = labels == "BREAKDOWN"
    semantic = labels == "SEM_DRIFT"
    structural = labels == "STRUCT_RECONFIG"

    strong = enforce_min_persistence(strong, min_len=min_strong_len)
    semantic = enforce_min_persistence(semantic, min_len=min_sem_len)
    structural = enforce_min_persistence(structural, min_len=min_struct_len)

    return {
        "strong": strong,
        "semantic": semantic,
        "structural": structural,
    }

def masks_to_display_labels(strong_mask, semantic_mask, structural_mask):
    strong_mask = np.asarray(strong_mask, dtype=bool)
    semantic_mask = np.asarray(semantic_mask, dtype=bool)
    structural_mask = np.asarray(structural_mask, dtype=bool)

    n = len(strong_mask)
    labels = np.array(["STABLE"] * n, dtype=object)

    # priority order for display
    labels[semantic_mask] = "SEM_DRIFT"
    labels[structural_mask] = "STRUCT_RECONFIG"
    labels[strong_mask] = "BREAKDOWN"

    return labels
    
def merge_nearby_sem_struct_events(
    semantic_mask: np.ndarray,
    structural_mask: np.ndarray,
    window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic_mask = np.asarray(semantic_mask, dtype=bool).copy()
    structural_mask = np.asarray(structural_mask, dtype=bool).copy()

    n = min(len(semantic_mask), len(structural_mask))
    semantic_mask = semantic_mask[:n]
    structural_mask = structural_mask[:n]

    complex_mask = np.zeros(n, dtype=bool)

    sem_idx = np.where(semantic_mask)[0]
    struct_idx = np.where(structural_mask)[0]

    for s in sem_idx:
        nearby = struct_idx[np.abs(struct_idx - s) <= int(window)]
        if nearby.size:
            a = max(0, min(int(s), int(nearby.min())) - int(window))
            b = min(n - 1, max(int(s), int(nearby.max())) + int(window))
            complex_mask[a:b + 1] = True

    semantic_mask = semantic_mask & (~complex_mask)
    structural_mask = structural_mask & (~complex_mask)

    return semantic_mask, structural_mask, complex_mask
        

# -------------------------------
# Ci trajectories per participant (embedding-based)
# -------------------------------
def compute_ci_series(
    E: np.ndarray,
    participants: List[str],
    method: str = "ctx",
    alpha: float = 0.90,
    eps: float = 1e-12,
) -> pd.DataFrame:
    E = np.asarray(E, float)
    n = E.shape[0]
    parts = [str(p) for p in participants]
    uniq = list(dict.fromkeys(parts))
    out = pd.DataFrame(index=np.arange(n))

    if n == 0 or len(uniq) == 0:
        return out

    e = np.zeros_like(E, float)
    for i in range(n):
        v = E[i]
        nv = float(np.linalg.norm(v))
        e[i] = v / (nv + eps) if nv > eps else v * 0.0

    method = (method or "ctx").lower().strip()

    if method == "im":
        for p in uniq:
            idx = [i for i, pp in enumerate(parts) if pp == p]
            if not idx:
                continue
            centroid = np.mean(e[idx], axis=0)
            centroid /= (float(np.linalg.norm(centroid)) + eps)
            ci = np.zeros(n, float)
            for t in range(n):
                ci[t] = 0.5 * (1.0 + _cos(e[t], centroid, eps=eps))
            out[f"Ci_{p}"] = np.clip(ci, 0.0, 1.0)
        return out

    a = float(np.clip(alpha, 0.0, 0.999))
    ctx = {p: e[0].copy() for p in uniq}
    ci_cols = {p: np.zeros(n, float) for p in uniq}

    for t in range(n):
        speaker = parts[t]
        for p in uniq:
            ci_cols[p][t] = 0.5 * (1.0 + _cos(e[t], ctx[p], eps=eps))

        c_prev = ctx[speaker]
        u = a * c_prev + (1.0 - a) * e[t]
        nu = float(np.linalg.norm(u))
        ctx[speaker] = (u / (nu + eps)) if nu > eps else c_prev

    for p in uniq:
        out[f"Ci_{p}"] = np.clip(ci_cols[p], 0.0, 1.0)

    return out


# -------------------------------
# Plot helpers
# -------------------------------
def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig


def plot_ct_main(
    Ct: np.ndarray,
    participants: List[str],
    title: str,
    height: int,
    C_inv: Optional[np.ndarray] = None,
    event_labels: Optional[Sequence[str]] = None,
) -> go.Figure:

    Ct = np.asarray(Ct, dtype=float)
    n = int(Ct.size)
    x = np.arange(1, n + 1, dtype=int)

    fig = go.Figure()
    fig = _base_layout(fig, title, height)
    
        # =========================================================
    # Operational event regions
    # =========================================================
    if event_labels is not None:
        labels = np.asarray(event_labels, dtype=object)

        if labels.size == n:

            region_styles = {
                "SEM_DRIFT": {
                    "fillcolor": "rgba(255, 193, 7, 0.10)",
                    "legend_color": "rgba(255, 193, 7, 0.65)",
                    "name": "SEM_DRIFT region",
                },
                "STRUCT_RECONFIG": {
                    "fillcolor": "rgba(33, 150, 243, 0.10)",
                    "legend_color": "rgba(33, 150, 243, 0.65)",
                    "name": "STRUCT_RECONFIG region",
                },
                "BREAKDOWN": {
                    "fillcolor": "rgba(220, 53, 69, 0.13)",
                    "legend_color": "rgba(220, 53, 69, 0.70)",
                    "name": "BREAKDOWN region",
                },
            }

            for label, style in region_styles.items():
                mask = labels == label

                if not np.any(mask):
                    continue

                segments = mask_to_segments(mask)

                for start_idx, end_idx in segments:
                    fig.add_vrect(
                        x0=(start_idx + 1) - 0.5,
                        x1=(end_idx + 1) + 0.5,
                        fillcolor=style["fillcolor"],
                        opacity=1.0,
                        line_width=0,
                        layer="below",
                    )

                # Invisible/dummy trace so each region appears once in legend
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=10,
                            symbol="square",
                            color=style["legend_color"],
                        ),
                        name=style["name"],
                        hoverinfo="skip",
                    )
                )

    # Main contextual continuity curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.clip(Ct, 0.0, 1.0),
            mode="lines",
            name="Cₜ (contextual continuity)",
            line=dict(width=3),
        )
    )

    # Optional structural continuity overlay
    if C_inv is not None:
        C_inv = np.asarray(C_inv, dtype=float)
        if C_inv.size == n:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.clip(C_inv, 0.0, 1.0),
                    mode="lines",
                    name="C_inv (structural continuity)",
                    line=dict(width=2, dash="dot"),
                )
            )

    # Participant markers
    parts = list(dict.fromkeys([str(p) for p in participants]))

    for name in parts:
        y = np.full(n, np.nan, float)
        idx = [
            i for i, p in enumerate(participants)
            if str(p) == name
        ]

        if idx:
            y[idx] = Ct[idx]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"Participant: {name}",
                marker=dict(size=7),
            )
        )

    fig = _base_layout(fig, title, height)
    return fig

def plot_embedding_comparison_overlay(
    runs: Dict[str, dict],
    height: int = 500,
    title: str = "Embedding comparison — overlaid Ct trajectories",
) -> go.Figure:
    fig = go.Figure()

    for name, run in runs.items():
        Ct = np.asarray(run["Ct"], float)
        x = np.arange(1, len(Ct) + 1)

        fig.add_trace(go.Scatter(
            x=x,
            y=np.clip(Ct, 0.0, 1.0),
            mode="lines",
            name=f"{name} — Ct",
            line=dict(width=3),
        ))

    fig.update_layout(
        title=title,
        height=int(height),
        xaxis_title="Turn",
        yaxis_title="Ct (0–1)",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=200, t=40, b=40),
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig
    
# =========================================================
# PDF REPORT (Matplotlib + PdfPages) 
# =========================================================

def _fig_text_page(title: str, lines: List[str], footer: str = ""):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")

    y = 0.92
    fig.text(0.07, y, title, fontsize=18, fontweight="bold")
    y -= 0.04
    fig.text(0.07, y, "Generated by CNøde — TIE–Dialog (Conversational Dynamics Lab)", fontsize=10, alpha=0.8)
    y -= 0.04

    for ln in lines:
        if ln.strip() == "":
            y -= 0.02
            continue
        fig.text(0.07, y, ln, fontsize=11)
        y -= 0.022
        if y < 0.10:
            break

    if footer:
        fig.text(0.07, 0.05, footer, fontsize=9, alpha=0.7)

    plt.axis("off")
    return fig


def _fig_timeseries(
    title: str,
    x: np.ndarray,
    series: List[Tuple[str, np.ndarray]],
    hlines: Optional[List[Tuple[str, float]]] = None,
    vmarks: Optional[List[Tuple[str, List[int]]]] = None,
    ylim: Tuple[float, float] = (0.0, 1.0),
):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for name, y in series:
        y = np.asarray(y, float)
        m = min(len(x), len(y))
        ax.plot(x[:m], y[:m], linewidth=2, label=name)

    if hlines:
        for label, yv in hlines:
            ax.axhline(float(yv), linestyle="--", linewidth=1)
            ax.text(x[0], float(yv) + 0.01, label, fontsize=9)

    if vmarks:
        for label, idxs in vmarks:
            for t in idxs:
                xv = int(t) + 1  # 0-index -> turn
                ax.axvline(xv, linestyle=":", linewidth=1, alpha=0.6)
            if idxs:
                ax.text(int(idxs[0]) + 1, 0.03, label, fontsize=9, alpha=0.9)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Value (0–1)")
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _wrap_text_cell(text: str, width: int = 45) -> str:
    if not isinstance(text, str):
        return str(text)
    return "\n".join(textwrap.wrap(text, width=width))


def _fig_table_page(title: str, table_df: pd.DataFrame, note: str = ""):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    df_show = table_df.copy()

    # --- WRAP TEXT COLUMN ---
    if "text" in df_show.columns:
        df_show["text"] = df_show["text"].apply(lambda x: _wrap_text_cell(x, width=55))

    # Convert all to string
    for c in df_show.columns:
        df_show[c] = df_show[c].astype(str)

    # Column widths (last column wider for text)
    n_cols = len(df_show.columns)
    col_widths = [0.08] * n_cols
    if "text" in df_show.columns:
        text_idx = list(df_show.columns).index("text")
        col_widths[text_idx] = 0.45

    tbl = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    if note:
        fig.text(0.02, 0.03, note, fontsize=9, alpha=0.75)

    fig.tight_layout()
    return fig

def build_pdf_report_bytes(
    *,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    ic2_df: pd.DataFrame,
    ic3_df: pd.DataFrame,
    Ct_base: np.ndarray,
    Ct_smooth: np.ndarray,
    C_inv: Optional[np.ndarray],
    used_mode: str,
    emb_msg: str,
    params: Dict[str, object],
) -> bytes:
    """
    Returns PDF bytes (downloadable).
    geom is assumed 0-indexed.
    """
    buf = BytesIO()

    n = int(len(Ct_base))
    turns = np.arange(1, n + 1, dtype=int)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _safe_mean(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")

    # ---------------------------
    # PAGE 1 — Summary
    # ---------------------------
    summary_lines = [
        f"Report time: {now}",
        f"Embedding mode: {used_mode}",
        f"Embedding status: {emb_msg}",
        "",
        f"Turns: {n}",
        f"Participants: {df_in['participant'].nunique() if 'participant' in df_in.columns else 'n/a'}",
        "",
        f"Mean coherence (Ct): {_safe_mean(Ct_base):.3f}",
        f"Mean smoothed coherence: {_safe_mean(Ct_smooth):.3f}",
        "",
        "Interpretation (high level):",
        "• Ct captures alignment with the evolving conversational context.",
        "• Primary state: z_t=(S_t,R_t,d_t,kappa_t,u_t).",
        "• S_t = 1 - C_t is Contextual Discontinuity.",
        "• R_t = 1 - C_inv is Structural Instability / low persistence.",
        "• d_t is angular displacement; kappa_t is turning-angle curvature.",
        "• u_t = 1 - rho_t is local semantic dispersion.",
        "• D_t is retained only as an exploratory legacy geometry magnitude.",
        "• C_inv (if enabled) tracks structural persistence via rolling graph invariants.",
    ]

    # ---------------------------
    # PAGE 2 — Parameters
    # ---------------------------
    param_lines = ["Parameters used in this run:", ""]
    for k, v in params.items():
        param_lines.append(f"- {k}: {v}")

    # ---------------------------
    # Tables
    # ---------------------------
    dCt = np.zeros_like(np.asarray(Ct_base, float))
    if len(dCt) > 1:
        dCt[1:] = Ct_base[1:] - Ct_base[:-1]
    idx_drop = np.argsort(dCt)[:10] if len(dCt) else np.array([], int)

    cols = ["turn", "participant", "Ct", "dCt", "S_t", "R_t", "d_t", "kappa_t", "u_t", "text"]
    table_df = df_out.copy()
    for c in cols:
        if c not in table_df.columns:
            table_df[c] = ""

    top_df = table_df.iloc[idx_drop][cols].copy() if len(idx_drop) else table_df.head(0)[cols].copy()
    if len(idx_drop):
        top_df["dCt"] = [f"{float(dCt[i]):.3f}" for i in idx_drop]
        top_df["Ct"] = [f"{float(Ct_base[i]):.3f}" for i in idx_drop]
    if "text" in top_df.columns:
        top_df["text"] = top_df["text"].astype(str).str.slice(0, 110)


    # ---------------------------
    # Write PDF
    # ---------------------------
    with PdfPages(buf) as pdf:
        fig = _fig_text_page("TIE–Dialog Report (English)", summary_lines, footer="CNøde — Informational Systems Research")
        pdf.savefig(fig); plt.close(fig)

        fig = _fig_text_page("Run Configuration", param_lines, footer="CNøde — TIE–Dialog computational analysis.")
        pdf.savefig(fig); plt.close(fig)

        # Main coherence page
        main_series = [("Ct (raw)", Ct_base), ("Ct_smooth", Ct_smooth)]
        hls = None
        fig = _fig_timeseries(
            "Coherence Dynamics (Ct)",
            turns,
            main_series,
            hlines=hls,
            vmarks=None,
            ylim=(0.0, 1.0),
        )
        pdf.savefig(fig); plt.close(fig)

        # Optional C_inv page
        if C_inv is not None and np.asarray(C_inv).size == len(Ct_base):
            fig = _fig_timeseries(
                "Invariant Coherence (C_inv) — Structural Persistence",
                turns,
                [("C_inv", np.asarray(C_inv, float))],
                hlines=None,
                vmarks=None,
                ylim=(0.0, 1.0),
            )
            pdf.savefig(fig); plt.close(fig)

        # Primary geometric observables page
        if ic3_df is not None and len(ic3_df) == len(turns):
            fig = _fig_timeseries(
                "Primary geometric observables",
                turns,
                [
                    ("d_t", np.asarray(ic3_df["d_t"], float)),
                    ("kappa_t", np.asarray(ic3_df["kappa_t"], float)),
                    ("u_t", np.asarray(ic3_df["u_t"], float)),
                ],
                hlines=None,
                vmarks=[],
                ylim=(0.0, 1.0),
            )
            pdf.savefig(fig); plt.close(fig)

        # Tables
        fig = _fig_table_page(
            "Top Ct Drop Candidates (most negative ΔCt)",
            top_df,
            note="Tip: these are the strongest local candidates for misalignment or rupture-like transitions.",
        )
        pdf.savefig(fig); plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to downloadable UTF-8 CSV bytes once."""
    if df is None or df.empty:
        return b""
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)
    return buffer.getvalue()


def read_dialogue_file(uploaded_file) -> pd.DataFrame:
    """Read and clean one uploaded CSV or XLSX dialogue."""
    name = str(uploaded_file.name).lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    df, missing = _clean_df(df)

    if df.empty:
        raise ValueError("The file contains no valid dialogue turns.")

    return df


def infer_dialog_id(uploaded_file, df: pd.DataFrame) -> str:
    """
    Prefer a dialog_id column. Otherwise use the filename.
    """
    if "dialog_id" in df.columns:
        values = df["dialog_id"].dropna().astype(str).str.strip()
        values = values[values != ""]

        if not values.empty:
            return values.iloc[0]

    filename = str(uploaded_file.name)
    return re.sub(r"\.(csv|xlsx)$", "", filename, flags=re.IGNORECASE)


def run_batch_embedding_validation(
    uploaded_files,
    selected_embeddings,
    smooth_method,
    env_alpha,
    env_span,
    use_cinv,
    cinv_window,
    cinv_knn,
    cinv_thr,
    cinv_keigs,
):
    """
    Run every uploaded dialogue with every selected embedding.

    Returns:
    - turn-level results
    - pairwise embedding comparisons per dialogue
    - per-embedding summaries per dialogue
    - errors
    """

    model_map = {
        "MiniLM": (
            "sbert",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        "E5": (
            "e5",
            "intfloat/e5-base-v2",
        ),
        "BGE": (
            "bge",
            "BAAI/bge-base-en-v1.5",
        ),
        "Instructor": (
            "instructor",
            "hkunlp/instructor-base",
        ),
    }

    turn_level_frames = []
    pairwise_frames = []
    summary_frames = []
    error_rows = []

    total_runs = len(uploaded_files) * len(selected_embeddings)
    completed_runs = 0

    progress_bar = st.progress(0.0)
    status_box = st.empty()

    for uploaded_file in uploaded_files:
        try:
            df_dialogue = read_dialogue_file(uploaded_file)
            dialog_id = infer_dialog_id(uploaded_file, df_dialogue)

            texts = df_dialogue["text"].astype(str).tolist()
            participants = df_dialogue["participant"].astype(str).tolist()
            turns = df_dialogue["turn"].to_numpy(dtype=int)

            embedding_runs = {}

            for embedding_name in selected_embeddings:
                completed_runs += 1

                status_box.write(
                    f"Processing {dialog_id} — {embedding_name} "
                    f"({completed_runs}/{total_runs})"
                )

                mode_name, model_name = model_map[embedding_name]

                try:
                    E, used_mode, embedding_message = embed_texts(
                        texts=texts,
                        mode=mode_name,
                        sbert_model=model_name,
                    )

                    # Important: detect silent fallback to TF-IDF
                    expected_prefixes = {
                        "MiniLM": "sbert:",
                        "E5": "e5:",
                        "BGE": "bge:",
                        "Instructor": "instructor:",
                    }

                    expected_prefix = expected_prefixes[embedding_name]

                    if not str(used_mode).startswith(expected_prefix):
                        raise RuntimeError(
                            f"{embedding_name} did not run correctly. "
                            f"Actual mode: {used_mode}. "
                            f"Message: {embedding_message}"
                        )

                    run = run_pipeline_from_embeddings(
                        E=E,
                        texts=texts,
                        participants=participants,
                        smooth_method=smooth_method,
                        env_alpha=float(env_alpha),
                        env_span=int(env_span),
                        use_cinv=bool(use_cinv),
                        cinv_window=int(cinv_window),
                        cinv_knn=int(cinv_knn),
                        cinv_thr=float(cinv_thr),
                        cinv_keigs=int(cinv_keigs),
                    )

                    embedding_runs[embedding_name] = run

                    scores = run["event_scores"]

                    turn_df = pd.DataFrame({
                        "dialog_id": dialog_id,
                        "source_file": uploaded_file.name,
                        "embedding": embedding_name,
                        "used_mode": used_mode,
                        "turn": turns,
                        "participant": participants,
                        "text": texts,
                        "Ct": np.asarray(run["Ct"], dtype=float),
                        "Ct_smooth": np.asarray(
                            run["Ct_smooth"],
                            dtype=float,
                        ),
                        "C_inv": (
                            np.asarray(run["C_inv"], dtype=float)
                            if run["C_inv"] is not None
                            else np.full(len(turns), np.nan)
                        ),
                        "rho_t": np.asarray(
                            run["rho_t"],
                            dtype=float,
                        ),
                        "S_t": np.asarray(run["S_t"], dtype=float),
                        "R_t": np.asarray(run["R_t"], dtype=float),
                        "D_t": np.asarray(run["D_t"], dtype=float),
                        "semantic_event_drop": np.asarray(
                            scores["sem_drop"],
                            dtype=float,
                        ),
                        "structural_event_drop": np.asarray(
                            scores["struct_drop"],
                            dtype=float,
                        ),
                        "semantic_score": np.asarray(
                            scores["semantic_score"],
                            dtype=float,
                        ),
                        "structural_score": np.asarray(
                            scores["structural_score"],
                            dtype=float,
                        ),
                        "strong_score": np.asarray(
                            scores["strong_score"],
                            dtype=float,
                        ),
                        "transition_pressure": np.asarray(
                            scores["transition_pressure"],
                            dtype=float,
                        ),
                    })

                    turn_level_frames.append(turn_df)

                except Exception as embedding_error:
                    error_rows.append({
                        "dialog_id": dialog_id,
                        "source_file": uploaded_file.name,
                        "embedding": embedding_name,
                        "error": str(embedding_error),
                    })

                progress_bar.progress(
                    min(completed_runs / total_runs, 1.0)
                )

            # Compare embeddings inside this dialogue
            if len(embedding_runs) >= 2:
                summary_df, pairs_df = compare_embedding_runs(
                    embedding_runs
                )

                summary_df.insert(0, "dialog_id", dialog_id)
                pairs_df.insert(0, "dialog_id", dialog_id)

                summary_frames.append(summary_df)
                pairwise_frames.append(pairs_df)

        except Exception as dialogue_error:
            error_rows.append({
                "dialog_id": None,
                "source_file": uploaded_file.name,
                "embedding": None,
                "error": str(dialogue_error),
            })

    progress_bar.progress(1.0)
    status_box.success(
        f"Batch completed: {completed_runs}/{total_runs} runs attempted."
    )

    turn_level_df = (
        pd.concat(turn_level_frames, ignore_index=True)
        if turn_level_frames
        else pd.DataFrame()
    )

    pairwise_df = (
        pd.concat(pairwise_frames, ignore_index=True)
        if pairwise_frames
        else pd.DataFrame()
    )

    summary_df = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )

    errors_df = pd.DataFrame(error_rows)

    return turn_level_df, pairwise_df, summary_df, errors_df



def build_hyperparameter_configs(base: dict) -> List[dict]:
    """Create a reproducible one-factor-at-a-time sensitivity grid."""
    configs = []

    def add_config(parameter, value, updates):
        cfg = dict(base)
        cfg.update(updates)
        cfg["varied_parameter"] = parameter
        cfg["parameter_value"] = str(value)
        cfg["is_base_config"] = parameter == "base"
        cfg["config_id"] = f"cfg_{len(configs):03d}_{parameter}_{str(value).replace(' ', '')}"
        configs.append(cfg)

    add_config("base", "base", {})

    grids = {
        "alpha_context": [0.70, 0.78, 0.90, 0.94],
        "env_alpha": [0.10, 0.30, 0.40],
        "env_span": [5, 7, 11, 13],
        "cinv_window": [6, 10, 12],
        "cinv_knn": [2, 5, 7],
        "cinv_thr": [0.05, 0.10, 0.20],
        "cinv_keigs": [4, 8],
    }

    for parameter, values in grids.items():
        base_value = base.get(parameter)
        for value in values:
            if base_value is not None and float(value) == float(base_value):
                continue
            add_config(parameter, value, {parameter: value})

    weight_variants = [
        ("balanced", (0.33, 0.33, 0.34)),
        ("semantic_heavy", (0.40, 0.30, 0.30)),
        ("structural_heavy", (0.20, 0.50, 0.30)),
        ("geometric_heavy", (0.20, 0.30, 0.50)),
    ]
    for name, weights in weight_variants:
        add_config("strong_weights", name, {"strong_weights": weights})

    transition_variants = [
        ("balanced", (0.33, 0.33, 0.34)),
        ("semantic_heavy", (0.40, 0.25, 0.35)),
        ("structural_heavy", (0.20, 0.50, 0.30)),
        ("geometric_heavy", (0.15, 0.25, 0.60)),
    ]
    for name, weights in transition_variants:
        if tuple(weights) == tuple(base.get("transition_weights", ())):
            continue
        add_config("transition_weights", name, {"transition_weights": weights})

    return configs


def run_batch_hyperparameter_validation(
    uploaded_files,
    embedding_name,
    smooth_method,
    env_alpha,
    env_span,
    use_cinv,
    cinv_window,
    cinv_knn,
    cinv_thr,
    cinv_keigs,
):
    """Run all dialogues over an OFAT hyperparameter grid with one fixed embedding."""
    model_map = {
        "MiniLM": ("sbert", "sentence-transformers/all-MiniLM-L6-v2"),
        "E5": ("e5", "intfloat/e5-base-v2"),
        "BGE": ("bge", "BAAI/bge-base-en-v1.5"),
        "Instructor": ("instructor", "hkunlp/instructor-base"),
    }
    mode, model_name = model_map[embedding_name]

    base = {
        "smooth_method": str(smooth_method),
        "env_alpha": float(env_alpha),
        "env_span": int(env_span),
        "use_cinv": bool(use_cinv),
        "cinv_window": int(cinv_window),
        "cinv_knn": int(cinv_knn),
        "cinv_thr": float(cinv_thr),
        "cinv_keigs": int(cinv_keigs),
        "alpha_context": 0.84,
        "strong_weights": (0.25, 0.35, 0.40),
        "transition_weights": (0.15, 0.25, 0.60),
    }
    configs = build_hyperparameter_configs(base)

    turn_frames, error_rows = [], []
    total_runs = len(uploaded_files) * len(configs)
    completed = 0
    progress_bar = st.progress(0.0)
    status_box = st.empty()

    for uploaded_file in uploaded_files:
        try:
            df_dialogue = read_dialogue_file(uploaded_file)
            dialog_id = infer_dialog_id(uploaded_file, df_dialogue)
            texts = df_dialogue["text"].astype(str).tolist()
            participants = df_dialogue["participant"].astype(str).tolist()
            turns = df_dialogue["turn"].to_numpy(dtype=int)

            E, used_mode, status_msg = embed_texts(texts, mode=mode, sbert_model=model_name)
            if not str(used_mode).lower().startswith(mode):
                raise RuntimeError(
                    f"Requested {embedding_name}, but embedding backend returned {used_mode}. {status_msg}"
                )

            for cfg in configs:
                status_box.info(
                    f"Dialogue {dialog_id} | {cfg['config_id']} | {completed + 1}/{total_runs}"
                )
                try:
                    run = run_pipeline_from_embeddings(
                        E=E,
                        texts=texts,
                        participants=participants,
                        smooth_method=cfg["smooth_method"],
                        env_alpha=cfg["env_alpha"],
                        env_span=cfg["env_span"],
                        use_cinv=cfg["use_cinv"],
                        cinv_window=cfg["cinv_window"],
                        cinv_knn=cfg["cinv_knn"],
                        cinv_thr=cfg["cinv_thr"],
                        cinv_keigs=cfg["cinv_keigs"],
                        alpha_context=cfg["alpha_context"],
                        strong_weights=cfg["strong_weights"],
                        transition_weights=cfg["transition_weights"],
                    )

                    ev = run["event_scores"]
                    n = len(turns)
                    frame = pd.DataFrame({
                        "dialog_id": dialog_id,
                        "source_file": uploaded_file.name,
                        "embedding": embedding_name,
                        "used_mode": used_mode,
                        "config_id": cfg["config_id"],
                        "varied_parameter": cfg["varied_parameter"],
                        "parameter_value": cfg["parameter_value"],
                        "is_base_config": cfg["is_base_config"],
                        "turn": turns,
                        "smooth_method": cfg["smooth_method"],
                        "env_alpha": cfg["env_alpha"],
                        "env_span": cfg["env_span"],
                        "cinv_window": cfg["cinv_window"],
                        "cinv_knn": cfg["cinv_knn"],
                        "cinv_thr": cfg["cinv_thr"],
                        "cinv_keigs": cfg["cinv_keigs"],
                        "alpha_context": cfg["alpha_context"],
                        "strong_weights": [str(cfg["strong_weights"])] * n,
                        "transition_weights": [str(cfg["transition_weights"])] * n,
                        "Ct": run["Ct"],
                        "Ct_smooth": run["Ct_smooth"],
                        "C_inv": run["C_inv"] if run["C_inv"] is not None else np.nan,
                        "rho_t": run["rho_t"],
                        "S_t": run["S_t"],
                        "R_t": run["R_t"],
                        "D_t": run["D_t"],
                        "semantic_event_drop": ev.get("sem_drop", np.zeros(n)),
                        "structural_event_drop": ev.get("struct_drop", np.zeros(n)),
                        "semantic_score": ev.get("semantic_score", np.zeros(n)),
                        "structural_score": ev.get("structural_score", np.zeros(n)),
                        "strong_score": ev.get("strong_score", np.zeros(n)),
                        "transition_pressure": ev.get("transition_pressure", np.zeros(n)),
                    })
                    turn_frames.append(frame)
                except Exception as e:
                    error_rows.append({
                        "dialog_id": dialog_id,
                        "source_file": uploaded_file.name,
                        "config_id": cfg["config_id"],
                        "varied_parameter": cfg["varied_parameter"],
                        "error": str(e),
                    })
                completed += 1
                progress_bar.progress(min(completed / max(total_runs, 1), 1.0))
        except Exception as e:
            error_rows.append({
                "dialog_id": None,
                "source_file": uploaded_file.name,
                "config_id": None,
                "varied_parameter": None,
                "error": str(e),
            })

    progress_bar.progress(1.0)
    status_box.success(f"Hyperparameter batch completed: {completed}/{total_runs} runs attempted.")
    turn_df = pd.concat(turn_frames, ignore_index=True) if turn_frames else pd.DataFrame()
    config_df = pd.DataFrame(configs)
    errors_df = pd.DataFrame(error_rows)
    return turn_df, config_df, errors_df

# ============================
# app.py — PART 3/3
# (demo + Streamlit app main + downloads)
# State_alpha slider exists and matches the call
# Undefined vars deleted (Ct_action/Ft/PDF/geom_* vars)
# ============================

def load_demo(n_turns: int = 36) -> pd.DataFrame:
    speakers = ["A", "B", "C", "D"]
    turns, parts, texts = [], [], []

    all_lines = [
        # --- Stable opening: planning the explanation ---
        "Before presenting the app, we should explain that it tracks how conversational coherence changes over time.",
        "Yes, and we should make clear that it models dialogue as a dynamic process rather than a static exchange of sentences.",
        "Exactly. The point is that each turn either sustains or destabilizes the evolving trajectory of the conversation.",
        "So the app is not just reading text. It is estimating how each contribution fits into a shared conversational structure.",
        "One useful angle is to say that semantic coherence measures how strongly each turn aligns with the evolving context.",
        "And structural coherence adds another layer by capturing changes in the relational organization of the dialogue.",
        "Right, so the system can distinguish ordinary continuation from deeper forms of disruption.",
        "That distinction is probably what makes the tool most interesting for analysis.",

        # --- Stable elaboration ---
        "We should also mention that the output is visual, so people can see where coherence rises, drops, and stabilizes again.",
        "Yes, because the value is not just a final score but the temporal pattern across turns.",
        "That makes the app useful for identifying possible rupture zones, transitions, and recoveries.",
        "Exactly. It gives a structured view of dialogue as something measurable in time.",

        # --- Rupture 1: semantic drift ---
        "Did anyone remember to buy coffee filters for tomorrow morning?",
        "That feels unrelated. We were still defining what the app actually measures.",
        "Yes, that move shifts the topic away from the explanation we were building.",
        "Let us return to the app itself and leave practical things for later.",

        # --- Recovery after rupture 1 ---
        "Right. Back to the app: another key point is that coherence is tracked turn by turn.",
        "And that allows us to compare moments of stability against moments of local breakdown.",
        "We should probably say that the system supports interpretation rather than replacing human judgment.",
        "Yes, because the graphs are useful precisely when they can later be compared with annotation or close reading.",

        # --- Stable continuation ---
        "We can also mention that the app compares participants, not only the dialogue as a whole.",
        "That matters because some breakdowns are local to one speaker while others affect the whole interaction.",
        "So the contribution is both global and participant-sensitive.",
        "Exactly. That makes the analysis richer than using one overall average.",

        # --- Rupture 2: stronger / frame-breaking ---
        "Unless coherence is actually controlled by a hidden committee of underwater mathematicians rewriting the dialogue in secret.",
        "Okay, that breaks the frame much more strongly.",
        "Yes, that is no longer a simple topic drift. It disrupts the explanatory frame itself.",
        "Let us reset and return to the task in a grounded way.",

        # --- Recovery after rupture 2 ---
        "Fine. The central claim is that the app detects stability, breakdown, and recovery in conversational structure.",
        "And it does so by combining semantic continuity with structural variation over time.",
        "That combination is what allows different rupture types to be distinguished instead of collapsed into one signal.",
        "We should end by saying that this makes dialogue dynamics observable rather than merely intuitive.",

        # --- Stable closing ---
        "So the simplest explanation is that the app measures how conversations hold together, drift, break, and recover.",
        "And because it does that turn by turn, it becomes possible to inspect the dynamics rather than only the outcome.",
        "That is probably the clearest closing: the app makes conversational structure visible.",
        "Agreed. That gives us a concise and realistic summary of what the system does.",
    ]

    all_lines = all_lines[:max(1, int(n_turns))]

    for i, text in enumerate(all_lines):
        turns.append(i + 1)
        parts.append(speakers[i % len(speakers)])
        texts.append(text)

    return pd.DataFrame(
        {
            "turn": np.array(turns, dtype=int),
            "timestamp": "",
            "participant": parts,
            "text": texts,
        }
    )        
# =========================
# Streamlit app
# =========================
lang = st.sidebar.selectbox(LABELS["en"]["lang"], options=["en", "es"], index=0)
L = LABELS[lang]

st.title(L["app_title"])
st.caption(L["app_subtitle"])

with st.expander(L["what_does"], expanded=False):
    st.markdown("""
This application represents conversation as a continuously evolving organizational system rather than as a static sequence of independent utterances.

Instead of analysing dialogue solely through turn-to-turn semantic similarity, TIE–Dialog estimates an evolving conversational state that changes as each new contribution reshapes the accumulated interaction.

To represent this evolving organizational state, the framework combines multiple complementary computational observables:

- **Contextual Continuity (Cₜ)** estimates how naturally each conversational turn integrates into the evolving conversational context.

- **Organizational Persistence (C_inv)** estimates the extent to which the underlying conversational organization remains stable as the dialogue develops.

- **Conversational Trajectory Dynamics** characterizes how the overall interaction evolves through semantic space, capturing changes in conversational direction that cannot be explained by local continuity alone.

TIE–Dialog keeps these observables separate rather than collapsing them into a single categorical state.

The core representation is multivariate and continuous:

z_t = (S_t, R_t, d_t, κ_t, u_t)

with **S_t = 1 - C_t** (Contextual Discontinuity), **R_t = 1 - C_inv(t)** (Structural Instability / low persistence), **d_t** as angular displacement, **κ_t** as turning-angle curvature, and **u_t = 1 - ρ_t** as local semantic dispersion. These coordinates are kept separate in the primary analysis. The composite Geometry Driver D_t is retained only as an exploratory legacy summary and does not define the primary state.

The framework therefore focuses on the temporal relationships among continuous conversational observables rather than imposing regime labels.

Unlike traditional approaches that focus primarily on neighbouring-turn similarity, TIE–Dialog represents conversation as a dynamic organizational process. Different conversational phenomena may therefore produce distinct organizational signatures even when local semantic similarity appears relatively stable.

Internally, the framework analyses the evolving trajectory through semantic space using separate geometric observables: local angular displacement, turning-angle curvature, and local semantic dispersion. A scalar geometry summary can still be inspected for backwards compatibility, but primary inference remains component-wise so that distinct geometric behaviours are not collapsed prematurely.

The goal of TIE–Dialog is not simply to measure conversational coherence, but to provide a computational representation of the temporal organization of conversation. By making this evolving organization observable, the framework enables conversational dynamics to be visualized, interpreted, and empirically investigated.
""")

    st.write("Expected columns: `turn`, `timestamp` (optional), `participant`, `text`.")
    
# =========================
# Mode: Canonical / Explore
# =========================
mode = st.sidebar.radio(
    "Mode",
    ["Canonical", "Explore"],
    index=0,
    help="Canonical = defaults fixed (clean UI). Explore = full controls."
)
IS_CANON = (mode == "Canonical")

def ui_slider(label, min_value, max_value, value, step, key=None, help=None):
    if IS_CANON:
        return value
    return st.slider(label, min_value, max_value, value, step, key=key, help=help)

def ui_checkbox(label, value=False, key=None, help=None):
    if IS_CANON:
        return value
    return st.checkbox(label, value=value, key=key, help=help)

def ui_selectbox(label, options, index=0, key=None, help=None):
    if IS_CANON:
        return options[index]
    return st.selectbox(label, options, index=index, key=key, help=help)

def ui_text_input(label, value="", key=None, help=None):
    if IS_CANON:
        return value
    return st.text_input(label, value=value, key=key, help=help)
    
with st.expander("Parameter guide", expanded=False):
    st.markdown("""
This guide follows the current sidebar structure: **Data**, **Core**, **Events**, **Visual**, and **Robust**.

---

## Data

**Load demo**  
Uses the built-in example dialogue. Disable it to upload your own CSV or Excel file.

**Upload dataset**  
Expected columns: `turn`, `timestamp` (optional), `participant`, and `text`.

---

## Core

**Embeddings mode**  
Selects how turns are represented semantically.  
Options include MiniLM/SBERT, E5, BGE, Instructor, or TF-IDF fallback.

**SBERT model**  
Allows a custom sentence-transformer model name. In Canonical mode, the default model is used automatically.

**Smoothing method**  
Controls how the coherence trajectory is smoothed.  
EMA reacts faster to local changes. EWMA produces a smoother trajectory over a longer span.

**EMA alpha**  
Controls responsiveness of EMA smoothing.  
Higher alpha = more reactive. Lower alpha = smoother and more stable.

**EWMA span**  
Controls the smoothing window for EWMA.  
Higher span = smoother curve. Lower span = more local sensitivity.

---

## Structural channel

**Enable C_inv**  
Activates the structural coherence channel based on graph invariants.

**C_inv window W**  
Controls the rolling window used to build local similarity graphs.  
Larger windows produce more stable structural estimates. Smaller windows are more sensitive.

**C_inv k-NN**  
Controls how many nearest neighbors are used when building the graph.  
Higher k creates denser graphs. Lower k creates simpler, more conservative graphs.

**C_inv edge threshold**  
Filters weak graph edges.  
Higher threshold removes weaker connections and may reduce noise.

**C_inv eigenfeatures (k)**  
Controls how many spectral graph features are retained.  
Higher values capture more structural detail. Lower values emphasize global stability.

---

## Visual

**Ci method**  
Controls how participant-level coherence trajectories are computed.

**Ci alpha**  
Controls participant-level inertia.  
Higher alpha = participant trajectories preserve more past context. Lower alpha = more responsiveness.

**State alpha**  
Controls smoothing/inertia in continuous participant state trajectories.

**Public View span**  
Controls the EWMA smoothing of the simplified public S–B–R visualization.

**Show Φ thresholds in public plot**  
Displays Φ_low and Φ_high in the public-facing plot.

**Show smoothed coherence plot**  
Displays an additional smoothed coherence view.

**Overlay smoothed curve on main plot**  
Adds the smoothed Cₜ curve to the main plot.

**Overlay C_inv on main plot**  
Adds the structural coherence channel to the main plot.

---

## Robust

**Compare embeddings automatically**  
Runs the same dialogue through multiple embedding models and compares whether the coherence trajectories and detected events remain similar.

**Embeddings to compare**  
Selects which representation models are included in the comparison.

**Event alignment window**  
Tolerance window used to decide whether events from different embeddings occur in the same region.

**Show baseline diagnostic**  
Compares TIE–Dialog against simpler baselines: turn-to-turn cosine, moving-context cosine, geometric displacement, and shuffled order.

**Moving average window**  
Controls the context size for the moving-context baseline.

**Run robustness test**  
Perturbs parameters while keeping embeddings fixed, testing whether trajectories and event masks remain stable.

**Robustness runs**  
Number of perturbed configurations to test.

**Parameter perturbation (%)**  
How strongly parameters are randomly varied around the current configuration.

**Robustness random seed**  
Controls reproducibility of the robustness test.

---

## Practical tip

Use **Canonical mode** for clean demos and stable outputs.  
Use **Explore mode** when tuning parameters, testing robustness, or preparing validation results.
""")

with st.sidebar:
    st.header(L["params"])

    data_tab, core_tab, event_tab, visual_tab, robust_tab = st.tabs([
        "Data",
        "Core",
        "Structure",
        "Visual",
        "Robust"
    ])

    with data_tab:
        validation_mode = st.radio(
            "Analysis mode",
            options=["Single dialogue", "Embedding batch", "Hyperparameter batch"],
            index=0,
            key="analysis_mode",
        )

    batch_mode = validation_mode == "Embedding batch"
    hyper_batch_mode = validation_mode == "Hyperparameter batch"

    if batch_mode or hyper_batch_mode:
        use_demo = False
        uploaded = None

        batch_uploaded_files = st.file_uploader(
            "Upload dialogue files",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key="batch_dialogues",
            help=(
                "Upload the 12 dialogue CSV/XLSX files. "
                "Each file must contain turn, participant and text."
            ),
        )
    else:
        batch_uploaded_files = []

        use_demo = st.checkbox(
            L["load_demo"],
            value=True,
        )

        uploaded = (
            None
            if use_demo
            else st.file_uploader(
                L["upload"],
                type=["csv", "xlsx"],
                key="single_dialogue",
            )
        )
        

    with core_tab:
        st.subheader(L["sem_repr"])

        emb_mode = ui_selectbox(
            L["emb_mode"],
            ["auto", "sbert", "e5", "bge", "instructor", "tfidf"],
            index=0
        )
                

        st.markdown("### Smoothed coherence")
        smooth_method = ui_selectbox("Smoothing method", ["ema", "ewma"], index=0)
        env_alpha = ui_slider("EMA alpha", 0.05, 0.60, 0.20, 0.01)
        env_span = ui_slider("EWMA span", 3, 25, 9, 1)

        # Default model logic inside tab, before text input
        default_model = "sentence-transformers/all-MiniLM-L6-v2"

        if emb_mode == "e5":
            default_model = "intfloat/e5-base-v2"
        elif emb_mode == "bge":
            default_model = "BAAI/bge-base-en-v1.5"
        elif emb_mode == "instructor":
            default_model = "hkunlp/instructor-base"
        elif emb_mode == "tfidf":
            default_model = ""

        sbert_model = ui_text_input(L["sbert_model"], value=default_model)

    with event_tab:
        st.subheader("Structural channel")
        

        st.markdown("### Invariant coherence (C_inv)")
        use_cinv = ui_checkbox("Enable C_inv (graph invariants)", value=True)
        cinv_window = ui_slider("C_inv window W", 6, 24, 8, 1)
        cinv_knn = ui_slider("C_inv k-NN", 2, 10, 3, 1)
        cinv_thr = ui_slider("C_inv edge threshold", 0.00, 0.30, 0.16, 0.01)
        cinv_keigs = ui_slider("C_inv eigenfeatures (k)", 3, 12, 6, 1)

    with visual_tab:
        st.subheader("Research Tutorial view")
        ci_method = ui_selectbox(L["ci_method"], ["ctx", "im"], index=0)
        ci_alpha = ui_slider(L["ci_alpha"], 0.70, 0.99, 0.90, 0.01)

        # Diagnostics are available, but not part of the primary tutorial result flow.
        show_envelope = ui_checkbox("Show smoothed-coherence diagnostic", value=False)
        overlay_cinv_on_main = ui_checkbox("Overlay C_inv on main plot", value=True)

        # Legacy operational event regions are deliberately hidden in the tutorial view.
        # Human consensus should remain an external validation target rather than being
        # conflated with model-defined event labels.
        show_event_regions = False

        st.markdown("### Continuous multivariate dynamics")
        show_crossdim = True
        crossdim_window = ui_slider(
            "Rolling dependency window",
            4, 15, 7, 1,
        )
        crossdim_max_lag = ui_slider(
            "Lead–lag maximum (± turns)",
            1, 10, 6, 1,
        )
        st.caption(
            "Primary tutorial analysis: z_t=(S_t,R_t,d_t,κ_t,u_t), where S_t=1-C_t is Contextual "
            "Discontinuity, R_t=1-C_inv is Structural Instability / low persistence, d_t is angular "
            "displacement, κ_t is turning-angle curvature, and u_t=1-ρ_t is local semantic dispersion. "
            "The geometry composite D_t is exploratory only and is excluded from the primary multivariate analysis."
        )

    with robust_tab:
        if batch_mode:
            st.markdown("### Batch embedding validation")

            batch_embeddings = st.multiselect(
                "Embeddings for batch run",
                options=["MiniLM", "E5", "BGE", "Instructor"],
                default=["MiniLM", "E5", "BGE", "Instructor"],
                key="batch_embeddings",
            )

            run_batch_button = st.button(
                "Run all dialogues",
                type="primary",
                key="run_batch_embeddings",
            )

            st.caption(
                "All TIE-Dialog parameters remain fixed. "
                "Only the embedding model changes."
            )
        else:
            run_batch_button = False
            batch_embeddings = []

        if hyper_batch_mode:
            st.markdown("### Batch hyperparameter robustness")
            hyper_embedding = st.selectbox(
                "Fixed embedding",
                options=["MiniLM", "E5", "BGE", "Instructor"],
                index=0,
                key="hyper_batch_embedding",
            )
            hyper_configs_preview = build_hyperparameter_configs({
                "smooth_method": str(smooth_method),
                "env_alpha": float(env_alpha), "env_span": int(env_span),
                "use_cinv": bool(use_cinv), "cinv_window": int(cinv_window),
                "cinv_knn": int(cinv_knn), "cinv_thr": float(cinv_thr),
                "cinv_keigs": int(cinv_keigs), "alpha_context": 0.84,
                "strong_weights": (0.25, 0.35, 0.40),
                "transition_weights": (0.15, 0.25, 0.60),
            })
            st.caption(
                f"One-factor-at-a-time grid: {len(hyper_configs_preview)} configurations per dialogue. "
                "The embedding and all non-varied parameters remain fixed."
            )
            run_hyper_batch_button = st.button(
                "Run hyperparameter batch",
                type="primary",
                key="run_batch_hyperparameters",
            )
        else:
            hyper_embedding = "MiniLM"
            run_hyper_batch_button = False
            
        st.markdown("### Embedding comparison")

        compare_embeddings = ui_checkbox("Compare embeddings automatically", value=False)

        embedding_compare_options = ["MiniLM", "E5", "BGE", "Instructor"]
        selected_compare_embeddings = []

        if compare_embeddings and not IS_CANON:
            selected_compare_embeddings = st.multiselect(
                "Embeddings to compare",
                options=embedding_compare_options,
                default=["MiniLM", "E5", "BGE", "Instructor"],
            )

            event_alignment_window = st.slider(
                "Event alignment window (± turns)",
                1, 6, 3, 1
            )
        else:
            event_alignment_window = 3

        # Legacy strong-score baseline and ablation diagnostics are intentionally
        # disabled in the Research Tutorial UI. Their functions remain in the file
        # for archival/reproducibility purposes, but the tutorial focuses on the
        # continuous S/R/d/kappa/u dynamics rather than the former composite event score.
        show_baseline_comparison = False
        baseline_moving_window = 5
        show_ablation_comparison = False

        st.markdown("### Robustness test")
        run_param_robustness = ui_checkbox(
            "Run robustness test",
            value=False,
            key="robust_run_toggle"
        )

        robustness_n_runs = ui_slider(
            "Robustness runs",
            10, 100, 30, 5,
            key="robust_runs_n"
        )

        robustness_pct = ui_slider(
            "Parameter perturbation (%)",
            0.0, 0.50, 0.15, 0.01,
            key="robust_pct"
        )

        robustness_seed = ui_slider(
             "Robustness random seed",
             0, 9999, 42, 1,
             key="robust_seed"
        )
            
# =========================================================
# Batch hyperparameter robustness
# =========================================================
if hyper_batch_mode:
    st.header("Batch hyperparameter robustness")

    if not batch_uploaded_files:
        st.info("Upload the 12 dialogue files in the Data tab.")
        st.stop()

    preview_base = {
        "smooth_method": str(smooth_method),
        "env_alpha": float(env_alpha), "env_span": int(env_span),
        "use_cinv": bool(use_cinv), "cinv_window": int(cinv_window),
        "cinv_knn": int(cinv_knn), "cinv_thr": float(cinv_thr),
        "cinv_keigs": int(cinv_keigs), "alpha_context": 0.84,
        "strong_weights": (0.25, 0.35, 0.40),
        "transition_weights": (0.15, 0.25, 0.60),
    }
    preview_configs = build_hyperparameter_configs(preview_base)
    expected_runs = len(batch_uploaded_files) * len(preview_configs)
    st.write(
        f"Planned executions: **{expected_runs}** "
        f"({len(batch_uploaded_files)} dialogues × {len(preview_configs)} configurations)"
    )

    if run_hyper_batch_button:
        hyper_turn_df, hyper_config_df, hyper_errors_df = run_batch_hyperparameter_validation(
            uploaded_files=batch_uploaded_files,
            embedding_name=hyper_embedding,
            smooth_method=smooth_method,
            env_alpha=env_alpha, env_span=env_span,
            use_cinv=use_cinv,
            cinv_window=cinv_window, cinv_knn=cinv_knn,
            cinv_thr=cinv_thr, cinv_keigs=cinv_keigs,
        )
        st.session_state["hyper_turn_df"] = hyper_turn_df
        st.session_state["hyper_config_df"] = hyper_config_df
        st.session_state["hyper_errors_df"] = hyper_errors_df
        st.session_state["hyper_turn_csv"] = dataframe_to_csv_bytes(hyper_turn_df)
        st.session_state["hyper_config_csv"] = dataframe_to_csv_bytes(hyper_config_df)
        st.session_state["hyper_errors_csv"] = dataframe_to_csv_bytes(hyper_errors_df)

    hyper_turn_df = st.session_state.get("hyper_turn_df", pd.DataFrame())
    hyper_config_df = st.session_state.get("hyper_config_df", pd.DataFrame())
    hyper_errors_df = st.session_state.get("hyper_errors_df", pd.DataFrame())

    if not hyper_config_df.empty:
        st.subheader("Configuration manifest")
        st.dataframe(hyper_config_df, use_container_width=True)
        st.download_button(
            "Download configuration manifest",
            data=st.session_state.get("hyper_config_csv", dataframe_to_csv_bytes(hyper_config_df)),
            file_name="hyperparameter_config_manifest.csv",
            mime="text/csv",
            key="download_hyper_configs",
            on_click="ignore",
        )

    if not hyper_turn_df.empty:
        st.subheader("Turn-level hyperparameter runs")
        st.write(f"{len(hyper_turn_df):,} rows generated.")
        st.download_button(
            "Download all hyperparameter runs",
            data=st.session_state.get("hyper_turn_csv", dataframe_to_csv_bytes(hyper_turn_df)),
            file_name="hyperparameter_runs_turn_level.csv",
            mime="text/csv",
            key="download_hyper_turns",
            on_click="ignore",
        )

    if not hyper_errors_df.empty:
        st.subheader("Errors")
        st.dataframe(hyper_errors_df, use_container_width=True)
        st.download_button(
            "Download error log",
            data=st.session_state.get("hyper_errors_csv", dataframe_to_csv_bytes(hyper_errors_df)),
            file_name="hyperparameter_batch_errors.csv",
            mime="text/csv",
            key="download_hyper_errors",
            on_click="ignore",
        )

    st.stop()

# =========================================================
# Batch embedding validation
# =========================================================
if batch_mode:
    st.header("Batch embedding validation")

    if not batch_uploaded_files:
        st.info("Upload the 12 dialogue files in the Data tab.")
        st.stop()

    st.write(
        f"Files loaded: **{len(batch_uploaded_files)}**"
    )

    if len(batch_uploaded_files) != 12:
        st.warning(
            f"You uploaded {len(batch_uploaded_files)} files. "
            "The pilot currently expects 12 dialogues."
        )

    if len(batch_embeddings) < 2:
        st.warning("Select at least two embedding models.")
        st.stop()

    expected_runs = (
        len(batch_uploaded_files) *
        len(batch_embeddings)
    )

    st.write(
        f"Planned executions: **{expected_runs}** "
        f"({len(batch_uploaded_files)} dialogues × "
        f"{len(batch_embeddings)} embeddings)"
    )

    if run_batch_button:
        (
            batch_turn_level_df,
            batch_pairwise_df,
            batch_summary_df,
            batch_errors_df,
        ) = run_batch_embedding_validation(
            uploaded_files=batch_uploaded_files,
            selected_embeddings=batch_embeddings,
            smooth_method=smooth_method,
            env_alpha=env_alpha,
            env_span=env_span,
            use_cinv=use_cinv,
            cinv_window=cinv_window,
            cinv_knn=cinv_knn,
            cinv_thr=cinv_thr,
            cinv_keigs=cinv_keigs,
        )

        st.session_state["batch_turn_level_df"] = batch_turn_level_df
        st.session_state["batch_pairwise_df"] = batch_pairwise_df
        st.session_state["batch_summary_df"] = batch_summary_df
        st.session_state["batch_errors_df"] = batch_errors_df

        # Prepare downloads once, immediately after the batch finishes.
        # This prevents each download click from rebuilding large CSVs.
        st.session_state["batch_turn_level_csv"] = dataframe_to_csv_bytes(
            batch_turn_level_df
        )
        st.session_state["batch_pairwise_csv"] = dataframe_to_csv_bytes(
            batch_pairwise_df
        )
        st.session_state["batch_summary_csv"] = dataframe_to_csv_bytes(
            batch_summary_df
        )
        st.session_state["batch_errors_csv"] = dataframe_to_csv_bytes(
            batch_errors_df
        )

    batch_turn_level_df = st.session_state.get(
        "batch_turn_level_df",
        pd.DataFrame(),
    )
    batch_pairwise_df = st.session_state.get(
        "batch_pairwise_df",
        pd.DataFrame(),
    )
    batch_summary_df = st.session_state.get(
        "batch_summary_df",
        pd.DataFrame(),
    )
    batch_errors_df = st.session_state.get(
        "batch_errors_df",
        pd.DataFrame(),
    )

    if not batch_summary_df.empty:
        st.subheader("Per-dialogue embedding summary")
        st.dataframe(
            batch_summary_df,
            use_container_width=True,
        )

        st.download_button(
            "Download embedding summary",
            data=st.session_state.get(
                "batch_summary_csv",
                dataframe_to_csv_bytes(batch_summary_df),
            ),
            file_name="embedding_summary_by_dialogue.csv",
            mime="text/csv",
            key="download_batch_summary",
            on_click="ignore",
        )

    if not batch_pairwise_df.empty:
        st.subheader("Pairwise comparisons by dialogue")
        st.dataframe(
            batch_pairwise_df,
            use_container_width=True,
        )

        st.download_button(
            "Download pairwise comparisons",
            data=st.session_state.get(
                "batch_pairwise_csv",
                dataframe_to_csv_bytes(batch_pairwise_df),
            ),
            file_name="embedding_pairwise_by_dialogue.csv",
            mime="text/csv",
            key="download_batch_pairwise",
            on_click="ignore",
        )

    if not batch_turn_level_df.empty:
        st.subheader("Turn-level results")

        st.write(
            f"{len(batch_turn_level_df):,} turn-level rows generated."
        )

        st.download_button(
            "Download all turn-level runs",
            data=st.session_state.get(
                "batch_turn_level_csv",
                dataframe_to_csv_bytes(batch_turn_level_df),
            ),
            file_name="embedding_runs_turn_level.csv",
            mime="text/csv",
            key="download_batch_turn_level",
            on_click="ignore",
        )

    if not batch_errors_df.empty:
        st.subheader("Errors and fallbacks")
        st.dataframe(
            batch_errors_df,
            use_container_width=True,
        )

        st.download_button(
            "Download error log",
            data=st.session_state.get(
                "batch_errors_csv",
                dataframe_to_csv_bytes(batch_errors_df),
            ),
            file_name="embedding_batch_errors.csv",
            mime="text/csv",
            key="download_batch_errors",
            on_click="ignore",
        )

    # Prevent the normal single-dialogue app from continuing
    st.stop()

if use_demo:
    df = load_demo(n_turns=34)
else:
    if uploaded is None:
        st.info("Upload a dataset or enable demo.")
        st.stop()
        
    if str(uploaded.name).lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

df, missing = _clean_df(df)

# =========================
# Core dialogue arrays + embeddings
# =========================
turns = df["turn"].to_numpy(dtype=int)
texts = df["text"].astype(str).tolist()
participants = df["participant"].astype(str).tolist()

E, used_mode, emb_msg = embed_texts(
    texts=texts,
    mode=emb_mode,
    sbert_model=sbert_model,
)

st.caption(emb_msg)

# =========================
# EMBEDDING COMPARISON
# =========================
embedding_summary_df = None
embedding_pairs_df = None
embedding_alignment_df = None
embedding_shuffle_df = None
embedding_variance_df = None

baseline_compare_df = None
event_overlap_df = None
shuffled_diagnostic_df = None

ablation_df = None
semantic_ablation_df = None

robustness_runs_df = None
robustness_summary_df = None
robustness_movement_df = None

if compare_embeddings and len(selected_compare_embeddings) >= 2:

    model_map = {
        "MiniLM": ("sbert", "sentence-transformers/all-MiniLM-L6-v2"),
        "E5": ("e5", "intfloat/e5-base-v2"),
        "BGE": ("bge", "BAAI/bge-base-en-v1.5"),
        "Instructor": ("instructor", "hkunlp/instructor-base"),
    }

    embedding_runs = {}

    for emb_name in selected_compare_embeddings:
        if emb_name not in model_map:
            continue

        emb_mode_cmp, emb_model_cmp = model_map[emb_name]

        # --- embeddings ---
        E_cmp, used_mode_cmp, emb_msg_cmp = embed_texts(
            texts=texts,
            mode=emb_mode_cmp,
            sbert_model=emb_model_cmp,
        )

        # --- pipeline ---
        run_cmp = run_pipeline_from_embeddings(
            E=E_cmp,
            texts=texts,
            participants=participants,
            smooth_method=smooth_method,
            env_alpha=env_alpha,
            env_span=env_span,
            use_cinv=use_cinv,
            cinv_window=cinv_window,
            cinv_knn=cinv_knn,
            cinv_thr=cinv_thr,
            cinv_keigs=cinv_keigs,
        )

        run_cmp["used_mode"] = used_mode_cmp
        run_cmp["emb_msg"] = emb_msg_cmp

        embedding_runs[emb_name] = run_cmp

    # --- compare ---
    if len(embedding_runs) >= 2:
       embedding_summary_df, embedding_pairs_df = compare_embedding_runs(embedding_runs)

       embedding_alignment_df = build_event_alignment_matrix(
           embedding_runs,
           window=int(event_alignment_window),
           event_q=0.85,
       )

       embedding_shuffle_df = build_real_vs_shuffled_embedding_table(
           embedding_runs,
           seed=42,
       )

       embedding_variance_df = build_variance_decomposition_table(
           embedding_runs,
       )

# =========================
# DISPLAY RESULTS
# =========================
if compare_embeddings and embedding_summary_df is not None and not embedding_summary_df.empty:

    st.subheader("Embedding comparison")

    # --- summary ---
    st.markdown("### Per-embedding summary")
    st.dataframe(embedding_summary_df, use_container_width=True)

    st.caption(
        "This table summarizes how each embedding model represents the same dialogue. "
        "mean_Ct and std_Ct describe the average level and variability of coherence. "
        "strong_events_n and broken_turns_n show how many disruption-like moments each embedding detects. "
        "If the values are similar across embeddings, the signal is less dependent on one specific representation."
    )

    # --- pairwise ---
    if embedding_pairs_df is not None and not embedding_pairs_df.empty:
        st.markdown("### Pairwise comparison")
        st.dataframe(embedding_pairs_df, use_container_width=True)

        st.caption(
            "This table compares pairs of embedding models after normalization. "
            "Ct_corr_zscore measures whether the coherence trajectories have a similar shape after removing mean and scale differences. "
            "Ct_dtw_similarity_minmax measures dynamic similarity after rescaling the curves to the same 0–1 range. "
            "strong_score_corr_zscore and strong_score_dtw_similarity_minmax do the same for event-pressure trajectories. "
            "Higher values suggest partial invariance across embedding representations."
        )

    if embedding_alignment_df is not None and not embedding_alignment_df.empty:
        st.markdown(f"### Event alignment score (±{event_alignment_window} turns)")
        st.dataframe(embedding_alignment_df, use_container_width=True)

        st.caption(
            f"This table measures whether strong events detected by different embeddings occur in approximately the same region of the dialogue. "
            f"An event is counted as aligned when it appears within ±{event_alignment_window} turns. "
            "High alignment means the model is detecting shared conversational structure rather than embedding-specific noise."
        )

    if embedding_shuffle_df is not None and not embedding_shuffle_df.empty:
        st.markdown("### Cross-embedding permutation test")
        st.dataframe(embedding_shuffle_df, use_container_width=True)

        st.caption(
            "This table tests whether cross-embedding similarity depends on the real temporal order of the dialogue. "
            "real_corr compares the original coherence/event trajectories. "
            "shuffled_corr compares the same signals after temporal order is randomly disrupted. "
            "A positive delta means the real dialogue has more structure than its shuffled version."
        )

    if embedding_variance_df is not None and not embedding_variance_df.empty:
        st.markdown("### Variance decomposition")
        st.dataframe(embedding_variance_df, use_container_width=True)

        st.caption(
            "This table estimates where the variation in the signal comes from. "
            "The embedding component reflects differences caused by the representation model. "
            "The dialogue_turn_structure component reflects differences caused by the actual progression of the dialogue. "
            "If dialogue_turn_structure is larger, the signal is mainly driven by conversational structure rather than embedding choice."
        )

    # --- overlay plot ---
    if len(embedding_runs) >= 2:
        st.markdown("### Overlaid coherence trajectories")

        fig_emb_overlay = plot_embedding_comparison_overlay(
            embedding_runs,
            height=500,
            title="Embedding comparison — overlaid Ct trajectories",
        )

        st.plotly_chart(fig_emb_overlay, use_container_width=True)
        st.caption(
            "Each line shows the same dialogue's contextual-continuity trajectory Cₜ computed with a different "
            "embedding model. Close trajectories indicate that the broad temporal pattern is relatively robust to "
            "the semantic representation, whereas sustained divergences identify embedding-sensitive regions. "
            "This plot evaluates representation robustness only; agreement across embeddings does not by itself "
            "establish agreement with human transition judgments."
        )
            
# -------------------------------
# C_inv (graph-invariant coherence)
# -------------------------------
C_inv = None
if use_cinv:
    C_inv = compute_C_inv_series(
        E,
        window=int(cinv_window),
        k_nn=int(cinv_knn),
        thr=float(cinv_thr),
        k_eigs=int(cinv_keigs),
        D_max=None,
    )

# --- Smooth C_inv for visualization only ---
C_inv_plot = None
if C_inv is not None:
    C_inv_plot = np.asarray(C_inv, float).copy()
    mask = np.isfinite(C_inv_plot)
    if mask.sum() >= 5:
        valid_idx = np.where(mask)[0]
        a, b = valid_idx[0], valid_idx[-1]
        seg = C_inv_plot[a:b+1]                       # only valid segment
        seg_s = smooth_coherence(seg, method="ema", ema_alpha=0.18, ewma_span=9)
        C_inv_plot[a:b+1] = seg_s

ic2 = compute_ic2_dynamics(
    E,
    alpha_context=0.84,
    b=0.40,
)

# 1) Raw IC-II coherence (sigmoid output)
Ct_raw = np.asarray(ic2["C_t"], float)
Ct_raw = np.clip(Ct_raw, 0.0, 1.0)

# 2) Canonical coherence used by the app everywhere
Ct_base = np.clip(Ct_raw, 0.0, 1.0)
Ct_base = apply_warmup_ramp(
    Ct_base,
    warm=WARMUP_TURNS,
    floor=0.10,
)
Ct_base = np.clip(Ct_base, 0.0, 1.0)

# 3) Alignment with I_m (kept as auxiliary signal)
Ct_im = 0.5 * (1.0 + np.asarray(ic2["res"], float))
Ct_im = np.clip(Ct_im, 0.0, 1.0)

# 4) Smoothed coherence for visualization/support only
Ct_smooth = smooth_coherence(
    Ct_base,
    method=smooth_method,
    ema_alpha=float(env_alpha),
    ewma_span=int(env_span),
)
Ct_smooth = np.clip(Ct_smooth, 0.0, 1.0)

# ============================
# IC-III => IC-II (driver/lag)
# ============================
ic3 = compute_ic3_geometry(E=E)

signals = compute_all_signals(
    E=E,
    texts=texts,
    Ct_base=Ct_base,
    C_inv=C_inv,
    ic3=ic3
)

rho_t = signals["rho_t"]
u_t = signals["u_t"]
D_t = signals["D_t"]  # exploratory legacy summary only
d_i = signals["d_i"]
kappa_i = signals["kappa_i"]
# Backward-compatible aliases used by older diagnostics/download code.
d_i_driver = d_i
kappa_i_driver = kappa_i

event_scores = compute_event_scores(
    Ct=signals["Ct"],
    C_inv=signals["C_inv"],
    D_t=signals["D_t"]
)

event_labels_v2 = classify_event_scores(
    event_scores,
    D_t=D_t,
)

event_masks = labels_to_event_masks(
    event_labels_v2,
    min_sem_len=EVENT_MIN_SEM_LEN,
    min_struct_len=EVENT_MIN_STRUCT_LEN,
    min_strong_len=EVENT_MIN_STRONG_LEN,
)


strong_mask = np.asarray(event_masks["strong"], dtype=bool)

semantic_mask = np.asarray(event_masks["semantic"], dtype=bool)
structural_mask = np.asarray(event_masks["structural"], dtype=bool)

semantic_mask = points_to_mask(np.where(semantic_mask)[0].tolist(), len(semantic_mask), w=0) if semantic_mask.any() else semantic_mask
structural_mask = points_to_mask(np.where(structural_mask)[0].tolist(), len(structural_mask), w=0) if structural_mask.any() else structural_mask
strong_mask = points_to_mask(np.where(strong_mask)[0].tolist(), len(strong_mask), w=1) if strong_mask.any() else strong_mask

# remove strong ruptures first
semantic_mask = semantic_mask & (~strong_mask)
structural_mask = structural_mask & (~strong_mask)


event_labels_final = masks_to_display_labels(
    strong_mask=strong_mask,
    semantic_mask=semantic_mask,
    structural_mask=structural_mask,
)

event_labels_final = np.asarray(event_labels_final, dtype=object)


# =========================
# BUILD OUTPUT DATAFRAME
# =========================

df_out = df.copy()

# --- Core signals ---
df_out["Ct"] = Ct_base
df_out["Ct_im"] = Ct_im
df_out["rho_t"] = rho_t
df_out["d_i_raw"] = np.asarray(signals["d_i"], float)
df_out["kappa_i_raw"] = np.asarray(signals["kappa_i"], float)
df_out["d_t"] = np.asarray(d_i, float)
df_out["kappa_t"] = np.asarray(kappa_i, float)
df_out["u_t"] = np.asarray(u_t, float)
df_out["D_t_exploratory"] = np.asarray(D_t, float)

# =========================
# C_inv (if available)
# =========================
if C_inv is not None and np.asarray(C_inv).size == len(df_out):
    df_out["C_inv"] = np.asarray(C_inv, float)
else:
    df_out["C_inv"] = np.nan

# =========================================================
# PRIMARY CONTINUOUS COMPUTATIONAL STATE
# =========================================================
# Primary Research Tutorial state: z_t=(S_t,R_t,d_t,kappa_t,u_t).
# D_t is intentionally excluded; it is retained only as an exploratory legacy
# geometry magnitude for backwards-compatible diagnostics.
primary_state = compute_primary_state(
    Ct=np.asarray(Ct_base, float),
    C_inv=(np.asarray(C_inv, float) if C_inv is not None else None),
    d_i=np.asarray(d_i, float),
    kappa_i=np.asarray(kappa_i, float),
    rho_t=np.asarray(rho_t, float),
)

df_out["S_t"] = primary_state["S_t"]
df_out["R_t"] = primary_state["R_t"]
df_out["d_t"] = primary_state["d_t"]
df_out["kappa_t"] = primary_state["kappa_t"]
df_out["u_t"] = primary_state["u_t"]

# =========================================================
# PRIMARY CROSS-DIMENSIONAL ANALYSIS
# =========================================================
crossdim = compute_cross_dimensional_analysis(
    S=df_out["S_t"].to_numpy(dtype=float),
    R=df_out["R_t"].to_numpy(dtype=float),
    d=df_out["d_t"].to_numpy(dtype=float),
    kappa=df_out["kappa_t"].to_numpy(dtype=float),
    u=df_out["u_t"].to_numpy(dtype=float),
    rolling_window=int(crossdim_window),
    max_lag=int(crossdim_max_lag),
)

# First differences
change_col_map = {
    "S": "dS_t",
    "R": "dR_t",
    "d": "dd_t",
    "kappa": "dkappa_t",
    "u": "du_t",
}
for name, col in change_col_map.items():
    df_out[col] = crossdim["changes"][name]

# Secondary movement summaries
df_out["J_t"] = crossdim["J_t"]
for name, col in {
    "S": "J_share_S",
    "R": "J_share_R",
    "d": "J_share_d",
    "kappa": "J_share_kappa",
    "u": "J_share_u",
}.items():
    df_out[col] = crossdim["shares"][name]

for name, col in {
    "S": "dir_S",
    "R": "dir_R",
    "d": "dir_d",
    "kappa": "dir_kappa",
    "u": "dir_u",
}.items():
    df_out[col] = crossdim["directions"][name]

# Export all pairwise local coupling series under stable column names.
pair_alias = {"S": "S", "R": "R", "d": "d", "kappa": "kappa", "u": "u"}
for key, arr in crossdim["rolling_change_corr"].items():
    a, b = key.split("|")
    df_out[f"r_d{pair_alias[a]}_d{pair_alias[b]}"] = arr

df_out["Q_t"] = crossdim["Q_t"]

# =========================
# CLEAN FINAL OUTPUT
# =========================

df_out = df_out.drop(columns=[
    "semantic_score",
    "structural_score",
    "strong_score",
    "sem_drop",
    "struct_drop",
], errors="ignore")

fig_main = plot_ct_main(
    Ct=Ct_base,
    participants=participants,
    title="TIE–Dialog - Continuity Dynamics",
    height=560,
    C_inv=(
        C_inv
        if overlay_cinv_on_main and C_inv is not None
        else None
    ),
    event_labels=(
        event_labels_final
        if show_event_regions
        else None
    ),
)

st.plotly_chart(fig_main, use_container_width=True)
st.caption(
    "The main trajectory shows contextual continuity Cₜ across conversational turns: higher values indicate that "
    "the current contribution remains more compatible with the accumulated context, while lower values indicate "
    "stronger contextual departure. If enabled, C_inv adds structural continuity of the rolling semantic graph, "
    "and shaded legacy event regions show the app's operational classifications. Participant markers identify who "
    "produced each turn. The curves should be read as descriptive dynamics; a local drop is not automatically a "
    "human-perceived conversational transition."
)


# =========================
# Baseline diagnostic: Added Structural Value
# =========================
if show_baseline_comparison:
    st.subheader("Baseline Diagnostic: Added Structural Value")

    st.caption(
        "This diagnostic tests whether strong TIE–Dialog events can be reproduced by simpler baselines: "
        "turn-to-turn cosine disruption, moving-context cosine disruption, and geometric displacement. "
        "The goal is not only curve similarity, but event-level reconstruction."
    )

    baseline_dyn = compute_baseline_dynamics(
        E=E,
        moving_window=int(baseline_moving_window),
    )

    random_shuffled_baseline = compute_random_shuffled_baseline(
        E=E,
        texts=texts,
        participants=participants,
        smooth_method=smooth_method,
        env_alpha=env_alpha,
        env_span=env_span,
        use_cinv=use_cinv,
        cinv_window=cinv_window,
        cinv_knn=cinv_knn,
        cinv_thr=cinv_thr,
        cinv_keigs=cinv_keigs,
        seed=42,
    )

    comparison_signals = {
        "TIE–Dialog composite": np.asarray(event_scores["strong_score"], float),
        "Turn-to-turn cosine": np.asarray(baseline_dyn["cosine"], float),
        "Moving-context cosine": np.asarray(baseline_dyn["moving_avg"], float),
        "Geometric displacement": np.asarray(D_t, float),
        "Random shuffled baseline": np.asarray(random_shuffled_baseline, float),
    }

    fig_baseline = plot_baseline_comparison(
        comparison_signals,
        title="TIE–Dialog composite vs simple baselines",
    )
    st.plotly_chart(fig_baseline, use_container_width=True)
    st.caption(
        "This overlay compares the TIE–Dialog composite with simpler signals derived from local cosine disruption, "
        "moving-context disruption, geometric displacement, and a shuffled control. Similar trajectories suggest "
        "that part of the composite may be reproducible by a simpler baseline; departures or unique peaks suggest "
        "additional structure worth testing. Visual similarity alone is descriptive, so the correlations and event-"
        "reconstruction measures below are needed to quantify reducibility."
    )

    baseline_names = [
        "Turn-to-turn cosine",
        "Moving-context cosine",
        "Geometric displacement",
    ]

    tie_signal = np.asarray(comparison_signals["TIE–Dialog composite"], float)

    # -------------------------
    # Global reducibility: correlations + ASV_global
    # -------------------------
    corr_rows = []
    corr_values = []

    for name in baseline_names:
        b = np.asarray(comparison_signals[name], float)
        corr = _safe_corr(tie_signal, b)
        corr_values.append(corr)
        corr_rows.append({
            "baseline": name,
            "correlation_with_TIE": corr,
        })

    max_corr = np.nanmax(corr_values) if len(corr_values) else np.nan
    ASV_global = 1.0 - max_corr if np.isfinite(max_corr) else np.nan

    st.markdown("### Global reducibility")
    c1, c2 = st.columns(2)
    c1.metric("Added Structural Value — global", f"{ASV_global:.3f}" if np.isfinite(ASV_global) else "n/a")
    c2.metric("Max baseline correlation", f"{max_corr:.3f}" if np.isfinite(max_corr) else "n/a")

    baseline_compare_df = pd.DataFrame(corr_rows)
    st.dataframe(baseline_compare_df, use_container_width=True)

    st.caption(
        "ASV_global = 1 - max correlation between the TIE composite and the strongest simple baseline. "
        "Higher values suggest that the composite is less reducible to any single baseline signal."
    )

    # -------------------------
    # Event-level reconstruction: overlap + ASV_event
    # -------------------------
    st.markdown("### Event-level reconstruction — strong/composite events")

    tie_event_thr = 0.20
    baseline_event_percentile = 0.80
    event_tolerance = 2

    tie_events, tie_strengths = extract_event_centers(tie_signal, tie_event_thr, gap=1)
    
    shuffled_signal = np.asarray(comparison_signals["Random shuffled baseline"], float)

    shuffled_events, shuffled_strengths = extract_event_centers(
        shuffled_signal,
        tie_event_thr  
    )

    event_rows = []
    for name, signal in comparison_signals.items():
        x = np.asarray(signal, float)
        x_valid = x[np.isfinite(x)]

        if x_valid.size == 0:
            thr = np.nan
            threshold_type = "none"
            threshold_percentile = np.nan
            events, strengths = [], []
        else:
            if name == "TIE–Dialog composite":
                thr = tie_event_thr
                threshold_type = "absolute"
                threshold_percentile = np.nan
            else:
                thr = float(np.quantile(x_valid, baseline_event_percentile))
                threshold_type = "percentile"
                threshold_percentile = baseline_event_percentile

            events, strengths = extract_event_centers(x, thr, gap=1)

        if name == "TIE–Dialog composite":
            overlap = np.nan
            matched_tie_events = np.nan
            unique_tie_events = np.nan
        else:
            overlap = event_overlap_rate(tie_events, events, tolerance=event_tolerance)
            matched_tie_events = sum(
                1 for e in tie_events
                if any(abs(int(e) - int(b)) <= event_tolerance for b in events)
            )
            unique_tie_events = len(tie_events) - int(matched_tie_events)

        event_rows.append({
            "signal": name,
            "threshold_type": threshold_type,
            "threshold_value": thr,
            "threshold_percentile": threshold_percentile,
            "events_detected": len(events),
            "match_with_TIE_events_±2": overlap,
            "matched_TIE_events": matched_tie_events,
            "unique_TIE_events_not_matched": unique_tie_events,
            "mean_event_strength": float(np.nanmean(strengths)) if strengths else np.nan,
            "event_centers_0_indexed": events,
            "event_turns_1_indexed": [int(e) + 1 for e in events],
        })

    event_overlap_df = pd.DataFrame(event_rows)
    st.dataframe(event_overlap_df, use_container_width=True)

    baseline_overlaps = event_overlap_df[
        event_overlap_df["signal"].isin(baseline_names)
    ]["match_with_TIE_events_±2"].values

    max_overlap = np.nanmax(baseline_overlaps) if len(baseline_overlaps) else np.nan
    ASV_event = 1.0 - max_overlap if np.isfinite(max_overlap) else np.nan

    c3, c4 = st.columns(2)
    c3.metric("Added Structural Value — event level", f"{ASV_event:.3f}" if np.isfinite(ASV_event) else "n/a")
    c4.metric("Max event overlap", f"{max_overlap:.3f}" if np.isfinite(max_overlap) else "n/a")

    st.caption(
        "Event overlap measures how many TIE events are reconstructed by each baseline within ±2 turns. "
        "ASV_event = 1 - max baseline overlap. Higher values indicate more TIE events that are not reconstructed "
        "by simpler local/contextual/geometric baselines."
    )


    # -------------------------
    # Shuffled event-localization diagnostic
    # -------------------------
    st.markdown("### Shuffled event-localization diagnostic")

    real_strength = float(np.nanmean(tie_strengths)) if tie_strengths else np.nan

    shuffled_x = np.asarray(random_shuffled_baseline, float)
    shuffled_valid = shuffled_x[np.isfinite(shuffled_x)]

    shuffled_thr = tie_event_thr if shuffled_valid.size else np.nan

    shuffled_events, shuffled_strengths = extract_event_centers(
        shuffled_x,
        shuffled_thr,
        gap=1,
    )

    shuffled_strength = float(np.nanmean(shuffled_strengths)) if shuffled_strengths else np.nan

    shuffle_location_overlap = event_overlap_rate(
        tie_events,
        shuffled_events,
        tolerance=event_tolerance,
    )

    shuffle_displacement = mean_event_displacement(
        tie_events,
        shuffled_events,
    )

    unique_real_events_vs_shuffle = (
        len(tie_events) - sum(
            1 for e in tie_events
            if any(abs(int(e) - int(b)) <= event_tolerance for b in shuffled_events)
        )
    ) if tie_events else np.nan

    shuffled_rows = [{
        "comparison": "Real TIE events vs shuffled TIE events",
        "real_events": len(tie_events),
        "shuffled_events": len(shuffled_events),
        "event_location_overlap_±2": shuffle_location_overlap,
        "mean_event_displacement_turns": shuffle_displacement,
        "unique_real_events_not_matched_by_shuffle": unique_real_events_vs_shuffle,
        "real_mean_event_strength": real_strength,
        "shuffled_mean_event_strength": shuffled_strength,
        "difference_real_minus_shuffled": (
            real_strength - shuffled_strength
            if np.isfinite(real_strength) and np.isfinite(shuffled_strength)
            else np.nan
        ),
        "real_event_turns_1_indexed": [int(e) + 1 for e in tie_events],
        "shuffled_event_turns_1_indexed": [int(e) + 1 for e in shuffled_events],
    }]

    shuffled_diagnostic_df = pd.DataFrame(shuffled_rows)
    st.dataframe(shuffled_diagnostic_df, use_container_width=True)

    c5, c6 = st.columns(2)
    c5.metric(
        "Shuffled event-location overlap",
        f"{shuffle_location_overlap:.3f}" if np.isfinite(shuffle_location_overlap) else "n/a",
    )
    c6.metric(
        "Mean event displacement",
        f"{shuffle_displacement:.2f} turns" if np.isfinite(shuffle_displacement) else "n/a",
    )

    st.caption(
        "This diagnostic compares where events occur in the real dialogue versus the shuffled-order control. "
        "The shuffled signal may preserve similar overall event strength, but if event-location overlap is low "
        "and mean displacement is high, the temporal placement of coherence events has been disrupted."
    )

    st.info(
        "Interpretation: if the TIE–Dialog composite shows moderate or low reducibility to simple baselines, "
        "and detects stable events not reconstructed by them, this suggests added trajectory-dependent structure "
        "beyond local similarity alone."
    )
    
# =========================
# Ablation diagnostic
# =========================
if show_ablation_comparison:
    st.subheader("Ablation Diagnostic: Component Contribution")

    st.caption(
        "This diagnostic tests whether the full TIE–Dialog composite depends on the interaction "
        "between semantic, structural, and geometric components. Each ablation removes one component "
        "and checks whether the same event structure can still be reconstructed."
    )

    zero_D = np.zeros_like(np.asarray(D_t, float))

    ablation_full = np.asarray(event_scores["strong_score"], float)

    ablation_no_cinv = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=None,
        D_t=signals["D_t"],
    )["strong_score"]

    ablation_no_dt = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=signals["C_inv"],
        D_t=zero_D,
    )["strong_score"]

    ablation_ct_only = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=None,
        D_t=zero_D,
    )["strong_score"]

    ablation_signals = {
        "Full TIE composite": np.asarray(ablation_full, float),
        "No C_inv": np.asarray(ablation_no_cinv, float),
        "No D_t": np.asarray(ablation_no_dt, float),
        "Ct only": np.asarray(ablation_ct_only, float),
    }

    fig_ablation = plot_baseline_comparison(
        ablation_signals,
        title="Ablation diagnostic — component contribution",
    )

    st.plotly_chart(fig_ablation, use_container_width=True)
    st.caption(
        "The full composite is compared with versions in which structural information (C_inv), geometric information "
        "(Dₜ), or both are removed. If an ablated curve closely follows the full signal, the removed component adds "
        "relatively little to that part of the trajectory; larger divergences or disappearing peaks indicate a stronger "
        "contribution. Because these variants are derived from the same computational architecture, this is an internal "
        "component-contribution diagnostic rather than independent external validation."
    )

    ablation_event_thr = 0.20
    ablation_event_tolerance = 2

    full_events, full_strengths = extract_event_centers(
        ablation_signals["Full TIE composite"],
        ablation_event_thr,
        gap=1,
    )

    ablation_rows = []

    for name, sig in ablation_signals.items():
        events, strengths = extract_event_centers(
            sig,
            ablation_event_thr,
            gap=1,
        )

        if name == "Full TIE composite":
            overlap = np.nan
            matched_full_events = np.nan
            unique_full_events = np.nan
            corr_with_full = np.nan
            displacement = np.nan
        else:
            overlap = event_overlap_rate(
                full_events,
                events,
                tolerance=ablation_event_tolerance,
            )

            matched_full_events = sum(
                1 for e in full_events
                if any(abs(int(e) - int(b)) <= ablation_event_tolerance for b in events)
            )

            unique_full_events = len(full_events) - int(matched_full_events)

            corr_with_full = _safe_corr(
                ablation_signals["Full TIE composite"],
                sig,
            )
            
            displacement = mean_event_displacement(
                full_events,
                events,
            )

        ablation_rows.append({
            "signal": name,
            "correlation_with_full": corr_with_full,
            "events_detected": len(events),
            "match_with_full_events_±2": overlap,
            "matched_full_events": matched_full_events,
            "unique_full_events_not_reconstructed": unique_full_events,
            "mean_event_displacement_turns": displacement,
            "mean_event_strength": float(np.nanmean(strengths)) if strengths else np.nan,
            "event_turns_1_indexed": [int(e) + 1 for e in events],
        })

    ablation_df = pd.DataFrame(ablation_rows)

    st.markdown("### Ablation reconstruction — strong/structural composite events")
    st.dataframe(ablation_df, use_container_width=True)

    ablation_overlaps = ablation_df[
        ablation_df["signal"] != "Full TIE composite"
    ]["match_with_full_events_±2"].values

    max_ablation_overlap = np.nanmax(ablation_overlaps) if len(ablation_overlaps) else np.nan
    ASV_ablation = 1.0 - max_ablation_overlap if np.isfinite(max_ablation_overlap) else np.nan

    c_ab1, c_ab2 = st.columns(2)
    c_ab1.metric(
        "Added Value — ablation",
        f"{ASV_ablation:.3f}" if np.isfinite(ASV_ablation) else "n/a",
    )
    c_ab2.metric(
        "Max ablation overlap",
        f"{max_ablation_overlap:.3f}" if np.isfinite(max_ablation_overlap) else "n/a",
    )
    
# -------------------------
# Semantic drift ablation
# -------------------------
    st.markdown("### Ablation reconstruction — semantic drift events")

    semantic_event_thr = 0.20
    semantic_event_tolerance = 2

    full_semantic_signal = np.asarray(event_scores["semantic_score"], float)

    no_cinv_semantic_signal = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=None,
        D_t=signals["D_t"],
    )["semantic_score"]

    no_dt_semantic_signal = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=signals["C_inv"],
        D_t=zero_D,
    )["semantic_score"]

    ct_only_semantic_signal = compute_event_scores(
        Ct=signals["Ct"],
        C_inv=None,
        D_t=zero_D,
    )["semantic_score"]

    semantic_ablation_signals = {
        "Full semantic signal": np.asarray(full_semantic_signal, float),
        "No C_inv": np.asarray(no_cinv_semantic_signal, float),
        "No D_t": np.asarray(no_dt_semantic_signal, float),
        "Ct only": np.asarray(ct_only_semantic_signal, float),
    }

    full_semantic_events, full_semantic_strengths = extract_event_centers(
        semantic_ablation_signals["Full semantic signal"],
        semantic_event_thr,
        gap=1,
    )

    semantic_rows = []

    for name, sig in semantic_ablation_signals.items():

        events, strengths = extract_event_centers(
            sig,
            semantic_event_thr,
            gap=1,
        )

        displacement = np.nan

        if name == "Full semantic signal":
            overlap = np.nan
            matched_full_events = np.nan
            unique_full_events = np.nan
            corr_with_full = np.nan

        else:
            overlap = event_overlap_rate(
                full_semantic_events,
                events,
                tolerance=semantic_event_tolerance,
            )

            matched_full_events = sum(
                1 for e in full_semantic_events
                if any(abs(int(e) - int(b)) <= semantic_event_tolerance for b in events)
            )

            unique_full_events = (
                len(full_semantic_events) - int(matched_full_events)
            )

            corr_with_full = _safe_corr(
                semantic_ablation_signals["Full semantic signal"],
                sig,
            )

            displacement = mean_event_displacement(
                full_semantic_events,
                events,
            )

        semantic_rows.append({
            "signal": name,
            "correlation_with_full_semantic": corr_with_full,
            "events_detected": len(events),
            "match_with_full_semantic_events_±2": overlap,
            "matched_full_semantic_events": matched_full_events,
            "unique_full_semantic_events_not_reconstructed": unique_full_events,
            "mean_event_displacement_turns": displacement,
            "mean_event_strength": float(np.nanmean(strengths)) if strengths else np.nan,
            "event_turns_1_indexed": [int(e) + 1 for e in events],
        })

    semantic_ablation_df = pd.DataFrame(semantic_rows)

    st.dataframe(semantic_ablation_df, use_container_width=True)

    st.caption(
        "This table evaluates semantic drift events separately from strong composite ruptures. "
        "It measures whether Ct-based semantic disruptions are still reconstructed after removing "
        "structural (C_inv) or geometric (D_t) information. "
        "If semantic events disappear after removing Ct-related structure, this suggests that "
        "semantic drift depends specifically on contextual coherence dynamics rather than on structural reconfiguration alone."
    )

    st.caption(
        "Ablation overlap measures whether the events detected by the full semantic signal are still recovered "
        "after removing individual components. Low overlap or unique unreconstructed semantic events suggest that "
        "semantic drift depends on the interaction between contextual, structural, and geometric dynamics "
        "rather than on a single component alone."
    )
    
def make_full_diagnostics_pdf(
    embedding_summary_df=None,
    embedding_pairs_df=None,
    embedding_alignment_df=None,
    embedding_shuffle_df=None,
    embedding_variance_df=None,

    baseline_compare_df=None,
    event_overlap_df=None,
    shuffled_diagnostic_df=None,

    ablation_df=None,
    semantic_ablation_df=None,
    robustness_summary_df=None,
    robustness_movement_df=None,
    robustness_runs_df=None,

    setup=None,
):
    buffer = BytesIO()

    def add_df_page(pdf, df, title, caption=None, max_rows=18, max_cols=6):
        if df is None or len(df) == 0:
            return

        df_show = df.copy()

    # Format floats
        for col in df_show.columns:
            if pd.api.types.is_float_dtype(df_show[col]):
                df_show[col] = df_show[col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else ""
                )

    # Convert all cells to string and truncate long values
        df_show = df_show.astype(str)

        for col in df_show.columns:
            df_show[col] = df_show[col].map(
                lambda x: x[:34] + "…" if len(x) > 34 else x
            )

    # Split by rows and columns
        row_chunks = [
            df_show.iloc[i:i + max_rows]
            for i in range(0, len(df_show), max_rows)
        ]

        col_chunks = [
            list(df_show.columns[i:i + max_cols])
            for i in range(0, len(df_show.columns), max_cols)
        ]

        total_pages = max(1, len(row_chunks) * len(col_chunks))
        page_n = 1

        for row_chunk in row_chunks:
            for col_chunk in col_chunks:
                chunk = row_chunk.loc[:, col_chunk]

                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")

                page_title = title
                if total_pages > 1:
                    page_title = f"{title} — page {page_n}/{total_pages}"

                ax.set_title(page_title, fontsize=13, pad=12)

                table = ax.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc="center",
                    cellLoc="center",
                    colLoc="center",
                )

                table.auto_set_font_size(False)
                table.set_fontsize(6.2)
                table.scale(1.0, 1.45)

                for _, cell in table.get_celld().items():
                    cell.set_linewidth(0.25)
                    cell.set_text_props(wrap=True)

                if caption:
                    fig.text(
                        0.05,
                        0.025,
                        caption,
                        ha="left",
                        va="bottom",
                        fontsize=7.2,
                        wrap=True,
                    )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                page_n += 1
            
    with PdfPages(buffer) as pdf:

        # =========================
        # COVER PAGE
        # =========================

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        cover_text = (
            "TIE–Dialog Robustness & Diagnostics Report\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            "This report summarizes embedding robustness, baseline comparisons, "
            "ablation diagnostics, and parameter robustness analyses for the current run."
        )

        if setup:
            cover_text += (
                f"\n\nEmbedding mode: {setup.get('embedding_mode')}"
                f"\nRobustness runs: {setup.get('n_runs')}"
                f"\nPerturbation: ±{setup.get('pct', 0) * 100:.1f}%"
                f"\nSeed: {setup.get('seed')}"
            )

        ax.text(
            0.08,
            0.92,
            cover_text,
            va="top",
            ha="left",
            fontsize=12,
            wrap=True,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================
        # EMBEDDINGS
        # =========================

        add_df_page(
            pdf,
            embedding_summary_df,
            "Embedding summary",
            "Summary statistics across embedding trajectories."
        )

        add_df_page(
            pdf,
            embedding_pairs_df,
            "Embedding pairwise comparison",
            "Pairwise robustness comparison between embedding trajectories."
        )

        add_df_page(
            pdf,
            embedding_alignment_df,
            "Embedding event alignment",
            "Overlap and alignment of event regions across embeddings."
        )

        add_df_page(
            pdf,
            embedding_shuffle_df,
            "Embedding shuffled diagnostics",
            "Comparison against shuffled-order controls."
        )

        add_df_page(
            pdf,
            embedding_variance_df,
            "Embedding variance decomposition",
            "Variance decomposition across embeddings and dialogue structure."
        )

        # =========================
        # BASELINES
        # =========================

        add_df_page(
            pdf,
            baseline_compare_df,
            "Baseline comparison",
            "Comparison between TIE–Dialog and simpler baseline signals."
        )

        add_df_page(
            pdf,
            event_overlap_df,
            "Baseline event overlap",
            "Overlap between TIE–Dialog event regions and baseline event regions."
        )
        
        add_df_page(
            pdf,
            shuffled_diagnostic_df,
            "Shuffled event-localization diagnostic",
            "Comparison between real TIE event locations and shuffled-order event locations."
        )

        # =========================
        # ABLATION
        # =========================

        add_df_page(
            pdf,
            ablation_df,
            "Ablation diagnostics",
            "Effect of removing semantic, structural, or geometric layers."
        )
        
        add_df_page(
            pdf,
            semantic_ablation_df,
            "Semantic drift ablation",
            "Effect of removing structural or geometric information from semantic drift event reconstruction."
        )

        # =========================
        # ROBUSTNESS
        # =========================

        add_df_page(
            pdf,
            robustness_summary_df,
            "Parameter robustness summary",
            "Summary of output stability under parameter perturbation."
        )

        add_df_page(
            pdf,
            robustness_movement_df,
            "Parameter movement summary",
            "Amount of perturbation applied to each parameter during robustness testing."
        )

        add_df_page(
            pdf,
            robustness_runs_df,
            "Per-run robustness results",
            "Detailed results for each perturbed parameter configuration."
        )

    buffer.seek(0)
    return buffer.getvalue()

# =========================
# Parameter robustness
# =========================
robustness_runs_df = None
robustness_summary_df = None
robustness_movement_df = None

if run_param_robustness:
    base_params = {
        "smooth_method": smooth_method,
        "env_alpha": float(env_alpha),
        "env_span": int(env_span),
        "use_cinv": bool(use_cinv),
        "cinv_window": int(cinv_window),
        "cinv_knn": int(cinv_knn),
        "cinv_thr": float(cinv_thr),
        "cinv_keigs": int(cinv_keigs),
        "alpha_context": 0.84,
        "b": 0.40,
    }

    base_event_masks = {
        "strong": np.asarray(strong_mask, dtype=bool),
        "semantic": np.asarray(semantic_mask, dtype=bool),
        "structural": np.asarray(structural_mask, dtype=bool),
    }

    robustness_runs_df, robustness_summary_df, robustness_movement_df = run_parameter_robustness(
        E=E,
        texts=texts,
        participants=participants,
        base_params=base_params,
        base_ct=np.asarray(Ct_base, float),
        base_event_masks=base_event_masks,
        n_runs=int(robustness_n_runs),
        pct=float(robustness_pct),
        seed=int(robustness_seed),
    )

    st.subheader("Robustness test — fixed embeddings")

    st.markdown("### Summary")
    st.dataframe(robustness_summary_df, use_container_width=True)
    
    st.caption(
        "Summary of the robustness test under parameter perturbation. "
        "Each value summarizes how stable the TIE–Dialog outputs remain when key parameters "
        "such as smoothing, context inertia, and graph-invariant settings are randomly varied. "
        "Higher Ct correlation, DTW similarity and event-mask Jaccard scores indicate greater robustness."
    )

    st.markdown("### Parameter movement summary")
    st.dataframe(robustness_movement_df, use_container_width=True)

    st.caption(
        "This table shows how much each parameter was perturbed during the robustness test. "
        "The baseline column reports the original value used in the main run. "
        "mean_perturbed, min_perturbed, and max_perturbed summarize the sampled parameter range across robustness runs. "
        "mean_abs_delta reports the average absolute movement from baseline, while mean_pct_delta reports the average relative movement. "
        "This makes explicit which parameters were modified and how strongly the robustness test stressed the pipeline."
    )
    
    st.markdown("### Per-run results")
    st.dataframe(robustness_runs_df, use_container_width=True)
    
    st.caption(
        "Per-run robustness results. Each row corresponds to one perturbed parameter configuration. "
        "Ct_corr and Ct_dtw_similarity compare the coherence trajectory against the original run. "
        "The Jaccard scores measure how much the detected strong, semantic, and structural event masks overlap "
        "with the original event masks. The remaining columns show the exact parameter values used in each run."
    )
    
    full_diag_pdf = make_full_diagnostics_pdf(
        embedding_summary_df=embedding_summary_df,
        embedding_pairs_df=embedding_pairs_df,
        embedding_alignment_df=embedding_alignment_df,
        embedding_shuffle_df=embedding_shuffle_df,
        embedding_variance_df=embedding_variance_df,

        baseline_compare_df=baseline_compare_df,
        event_overlap_df=event_overlap_df,
        shuffled_diagnostic_df=shuffled_diagnostic_df,

        ablation_df=ablation_df,
        semantic_ablation_df=semantic_ablation_df,

        robustness_summary_df=robustness_summary_df,
        robustness_movement_df=robustness_movement_df,
        robustness_runs_df=robustness_runs_df,

        setup={
            "embedding_mode": used_mode,
            "n_runs": robustness_n_runs,
            "pct": robustness_pct,
            "seed": robustness_seed,
        },
    )

    if full_diag_pdf is not None:
        st.download_button(
            label="Download robustness & diagnostics report (PDF)",
            data=full_diag_pdf,
            file_name="tie_dialog_robustness_diagnostics_report.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("Robustness diagnostics PDF could not be generated.")
    
ci_df = compute_ci_series(
    E=E,
    participants=participants,
    method=ci_method,
    alpha=float(ci_alpha),
)

if ci_df is not None and ci_df.shape[1] > 0:
    for col in ci_df.columns:
        df_out[col] = ci_df[col].to_numpy()

# =========================
# FULL DIALOGUE TABLE
# =========================
st.subheader(L["table_title"])

ci_cols = [c for c in df_out.columns if str(c).startswith("Ci_")]
cols_order = [
    "turn", "participant", "text",
    "Ct", "Ct_im", "C_inv", *ci_cols,
    "S_t", "R_t", "d_t", "kappa_t", "u_t",
    "dS_t", "dR_t", "dd_t", "dkappa_t", "du_t",
    "J_t", "J_share_S", "J_share_R", "J_share_d", "J_share_kappa", "J_share_u",
    "Q_t", "rho_t", "D_t_exploratory",
]
cols_show = [c for c in cols_order if c in df_out.columns]
st.dataframe(df_out[cols_show], use_container_width=True)
st.caption(
    "Primary turn-level state: z_t=(S_t,R_t,d_t,κ_t,u_t). S_t=1-C_t is Contextual Discontinuity; "
    "R_t=1-C_inv is Structural Instability / low persistence; d_t is angular displacement; κ_t is "
    "turning-angle curvature; and u_t=1-ρ_t is local semantic dispersion. D_t_exploratory is retained "
    "only for backwards-compatible diagnostics and is not used in the primary multivariate analysis."
)

# =========================================================
# Research Tutorial — Primary Multicomponent Dynamics
# =========================================================
if show_crossdim:
    with st.expander("Research Tutorial — Primary Multicomponent Dynamics", expanded=True):
        st.caption(
            "Primary analysis keeps five continuous observables separate: z_t=(S_t,R_t,d_t,κ_t,u_t). "
            "This avoids repeating the Transition Pressure mistake of assuming that distinct signals should be "
            "collapsed into one scalar before their relationships are empirically established. d_t, κ_t and u_t "
            "are temporally causal; S_t and R_t remain offline within-dialogue descriptors because C_t and C_inv "
            "use dialogue-level scaling. Human consensus should remain an external validation target."
        )

        primary_series = {
            "S": np.asarray(df_out["S_t"], float),
            "R": np.asarray(df_out["R_t"], float),
            "d": np.asarray(df_out["d_t"], float),
            "κ": np.asarray(df_out["kappa_t"], float),
            "u": np.asarray(df_out["u_t"], float),
        }
        finite_R = int(np.isfinite(primary_series["R"]).sum())
        if finite_R < 4:
            st.warning(
                "R_t has too few valid observations for a full five-dimensional analysis. "
                "Enable C_inv and use a dialogue long enough to fill the structural window."
            )

        tab_dyn, tab_dep, tab_lag = st.tabs([
            "State & Change", "Dynamic Coupling", "Lead–Lag"
        ])

        with tab_dyn:
            x_cd = np.arange(1, len(df_out) + 1)

            fig_state = go.Figure()
            state_labels = {
                "S": "S_t — Contextual Discontinuity (1-C_t)",
                "R": "R_t — Structural Instability (1-C_inv)",
                "d": "d_t — Angular displacement",
                "κ": "κ_t — Turning-angle curvature",
                "u": "u_t — Local semantic dispersion (1-ρ_t)",
            }
            for key, y in primary_series.items():
                fig_state.add_trace(go.Scatter(x=x_cd, y=y, mode="lines", name=state_labels[key]))
            fig_state.update_layout(
                title="Primary five-dimensional computational state",
                xaxis_title="Turn", yaxis_title="State value (0–1)",
                yaxis=dict(range=[0, 1]), height=470,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_state, use_container_width=True)
            st.caption(
                "The five curves are primary observables, not components of a pre-weighted transition score. "
                "S and R describe contextual/structural state levels; d, κ and u describe distinct aspects of the "
                "embedding trajectory. Their value lies partly in possible dissociations: a human transition may be "
                "associated with only one or a particular combination rather than with all dimensions rising together."
            )

            fig_delta = go.Figure()
            delta_specs = [
                ("dS_t", "ΔS_t"), ("dR_t", "ΔR_t"), ("dd_t", "Δd_t"),
                ("dkappa_t", "Δκ_t"), ("du_t", "Δu_t"),
            ]
            for col, label in delta_specs:
                fig_delta.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(df_out[col], float), mode="lines", name=label
                ))
            fig_delta.update_layout(
                title="Turn-to-turn changes in the five primary dimensions",
                xaxis_title="Turn", yaxis_title="Change", height=470,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption(
                "First differences show directional movement rather than absolute level. These component-wise curves "
                "are more important than any composite magnitude for testing whether human-perceived transitions have "
                "a recurrent multivariate temporal signature."
            )

            fig_J = go.Figure()
            fig_J.add_trace(go.Scatter(
                x=x_cd, y=np.asarray(df_out["J_t"], float), mode="lines",
                name="J_t — 5D movement magnitude"
            ))
            fig_J.update_layout(
                title="Secondary five-dimensional movement magnitude",
                xaxis_title="Turn", yaxis_title="J_t (0–1)",
                yaxis=dict(range=[0, 1]), height=360,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_J, use_container_width=True)
            st.caption(
                "J_t = ||Δz_t||/√5 summarizes how far the complete five-dimensional state moved between adjacent "
                "turns. It is intentionally secondary: equal numerical scaling does not prove that the five constructs "
                "have equal reliability or should be treated as one latent variable. Use J_t as a movement descriptor, "
                "not as a transition detector."
            )

            fig_J_comp = go.Figure()
            for col, label in [
                ("J_share_S", "S contribution"), ("J_share_R", "R contribution"),
                ("J_share_d", "d contribution"), ("J_share_kappa", "κ contribution"),
                ("J_share_u", "u contribution"),
            ]:
                fig_J_comp.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(df_out[col], float), mode="lines",
                    stackgroup="movement", name=label
                ))
            fig_J_comp.update_layout(
                title="Composition of five-dimensional movement",
                xaxis_title="Turn", yaxis_title="Share of squared change",
                yaxis=dict(range=[0, 1]), height=390,
                margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_J_comp, use_container_width=True)
            st.caption(
                "These shares identify which coordinate produced each non-zero 5D movement. They prevent a high J_t "
                "from hiding whether the movement was primarily contextual, structural, displacement-based, curvature-based, "
                "or dispersion-based. Shares are undefined when total movement is exactly zero."
            )

        with tab_dep:
            x_cd = np.arange(1, len(df_out) + 1)
            rolling_change_corr = crossdim["rolling_change_corr"]
            pair_pretty = {
                "S|R": "r(ΔS,ΔR)", "S|d": "r(ΔS,Δd)", "S|kappa": "r(ΔS,Δκ)",
                "S|u": "r(ΔS,Δu)", "R|d": "r(ΔR,Δd)", "R|kappa": "r(ΔR,Δκ)",
                "R|u": "r(ΔR,Δu)", "d|kappa": "r(Δd,Δκ)", "d|u": "r(Δd,Δu)",
                "kappa|u": "r(Δκ,Δu)",
            }
            pair_options = list(rolling_change_corr.keys())
            default_pairs = [k for k in ["S|R", "S|d", "S|kappa", "R|d", "R|kappa", "d|kappa"] if k in pair_options]
            selected_pairs = st.multiselect(
                "Pairs shown in coupling plot",
                options=pair_options,
                default=default_pairs,
                format_func=lambda k: pair_pretty.get(k, k),
                key="coupling_pairs_5d",
            )
            fig_dep_delta = go.Figure()
            for key in selected_pairs:
                fig_dep_delta.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(rolling_change_corr[key], float), mode="lines",
                    name=pair_pretty.get(key, key)
                ))
            fig_dep_delta.update_layout(
                title=f"Trailing coupling among changes — window={int(crossdim_window)} turns",
                xaxis_title="Turn", yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]),
                height=460, margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_dep_delta, use_container_width=True)
            st.caption(
                f"Pairwise trailing correlations over {int(crossdim_window)} turns describe local coordination among "
                "first differences. Positive values indicate same-direction movement, negative values opposing movement, "
                "and values near zero little linear coupling. The plot is selectable because ten pairwise relationships "
                "exist in a five-dimensional state."
            )

            show_q_exploratory = st.checkbox(
                "Show exploratory Q_t dependency-structure diagnostic",
                value=False, key="show_q_t_exploratory",
            )
            if show_q_exploratory:
                fig_Q = go.Figure()
                fig_Q.add_trace(go.Scatter(
                    x=x_cd, y=np.asarray(df_out["Q_t"], float), mode="lines",
                    name="Q_t — 5D dependency-structure change"
                ))
                fig_Q.update_layout(
                    title="Exploratory change in the 5D rolling dependency structure",
                    xaxis_title="Turn", yaxis_title="Q_t (0–1)", yaxis=dict(range=[0, 1]),
                    height=360, margin=dict(l=40, r=40, t=45, b=40),
                )
                st.plotly_chart(fig_Q, use_container_width=True)
                st.caption(
                    "Q_t measures how much the complete 5×5 rolling correlation structure changes from one turn to the "
                    "next. It is a second-order exploratory descriptor, not a primary transition metric."
                )

        with tab_lag:
            st.caption(
                "Lead–lag is evaluated on first differences. The table corrects the retrospective search over candidate "
                "lags with a max-over-lags circular-shift null, then applies Holm adjustment across the ten pairwise tests. "
                "These remain descriptive temporal associations, not causal estimates."
            )
            lag_tables = crossdim["lag_changes"]
            lag_null_summary = crossdim["lag_null_summary"].copy()
            lag_options = list(lag_tables.keys())
            default_lags = lag_options[:6]
            selected_lags = st.multiselect(
                "Pairs shown in lead–lag plot",
                options=lag_options,
                default=default_lags,
                key="leadlag_pairs_5d",
            )
            fig_lag = go.Figure()
            for pair_name in selected_lags:
                lag_df = lag_tables[pair_name]
                fig_lag.add_trace(go.Scatter(
                    x=lag_df["lag"], y=lag_df["corr"], mode="lines+markers", name=pair_name
                ))
            fig_lag.update_layout(
                title="Lead–lag association among first differences",
                xaxis_title="Lag (turns)", yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]),
                height=460, margin=dict(l=40, r=40, t=45, b=40),
            )
            st.plotly_chart(fig_lag, use_container_width=True)
            st.caption(
                "Positive lag means the first named change precedes the second; negative lag means the reverse. The raw "
                "best lag should never be interpreted alone because the analysis searches several candidate lags."
            )

            st.markdown("#### Strongest lag with search and family-wise correction")
            lag_cols = [
                "pair", "best_lag", "corr", "abs_corr", "n_pairs",
                "null_mean_max_abs_r", "null_q95_max_abs_r",
                "p_max_over_lags", "p_holm", "null_n",
            ]
            lag_show = lag_null_summary[[c for c in lag_cols if c in lag_null_summary.columns]].copy()
            st.dataframe(lag_show, use_container_width=True, hide_index=True)
            st.caption(
                "p_max_over_lags controls the search across lags within each pair. p_holm additionally controls family-wise "
                "error across the ten tested dimension pairs. Different coordinates also have different intrinsic temporal "
                "response properties, so any proposed sequence should still be checked with synthetic latency calibration."
            )


st.session_state["last_main_fig"] = fig_main
st.session_state["last_main_fig"] = fig_main
html = fig_main.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
st.download_button(
    label="Download main plot (HTML)",
    data=html,
    file_name="tie_dialog_main_plot.html",
    mime="text/html",
)

with st.expander("Exploratory / implementation diagnostics", expanded=False):
    st.caption(
        "These plots document implementation-level signals and optional legacy summaries. The primary "
        "Research Tutorial state is S/R/d/κ/u; composite D_t is exploratory only."
    )

    if show_envelope:
        fig_s = go.Figure()
        x_env = np.arange(1, len(Ct_base) + 1)

        fig_s.add_trace(go.Scatter(
            x=x_env,
            y=np.asarray(Ct_base, float),
            mode="lines",
            name="Cₜ (raw)",
            line=dict(width=2),
        ))

        fig_s.add_trace(go.Scatter(
            x=x_env,
            y=np.asarray(Ct_smooth, float),
            mode="lines",
            name="Cₜ_smooth",
            line=dict(width=4),
        ))

        if C_inv_plot is not None:
            fig_s.add_trace(go.Scatter(
                x=x_env,
                y=np.clip(np.asarray(C_inv_plot, float), 0.0, 1.0),
                mode="lines",
                name="C_inv_smooth",
                line=dict(width=2, dash="dot"),
            ))

        fig_s.update_layout(
            title="Diagnostic: smoothed coherence",
            height=420,
            margin=dict(l=40, r=200, t=30, b=40),
            xaxis_title="Turn",
            yaxis_title="Coherence (0–1)",
            yaxis=dict(range=[0, 1]),
            legend=dict(
                orientation="v",
                x=1.02,
                xanchor="left",
                y=1.0,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption(
            "This diagnostic contrasts the raw contextual-continuity trajectory with its smoothed visualization and, "
            "when available, the smoothed structural-continuity trajectory C_inv. Smoothing makes broader trends easier "
            "to inspect but can attenuate or slightly shift short-lived local fluctuations; C_inv also remains undefined "
            "until the rolling graph window can be estimated. The figure is therefore useful for inspecting signal shape "
            "and implementation behaviour, not for defining transition onsets."
        )

    fig_g = plot_ic3_geometry(
        d_i=d_i,
        kappa_i=kappa_i,
        u_t=u_t,
        height=560,
    )
    st.plotly_chart(fig_g, use_container_width=True)
    st.caption(
        "These are the exact three geometric coordinates used separately in the primary state: d_t is causal angular "
        "displacement, κ_t is causal turning-angle curvature, and u_t=1-ρ_t is causal local semantic dispersion over a "
        f"trailing {GEOM_COMPACTNESS_WINDOW + 1}-turn neighbourhood. No dialogue-wise calibration or composite weighting "
        "is applied to these primary coordinates. The legacy D_t summary is intentionally omitted from this figure."
    )


def _df_to_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")

ic2_df = pd.DataFrame({
    "turn": turns,
    "res": np.asarray(ic2["res"], float),
    "dI_norm": np.asarray(ic2["dI_norm"], float),
    "Ct": np.asarray(ic2["C_t"], float),
})

ic3_df = pd.DataFrame({
    "turn": turns,
    "d_t": np.asarray(d_i, float),
    "kappa_t": np.asarray(kappa_i, float),
    "rho_t": np.asarray(rho_t, float),
    "u_t": np.asarray(u_t, float),
    "D_t_exploratory": np.asarray(D_t, float),
})

# =========================
# PDF Report (English) — Download
# =========================
st.subheader("Report (PDF)")

report_params = {
    "mode": mode,
    "emb_mode": emb_mode,
    "sbert_model": sbert_model,
    "smooth_method": smooth_method,
    "env_alpha": float(env_alpha),
    "env_span": int(env_span),
    "use_cinv": bool(use_cinv),
    "cinv_window": int(cinv_window),
    "cinv_knn": int(cinv_knn),
    "cinv_thr": float(cinv_thr),
    "cinv_keigs": int(cinv_keigs),
    "ci_method": ci_method,
    "ci_alpha": float(ci_alpha),
    "crossdim_window": int(crossdim_window),
    "crossdim_max_lag": int(crossdim_max_lag),
    "primary_state": "S/R/d/kappa/u",
    "geometry_composite_primary": False,
    "geom_compactness_window": int(GEOM_COMPACTNESS_WINDOW),
}

pdf_bytes = build_pdf_report_bytes(
    df_in=df,
    df_out=df_out,
    ic2_df=ic2_df,
    ic3_df=ic3_df,
    Ct_base=Ct_base,
    Ct_smooth=Ct_smooth,
    C_inv=(C_inv if (use_cinv and C_inv is not None) else None),
    used_mode=used_mode,
    emb_msg=emb_msg,
    params=report_params,
)

st.download_button(
    label="Download report (PDF)",
    data=pdf_bytes,
    file_name="tie_dialog_report.pdf",
    mime="application/pdf",
)
st.caption("This PDF includes summary, parameters, key plots, and event tables (English).")

st.subheader("Downloads")
cA, cC, cD = st.columns(3)

with cA:
    st.download_button(
        L["download_full_csv"],
        data=_df_to_csv_bytes(df_out),
        file_name="tie_dialog_full_results.csv",
        mime="text/csv"
    )

with cC:
    st.download_button(
        L["download_ic2_csv"],
        data=_df_to_csv_bytes(ic2_df),
        file_name="tie_dialog_ic2_dynamics.csv",
        mime="text/csv"
    )

with cD:
    st.download_button(
        L["download_ic3_csv"],
        data=_df_to_csv_bytes(ic3_df),
        file_name="tie_dialog_ic3_geometry.csv",
        mime="text/csv"
    )
