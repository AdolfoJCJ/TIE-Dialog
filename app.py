# ============================
# app.py — PART 1/3
# (imports + labels + helpers + Public View + IC-II/IC-III core)
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
        "what_does": "What does this app do?",
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
        "public_header": "Public View (Smoothed S–B–R)",
        "public_span": "Smoothing span (EWMA)",
        "public_show_thresholds": "Show Φ thresholds in public plot",
        "public_title": "Public plot: Smoothed coherence + S–B–R regimes",
        "ci_header": "Participant trajectories (Ci)",
        "ci_alpha": "Ci context inertia α (per-participant)",
        "ci_method": "Ci method",
        "ci_title": "Per-participant coherence trajectories (Ci) + Φ thresholds",
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
        "table_title": "Diálogo con etiquetas S/B/R, potencialidad ℘ₜ y geometría (IC–III)",
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
WARMUP_TURNS = 5

def apply_warmup_ramp(Ct: np.ndarray, warm: int = WARMUP_TURNS, floor: float = 0.10) -> np.ndarray:
    Ct = np.asarray(Ct, float).copy()
    n = len(Ct)
    if n == 0:
        return Ct
    warm = int(max(0, warm))
    if warm <= 0:
        return Ct
    if n <= warm:
        Ct[:] = np.linspace(floor, Ct[-1], n)
        return np.clip(Ct, 0.0, 1.0)
    Ct[:warm] = np.linspace(floor, Ct[warm], warm)
    return np.clip(Ct, 0.0, 1.0)


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
    IMPORTANT: It's only start producing values when t >= window-1 (full window).
    Before that: NaN (so it can't pollute scaling / event logic).
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

# =====================
# Public View smoothing + SBR regimes
# =====================
def smooth_public_ewma(y: np.ndarray, span: int = 9) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    span = int(max(3, span))
    return pd.Series(y).ewm(span=span, adjust=False).mean().to_numpy()

def sbr_labels_public(C: np.ndarray, theta_S: float, theta_B: float, warmup_turns: int = WARMUP_TURNS) -> np.ndarray:
    C = np.asarray(C, float)
    n = C.size
    lab = np.array(["S"] * n, dtype=object)
    in_repair = False
    for t in range(n):
        if t < int(warmup_turns):
            lab[t] = "W"
            continue
        if C[t] <= float(theta_B):
            lab[t] = "B"
            in_repair = True
        elif C[t] >= float(theta_S):
            lab[t] = "S"
            in_repair = False
        else:
            lab[t] = "R" if in_repair else "S"
    return lab

def _segments_from_labels(labels: np.ndarray) -> List[Tuple[str, int, int]]:
    segs = []
    if labels.size == 0:
        return segs
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            segs.append((str(labels[i-1]), start, i-1))
            start = i
    segs.append((str(labels[-1]), start, len(labels)-1))
    return segs

def add_regime_bands(
    fig: go.Figure,
    turns: np.ndarray,
    labels: np.ndarray,
    *,
    strip: bool = True,
    strip_y0: float = 0.00,
    strip_y1: float = 0.12,
    fullheight_breaks: bool = True,
):
    """
    Paint vertical regime bands (S/B/R/W) using Plotly SHAPES (robust with yaxis2).
    - strip=True paints a thin strip at the bottom (yref='paper')
    - fullheight_breaks=True softly shades full height only for B-regions
    """

    turns = np.asarray(turns, int)
    labels = np.asarray(labels, object)

    # colors (soft)
    band = {
        "S": "rgba(0,200,0,0.14)",
        "B": "rgba(220,0,0,0.18)",
        "R": "rgba(0,120,255,0.14)",
        "W": "rgba(150,150,150,0.10)",
    }

    segs = _segments_from_labels(labels)  # returns (label, start_idx, end_idx)

    for lab, a, b in segs:
        x0 = int(turns[a])

        # cover full “turn width”
        if b + 1 < len(turns):
            x1 = int(turns[b + 1])
        else:
            x1 = int(turns[b]) + 1

        color = band.get(str(lab), band["R"])

        # 1) thin strip at bottom
        if strip:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=float(strip_y0),
                y1=float(strip_y1),
                fillcolor=color,
                line=dict(width=0),
                layer="below",
            )

        # 2) optional full-height shading ONLY for breaks
        if fullheight_breaks and str(lab) == "B":
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0.0,
                y1=1.0,
                fillcolor="rgba(220,0,0,0.07)",
                line=dict(width=0),
                layer="below",
            )

    return fig

def plot_public_sbr(
    turns: np.ndarray,
    C_smooth: np.ndarray,
    labels: np.ndarray,
    theta_S: float,
    theta_B: float,
    show_thresholds: bool = False,
    height: int = 420,
    title: str = "Public View: Smoothed Coherence + S–B–R",
) -> go.Figure:
    turns = np.asarray(turns, int)
    C_smooth = np.asarray(C_smooth, float)
    labels = np.asarray(labels, object)

    fig = go.Figure()
    band = {
        "S": dict(fillcolor="rgba(0,200,0,0.10)", line_width=0),
        "B": dict(fillcolor="rgba(200,0,0,0.12)", line_width=0),
        "R": dict(fillcolor="rgba(0,120,255,0.10)", line_width=0),
        "W": dict(fillcolor="rgba(150,150,150,0.06)", line_width=0),
    }
    for lab, a, b in _segments_from_labels(labels):
        fig.add_vrect(x0=int(turns[a]), x1=int(turns[b]) + 1, **band.get(lab, band["R"]))

    fig.add_trace(go.Scatter(
        x=turns, y=C_smooth, mode="lines", name="Cₜ (smoothed)",
        line=dict(width=3),
    ))

    if show_thresholds:
        fig.add_hline(y=float(theta_B), line_dash="dash", opacity=0.35, annotation_text="Φ_low")
        fig.add_hline(y=float(theta_S), line_dash="dash", opacity=0.35, annotation_text="Φ_high")

    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=35, b=40),
        xaxis_title="Turn",
        yaxis_title="Coherence (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        )
    )
    return fig

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

def compute_ic2_dynamics(
    E: np.ndarray,
    alpha_context: float = 0.84,
    beta: float = 0.70,
    b: float = 0.40,
    eps: float = 1e-12,
) -> dict:
    """
    Canonical IC-II:
    I_m(t) = α I_m(t-1) + (1-α) E_{t-1}
    r_t = cos(E_t, I_m(t))
    ΔI_t = ||E_t - E_{t-1}||
    C_t = σ( β r_t - (1-β) ΔI_t - b )
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
    eps: float = 1e-12
) -> dict:
    """
    IC-III (intrinsic geometry): computed from the embedding trajectory.

    Uses cosine-based local displacement instead of raw Euclidean step length.
    This makes d_i more interpretable for dialogue, because it reflects
    semantic deviation between consecutive turns without inflating normal variation.

    Definitions:
    - d_i(t): cosine-based local displacement in [0,1]
    - kappa_i(t): local change in displacement (discrete curvature proxy)
    """
    E = np.asarray(E, float)
    n = E.shape[0]
    if n == 0:
        return {
            "d_i": np.zeros(0, float),
            "kappa_i": np.zeros(0, float),
            "tau_t": np.zeros(0, float),
        }

    # unit-normalize embeddings
    En = E.copy()
    norms = np.linalg.norm(En, axis=1, keepdims=True) + eps
    En = En / norms

    # cosine-based local displacement
    d_i = np.zeros(n, float)
    for t in range(1, n):
        cos_sim = float(np.dot(En[t], En[t - 1]))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        # cosine distance rescaled to [0,1]
        d_i[t] = 0.5 * (1.0 - cos_sim)

    # mild smoothing to reduce embedding jitter
    d_i = _ema(d_i, alpha=0.35)

    # robust dialogue-level renormalization
    valid = d_i[np.isfinite(d_i)]
    if valid.size > 3:
        lo = float(np.nanpercentile(valid, 10))
        hi = float(np.nanpercentile(valid, 90))
        if hi - lo > 1e-9:
            d_i = (d_i - lo) / (hi - lo)
            d_i = np.clip(d_i, 0.0, 1.0)
        else:
            d_i = np.zeros_like(d_i)

    # curvature proxy = local change in displacement
    dd = np.zeros(n, float)
    dd[1:] = d_i[1:] - d_i[:-1]
    kappa_i = np.abs(dd)

    return {
        "d_i": np.clip(d_i, 0.0, 1.0),
        "kappa_i": np.clip(kappa_i, 0.0, 1.0),
    }
    
def compute_all_signals(E, texts, Ct_base, C_inv, ic3):
    rho_t = semantic_compactness_rho(E, texts, w=2, mode="centroid", min_tokens=3)
    di_n = _norm01(ic3["d_i"])
    kappa_n = _norm01(ic3["kappa_i"])

    D_t = manifold_driver_D(
        di=di_n,
        kappa=kappa_n,
        rho=rho_t,
        w_d=0.45,
        w_k=0.35,
        w_r=0.20,
        gating=True,
    )

    return {
        "Ct": np.asarray(Ct_base, float),
        "C_inv": np.asarray(C_inv, float) if C_inv is not None else None,
        "rho_t": np.asarray(rho_t, float),
        "d_i": np.asarray(ic3["d_i"], float),
        "kappa_i": np.asarray(ic3["kappa_i"], float),
        "D_t": np.asarray(D_t, float),
    }
    
# -------------------------------------------------
# IC-II helper — Semantic Compactness ρ_t
# -------------------------------------------------
def semantic_compactness_rho(
    E: np.ndarray,
    texts: list,
    w: int = 2,
    mode: str = "centroid",
    min_tokens: int = 3,
) -> np.ndarray:
    """
    Computes local semantic compactness ρ_t.
    Measures how tightly clustered embeddings are in a rolling window.
    Returns a signal in [0,1].
    """
    E = np.asarray(E, float)
    n = int(E.shape[0]) if E.ndim == 2 else 0
    if n == 0:
        return np.zeros((0,), float)

    # light token filter (optional)
    tok_ok = np.ones(n, dtype=bool)
    if texts is not None and min_tokens is not None and int(min_tokens) > 0:
        for i, t in enumerate(texts[:n]):
            s = t if isinstance(t, str) else ""
            tok_ok[i] = (len(s.strip().split()) >= int(min_tokens))

    rho = np.zeros(n, float)

    for t in range(n):
        start = max(0, t - int(w))
        end = min(n, t + int(w) + 1)

        idx = np.arange(start, end)
        idx = idx[tok_ok[idx]]
        if idx.size < 2:
            rho[t] = 1.0
            continue

        window_vecs = E[idx]

        if str(mode).lower().strip() == "pairwise":
            dists = []
            for i in range(len(window_vecs)):
                for j in range(i + 1, len(window_vecs)):
                    dists.append(float(np.linalg.norm(window_vecs[i] - window_vecs[j])))
            mean_d = float(np.mean(dists)) if dists else 0.0
            rho[t] = 1.0 / (1.0 + mean_d)
        else:
            centroid = window_vecs.mean(axis=0)
            dists = np.linalg.norm(window_vecs - centroid, axis=1)
            rho[t] = 1.0 / (1.0 + float(dists.mean()))

    # normalize safely to [0,1]
    mx = float(np.max(rho)) if rho.size else 1.0
    if mx > 1e-12:
        rho = rho / mx
    return np.clip(rho, 0.0, 1.0)


# -------------------------------------------------
# IC-III → IC-II driver D_t
# -------------------------------------------------
def manifold_driver_D(
    *,
    di: np.ndarray,
    kappa: np.ndarray,
    rho: np.ndarray,
    w_d: float = 0.45,
    w_k: float = 0.35,
    w_r: float = 0.20,
    gating: bool = True,
) -> np.ndarray:
    """
    Combines IC-III channels into a driver D_t in [0,1].
    Intuition: high displacement + high curvature + low compactness => high driver (reconfiguration pressure).
    """
    di = np.asarray(di, float)
    kappa = np.asarray(kappa, float)
    rho = np.asarray(rho, float)
    n = min(di.size, kappa.size, rho.size)
    if n == 0:
        return np.zeros((0,), float)

    di = np.clip(di[:n], 0.0, 1.0)
    kappa = np.clip(kappa[:n], 0.0, 1.0)
    rho = np.clip(rho[:n], 0.0, 1.0)

    inv_rho = 1.0 - rho

    D = float(w_d) * di + float(w_k) * kappa + float(w_r) * inv_rho

    if gating:
        # if compactness is high, damp the driver slightly
        gate = np.clip(0.40 + 0.60 * inv_rho, 0.0, 1.0)
        D = D * gate

    return np.clip(D, 0.0, 1.0)


# -------------------------------------------------
# Break detectors on driver / proxy channels
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
    """
    Match indices in a to nearest indices in b within ±delta_max.
    Returns list of dicts with lag = b-a.
    """
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

# -------------------------------
# IC-III geometry plot
# -------------------------------
def plot_ic3_geometry(
    Ct: np.ndarray,
    d_i: np.ndarray,
    kappa_i: np.ndarray,
    phi_low: float,
    phi_high: float,
    height: int = 600,
) -> go.Figure:
    Ct = np.asarray(Ct, float)
    d_i = np.asarray(d_i, float)
    kappa_i = np.asarray(kappa_i, float)
    n = len(Ct)
    x = np.arange(1, n + 1)

    def _norm_series(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        if y.size == 0:
            return y

        lo = float(np.nanpercentile(y, 5))
        hi = float(np.nanpercentile(y, 95))

        if hi - lo < 1e-9:
            return np.zeros_like(y)

        z = (y - lo) / (hi - lo)
        return np.clip(z, 0.0, 1.0)

    d_norm = _norm_series(d_i)
    kappa_norm = _norm_series(kappa_i)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=Ct, mode="lines", name="Cₜ (coherence)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=d_norm, mode="lines", name="dᵢ (normalized)", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=kappa_norm, mode="lines", name="κᵢ (normalized)", line=dict(width=2, dash="dot")))

    fig.add_hline(y=float(phi_low), line_dash="dash", opacity=0.40, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.40, annotation_text="Φ_high")

    fig.update_layout(
        title="IC–III geometric layer: dᵢ and κᵢ over the coherence manifold",
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Normalized value (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        )
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


# -------------------------------
# Potentiality metric ℘ₜ
# -------------------------------
def compute_potentiality(texts: List[str]) -> np.ndarray:
    interrogatives = {"what", "why", "how", "where", "when", "which", "who"}
    modals = {"maybe", "might", "could", "would", "can", "possibly", "perhaps", "seems", "seem", "appear", "appears"}
    conditionals = {"if", "unless"}
    openness_phrases = ["i wonder", "what if", "let's think", "let us think", "could we", "might we"]

    scores = []
    for raw in texts:
        t = raw if isinstance(raw, str) else ""
        lower = t.lower()
        tokens = lower.split()
        s = 0.0
        if "?" in t:
            s += 0.35
        if any(tok in interrogatives for tok in tokens):
            s += 0.20
        if any(tok in conditionals for tok in tokens):
            s += 0.15
        if any(tok in modals for tok in tokens):
            s += 0.15
        if any(phrase in lower for phrase in openness_phrases):
            s += 0.20
        s = min(1.0, s)
        scores.append(s)

    arr = np.asarray(scores, float)
    arr = _ema(arr, alpha=0.4)
    return np.clip(arr, 0.0, 1.0)

# -------------------------------
# Continuous state trajectories
# -------------------------------
def compute_participant_state_trajectories(
    Ct: np.ndarray,
    participants: List[str],
    alpha: float = 0.88,
) -> Dict[str, np.ndarray]:
    Ct = np.asarray(Ct, float)
    n = len(Ct)
    uniq = list(dict.fromkeys([str(p) for p in participants]))

    C_parts = {p: np.zeros(n, float) for p in uniq}
    if n == 0:
        return C_parts

    for p in uniq:
        C_parts[p][0] = float(Ct[0])

    a = float(np.clip(alpha, 0.0, 0.999))

    for t in range(1, n):
        spk = str(participants[t])
        for p in uniq:
            if spk == p:
                C_parts[p][t] = float(Ct[t])
            else:
                C_parts[p][t] = a * float(C_parts[p][t - 1]) + (1.0 - a) * float(Ct[t])

    return C_parts

def plot_participant_state_lines(
    Ct: np.ndarray,
    participants: List[str],
    phi_low: float,
    phi_high: float,
    alpha_state: float = 0.88,
    title: str = "State trajectories",
    height: int = 520,
) -> go.Figure:
    turns = np.arange(1, len(Ct) + 1)
    states = compute_participant_state_trajectories(Ct=Ct, participants=participants, alpha=float(alpha_state))

    fig = go.Figure()
    for p, y in states.items():
        fig.add_trace(go.Scatter(x=turns, y=y, mode="lines", name=str(p)))

    fig.add_hline(y=float(phi_low), line_dash="dot")
    fig.add_hline(y=float(phi_high), line_dash="dot")

    fig.update_layout(
        title=title,
        height=int(height),
        xaxis_title="Turn",
        yaxis_title="State coherence",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def plot_ci_lines(
    turns: np.ndarray,
    ci_df: pd.DataFrame,
    phi_low: float,
    phi_high: float,
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

    # Thresholds
    fig.add_hline(y=float(phi_low), line_dash="dash", opacity=0.35, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.35, annotation_text="Φ_high")

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

def extract_transition_zones(
    pressure,
    threshold=None,
    min_len=2,
):
    pressure = np.asarray(pressure, float)

    if threshold is None:
        threshold = np.quantile(pressure, 0.55)

    mask = pressure >= float(threshold)

    segments = mask_to_segments(mask)

    zones = []

    for s, e in segments:

        if (e - s + 1) < int(min_len):
            continue

        peak_idx = s + np.argmax(pressure[s:e+1])

        zones.append({
            "start": int(s),
            "end": int(e),
            "peak": int(peak_idx),
            "mean_pressure": float(np.mean(pressure[s:e+1])),
            "max_pressure": float(np.max(pressure[s:e+1])),
            "duration": int(e - s + 1),
        })

    return zones
    
    pressure = np.asarray(pressure, float)

    mask = pressure >= float(threshold)

    segments = mask_to_segments(mask)

    zones = []

    for s, e in segments:

        if (e - s + 1) < int(min_len):
            continue

        peak_idx = s + np.argmax(pressure[s:e+1])

        zones.append({
            "start": int(s),
            "end": int(e),
            "peak": int(peak_idx),
            "mean_pressure": float(np.mean(pressure[s:e+1])),
            "max_pressure": float(np.max(pressure[s:e+1])),
            "duration": int(e - s + 1),
        })

    return zones


def build_transition_zone_dataframe(zones):
    rows = []

    for i, z in enumerate(zones):
        rows.append({
            "zone_id": i + 1,
            "start_turn": int(z["start"]) + 1,
            "end_turn": int(z["end"]) + 1,
            "peak_turn": int(z["peak"]) + 1,
            "duration": int(z["duration"]),
            "mean_pressure": float(z["mean_pressure"]),
            "max_pressure": float(z["max_pressure"]),
        })

    return pd.DataFrame(rows)

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
def compute_event_scores(Ct, C_inv, D_t):
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

    strong_score = (
        0.4 * sem_drop +
        0.35 * struct_drop +
        0.25 * D_t
    )
    strong_score = np.clip(strong_score, 0.0, 1.0)

    semantic_score = 0.55 * sem_drop + 0.20 * (1.0 - struct_drop) + 0.25 * D_t
    structural_score = 0.70 * struct_drop + 0.15 * (1.0 - sem_drop) + 0.15 * D_t

    transition_pressure = (
        0.50 * sem_drop +
        0.35 * struct_drop +
        0.40 * np.asarray(D_t, float)
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
    q_low: float,
    q_high: float,
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
        q_low=float(q_low),
        q_high=float(q_high),
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
    q_low: float,
    q_high: float,
    smooth_method: str,
    env_alpha: float,
    env_span: int,
    use_cinv: bool,
    cinv_window: int,
    cinv_knn: int,
    cinv_thr: float,
    cinv_keigs: int,
    alpha_context: float = 0.84,
    beta: float = 0.70,
    b: float = 0.40,
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
        beta=float(beta),
        b=float(b),
    )

    Ct_raw_local = np.asarray(ic2_local["C_t"], float)
    Ct_raw_local = np.clip(Ct_raw_local, 0.0, 1.0)

    Ct_base_local = _normalize_ct(Ct_raw_local, lower=0.06, upper=0.94)
    Ct_base_local = apply_warmup_ramp(Ct_base_local, warm=WARMUP_TURNS, floor=0.10)
    Ct_base_local = np.clip(Ct_base_local, 0.0, 1.0)

    Ct_smooth_local = smooth_coherence(
        Ct_base_local,
        method=smooth_method,
        ema_alpha=float(env_alpha),
        ewma_span=int(env_span),
    )
    Ct_smooth_local = np.clip(Ct_smooth_local, 0.0, 1.0)

    # Φ
    if len(Ct_base_local):
        idx = np.arange(len(Ct_base_local))
        mask_valid = idx >= int(WARMUP_TURNS)
        Ct_for_phi_local = Ct_base_local[mask_valid] if np.any(mask_valid) else Ct_base_local
        phi_low_local = float(np.quantile(Ct_for_phi_local, float(q_low)))
        phi_high_local = float(np.quantile(Ct_for_phi_local, float(q_high)))
    else:
        phi_low_local, phi_high_local = 0.55, 0.75

    phi_low_local = float(np.clip(phi_low_local, 0.0, 0.95))
    phi_high_local = float(np.clip(phi_high_local, 0.05, 1.0))
    if phi_high_local <= phi_low_local + 0.08:
        phi_high_local = float(min(1.0, phi_low_local + 0.12))

    # simple SBR
    sbr_local = []
    for i, ct in enumerate(Ct_base_local):
        if i < WARMUP_TURNS:
            sbr_local.append("W")
        elif ct < phi_low_local:
            sbr_local.append("B")
        else:
            sbr_local.append("S")

    sbr_local = protect_conversation_ending(
        sbr_local,
        Ct_level=Ct_base_local,
        Ct_drop=Ct_smooth_local,
        n_end_protect=6,
        min_drop=0.25,
        stable_threshold=0.35,
    )

    # IC-III + event scores
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
    )

    event_labels_local = classify_event_scores(event_scores_local,
    D_t=signals_local["D_t"])
    event_masks_local = labels_to_event_masks(
        event_labels_local,
        min_sem_len=2,
        min_struct_len=2,
        min_strong_len=1,
    )

    if "transition" not in event_masks_local:
        transition_pressure_local = np.asarray(
            event_scores_local.get("transition_pressure", np.zeros(len(Ct_base_local))),
            dtype=float,
        )

        transition_mask_local = (
            transition_pressure_local >=
            np.quantile(transition_pressure_local, 0.80)
        )

        event_masks_local["transition"] = enforce_min_persistence(
            transition_mask_local,
            min_len=2,
        )

    return {
        "Ct": Ct_base_local,
        "Ct_smooth": Ct_smooth_local,
        "C_inv": C_inv_local,
        "phi_low": phi_low_local,
        "phi_high": phi_high_local,
        "sbr": sbr_local,
        "rho_t": signals_local["rho_t"],
        "D_t": signals_local["D_t"],
        "event_scores": event_scores_local,
        "event_labels": event_labels_local,
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
            elif k in {"q_low", "q_high", "env_alpha", "cinv_thr"}:
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


def compare_embedding_runs(runs: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        sbr = np.asarray(run.get("sbr", []), dtype=object)

        summary_rows.append({
            "embedding": name,
            "mean_Ct": float(np.nanmean(Ct)) if Ct.size else np.nan,
            "std_Ct": float(np.nanstd(Ct)) if Ct.size else np.nan,
            "min_Ct": float(np.nanmin(Ct)) if Ct.size else np.nan,
            "max_Ct": float(np.nanmax(Ct)) if Ct.size else np.nan,
            "mean_strong_score": float(np.nanmean(strong_score)) if strong_score.size else np.nan,
            "std_strong_score": float(np.nanstd(strong_score)) if strong_score.size else np.nan,
            "strong_events_n": int(np.sum(strong_mask)) if strong_mask.size else np.nan,
            "broken_turns_n": int(np.sum(sbr == "B")) if sbr.size else np.nan,
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
    base_sbr: Sequence[str],
    base_event_masks: Dict[str, np.ndarray],
    n_runs: int = 30,
    pct: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    rows = []

    for r in range(int(n_runs)):
        p = {}

        p["alpha_context"] = _perturb_value(base_params["alpha_context"], pct, lo=0.70, hi=0.95, rng=rng)
        p["beta"] = _perturb_value(base_params["beta"], pct, lo=0.40, hi=1.20, rng=rng)
        p["b"] = _perturb_value(base_params["b"], pct, lo=0.20, hi=0.70, rng=rng)
        
        p["q_low"] = _perturb_value(base_params["q_low"], pct, lo=0.05, hi=0.50, rng=rng)
        p["q_high"] = _perturb_value(base_params["q_high"], pct, lo=0.50, hi=0.95, rng=rng)
        if p["q_high"] <= p["q_low"] + 0.08:
            p["q_high"] = min(0.95, p["q_low"] + 0.12)

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
            q_low=float(p["q_low"]),
            q_high=float(p["q_high"]),
            smooth_method=str(base_params["smooth_method"]),
            env_alpha=float(p["env_alpha"]),
            env_span=int(p["env_span"]),
            use_cinv=bool(base_params["use_cinv"]),
            cinv_window=int(p["cinv_window"]),
            cinv_knn=int(p["cinv_knn"]),
            cinv_thr=float(p["cinv_thr"]),
            cinv_keigs=int(p["cinv_keigs"]),
            alpha_context=float(p["alpha_context"]),
            beta=float(p["beta"]),
            b=float(p["b"]),
        )

        ct = np.asarray(run["Ct"], float)
        sbr = run["sbr"]
        masks = run["event_masks"]

        rows.append({
            "run": r + 1,
            "Ct_corr": _safe_corr(base_ct, ct),
            "Ct_dtw_similarity": dtw_similarity(base_ct, ct),
            "SBR_agreement": label_agreement(base_sbr, sbr),
            "strong_jaccard": jaccard_similarity(base_event_masks["strong"], masks["strong"]),
            "semantic_jaccard": jaccard_similarity(base_event_masks["semantic"], masks["semantic"]),
            "structural_jaccard": jaccard_similarity(base_event_masks["structural"], masks["structural"]),
            "q_low": p["q_low"],
            "q_high": p["q_high"],
            "env_alpha": p["env_alpha"],
            "env_span": p["env_span"],
            "cinv_window": p["cinv_window"],
            "cinv_knn": p["cinv_knn"],
            "cinv_thr": p["cinv_thr"],
            "cinv_keigs": p["cinv_keigs"],
            "alpha_context": p["alpha_context"],
            "beta": p["beta"],
            "b": p["b"],
        })

    df_runs = pd.DataFrame(rows)

    df_summary = pd.DataFrame([{
        "Ct_corr_mean": float(df_runs["Ct_corr"].mean()),
        "Ct_corr_std": float(df_runs["Ct_corr"].std()),
        "Ct_dtw_similarity_mean": float(df_runs["Ct_dtw_similarity"].mean()),
        "SBR_agreement_mean": float(df_runs["SBR_agreement"].mean()),
        "strong_jaccard_mean": float(df_runs["strong_jaccard"].mean()),
        "semantic_jaccard_mean": float(df_runs["semantic_jaccard"].mean()),
        "structural_jaccard_mean": float(df_runs["structural_jaccard"].mean()),
    }])

    return df_runs, df_summary
        
def classify_event_scores(
    event_scores,
    D_t=None,
    strong_thr=0.68,
    sem_thr=0.50,
    struct_thr=0.40,
    d_thr=0.32,
    sem_margin=0.10,
    struct_margin=0.10,
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
            labels.append("RUPTURE_STRONG")

        # 2) semantic rupture when semantic clearly dominates structural
        elif sem[i] >= sem_thr and sem[i] >= struct[i] + sem_margin:
            labels.append("RUPTURE_SEM")

        # 3) structural rupture when structural clearly dominates semantic
        elif struct[i] >= struct_thr and struct[i] >= sem[i] + struct_margin and D[i] >= d_thr:
            labels.append("RUPTURE_STRUCT")

        else:
            labels.append("STABLE")

    return labels
    
def classify_event_structural(Ct, C_inv, phi_low):

    Ct = np.asarray(Ct, float)
    C_inv = np.asarray(C_inv, float) if C_inv is not None else None

    labels = []

    for t in range(len(Ct)):

        ct_drop = Ct[t] < phi_low

        if C_inv is not None and np.isfinite(C_inv[t]):
            cinv_drop = C_inv[t] < phi_low
        else:
            cinv_drop = False

        # 🔴 STRONG
        if ct_drop and cinv_drop:
            labels.append("RUPTURE_STRONG")

        # 🟡 SEMANTIC
        elif ct_drop and not cinv_drop:
            labels.append("RUPTURE_SEM")

        # 🔵 STRUCTURAL
        elif not ct_drop and cinv_drop:
            labels.append("RUPTURE_STRUCT")

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

    strong = labels == "RUPTURE_STRONG"
    semantic = labels == "RUPTURE_SEM"
    structural = labels == "RUPTURE_STRUCT"

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
    labels[semantic_mask] = "RUPTURE_SEM"
    labels[structural_mask] = "RUPTURE_STRUCT"
    labels[strong_mask] = "RUPTURE_STRONG"

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
# Quanto of Coherence
# -------------------------------
def _central_derivative(Ct: np.ndarray) -> np.ndarray:
    Ct = np.asarray(Ct, float)
    d = np.zeros_like(Ct)
    if Ct.size == 0:
        return d
    if Ct.size == 1:
        d[0] = 0.0
        return d
    d[0] = Ct[1] - Ct[0]
    d[-1] = Ct[-1] - Ct[-2]
    if Ct.size > 2:
        d[1:-1] = 0.5 * (Ct[2:] - Ct[:-2])
    return d


def _find_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            s = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            e = i
            segs.append((s, e))
        i += 1
    return segs


def _extrema_idxs(
    y: np.ndarray,
    start: int,
    end: int,
    eps: float,
    dCt: Optional[np.ndarray] = None,
) -> List[int]:
    if dCt is None:
        dCt = _central_derivative(y)

    idxs: List[int] = []
    # interior points only
    for i in range(max(start + 1, 1), min(end, len(y) - 2) + 1):
        left, mid, right = y[i - 1], y[i], y[i + 1]
        is_peak = (mid > left) and (mid > right)
        is_trough = (mid < left) and (mid < right)
        if (is_peak or is_trough) and (
            abs(dCt[i - 1]) > eps or abs(dCt[i]) > eps or abs(dCt[i + 1]) > eps
        ):
            idxs.append(int(i))
    return idxs


def compute_quanto_of_coherence(
    Ct: np.ndarray,
    phi_low: float,
    phi_high: float,
    eps: float = 1e-4,
) -> Tuple[float, List[Tuple[int, int, float]], dict]:
    Ct = np.asarray(Ct, float)
    dCt = _central_derivative(Ct)

    mask = (Ct > float(phi_low)) & (Ct <= float(phi_high))
    segs = _find_segments(mask)

    segments_info: List[Tuple[int, int, float]] = []
    all_osc: List[Tuple[float, int, int]] = []

    for s, e in segs:
        if not np.any(np.abs(dCt[s : e + 1]) > float(eps)):
            continue
        ex = _extrema_idxs(Ct, s, e, float(eps), dCt)
        if len(ex) < 2:
            continue

        amps: List[Tuple[float, int, int]] = []
        for k in range(len(ex) - 1):
            i, j = ex[k], ex[k + 1]
            if (phi_low < Ct[i] <= phi_high) and (phi_low < Ct[j] <= phi_high):
                amp = abs(float(Ct[j]) - float(Ct[i]))
                if amp > 0:
                    amps.append((amp, int(i), int(j)))

        if not amps:
            continue

        min_amp, ai, bi = min(amps, key=lambda x: x[0])
        segments_info.append((int(s), int(e), float(min_amp)))
        all_osc.extend(amps)

    if not all_osc:
        Qa = float("nan")
        chosen = None
    else:
        Qa, i0, j0 = min(all_osc, key=lambda x: x[0])
        chosen = (int(i0), int(j0))

    dbg = {
        "dCt": dCt,
        "chosen_extrema_pair": chosen,
        "phi_low": float(phi_low),
        "phi_high": float(phi_high),
    }
    return float(Qa), segments_info, dbg


# -------------------------------
# Enforce S-B-R mandatory repair rule
# -------------------------------
def _sbr_fullname(s: str) -> str:
    m = {
        "S": "stable",
        "B": "broken",
        "R": "repair",
        "stable": "stable",
        "broken": "broken",
        "repair": "repair",
    }
    key = str(s).strip()
    if key in {"S", "B", "R"}:
        return m.get(key, "stable")
    return m.get(key.lower(), "stable")


def protect_conversation_ending(
    sbr: Sequence[str],
    Ct_level: Sequence[float],
    Ct_drop: Optional[Sequence[float]] = None,
    n_end_protect: int = 6,
    min_drop: float = 0.25,
    stable_threshold: float = 0.35,
) -> List[str]:
    sbr_out = list(sbr)

    Ct_level = np.asarray(Ct_level, dtype=float)
    Ct_drop = Ct_level if Ct_drop is None else np.asarray(Ct_drop, dtype=float)

    n = min(len(sbr_out), len(Ct_level), len(Ct_drop))
    if n == 0:
        return sbr_out

    start = max(0, n - int(n_end_protect))

    for i in range(start, n):
        if sbr_out[i] != "B":
            continue
        if i == 0:
            sbr_out[i] = "S" if float(Ct_level[i]) > float(stable_threshold) else "R"
            continue
        drop = float(Ct_drop[i - 1]) - float(Ct_drop[i])
        if drop < float(min_drop):
            sbr_out[i] = "S" if float(Ct_level[i]) > float(stable_threshold) else "R"

    return sbr_out



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
        yaxis2=dict(
            title="Potentiality ℘ₜ",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
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


def add_window_bands(
    fig: go.Figure,
    turns: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    strip: bool = True,
    strip_y0: float = 0.00,
    strip_y1: float = 0.12,
    fullheight: bool = False,
    fullheight_color: Optional[str] = None,
) -> go.Figure:
    turns = np.asarray(turns, int)
    mask = np.asarray(mask, bool)

    if turns.size == 0 or mask.size == 0:
        return fig
    if turns.size != mask.size:
        m = min(turns.size, mask.size)
        turns = turns[:m]
        mask = mask[:m]

    segs = mask_to_segments(mask)

    for s, e in segs:
        x0 = int(turns[s])
        x1 = int(turns[e]) + 1  # cover full turn width

        if strip:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=float(strip_y0),
                y1=float(strip_y1),
                fillcolor=str(color),
                line=dict(width=0),
                layer="below",
            )

        if fullheight:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0.0,
                y1=1.0,
                fillcolor=str(fullheight_color or color),
                line=dict(width=0),
                layer="below",
            )

    return fig

def plot_ct_main(
    Ct: np.ndarray,
    participants: List[str],
    phi_low: float,
    phi_high: float,
    title: str,
    height: int,
    pilot_w: int = 2,
    potentiality: Optional[np.ndarray] = None,
    C_inv: Optional[np.ndarray] = None,
    sbr_labels: Optional[Sequence[str]] = None,
    strong_mask: Optional[np.ndarray] = None,
    semantic_mask: Optional[np.ndarray] = None,
    structural_mask: Optional[np.ndarray] = None,
) -> go.Figure:
    Ct = np.asarray(Ct, dtype=float)
    n = int(Ct.size)
    x = np.arange(1, n + 1, dtype=int)

    fig = go.Figure()
    fig = _base_layout(fig, title, height)

    # Main curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.clip(Ct, 0.0, 1.0),
            mode="lines",
            name="Cₜ (global)",
            line=dict(width=3),
            yaxis="y",
        )
    )

    # Optional overlays
    if C_inv is not None:
        C_inv = np.asarray(C_inv, dtype=float)
        if C_inv.size == n:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.clip(C_inv, 0.0, 1.0),
                    mode="lines",
                    name="C_inv (invariants)",
                    line=dict(width=2, dash="dot"),
                    yaxis="y",
                )
            )

    if potentiality is not None:
        potentiality = np.asarray(potentiality, dtype=float)
        if potentiality.size == n:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.clip(potentiality, 0.0, 1.0),
                    mode="lines",
                    name="℘ₜ (potentiality)",
                    line=dict(width=2, dash="dot"),
                    yaxis="y2",
                )
            )

    # Regime bands
    if sbr_labels is not None:
        lab = np.asarray(list(sbr_labels), dtype=object)
        if lab.size == n:
            fig = add_regime_bands(
                fig,
                turns=x,
                labels=lab,
                strip=False,
                strip_y0=0.00,
                strip_y1=0.12,
                fullheight_breaks=True,
            )

    # Typed event windows
    if structural_mask is not None:
        structural_mask = np.asarray(structural_mask, dtype=bool)
        if structural_mask.size == n:
            fig = add_window_bands(
                fig,
                turns=x,
                mask=structural_mask,
                color="rgba(70,130,180,0.22)",
                strip=False,
                fullheight=True,
                fullheight_color="rgba(70,130,180,0.08)",
            )

    if semantic_mask is not None:
        semantic_mask = np.asarray(semantic_mask, dtype=bool)
        if semantic_mask.size == n:
            fig = add_window_bands(
                fig,
                turns=x,
                mask=semantic_mask,
                color="rgba(255,215,0,0.30)",
                strip=False,
                fullheight=True,
                fullheight_color="rgba(255,215,0,0.12)",
            )

    if strong_mask is not None:
        strong_mask = np.asarray(strong_mask, dtype=bool)
        if strong_mask.size == n:
            fig = add_window_bands(
                fig,
                turns=x,
                mask=strong_mask,
                color="rgba(220,20,60,0.22)",
                strip=False,
                fullheight=True,
                fullheight_color="rgba(220,20,60,0.08)",
            )

    # Participant markers
    parts = list(dict.fromkeys([str(p) for p in participants]))
    for name in parts:
        y = np.full(n, np.nan, float)
        idx = [i for i, p in enumerate(participants) if str(p) == name]
        if idx:
            y[idx] = Ct[idx]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"Participant: {name}",
                marker=dict(size=7),
                yaxis="y",
            )
        )

    # Thresholds
    fig.add_hline(y=float(phi_low), line_dash="dash", opacity=0.45, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.45, annotation_text="Φ_high")

    # Dummy legend traces
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(220,20,60,0.6)"),
        name="Strong rupture",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(255,215,0,0.6)"),
        name="Semantic drift",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(70,130,180,0.6)"),
        name="Structural reorganization",
        showlegend=True,
    ))

    fig = _base_layout(fig, title, height=height)
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
    phi_low: float,
    phi_high: float,
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
        f"Φ_low: {phi_low:.3f}",
        f"Φ_high: {phi_high:.3f}",
        f"Mean coherence (Ct): {_safe_mean(Ct_base):.3f}",
        f"Mean smoothed coherence: {_safe_mean(Ct_smooth):.3f}",
        "",
        "Interpretation (high level):",
        "• Ct captures alignment with the evolving conversational context.",
        "• Event windows are derived from the multi-signal scoring layer",
        "• C_inv (if enabled) tracks structural reconfiguration via rolling graph invariants.",
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

    cols = ["turn", "participant", "Ct", "dCt", "sbr", "event_type", "text"]
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

        fig = _fig_text_page("Run Configuration", param_lines, footer="Notes: Φ thresholds are percentiles unless overridden.")
        pdf.savefig(fig); plt.close(fig)

        # Main coherence page
        main_series = [("Ct (raw)", Ct_base), ("Ct_smooth", Ct_smooth)]
        hls = [("Φ_low", float(phi_low)), ("Φ_high", float(phi_high))]
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
                "Invariant Coherence (C_inv) — Structural Reconfiguration Channel",
                turns,
                [("C_inv", np.asarray(C_inv, float))],
                hlines=None,
                vmarks=None,
                ylim=(0.0, 1.0),
            )
            pdf.savefig(fig); plt.close(fig)

        # IC-III page (uses ic3_df which you already create)
        if ic3_df is not None and len(ic3_df) == len(turns):
            fig = _fig_timeseries(
                "IC–III / Driver Layer (normalized channels)",
                turns,
                [
                    ("rho_t", np.asarray(ic3_df["rho_t"], float)),
                    ("D_t", np.asarray(ic3_df["D_t"], float)),
                    
                ],
                hlines=[("Φ_low", float(phi_low)), ("Φ_high", float(phi_high))],
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
This application models dialogue as a **dynamic informational process** rather than a static exchange of utterances.

It computes a turn-by-turn coherence signal (**Ct**), where coherence is defined as the **persistence of identity through continuous transformation**. Instead of measuring simple alignment, Ct captures how each contribution maintains or disrupts the evolving trajectory of the conversation.

In parallel, the app computes a **structural coherence** channel (**C_inv**), based on the stability of a rolling similarity graph. While Ct reflects **contextual continuity**, C_inv captures **structural reconfiguration** by tracking changes in the topology of the dialogue over time.

This dual perspective allows the system to distinguish between fundamentally different types of disruption:

– **Semantic drift**: Ct decreases while C_inv remains stable  
– **Structural reframe**: Ct remains stable while C_inv decreases  
– **Strong rupture**: both Ct and C_inv decrease  

The system segments dialogue into three regimes:

– **Stable (S)**: continuity of the informational trajectory  
– **Broken (B)**: rupture in semantic and/or structural coherence  
– **Repair (R)**: recovery or re-alignment of the trajectory  

Additionally, a geometric layer (**IC–III**) models the dialogue as a trajectory in semantic space:

– **d_i**: local displacement  
– **κ_i**: curvature (directional change)  

The goal is not only to measure coherence, but to make the **dynamics of conversational structure** observable, interpretable, and explorable.
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

**Φ_low percentile / Φ_high percentile**  
Adaptive thresholds used to separate lower and higher coherence regions.  
Lower Φ values make the system more permissive. Higher Φ values make it stricter.

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

## Events

**Min separation (turns)**  
Minimum distance between detected event candidates.  
Higher values group nearby disruptions. Lower values allow more fragmented detection.

**Min prominence**  
Minimum strength required for a change to count as salient.  
Higher values make event detection stricter.

**Min drop (Ct step)**  
Minimum turn-to-turn decrease in Cₜ required for drop-based detection.  
Higher values focus on abrupt breaks. Lower values allow gradual changes.

**Merge gap**  
Groups nearby micro-breaks into a larger event region.  
Higher values create broader event windows.

**Window half-width w**  
Expands detected event points into local windows around each turn.  
Higher values produce wider event regions.

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
        "Events",
        "Visual",
        "Robust"
    ])

    with data_tab:
        use_demo = st.checkbox(L["load_demo"], value=True)
        uploaded = None if use_demo else st.file_uploader(L["upload"], type=["csv", "xlsx"])

    with core_tab:
        st.subheader(L["sem_repr"])

        emb_mode = ui_selectbox(
            L["emb_mode"],
            ["auto", "sbert", "e5", "bge", "instructor", "tfidf"],
            index=0
        )

    
        st.subheader(L["phi"])
        q_low = ui_slider("Φ_low percentile", 0.05, 0.50, 0.20, 0.01)
        q_high = ui_slider("Φ_high percentile", 0.50, 0.95, 0.80, 0.01)

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
        st.subheader(L["events"])
        sep_min = ui_slider("min separation (turns)", 1, 8, 2, 1)
        prom_min = ui_slider("min prominence", 0.01, 0.30, 0.10, 0.01)
        min_drop = ui_slider("min drop (Ct step)", 0.00, 0.60, 0.25, 0.01)
        merge_gap = ui_slider("merge_gap (micro-break grouping)", 1, 6, 3, 1)
        pilot_w = ui_slider("window half-width w (turns)", 0, 6, 2, 1)

        st.markdown("### Invariant coherence (C_inv)")
        use_cinv = ui_checkbox("Enable C_inv (graph invariants)", value=True)
        cinv_window = ui_slider("C_inv window W", 6, 24, 8, 1)
        cinv_knn = ui_slider("C_inv k-NN", 2, 10, 3, 1)
        cinv_thr = ui_slider("C_inv edge threshold", 0.00, 0.30, 0.16, 0.01)
        cinv_keigs = ui_slider("C_inv eigenfeatures (k)", 3, 12, 6, 1)

    with visual_tab:
        st.subheader(L["ci_header"])
        ci_method = ui_selectbox(L["ci_method"], ["ctx", "im"], index=0)
        ci_alpha = ui_slider(L["ci_alpha"], 0.70, 0.99, 0.90, 0.01)
        state_alpha = ui_slider(L["state_alpha"], 0.60, 0.98, 0.88, 0.01)

        st.subheader(L["public_header"])
        public_span = ui_slider(L["public_span"], 3, 25, 9, 1)
        public_show_thr = ui_checkbox(L["public_show_thresholds"], value=False)

        show_envelope = ui_checkbox("Show smoothed coherence plot", value=True)
        overlay_smoothed_on_main = ui_checkbox("Overlay smoothed curve on main plot", value=False)
        overlay_cinv_on_main = ui_checkbox("Overlay C_inv on main plot", value=True)

        with robust_tab:
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

            st.markdown("### Baseline diagnostic")
            show_baseline_comparison = ui_checkbox(
                "Show baseline diagnostic",
                value=False,
                key="robust_baseline_toggle"
            )

            baseline_moving_window = ui_slider(
                "Moving average window",
                2, 12, 5, 1,
                key="robust_baseline_window"
            )

            st.markdown("### Ablation diagnostic")
            
            show_ablation_comparison = ui_checkbox(
                "Show ablation diagnostic",
                value=False,
                key="robust_ablation_toggle"
            )

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
            q_low=q_low,
            q_high=q_high,
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
    beta=0.70,
    b=0.40,
)

# 1) Raw IC-II coherence (sigmoid output)
Ct_raw = np.asarray(ic2["C_t"], float)
Ct_raw = np.clip(Ct_raw, 0.0, 1.0)

# 2) Canonical coherence used by the app everywhere
Ct_base = np.clip(Ct_raw, 0.0, 1.0)
Ct_base = np.clip(Ct_base, 0.0, 1.0)
Ct_base = apply_warmup_ramp(Ct_base, warm=WARMUP_TURNS, floor=0.10)
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

# -------------------------
# Φ thresholds (exclude warmup)
# -------------------------
if len(Ct_base):
    idx = np.arange(len(Ct_base))
    mask_valid = idx >= int(WARMUP_TURNS)
    Ct_for_phi = Ct_base[mask_valid] if np.any(mask_valid) else Ct_base

    phi_low_eff  = float(np.quantile(Ct_for_phi, float(q_low)))
    phi_high_eff = float(np.quantile(Ct_for_phi, float(q_high)))
else:
    phi_low_eff, phi_high_eff = 0.55, 0.75

phi_low_eff  = float(np.clip(phi_low_eff,  0.0, 0.95))
phi_high_eff = float(np.clip(phi_high_eff, 0.05, 1.0))
if phi_high_eff <= phi_low_eff + 0.08:
    phi_high_eff = float(min(1.0, phi_low_eff + 0.12))

P_t = compute_potentiality(texts)

# --- 4) optional: keep for UI/debug compatibility ---
sbr_corrections = []

# =========================
# 2) Automatic Interpretation Cheatsheet
# =========================

def _nanmedian(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

def _nanquantile(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")

def compute_cheatsheet(Ct: np.ndarray, Cinv: Optional[np.ndarray]):
    Ct = np.asarray(Ct, float)
    dCt = np.zeros_like(Ct)
    dCt[1:] = Ct[1:] - Ct[:-1]

    # "Ct decreases" means a big negative step compared to typical drops
    drop_thr = _nanquantile(dCt, 0.10)  # 10th percentile of step changes (more negative = bigger drop)
    stable_thr = _nanquantile(np.abs(dCt), 0.60)  # typical absolute change

    out = {
        "drop_thr": drop_thr,
        "stable_thr": stable_thr,
        "has_cinv": (Cinv is not None and np.asarray(Cinv).size == Ct.size)
    }

    if out["has_cinv"]:
        Cinv = np.asarray(Cinv, float)
        dCi = np.zeros_like(Cinv)
        dCi[1:] = Cinv[1:] - Cinv[:-1]
        # "C_inv decreases" threshold
        drop_inv_thr = _nanquantile(dCi, 0.10)
        stable_inv_thr = _nanquantile(np.abs(dCi), 0.60)
        out.update({
            "drop_inv_thr": drop_inv_thr,
            "stable_inv_thr": stable_inv_thr
        })

    return out

def interpret_turn(t: int, Ct: np.ndarray, Cinv: Optional[np.ndarray], cs: dict) -> str:
    # t is 0-indexed
    Ct = np.asarray(Ct, float)
    dCt = 0.0 if t <= 0 else float(Ct[t] - Ct[t-1])

    Ct_down = (dCt <= float(cs["drop_thr"]))  # more negative than threshold
    Ct_flat = (abs(dCt) <= float(cs["stable_thr"]))

    if not cs["has_cinv"]:
        # no structural channel
        if Ct_down:
            return "Possible SEMANTIC rupture (Ct drop)"
        return "Stable / drift"

    Cinv = np.asarray(Cinv, float)
    dCi = 0.0 if t <= 0 else float(Cinv[t] - Cinv[t-1])
    Ci_down = (dCi <= float(cs["drop_inv_thr"]))
    Ci_flat = (abs(dCi) <= float(cs["stable_inv_thr"]))

    if Ct_down and Ci_down:
        return "RUPTURE_STRONG (Ct decreases and C_inv decreases)"
    if Ct_down and (Ci_flat or not Ci_down):
        return "SEMANTIC_DRIFT (Ct decreases, C_inv~)"
    if Ct_flat and Ci_down:
        return "STRUCTURAL_REORGANIZATION (Ct~, C_inv decreases)"
    return "STABLE / smooth evolution"


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
D_t = signals["D_t"]
kappa_n = _norm01(signals["kappa_i"])

event_scores = compute_event_scores(
    Ct=signals["Ct"],
    C_inv=signals["C_inv"],
    D_t=signals["D_t"]
)

transition_zones = extract_transition_zones(
    event_scores["transition_pressure"],
    threshold=0.18,
    min_len=2,
)

transition_mask = np.zeros(len(Ct_base), dtype=bool)

for z in transition_zones:
    transition_mask[z["start"]:z["end"]+1] = True

event_labels_v2 = classify_event_scores(
    event_scores,
    D_t=D_t,
    strong_thr=0.48,
    sem_thr=0.30,
    struct_thr=0.32,
    d_thr=0.22,
    sem_margin=0.04,
    struct_margin=0.04,
)

event_masks = labels_to_event_masks(
    event_labels_v2,
    min_sem_len=1,
    min_struct_len=2,
    min_strong_len=1,
)

transition_mask=transition_mask,

if "transition" not in event_masks:
    transition_pressure = np.asarray(
        event_scores.get("transition_pressure", np.zeros(len(Ct_base))),
        dtype=float,
    )

    transition_mask = transition_pressure >= 0.18

    event_masks["transition"] = enforce_min_persistence(
        transition_mask,
        min_len=2,
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


# --- 1) sanitize thresholds FIRST ---
phi_low_eff  = float(np.clip(phi_low_eff,  0.0, 0.95))
phi_high_eff = float(np.clip(phi_high_eff, 0.05, 1.0))
if phi_high_eff <= phi_low_eff + 0.08:
    phi_high_eff = float(min(1.0, phi_low_eff + 0.12))

sbr_fixed = []

for i in range(len(Ct_base)):
    if i < WARMUP_TURNS:
        sbr_fixed.append("W")
    elif event_labels_final[i] == "RUPTURE_STRONG":
        sbr_fixed.append("B")
    elif event_labels_final[i] in ("RUPTURE_SEM", "RUPTURE_STRUCT"):
        sbr_fixed.append("R")
    else:
        sbr_fixed.append("S")
        
sbr_fixed = protect_conversation_ending(
    sbr_fixed,
    Ct_level=Ct_base,
    Ct_drop=Ct_smooth,
    n_end_protect=6,
    min_drop=0.25,
    stable_threshold=0.35,
)

# =========================
# BUILD OUTPUT DATAFRAME
# =========================

df_out = df.copy()

# --- Core signals ---
df_out["Ct"] = Ct_base
df_out["Ct_im"] = Ct_im
df_out["P_t"] = P_t
df_out["sbr"] = sbr_fixed
df_out["rho_t"] = rho_t
df_out["D_t"] = D_t

# --- Event scores (keep for debugging/analysis) ---
df_out["sem_drop"] = event_scores["sem_drop"]
df_out["struct_drop"] = event_scores["struct_drop"]
df_out["strong_score"] = event_scores["strong_score"]
df_out["semantic_score"] = event_scores["semantic_score"]
df_out["structural_score"] = event_scores["structural_score"]

# --- Final unified event label ---
df_out["event_type"] = event_labels_final


# =========================
# Event windows (ALREADY EXPANDED MASKS)
# =========================

df_out["strong_event_window"] = strong_mask.astype(int)
df_out["semantic_event_window"] = semantic_mask.astype(int)
df_out["structural_event_window"] = structural_mask.astype(int)
df_out["transition_pressure"] = event_scores["transition_pressure"]
df_out["transition_zone"] = transition_mask.astype(int)

# =========================
# C_inv (if available)
# =========================

if C_inv is not None and np.asarray(C_inv).size == len(df_out):
    df_out["C_inv"] = np.asarray(C_inv, float)
else:
    df_out["C_inv"] = np.nan

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
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    title="Conversational coherence dynamics (TIE–Dialog)",
    height=560,
    potentiality=P_t,
    pilot_w=int(pilot_w),
    C_inv=(C_inv if (overlay_cinv_on_main and C_inv is not None) else None),
    sbr_labels=sbr_fixed,
    strong_mask=strong_mask,
    semantic_mask=semantic_mask,
    structural_mask=structural_mask,
)

st.plotly_chart(fig_main, use_container_width=True)


# =========================
# Transition pressure zones
# =========================

transition_pressure_series = np.asarray(
    event_scores["transition_pressure"],
    dtype=float,
)

st.write({
    "transition_pressure_min": float(np.min(transition_pressure_series)),
    "transition_pressure_max": float(np.max(transition_pressure_series)),
    "transition_pressure_mean": float(np.mean(transition_pressure_series)),
})

transition_pressure_series = np.asarray(
    event_scores["transition_pressure"],
    dtype=float,
)

transition_zones = extract_transition_zones(
    transition_pressure_series,
    threshold=np.quantile(
        transition_pressure,
        0.55
    ),
    min_len=3,
)

transition_zone_df = build_transition_zone_dataframe(
    transition_zones
)

fig_pressure = go.Figure()

fig_pressure.add_trace(
    go.Scatter(
        x=np.arange(1, len(transition_pressure_series) + 1),
        y=transition_pressure_series,
        mode="lines",
        name="Transition pressure",
        line=dict(width=3),
    )
)

fig_pressure.add_hline(
    y=0.42,
    line_dash="dash",
    opacity=0.4,
    annotation_text="Transition threshold",
)

for z in transition_zones:
    fig_pressure.add_vrect(
        x0=int(z["start"]) + 1,
        x1=int(z["end"]) + 1,
        fillcolor="rgba(255,0,0,0.10)",
        line_width=0,
    )

fig_pressure.update_layout(
    title="Transition pressure landscape",
    height=320,
    xaxis_title="Turn",
    yaxis_title="Pressure (0–1)",
    yaxis=dict(range=[0, 1]),
    margin=dict(l=40, r=40, t=50, b=40),
)

st.plotly_chart(fig_pressure, use_container_width=True)

st.dataframe(
    transition_zone_df,
    use_container_width=True,
)

st.caption(
    "Transition pressure zones represent temporally extended regions of structural instability. "
    "These zones can be directly compared against human annotations of breakdown and repair. "
    "Each zone aggregates semantic disruption, structural instability, and geometric displacement "
    "into a unified transition-pressure field."
)


st.caption(
    "Main TIE–Dialog coherence dynamics. "
    "The solid Cₜ line shows turn-level coherence relative to the evolving dialogue context, capturing how each contribution maintains or disrupts the conversational trajectory. "
    "Participant markers indicate which speaker produced each turn. "
    "Φ_low and Φ_high are adaptive thresholds that define transitions between stable, transitional, and disrupted regions. "
    "Colored event windows mark detected structural changes: strong ruptures (combined disruption), semantic drift (contextual misalignment), and structural reorganization. "
    "The dotted C_inv line reflects structural stability derived from graph invariants, where lower values indicate stronger reconfiguration. "
    "The dotted ℘ₜ line represents turn-level potentiality or openness."
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
        q_low=q_low,
        q_high=q_high,
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

    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)

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

    st.dataframe(pd.DataFrame(shuffled_rows), use_container_width=True)

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
    
# =========================
# Parameter robustness
# =========================
robustness_runs_df = None
robustness_summary_df = None

if run_param_robustness:
    base_params = {
        "q_low": float(q_low),
        "q_high": float(q_high),
        "smooth_method": smooth_method,
        "env_alpha": float(env_alpha),
        "env_span": int(env_span),
        "use_cinv": bool(use_cinv),
        "cinv_window": int(cinv_window),
        "cinv_knn": int(cinv_knn),
        "cinv_thr": float(cinv_thr),
        "cinv_keigs": int(cinv_keigs),
        "alpha_context": 0.84,
        "beta": 0.70,
        "b": 0.40,
    }

    base_event_masks = {
        "strong": np.asarray(strong_mask, dtype=bool),
        "semantic": np.asarray(semantic_mask, dtype=bool),
        "structural": np.asarray(structural_mask, dtype=bool),
    }

    robustness_runs_df, robustness_summary_df = run_parameter_robustness(
        E=E,
        texts=texts,
        participants=participants,
        base_params=base_params,
        base_ct=np.asarray(Ct_base, float),
        base_sbr=sbr_fixed,
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
        "such as smoothing, Φ thresholds, context inertia, and graph-invariant settings are randomly varied. "
        "Higher Ct correlation, DTW similarity, S–B–R agreement, and Jaccard scores indicate greater robustness."
    )

    st.markdown("### Per-run results")
    st.dataframe(robustness_runs_df, use_container_width=True)
    
    st.caption(
        "Per-run robustness results. Each row corresponds to one perturbed parameter configuration. "
        "Ct_corr and Ct_dtw_similarity compare the coherence trajectory against the original run. "
        "SBR_agreement measures stability of the stable/break/repair regime labels. "
        "The Jaccard scores measure how much the detected strong, semantic, and structural event masks overlap "
        "with the original event masks. The remaining columns show the exact parameter values used in each run."
    )

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

# Orden lógico de columnas
ci_cols = [c for c in df_out.columns if str(c).startswith("Ci_")]

cols_order = [
    "turn",
    "participant",
    "text",
    "Ct",
    "Ct_im",
    "C_inv",
    *ci_cols,
    "P_t",
    "sbr",
    "event_type",
    "strong_event_window",
    "complex_event_window",
    "semantic_event_window",
    "structural_event_window",
    "rho_t",
    "D_t",
]


cols_show = [c for c in cols_order if c in df_out.columns]

st.dataframe(
    df_out[cols_show],
    use_container_width=True,
)

st.caption(
    "Turn-level representation of the dialogue as a dynamic informational system. "
    "Each row corresponds to a turn embedded within a multi-signal structure capturing "
    "contextual coherence (Cₜ), structural stability (C_inv), and geometric evolution (Dₜ, ρₜ). "
    "The S/B/R labels indicate regime segmentation (Stable, Broken, Repair), while event_type "
    "identifies localized structural transitions. "
    "Together, these features provide a trajectory-based view of conversational organization "
    "that goes beyond local similarity and static analysis."
)
        
# =========================
# Event score diagnostics
# =========================
with st.expander("Event score diagnostics", expanded=False):
    fig_diag = go.Figure()
    x_diag = np.arange(1, len(Ct_base) + 1)

    fig_diag.add_trace(go.Scatter(
        x=x_diag, 
        y=np.asarray(Ct_base, float), 
        mode="lines", 
        name="Ct"
    ))
    
    if C_inv is not None:
        fig_diag.add_trace(go.Scatter(
            x=x_diag, 
            y=np.asarray(C_inv, float), 
            mode="lines", 
            name="C_inv"
        ))

    fig_diag.add_trace(go.Scatter(
        x=x_diag, 
        y=event_scores["sem_drop"], 
        mode="lines", 
        name="sem_drop"
    ))

    fig_diag.add_trace(go.Scatter(
        x=x_diag, 
        y=event_scores["struct_drop"], 
        mode="lines", 
        name="struct_drop"
    ))

    fig_diag.add_trace(go.Scatter(
        x=x_diag, 
        y=event_scores["strong_score"], 
        mode="lines", 
        name="strong_score"
    ))

    fig_diag.update_layout(
        title="Event score diagnostics",
        height=500,
        xaxis_title="Turn",
        yaxis_title="Value (0–1)",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig_diag, use_container_width=True)
    
    st.caption(
        "Event score diagnostics showing how different signals contribute to event detection. "
        "Cₜ represents contextual coherence, while C_inv (if present) reflects structural stability. "
        "sem_drop captures decreases in coherence (semantic disruption), and struct_drop captures decreases in structural stability. "
        "strong_score combines these signals into a unified event pressure measure used to detect strong rupture. "
        "Peaks in strong_score typically correspond to moments where multiple signals indicate a significant structural change in the dialogue."
    )

st.session_state["last_main_fig"] = fig_main
html = fig_main.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
st.download_button(
    label="Download main plot (HTML)",
    data=html,
    file_name="tie_dialog_main_plot.html",
    mime="text/html",
)

if show_envelope:
    with st.expander("Smoothed coherence (C_smooth)", expanded=False):

        fig_s = go.Figure()
        x_env = np.arange(1, len(Ct_base) + 1)

        # Raw coherence
        fig_s.add_trace(
            go.Scatter(
                x=x_env,
                y=np.asarray(Ct_base, float),
                mode="lines",
                name="Cₜ (raw)",
                line=dict(width=2),
            )
        )

        # Smoothed Ct
        fig_s.add_trace(
            go.Scatter(
                x=x_env,
                y=np.asarray(Ct_smooth, float),
                mode="lines",
                name="Cₜ_smooth",
                line=dict(width=3),
            )
        )

        # Smoothed C_inv
        if C_inv_plot is not None:
            fig_s.add_trace(
                go.Scatter(
                    x=x_env,
                    y=np.clip(np.asarray(C_inv_plot, float), 0.0, 1.0),
                    mode="lines",
                    name="C_inv_smooth",
                    line=dict(width=2, dash="dot"),
                )
            )

        # Thresholds
        fig_s.add_hline(
            y=float(phi_low_eff),
            line_dash="dash",
            opacity=0.35,
            annotation_text="Φ_low",
        )

        fig_s.add_hline(
            y=float(phi_high_eff),
            line_dash="dash",
            opacity=0.35,
            annotation_text="Φ_high",
        )

        fig_s.update_layout(
            title="Smoothed coherence curve",
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
with st.expander(L["geom_plot"], expanded=False):
    fig_g = plot_ic3_geometry(
        Ct=Ct_base,
        d_i=ic3["d_i"],
        kappa_i=ic3["kappa_i"],
        phi_low=float(phi_low_eff),
        phi_high=float(phi_high_eff),
        height=600,
    )
    st.plotly_chart(fig_g, use_container_width=True)
    
    st.caption(
        "Geometric view of the dialogue trajectory in embedding space. "
        "dᵢ (displacement) measures how far each turn moves from the previous one, capturing the magnitude of change. "
        "κᵢ (curvature) measures directional change, indicating shifts in conversational direction. "
        "Together, these signals reflect the underlying geometry of the dialogue trajectory. "
        "High dᵢ and κᵢ values often correspond to moments of transition or disruption, complementing the coherence signal (Cₜ) shown in the main plot."
    )

with st.expander(L["state_title"], expanded=False):
    fig_state = plot_participant_state_lines(
        Ct=Ct_base,
        participants=participants,
        phi_low=float(phi_low_eff),
        phi_high=float(phi_high_eff),
        alpha_state=float(state_alpha),
        title=L["state_title"],
        height=520,
    )
    st.plotly_chart(fig_state, use_container_width=True)

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
    "d_i": np.asarray(ic3["d_i"], float),
    "kappa_i": np.asarray(ic3["kappa_i"], float),
    "rho_t": np.asarray(rho_t, float),
    "D_t": np.asarray(D_t, float),
})

# =========================
# PDF Report (English) — Download
# =========================
st.subheader("Report (PDF)")

report_params = {
    "mode": mode,
    "emb_mode": emb_mode,
    "sbert_model": sbert_model,
    "q_low": float(q_low),
    "q_high": float(q_high),
    "sep_min": int(sep_min),
    "prom_min": float(prom_min),
    "merge_gap": int(merge_gap),
    "pilot_w": int(pilot_w),
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
    "state_alpha": float(state_alpha),
    "public_span": int(public_span),
}

pdf_bytes = build_pdf_report_bytes(
    df_in=df,
    df_out=df_out,
    ic2_df=ic2_df,
    ic3_df=ic3_df,
    Ct_base=Ct_base,
    Ct_smooth=Ct_smooth,
    C_inv=(C_inv if (use_cinv and C_inv is not None) else None),
    phi_low=float(phi_low_eff),
    phi_high=float(phi_high_eff),
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
