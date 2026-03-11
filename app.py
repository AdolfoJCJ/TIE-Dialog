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

    # --- SBERT path ---
    if mode in ("auto", "sbert") and _SBERT_AVAILABLE:
        try:
            model_name = sbert_model.strip() if sbert_model and sbert_model.strip() else \
                         "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            E = model.encode(texts, normalize_embeddings=True)
            return np.asarray(E, dtype=float), f"sbert:{model_name}", f"✅ Using SBERT embeddings: {model_name}"
        except Exception as e:
            msg = f"⚠️ SBERT failed ({type(e).__name__}): {e}. Falling back to TF-IDF."
            # fall through

    if mode == "sbert" and not _SBERT_AVAILABLE:
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
        "table_title": "Dialogue with S/B/R labels, potentiality ℘ₜ and geometry (IC–III)",
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
    tie_eps: float = 1e-12,   # deterministic tie-break
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
    IMPORTANT: we only start producing values when t >= window-1 (full window).
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

    # context memory I_m
    I_m = np.zeros_like(I_s)
    I_m[0] = I_s[0]
    a = float(np.clip(alpha_context, 0.0, 0.999))
    for t in range(1, n):
        I_m[t] = a * I_m[t - 1] + (1.0 - a) * I_s[t - 1]

    # resonance r_t
    res = np.zeros(n, float)
    for t in range(n):
        res[t] = _cos(I_s[t], I_m[t], eps=eps)

    # local displacement ΔI_t
    dI = np.zeros(n, float)
    for t in range(1, n):
        dI[t] = float(np.linalg.norm(I_s[t] - I_s[t - 1]))

    # IMPORTANT: scale ΔI into a comparable range (otherwise it can dominate)
    # robust scaling by 95th percentile
    s = float(np.quantile(dI[1:], 0.95)) if n > 2 else float(np.max(dI) + eps)
    s = max(s, 1e-6)
    dI_s = np.clip(dI / s, 0.0, 1.0)

    z = float(beta) * res - (1.0 - float(beta)) * dI_s - float(b)
    C_t = _sigma(z)

    return {"I_s": I_s, "I_m": I_m, "res": res, "dI_norm": dI_s, "C_t": np.clip(C_t, 0.0, 1.0)}

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
    - tau(t): cumulative deformation (arc-length proxy)
    """
    E = np.asarray(E, float)
    n = E.shape[0]
    if n == 0:
        return {
            "d_i": np.zeros(0, float),
            "kappa_i": np.zeros(0, float),
            "tau_t": np.zeros(0, float),
            "tau_norm": np.zeros(0, float),
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

    # curvature proxy = local change in displacement
    dd = np.zeros(n, float)
    dd[1:] = d_i[1:] - d_i[:-1]
    kappa_i = np.abs(dd)

    # cumulative deformation / arc-length proxy
    tau_t = np.cumsum(d_i)

    # normalized tau
    if tau_t.size == 0:
        tau_norm = tau_t
    else:
        t_min = float(np.min(tau_t))
        t_max = float(np.max(tau_t))
        tau_norm = np.zeros_like(tau_t) if (t_max - t_min < 1e-9) else (tau_t - t_min) / (t_max - t_min)

    return {
        "d_i": np.clip(d_i, 0.0, 1.0),
        "kappa_i": np.clip(kappa_i, 0.0, 1.0),
        "tau_t": tau_t,
        "tau_norm": tau_norm,
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
# Lag estimation Δ* between D_t and Ct
# -------------------------------------------------
def estimate_lag_delta(
    *,
    D: np.ndarray,
    C: np.ndarray,
    phi_low: float,
    delta_max: int = 6,
    smooth_alpha: float = 0.30,
) -> Dict[str, float]:
    """
    Estimate lag Δ* by scanning correlations corr(D_{t-Δ}, 1-C_t) (or similar).
    Returns dict with delta_star and score.
    """
    D = np.asarray(D, float)
    C = np.asarray(C, float)
    n = min(D.size, C.size)
    if n < 5:
        return {"delta_star": 0.0, "score": float("nan")}

    D0 = _ema(np.clip(D[:n], 0.0, 1.0), alpha=float(np.clip(smooth_alpha, 0.0, 0.999)))
    R0 = 1.0 - np.clip(C[:n], 0.0, 1.0)  # rupture-ish proxy

    best_d = 0
    best_s = -1.0

    for d in range(0, int(delta_max) + 1):
        if d == 0:
            a = D0
            b = R0
        else:
            a = D0[:-d]
            b = R0[d:]
        s = _safe_corr(a, b)
        if np.isfinite(s) and s > best_s:
            best_s = float(s)
            best_d = int(d)

    return {"delta_star": float(best_d), "score": float(best_s)}


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
    tau_t: np.ndarray,
    phi_low: float,
    phi_high: float,
    height: int = 600,
) -> go.Figure:
    Ct = np.asarray(Ct, float)
    d_i = np.asarray(d_i, float)
    kappa_i = np.asarray(kappa_i, float)
    tau_t = np.asarray(tau_t, float)
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

    if tau_t.size == 0:
        tau_norm = tau_t
    else:
        tau_min = float(np.min(tau_t))
        tau_max = float(np.max(tau_t))
        tau_norm = np.zeros_like(tau_t) if (tau_max - tau_min < 1e-9) else (tau_t - tau_min) / (tau_max - tau_min)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=Ct, mode="lines", name="Cₜ (coherence)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=d_norm, mode="lines", name="dᵢ (normalized)", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=kappa_norm, mode="lines", name="κᵢ (normalized)", line=dict(width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=tau_norm, mode="lines", name="τ (normalized)", line=dict(width=2, dash="longdash")))

    fig.add_hline(y=float(phi_low), line_dash="dash", opacity=0.40, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.40, annotation_text="Φ_high")

    fig.update_layout(
        title="IC–III geometric layer: dᵢ, κᵢ and τ over the coherence manifold",
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
# ✅ Continuous state trajectories
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
# ✅ FIXED: Ci plot and State plot are both available (defined elsewhere)
# ✅ FIXED: plot_ct_main no longer references undefined friction/geom_* vars
# ✅ FIXED: removed dead/unreachable code under _base_layout
# ✅ FIXED: plot_ct_main now returns fig and includes markers/thresholds/annotations
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


# -------------------------------
# Events and S/B/R
# -------------------------------
def _detect_events_peaks(
    Ct: np.ndarray,
    phi_low_t: np.ndarray,
    phi_high_t: np.ndarray,
    sep_min: int = 2,
    prom_min: float = 0.05,
) -> Tuple[List[int], List[int]]:
    n = len(Ct)
    if n < 3:
        return [], []
    margin = 0.01

    if _HAS_FIND_PEAKS:
        inv = 1.0 - Ct
        v_idx, _ = find_peaks(inv, prominence=float(prom_min), distance=int(sep_min))
        valleys = [int(i) for i in v_idx if Ct[int(i)] < (phi_low_t[int(i)] - margin)]

        p_idx, _ = find_peaks(Ct, prominence=float(prom_min), distance=int(sep_min))
        p_idx = [int(i) for i in p_idx]

        peaks: List[int] = []
        for vi in valleys:
            nxt = [p for p in p_idx if p > vi and Ct[p] > (phi_high_t[p] + margin)]
            if nxt:
                peaks.append(int(nxt[0]))
        return valleys, peaks

    valleys, peaks = [], []
    last_v = -10**9

    for i in range(1, n - 1):
        if Ct[i] < Ct[i - 1] and Ct[i] <= Ct[i + 1] and (i - last_v) >= int(sep_min):
            prom = max(Ct[i - 1] - Ct[i], Ct[i + 1] - Ct[i])
            if prom >= float(prom_min) and Ct[i] < (phi_low_t[i] - margin):
                valleys.append(int(i))
                last_v = int(i)

    for vi in valleys:
        for j in range(vi + 1, n - 1):
            if Ct[j] > Ct[j - 1] and Ct[j] >= Ct[j + 1] and Ct[j] > (phi_high_t[j] + margin):
                peaks.append(int(j))
                break

    return valleys, peaks


def detect_events(
    Ct: np.ndarray,
    phi_low: float,
    phi_high: float,
    sep_min: int = 2,
    prom_min: float = 0.05,
    phi_low_t: Optional[np.ndarray] = None,
    phi_high_t: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    n = len(Ct)
    if n < 3:
        return [], []
    lo = phi_low_t if phi_low_t is not None else np.full(n, float(phi_low))
    hi = phi_high_t if phi_high_t is not None else np.full(n, float(phi_high))
    return _detect_events_peaks(Ct, lo, hi, sep_min=int(sep_min), prom_min=float(prom_min))


def assign_sbr(
    Ct: np.ndarray,
    valleys: List[int],
    peaks: List[int],
    phi_low: float,
    phi_high: float,
    warmup_turns: int = WARMUP_TURNS,
) -> List[str]:
    n = len(Ct)
    eB = set(int(v) for v in valleys)
    eR = set(int(p) for p in peaks)
    states: List[str] = []

    for i in range(n):
        if i < int(warmup_turns):
            states.append("W")
            continue
        if i in eB:
            states.append("B")
        elif i in eR:
            states.append("R")
        else:
            states.append("B" if float(Ct[i]) < float(phi_low) else "S")
    return states


# -------------------------------
# Quanto of Coherence (legacy Qa)
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
# ✅ Ci trajectories per participant (embedding-based)
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
    valleys: List[int],
    peaks: List[int],
    title: str,
    height: int,
    pilot_w: int = 2,
    potentiality: Optional[np.ndarray] = None,
    C_inv: Optional[np.ndarray] = None, 
    geom_breaks_struct: Optional[List[int]] = None,
    lag_label: Optional[str] = None,
    sbr_labels: Optional[Sequence[str]] = None,  
    ) -> go.Figure:
    Ct = np.asarray(Ct, dtype=float)
    n = int(Ct.size)
    x = np.arange(1, n + 1, dtype=int)

    fig = go.Figure()
    fig = _base_layout(fig, title, height)

    # -----------------------------
    # Main curve (Ct)
    # -----------------------------
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

    # -----------------------------
    # Optional overlays
    # -----------------------------
    # ✅ C_inv overlay (graph invariants)
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


    # ---------------------------------
    # Regime bands (S/B/R) — exact segments (no pilot inflation)
    # ---------------------------------
    if sbr_labels is not None:
        lab = np.asarray(list(sbr_labels), dtype=object)
        if lab.size == n:
            fig = add_regime_bands(
                fig,
                turns=x,
                labels=lab,
                strip=True,
                strip_y0=0.00,
                strip_y1=0.12,
                fullheight_breaks=True,
            )

    # participant markers (masked points)
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

    # thresholds
    fig.add_hline(y=float(phi_low), line_dash="dash", opacity=0.45, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.45, annotation_text="Φ_high")

    # event markers
    if valleys:
        fig.add_trace(
            go.Scatter(
                x=[int(v) + 1 for v in valleys],
                y=[float(Ct[int(v)]) for v in valleys],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12),
                name="Breaks (B)",
                yaxis="y",
            )
        )

    if peaks:
        fig.add_trace(
            go.Scatter(
                x=[int(p) + 1 for p in peaks],
                y=[float(Ct[int(p)]) for p in peaks],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12),
                name="Repairs (R)",
                yaxis="y",
            )
        )

    if lag_label:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            text=str(lag_label),
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        )

    return _base_layout(fig, title, height=height)

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
    valleys: List[int],
    peaks: List[int],
    geom_breaks_struct: List[int],
    lag_label: str,
    used_mode: str,
    emb_msg: str,
    params: Dict[str, object],
) -> bytes:
    """
    Returns PDF bytes (downloadable).
    valleys/peaks/geom/perceived are assumed 0-indexed.
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
        f"Detected breaks (valleys): {len(valleys)}",
        f"Detected repairs (peaks): {len(peaks)}",
        f"Structural breaks (geom): {len(geom_breaks_struct)}",
        "",
        f"{lag_label}",
        "",
        "Interpretation (high level):",
        "• Ct captures alignment with the evolving conversational context.",
        "• Valleys below Φ_low are rupture candidates; peaks above Φ_high are repair candidates.",
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

    cols = ["turn", "participant", "Ct", "dCt", "sbr", "rupture_type", "text"]
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

    events_rows = []
    for v in valleys:
        events_rows.append({"type": "B (valley)", "turn": int(v) + 1, "Ct": f"{float(Ct_base[int(v)]):.3f}"})
    for p in peaks:
        events_rows.append({"type": "R (peak)", "turn": int(p) + 1, "Ct": f"{float(Ct_base[int(p)]):.3f}"})
    events_df = pd.DataFrame(events_rows) if events_rows else pd.DataFrame({"type": [], "turn": [], "Ct": []})
    if not events_df.empty:
        events_df = events_df.sort_values(["turn", "type"]).reset_index(drop=True)

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
        marks = [("Break candidates", valleys), ("Repair candidates", peaks)]
        fig = _fig_timeseries(
            "Coherence Dynamics (Ct) with Break/Repair Events",
            turns,
            main_series,
            hlines=hls,
            vmarks=marks,
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
                vmarks=[("Geom breaks", geom_breaks_struct)],
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

        fig = _fig_table_page(
            "Detected Events (Valleys/Peaks)",
            events_df if not events_df.empty else pd.DataFrame({"type": ["(none)"], "turn": [""], "Ct": [""]}),
            note="Valleys are break candidates; peaks are repair candidates (based on your event detector).",
        )
        pdf.savefig(fig); plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

# ============================
# app.py — PART 3/3
# (demo + Streamlit app main + downloads)
# ✅ FIXED: state_alpha slider exists and matches the call
# ✅ FIXED: undefined vars deleted (Ct_action/Ft/PDF/geom_* vars)
# ============================

def load_demo(n_turns: int = 34) -> pd.DataFrame:
    speakers = ["A", "B"]
    turns, parts, texts = [], [], []

    base_lines = [
        "Can we do a quick recap of yesterday’s meeting? I want to make sure we aligned on the next steps.",
        "Sure. The main decision was to ship the onboarding changes first, then run the user test next week.",
        "Right, and we said we’d freeze new feature requests until the test results come back.",
        "Exactly. Otherwise we keep reopening the scope and the conversation drifts.",
        "That drift is what I want to measure: when we lose the shared frame and when we repair it.",
        "So in TIE–Dialog terms, we’d expect coherence to stay high while we’re on the same plan.",
        "Yes, and when someone introduces a new angle, we might see a small dip and then a quick recovery.",
        "Makes sense. Can you remind me what counts as a rupture versus normal topic evolution?",
        "A rupture is when a turn stops being compatible with the current context—like a sudden unrelated jump.",
        "And repair would be the re-attachment: explicitly reconnecting to the shared topic or clarifying intent.",
    ]

    rupture_block1 = [
        "Anyway, I’m thinking of buying a used motorcycle this weekend—do you know any good brands?",
        "Wait, that’s a complete switch. We were on the meeting decisions and measuring drift.",
        "True—sorry. Let me pull it back: I asked because I noticed we also switched topics in the meeting like that.",
        "So the motorcycle question is basically a toy example of an off-topic injection that creates a coherence drop.",
        "Exactly. And the repair is us naming the mismatch and reconnecting to the original frame.",
    ]

    tech_block = [
        "Okay, so how do you represent the ‘current frame’ computationally?",
        "We keep an evolving context vector—like a running summary of what the conversation is about.",
        "Then each new turn gets compared against that context to compute local coherence.",
        "And you also compute coherence with an emergent structuring field, like I_M, right?",
        "Yes. That helps distinguish ‘locally smooth drift’ from ‘global misalignment with the main topic’.",
        "So a turn can be coherent with the last turn but still diverge from the overall trajectory.",
        "Exactly. That’s why the two signals together are useful.",
        "And speaker-level coherence shows who is pulling the topic away or doing most of the repairs.",
        "Right—sometimes one participant is effectively acting as a stabilizer for the shared frame.",
    ]

    rupture_block2 = [
        "BREAKING: The meeting is actually a sandwich, and the action items are made of glitter.",
        "Okay, that’s not just drift—that breaks the frame completely. I can’t map that onto our topic.",
        "Yes, intentional rupture. Now the repair: we return to the agenda and the measurement idea.",
        "Specifically, we want the demo to show a steep drop followed by a clear recovery after re-alignment.",
    ]

    end_block = [
        "So after the repair, we restate the shared goal: track coherence turn-by-turn and flag rupture candidates.",
        "And we keep the language practical: recap, mismatch, repair, and back to the plan.",
        "Then the coherence curve should climb and stabilize as we stay within the same frame again.",
        "Exactly. A good demo ends with a stable phase so the viewer sees recovery clearly.",
        "We can also mention that mild dips are normal—real dialogue isn’t perfectly constant.",
        "Right, the point is interpretability: you can see transitions, not just a single average score.",
        "And if the last turns are stable, it avoids the impression that the conversation ends ‘broken’.",
        "Perfect. That should make the demo feel realistic while still illustrating the signal behavior.",
    ]

    all_lines = base_lines + rupture_block1 + tech_block + rupture_block2 + end_block
    all_lines = all_lines[:max(1, int(n_turns))]

    for i, text in enumerate(all_lines):
        turns.append(i + 1)
        parts.append(speakers[i % 2])
        texts.append(text)

    return pd.DataFrame(
        {"turn": np.array(turns, dtype=int), "timestamp": "", "participant": parts, "text": texts}
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
This application analyzes dialogue as a dynamic informational system.

It computes a turn-by-turn coherence signal (Ct) that captures how each contribution aligns with the evolving conversational context. Rather than treating coherence as a static score, the app models it as a continuous process that fluctuates over time.

In addition, the app can compute an invariant-based structural coherence channel (**C_inv**). While Ct tracks contextual/semantic alignment, **C_inv** tracks structural reconfiguration of the dialogue by measuring how the topology of a rolling similarity graph changes over time (via graph-invariant features). This helps distinguish:

- **Semantic drift** (Ct decreases while C_inv remains stable)
- **Structural reframe** (Ct remains stable while C_inv decreases)
- **Strong rupture** (Ct decreases and C_inv decreases)

The system detects three conversational regimes:

- **Stable (S)** — coherent continuation of the shared frame  
- **Broken (B)** — rupture candidate (semantic and/or structural)  
- **Repair (R)** — re-alignment or recovery of coherence  

Beyond coherence, the app incorporates a geometric layer (IC–III) that measures structural properties of the dialogue trajectory, including:

- **d_i** — semantic displacement  
- **kappa_i** — structural curvature  
- **tau** — cumulative conversational deformation  

The goal is to make conversational dynamics measurable, interpretable, and visually explorable.
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
    return st.sidebar.slider(label, min_value, max_value, value, step, key=key, help=help)

def ui_checkbox(label, value=False, key=None, help=None):
    if IS_CANON:
        return value
    return st.sidebar.checkbox(label, value=value, key=key, help=help)

def ui_selectbox(label, options, index=0, key=None, help=None):
    if IS_CANON:
        # canonical: fixed choice by index
        return options[index]
    return st.sidebar.selectbox(label, options, index=index, key=key, help=help)

def ui_text_input(label, value="", key=None, help=None):
    if IS_CANON:
        return value
    return st.sidebar.text_input(label, value=value, key=key, help=help)

st.sidebar.header(L["params"])

# Estos suelen ser "básicos" => los dejo visibles incluso en Canonical
use_demo = st.sidebar.checkbox(L["load_demo"], value=True)
uploaded = None if use_demo else st.sidebar.file_uploader(L["upload"], type=["csv", "xlsx"])

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

with st.expander(L["preview"], expanded=True):
    st.dataframe(df.head(25), use_container_width=True)

# -------------------------
# Semantic representation
# -------------------------
st.sidebar.subheader(L["sem_repr"])
emb_mode = ui_selectbox(L["emb_mode"], ["auto", "sbert", "tfidf"], index=0)
sbert_model = ui_text_input(L["sbert_model"], value="sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# Φ thresholds
# -------------------------
st.sidebar.subheader(L["phi"])
q_low = ui_slider("Φ_low percentile", 0.05, 0.50, 0.20, 0.01)
q_high = ui_slider("Φ_high percentile", 0.50, 0.95, 0.80, 0.01)

# -------------------------
# Events
# -------------------------
st.sidebar.subheader(L["events"])
sep_min = ui_slider("min separation (turns)", 1, 8, 2, 1)
prom_min = ui_slider("min prominence", 0.01, 0.30, 0.05, 0.01)
min_drop = ui_slider("min drop (Ct step)", 0.00, 0.60, 0.20, 0.01)
merge_gap = ui_slider("merge_gap (micro-break grouping)", 1, 6, 2, 1)
pilot_w = ui_slider("Pilot window half-width w (turns)", 0, 6, 2, 1)

# -------------------------
# Pilot evaluation (si en Canonical lo quieres oculto, aquí queda oculto)
# -------------------------
st.sidebar.subheader("Pilot evaluation")
k_annot = ui_slider("Annotators (K)", 5, 10, 5, 1)
min_votes = ui_slider("Consensus min votes", 1, 10, 3, 1)

# -------------------------
# Ci
# -------------------------
st.sidebar.subheader(L["ci_header"])
ci_method = ui_selectbox(L["ci_method"], ["ctx", "im"], index=0)
ci_alpha = ui_slider(L["ci_alpha"], 0.70, 0.99, 0.90, 0.01)

state_alpha = ui_slider(L["state_alpha"], 0.60, 0.98, 0.88, 0.01)

# -------------------------
# Public view
# -------------------------
st.sidebar.subheader(L["public_header"])
public_span = ui_slider(L["public_span"], 3, 25, 9, 1)
public_show_thr = ui_checkbox(L["public_show_thresholds"], value=False)

# -------------------------
# Smoothed coherence
# -------------------------
st.sidebar.markdown("### Smoothed coherence")
smooth_method = ui_selectbox("Smoothing method", ["ema", "ewma"], index=0)
env_alpha = ui_slider("EMA alpha", 0.05, 0.60, 0.20, 0.01)
env_span = ui_slider("EWMA span", 3, 25, 9, 1)
show_envelope = ui_checkbox("Show smoothed coherence plot", value=True)
overlay_smoothed_on_main = ui_checkbox("Overlay smoothed curve on main plot", value=False)

# -------------------------
# C_inv
# -------------------------
st.sidebar.markdown("### Invariant coherence (C_inv)")
use_cinv = ui_checkbox("Enable C_inv (graph invariants)", value=True)
cinv_window = ui_slider("C_inv window W", 6, 24, 8, 1)
cinv_knn = ui_slider("C_inv k-NN", 2, 10, 3, 1)
cinv_thr = ui_slider("C_inv edge threshold", 0.00, 0.30, 0.16, 0.01)
cinv_keigs = ui_slider("C_inv eigenfeatures (k)", 3, 12, 6, 1)
overlay_cinv_on_main = ui_checkbox("Overlay C_inv on main plot", value=True)

if not st.button(L["compute"]):
    st.stop()

texts = df["text"].tolist()
participants = df["participant"].tolist()
turns = df["turn"].to_numpy(dtype=int)

E, used_mode, emb_msg = embed_texts(texts, mode=emb_mode, sbert_model=sbert_model)
st.sidebar.info(emb_msg)


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

# 1) raw (sigmoid output)
Ct_raw = np.asarray(ic2["C_t"], float)
Ct_raw = np.clip(Ct_raw, 0.0, 1.0)

# 2) your canonical post-processing (this is what you WANT to plot/use)
Ct_ic2 = _normalize_ct(Ct_raw, lower=0.06, upper=0.94)
Ct_ic2 = apply_warmup_ramp(Ct_ic2, warm=WARMUP_TURNS, floor=0.10)

# 3) choose what Ct_base means in your app
Ct_base = Ct_ic2 

Ct_im = 0.5 * (1.0 + np.asarray(ic2["res"], float))
Ct_im = np.clip(Ct_im, 0.0, 1.0)

Ct_ic2 = np.clip(ic2["C_t"], 0.0, 1.0)
Ct_ic2 = _normalize_ct(Ct_ic2, lower=0.06, upper=0.94)
Ct_ic2 = apply_warmup_ramp(Ct_ic2, warm=WARMUP_TURNS, floor=0.10)

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

Ct_for_events = Ct_base.copy()
Ct_for_events[:WARMUP_TURNS] = np.nan  # ignore warmup turns for peak finding

valleys, peaks = detect_events(
    Ct_for_events,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    sep_min=sep_min,
    prom_min=prom_min
)

# -------------------------
# Extra filter: require a minimum local drop into the valley
# (reduces micro-valleys and makes breaks more "event-like")
# -------------------------
min_drop_eff = float(min_drop)
valleys_f = []
for v in valleys:
    v = int(v)
    if v <= 0:
        continue
    drop = float(Ct_for_events[v - 1]) - float(Ct_for_events[v])
    if drop >= min_drop_eff:
        valleys_f.append(v)

valleys = valleys_f
valleys = merge_consecutive(valleys, gap=merge_gap)

# peaks can stay as-is; they are tied to valleys in your detector

# --- 1) sanitize thresholds FIRST ---
phi_low_eff  = float(np.clip(phi_low_eff,  0.0, 0.95))
phi_high_eff = float(np.clip(phi_high_eff, 0.05, 1.0))
if phi_high_eff <= phi_low_eff + 0.08:
    phi_high_eff = float(min(1.0, phi_low_eff + 0.12))

# --- 2) compute SBR directly (no mandatory repair rule) ---
sbr_fixed = assign_sbr(
    Ct_base,
    valleys=valleys,
    peaks=peaks,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    warmup_turns=WARMUP_TURNS
)

# --- 3) optional: end-protection only (keeps endings from looking artificially broken) ---
sbr_fixed = protect_conversation_ending(
    sbr_fixed,
    Ct_level=Ct_base,
    Ct_drop=Ct_smooth,
    n_end_protect=6,
    min_drop=0.25,
    stable_threshold=0.35,
)

# --- 4) optional: keep for UI/debug compatibility ---
sbr_corrections = []

# =========================
# 3) Mini stability panel (quick scan)
# Heuristic: perturb 2 params ±10% and see how much event sets change.
# Place AFTER Ct_for_events is defined and detect_events exists.
# =========================

def jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(map(int, a)), set(map(int, b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def stability_quick_scan(
    Ct_signal: np.ndarray,
    phi_low: float,
    phi_high: float,
    sep_min: int,
    prom_min: float,
    merge_gap: int,
    *,
    delta_frac: float = 0.10
) -> Dict[str, float]:
    Ct_signal = np.asarray(Ct_signal, float)

    # baseline
    v0, p0 = detect_events(Ct_signal, phi_low=phi_low, phi_high=phi_high, sep_min=sep_min, prom_min=prom_min)
    v0 = merge_consecutive(v0, gap=merge_gap)
    p0 = merge_consecutive(p0, gap=merge_gap)

    # perturb 1) prom_min ±10%
    prom_lo = max(0.001, float(prom_min) * (1.0 - delta_frac))
    prom_hi = float(prom_min) * (1.0 + delta_frac)

    v1, p1 = detect_events(Ct_signal, phi_low=phi_low, phi_high=phi_high, sep_min=sep_min, prom_min=prom_lo)
    v2, p2 = detect_events(Ct_signal, phi_low=phi_low, phi_high=phi_high, sep_min=sep_min, prom_min=prom_hi)

    v1 = merge_consecutive(v1, gap=merge_gap); p1 = merge_consecutive(p1, gap=merge_gap)
    v2 = merge_consecutive(v2, gap=merge_gap); p2 = merge_consecutive(p2, gap=merge_gap)

    jv_prom = 0.5 * (jaccard(v0, v1) + jaccard(v0, v2))
    jp_prom = 0.5 * (jaccard(p0, p1) + jaccard(p0, p2))

    # perturb 2) phi_low ±10% of its value (clipped)
    phi_lo = float(np.clip(phi_low * (1.0 - delta_frac), 0.0, 0.95))
    phi_hi = float(np.clip(phi_low * (1.0 + delta_frac), 0.0, 0.95))

    v3, p3 = detect_events(Ct_signal, phi_low=phi_lo, phi_high=phi_high, sep_min=sep_min, prom_min=prom_min)
    v4, p4 = detect_events(Ct_signal, phi_low=phi_hi, phi_high=phi_high, sep_min=sep_min, prom_min=prom_min)

    v3 = merge_consecutive(v3, gap=merge_gap); p3 = merge_consecutive(p3, gap=merge_gap)
    v4 = merge_consecutive(v4, gap=merge_gap); p4 = merge_consecutive(p4, gap=merge_gap)

    jv_phi = 0.5 * (jaccard(v0, v3) + jaccard(v0, v4))
    jp_phi = 0.5 * (jaccard(p0, p3) + jaccard(p0, p4))

    # overall stability score in [0,1]
    score = float(np.nanmean([jv_prom, jp_prom, jv_phi, jp_phi]))

    return {
        "stability_score": score,
        "jv_prom": jv_prom,
        "jp_prom": jp_prom,
        "jv_phi": jv_phi,
        "jp_phi": jp_phi,
        "n_valleys": len(v0),
        "n_peaks": len(p0),
    }

def stability_label(score: float) -> str:
    if not np.isfinite(score):
        return "n/a"
    if score >= 0.80:
        return "LOW sensitivity (stable)"
    if score >= 0.55:
        return "MEDIUM sensitivity"
    return "HIGH sensitivity (unstable)"

stab = stability_quick_scan(
    Ct_signal=Ct_for_events,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    sep_min=int(sep_min),
    prom_min=float(prom_min),
    merge_gap=int(merge_gap),
    delta_frac=0.10
)

with st.expander("Stability (quick scan)", expanded=False):
    st.metric("Parameter sensitivity", stability_label(stab["stability_score"]))
    st.caption(f"Score={stab['stability_score']:.3f} (1.0 = identical events after ±10% tweaks)")
    st.write({
        "baseline valleys": stab["n_valleys"],
        "baseline peaks": stab["n_peaks"],
        "Jaccard valleys (prom)": round(stab["jv_prom"], 3),
        "Jaccard peaks (prom)": round(stab["jp_prom"], 3),
        "Jaccard valleys (phi_low)": round(stab["jv_phi"], 3),
        "Jaccard peaks (phi_low)": round(stab["jp_phi"], 3),
    })

# =========================
# 2) Automatic Interpretation Cheatsheet
# Place AFTER you compute Ct_base, C_inv (and have df_out if you want)
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
        return "RUPTURE_SEM (Ct decreases, C_inv~)"
    if Ct_flat and Ci_down:
        return "RUPTURE_STRUCT (Ct~, C_inv decreases)"
    return "STABLE / smooth evolution"

# --- UI block ---
cheat = compute_cheatsheet(Ct_base, C_inv)

with st.expander("Interpretation cheatsheet", expanded=True):
    st.markdown("""
**Rules of thumb (auto):**
- **Ct decreases and C_inv decreases => strong rupture** (semantic + structural reconfiguration)
- **Ct decreases and C_inv ~ stable => semantic drift** (topic/meaning shift, structure stable)
- **Ct ~ stable and C_inv decreases => structural reframe** (organization changes, meaning locally stable)
""")

    if cheat["has_cinv"]:
        st.caption(
            f"Auto thresholds: Ct drop <= {cheat['drop_thr']:.3f} | "
            f"C_inv drop <= {cheat['drop_inv_thr']:.3f}"
        )
    else:
        st.caption(f"Auto thresholds: Ct drop <= {cheat['drop_thr']:.3f} (C_inv disabled)")

    # show top candidates (biggest drops)
    dCt = np.zeros_like(Ct_base, float)
    dCt[1:] = Ct_base[1:] - Ct_base[:-1]

    top_ct = np.argsort(dCt)[:5]  # most negative
    rows = []
    for idx in top_ct:
        lab = interpret_turn(int(idx), Ct_base, C_inv, cheat)
        rows.append({
            "turn": int(idx) + 1,
            "dCt": float(dCt[idx]),
            "label": lab
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ============================
# IC-III => IC-II (driver/lag)
# ============================
ic3 = compute_ic3_geometry(E=E)

rho_t = semantic_compactness_rho(E, texts, w=2, mode="centroid", min_tokens=3)
di_n = _norm01(ic3["d_i"])
kappa_n = _norm01(ic3["kappa_i"])

D_t = manifold_driver_D(di=di_n, kappa=kappa_n, rho=rho_t, w_d=0.45, w_k=0.35, w_r=0.20, gating=True)

lag_info = estimate_lag_delta(D=D_t, C=Ct_base, phi_low=phi_low_eff, delta_max=6, smooth_alpha=0.30)
delta_star = int(round(float(lag_info.get("delta_star", 0.0)))) if np.isfinite(lag_info.get("delta_star", np.nan)) else 0
lag_score = lag_info.get("score", float("nan"))

geom_breaks_struct = detect_geom_breaks(D=D_t, kappa=kappa_n, D_hi=0.70, dD_hi=0.12, refractory=2)

observed_lag_mean = np.nan

lag_label = f"Estimated structural lag Δ* = {delta_star} turns (corr = {('n/a' if not np.isfinite(lag_score) else f'{lag_score:.3f}')})"

df_out = df.copy()
df_out["Ct"] = Ct_base
df_out["Ct_im"] = Ct_im
df_out["P_t"] = P_t
df_out["sbr"] = sbr_fixed
df_out["rho_t"] = rho_t
df_out["D_t"] = D_t
df_out["delta_star"] = float(delta_star)
df_out["geom_break_struct"] = np.isin(np.arange(len(df_out)), np.array(geom_breaks_struct, int)).astype(int)
df_out["perceived_break"] = 0
df_out["observed_lag_mean"] = float(observed_lag_mean) if np.isfinite(observed_lag_mean) else np.nan

if C_inv is not None and np.asarray(C_inv).size == len(df_out):
    df_out["C_inv"] = np.asarray(C_inv, float)
else:
    df_out["C_inv"] = np.nan

# -------------------------------------#
# Ct vs C_inv quadrants (event typing)
# -------------------------------------#
if C_inv is not None and np.asarray(C_inv).size == len(Ct_base):
    Ct0 = np.asarray(Ct_base, float)
    Ci0 = np.asarray(C_inv, float)

    # only define deltas where C_inv exists (after W-1)
    Delta_Ct = 1.0 - Ct0
    Delta_Cinv = 1.0 - Ci0

    thr_sem = float(np.nanquantile(Delta_Ct, 0.85))
    thr_inv = float(np.nanquantile(Delta_Cinv, 0.85))

    labels = []
    for t in range(len(Ct0)):
        if not np.isfinite(Ci0[t]):
            labels.append("WARMUP")
            continue
        if (Delta_Ct[t] > thr_sem) and (Delta_Cinv[t] > thr_inv):
            labels.append("RUPTURE_STRONG")
        elif (Delta_Ct[t] > thr_sem):
            labels.append("RUPTURE_SEM")
        elif (Delta_Cinv[t] > thr_inv):
            labels.append("RUPTURE_STRUCT")
        else:
            labels.append("STABLE")

    df_out["rupture_type"] = labels

with st.expander("Ct vs C_inv — rupture typing (quadrants)", expanded=False):
     st.dataframe(df_out[["turn","participant","Ct","C_inv","rupture_type","text"]], use_container_width=True)

ci_df = compute_ci_series(E=E, participants=participants, method=ci_method, alpha=float(ci_alpha))
df_out = pd.concat([df_out, ci_df], axis=1)

with st.expander(L["table_title"], expanded=False):
    st.dataframe(df_out, use_container_width=True)

fig_main = plot_ct_main(
    Ct=Ct_base,
    participants=participants,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    valleys=valleys,
    peaks=peaks,
    title=L["overview"],
    height=560,
    potentiality=P_t,
    pilot_w=int(pilot_w),
    C_inv=(C_inv if (overlay_cinv_on_main and C_inv is not None) else None),
    lag_label=lag_label,
         sbr_labels=sbr_fixed,  
)

st.plotly_chart(fig_main, use_container_width=True)

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
        tau_t=ic3["tau_t"],
        phi_low=float(phi_low_eff),
        phi_high=float(phi_high_eff),
        height=600,
    )
    st.plotly_chart(fig_g, use_container_width=True)

with st.expander(L["ci_title"], expanded=False):
    if ci_df.shape[1] == 0:
        st.info("No Ci columns available (check participants).")
    else:
        fig_ci = plot_ci_lines(
            turns=turns,
            ci_df=ci_df,
            phi_low=float(phi_low_eff),
            phi_high=float(phi_high_eff),
            height=520,
            title=L["ci_title"],
        )
        st.plotly_chart(fig_ci, use_container_width=True)

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
    "tau_t": np.asarray(ic3["tau_t"], float),
    "tau_norm": np.asarray(ic3["tau_norm"], float),
    "rho_t": np.asarray(rho_t, float),
    "D_t": np.asarray(D_t, float),
    "delta_star": float(delta_star),
    "lag_score": float(lag_score) if np.isfinite(lag_score) else np.nan,
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
    valleys=valleys,
    peaks=peaks,
    geom_breaks_struct=geom_breaks_struct,
    lag_label=lag_label,
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
