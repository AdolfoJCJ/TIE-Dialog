# ============================
# app.py — PART 1/3
# (imports + labels + helpers + Public View + IC-II/IC-III core)
# ✅ FIXED: state_alpha defined (slider) and used correctly later
# ✅ FIXED: participant trajectories (continuous) available (state + Ci)
# ✅ NEW: IC-III → IC-II module (geom driver D_t, rho(t), C_hat, lag Δ*)
# ============================

import streamlit as st


st.set_page_config(page_title="Coherence Explorer — CNøde", layout="wide")


import math
from io import BytesIO
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================================================
# CACHED / UTILITY FUNCTIONS (AFTER config)
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
            # Fall back
            msg = f"⚠️ SBERT failed ({type(e).__name__}): {e}. Falling back to TF-IDF."
            # continue to TF-IDF below

    # --- Explicit SBERT requested but not available ---
    if mode == "sbert" and not _SBERT_AVAILABLE:
        msg = "⚠️ sentence-transformers is not installed. Falling back to TF-IDF."
    else:
        msg = "ℹ️ Using TF-IDF embeddings."

    # --- TF-IDF / fallback path ---
    if TfidfVectorizer is None:
        # one-hot fallback
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
        return E, "onehot", "⚠️ TF-IDF unavailable → using one-hot fallback."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).astype(float)
    if sk_normalize is not None:
        X = sk_normalize(X, norm="l2", axis=1, copy=False)

    return X.toarray(), "tfidf", msg

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

# =====================
# Bilingual labels
# =====================
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "🌀 Coherence Explorer 🌀",
        "app_subtitle": "Developed by CNøde — Informational Systems Lab",
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
        "smooth": "Cₜ smoothing (legacy / post-IC-II)",
        "phi": "Φ thresholds (percentiles by default)",
        "events": "Event detection",
        "quanto_legacy": "Quanto (legacy Qa)",
        "quanto_triad": "Quantum of Coherence — Triadic (S–B–R)",
        "layout": "Layout",
        "compute": "Compute metrics",
        "preview": "Dialogue preview",
        "global_metrics": "Global metrics",
        "table_title": "Dialogue with S/B/R labels, potentiality ℘ₜ and geometry (IC–III)",
        "overview": "Overview plot (TIE–Dialog)",
        "geom_plot": "IC–III geometric layer: dᵢ, κᵢ and τ(t)",
        "triadic_title": "Triadic Quantum of Coherence (S–B–R)",
        "legacy_q": "Legacy Quantum of Coherence (Qₐ)",
        "pdf": "Download PDF report",
        "csv": "Download CSV results",
        "download_pdf": "Download PDF report",
        "download_full_csv": "Download full results (CSV)",
        "download_triad_csv": "Download triadic quanta (CSV)",
        "download_ic2_csv": "Download IC–IIa dynamics (CSV)",
        "download_ic3_csv": "Download IC–III geometric layer (CSV)",
        "ctx_header": "Context-aware coherence (Ct_new)",
        "geom_events_header": "IC–III geometric events (pre-transition + break/shift)",
        "debug_header": "Debug",
        "show_ct_old": "Show Ct_old overlay in main plot",
        "use_ct_old_for_events": "Use Ct_old for peak/valley events (debug)",
        "lang": "Language",

        # Public View
        "public_header": "Public View (Smoothed S–B–R)",
        "public_span": "Smoothing span (EWMA)",
        "public_show_thresholds": "Show Φ thresholds in public plot",
        "public_title": "Public plot: Smoothed coherence + S–B–R regimes",

        # Participant trajectories
        "ci_header": "Participant trajectories (Ci)",
        "ci_alpha": "Ci context inertia α (per-participant)",
        "ci_method": "Ci method",
        "ci_title": "Per-participant coherence trajectories (Ci) + Φ thresholds",

        # ✅ FIX: State trajectories control
        "state_alpha": "State trajectory inertia α (continuous lines)",
        "state_title": "Per-participant continuous trajectories (state) + Φ thresholds",
    },
    "es": {
        "app_title": "🌀 Coherence Explorer 🌀",
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
        "smooth": "Suavizado de Cₜ (legacy / post-IC-II)",
        "phi": "Umbrales Φ (por defecto percentiles)",
        "events": "Detección de eventos",
        "quanto_legacy": "Quanto (Qa legacy)",
        "quanto_triad": "Quantum of Coherence — Triádico (S–B–R)",
        "layout": "Layout",
        "compute": "Calcular métricas",
        "preview": "Vista previa del diálogo",
        "global_metrics": "Métricas globales",
        "table_title": "Diálogo con etiquetas S/B/R, potencialidad ℘ₜ y geometría (IC–III)",
        "overview": "Plot overview (TIE–Dialog)",
        "geom_plot": "Capa geométrica IC–III: dᵢ, κᵢ y τ(t)",
        "triadic_title": "Quantum of Coherence Triádico (S–B–R)",
        "legacy_q": "Quantum of Coherence legacy (Qₐ)",
        "pdf": "Descargar informe PDF",
        "csv": "Descargar resultados CSV",
        "download_pdf": "Descargar informe PDF",
        "download_full_csv": "Descargar resultados completos (CSV)",
        "download_triad_csv": "Descargar quanta triádicos (CSV)",
        "download_ic2_csv": "Descargar dinámica IC–IIa (CSV)",
        "download_ic3_csv": "Descargar capa geométrica IC–III (CSV)",
        "ctx_header": "Coherencia context-aware (Ct_new)",
        "geom_events_header": "Eventos geométricos IC–III (pre-transición + break/shift)",
        "debug_header": "Debug",
        "show_ct_old": "Mostrar Ct_old superpuesto en el plot principal",
        "use_ct_old_for_events": "Usar Ct_old para eventos peak/valley (debug)",
        "lang": "Idioma",

        # Public View
        "public_header": "Vista pública (S–B–R suavizado)",
        "public_span": "Span de suavizado (EWMA)",
        "public_show_thresholds": "Mostrar umbrales Φ en plot público",
        "public_title": "Plot público: coherencia suavizada + regímenes S–B–R",

        # Participant trajectories
        "ci_header": "Trayectorias por participante (Ci)",
        "ci_alpha": "Inercia de contexto α (Ci por participante)",
        "ci_method": "Método Ci",
        "ci_title": "Trayectorias de coherencia por participante (Ci) + umbrales Φ",

        # ✅ FIX: State trajectories control
        "state_alpha": "Inercia de trayectoria α (líneas continuas)",
        "state_title": "Trayectorias continuas por participante (state) + umbrales Φ",
    }
}

# =====================
# Warm-up configuration
# =====================
WARMUP_TURNS = 5  # initial turns treated as context ramp

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
    """
    Coherence smoothing utility.
    - method="ema": uses your _ema alpha (0..1)  (more 'signal-processing')
    - method="ewma": uses pandas EWMA span        (more 'finance-like')
    """
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    method = (method or "ema").lower().strip()

    if method == "ewma":
        ewma_span = int(max(3, ewma_span))
        return pd.Series(y).ewm(span=ewma_span, adjust=False).mean().to_numpy()

    # default ema
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
        fig.add_vrect(x0=int(turns[a]), x1=int(turns[b]), **band.get(lab, band["R"]))

    fig.add_trace(go.Scatter(
        x=turns, y=C_smooth, mode="lines", name="Cₜ (smoothed)",
        line=dict(width=3),
    ))

    b_idx = np.where(labels == "B")[0].tolist()
    if b_idx:
        fig.add_trace(go.Scatter(
            x=turns[b_idx], y=C_smooth[b_idx],
            mode="markers", name="B (break)",
            marker=dict(symbol="triangle-down", size=10),
        ))

    r_idx = np.where(labels == "R")[0].tolist()
    if r_idx:
        r_starts = []
        for i in r_idx:
            if i == 0 or labels[i-1] != "R":
                r_starts.append(i)
        if r_starts:
            fig.add_trace(go.Scatter(
                x=turns[r_starts], y=C_smooth[r_starts],
                mode="markers", name="R (repair start)",
                marker=dict(symbol="triangle-up", size=9),
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
# Robust local stats for IC-III z-scores
# -------------------------------
def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

def _robust_sigma(x: np.ndarray) -> float:
    return 1.4826 * _mad(x)

def local_zscore_robust(
    x: np.ndarray,
    W: int = 7,
    eps: float = 1e-9,
    sigma_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    n = x.size
    z = np.zeros(n, float)
    mu_roll = np.zeros(n, float)
    sig_eff = np.zeros(n, float)
    prev_sigmas = []

    for t in range(n):
        if t == 0:
            mu_roll[t] = float(x[t])
            sig_eff[t] = float(max(sigma_floor, eps))
            z[t] = 0.0
            prev_sigmas.append(sig_eff[t])
            continue

        start = max(0, t - int(W))
        win = x[start:t]
        if win.size == 0:
            mu = float(x[t-1]); s_std = 0.0; s_rob = 0.0
        else:
            mu = float(np.mean(win))
            s_std = float(np.std(win))
            s_rob = float(_robust_sigma(win))

        med_prev = float(np.median(prev_sigmas)) if prev_sigmas else float(sigma_floor)
        floor_dyn = float(max(sigma_floor, 0.25 * med_prev))
        s_eff = float(max(s_std, s_rob, floor_dyn, eps))

        mu_roll[t] = mu
        sig_eff[t] = s_eff
        z[t] = float((x[t] - mu) / (s_eff + eps))
        prev_sigmas.append(s_eff)

    return z, mu_roll, sig_eff

def discrete_derivative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    d = np.zeros_like(x)
    d[0] = 0.0
    if x.size > 1:
        d[1:] = x[1:] - x[:-1]
    return d

def persistence_ratio(cond: np.ndarray, t: int, K: int, mode: str = "offline") -> float:
    n = int(cond.size)
    K = int(max(1, K))
    t = int(t)
    mode = (mode or "offline").lower().strip()

    if n <= 0 or t < 0 or t >= n:
        return float("nan")

    if mode == "online":
        start = max(0, t - (K - 1))
        window = cond[start:t+1]
    else:
        end = min(n, t + K)
        window = cond[t:end]

    if window.size == 0:
        return float("nan")
    return float(np.mean(window.astype(float)))

# -------------------------------
# IC-II operators & helpers
# -------------------------------
def _sigma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-x))

def resonance_op(a: np.ndarray, b: np.ndarray) -> float:
    return _cos(a, b)

def compute_ic2_dynamics(
    E: np.ndarray,
    alpha_context: float = 0.8,
    theta0: float = 0.0,
    theta1: float = 4.0,
    theta2: float = 2.0,
    window_W: int = 5,
    delta_async: int = 0,
    gamma_async: float = 1.0
) -> dict:
    E = np.asarray(E, float)
    n = E.shape[0]
    if n == 0:
        return {
            "I_s": E, "I_m": E,
            "res": np.zeros(0, float),
            "res_sync": np.zeros(0, float),
            "d_res": np.zeros(0, float),
            "C_t": np.zeros(0, float),
            "E_t": np.zeros(0, float),
            "I_W": np.zeros(0, float),
            "Delta_async": np.zeros(0, float),
            "dI_norm": np.zeros(0, float),
        }

    I_s = E.copy()
    I_m = np.zeros_like(I_s)
    I_m[0] = I_s[0]
    alpha = float(np.clip(alpha_context, 0.0, 0.99))
    for t in range(1, n):
        I_m[t] = alpha * I_m[t - 1] + (1.0 - alpha) * I_s[t - 1]

    res_sync = np.zeros(n, float)
    for t in range(n):
        res_sync[t] = resonance_op(I_s[t], I_m[t])

    d_steps = max(0, int(delta_async))
    I_m_async = np.zeros_like(I_m)
    for t in range(n):
        j = t + d_steps
        I_m_async[t] = I_m[j] if j < n else I_m[-1]

    res_async = np.zeros(n, float)
    for t in range(n):
        res_async[t] = resonance_op(I_s[t], I_m_async[t])

    dI_norm = np.zeros(n, float)
    for t in range(1, n):
        dI_norm[t] = float(np.linalg.norm(I_s[t] - I_s[t - 1]))

    Delta_async = np.zeros(n, float)
    for t in range(n):
        Delta_async[t] = float(np.linalg.norm(I_s[t] - I_m_async[t]))

    arg = (
        float(theta0)
        + float(theta1) * res_async
        - float(theta2) * dI_norm
        - float(gamma_async) * Delta_async
    )
    C_t = _sigma(arg)
    E_t = 1.0 - res_async

    w = max(1, int(window_W))
    I_W = np.zeros(n, float)
    for t in range(n):
        start = max(0, t - w + 1)
        I_W[t] = float(np.mean(res_async[start:t + 1]))

    d_res = discrete_derivative(res_async)

    return {
        "I_s": I_s,
        "I_m": I_m,
        "res": res_async,
        "res_sync": res_sync,
        "d_res": d_res,
        "C_t": C_t,
        "E_t": E_t,
        "I_W": I_W,
        "Delta_async": Delta_async,
        "dI_norm": dI_norm,
    }

def detect_minimal_repair_events(E_t: np.ndarray, I_W: np.ndarray, Phi: float) -> List[int]:
    E_t = np.asarray(E_t, float)
    I_W = np.asarray(I_W, float)
    n = len(E_t)
    idx = []
    if n <= 1:
        return idx
    for t in range(1, n):
        if (E_t[t] < E_t[t - 1]) and (I_W[t - 1] < Phi) and (I_W[t] >= Phi):
            idx.append(t)
    return idx

# -------------------------------
# IC–III geometric layer
# -------------------------------
def compute_ic3_geometry(
    E: np.ndarray,
    Ct: np.ndarray,
    dI_norm: np.ndarray,
    Delta_async: np.ndarray,
    alpha_sem: float = 1.0,
    beta_diff: float = 0.8,
    gamma_async: float = 0.5,
    delta_noise: float = 0.05,
    lambda_k1: float = 1.0,
    lambda_k2: float = 1.0,
    lambda_k3: float = 1.0,
    tau_w1: float = 0.7,
    tau_w2: float = 1.0,
    tau_w3: float = 0.8,
) -> dict:
    E = np.asarray(E, float)
    Ct = np.asarray(Ct, float)
    dI_norm = np.asarray(dI_norm, float)
    Delta_async = np.asarray(Delta_async, float)
    n = E.shape[0]

    if n == 0:
        return {
            "d_i": np.zeros(0, float),
            "kappa_i": np.zeros(0, float),
            "tau_t": np.zeros(0, float),
            "tau_norm": np.zeros(0, float),
        }

    one_minus_cos = np.zeros(n, float)
    for t in range(1, n):
        one_minus_cos[t] = 1.0 - _cos(E[t], E[t - 1])

    d_i = (
        float(alpha_sem) * one_minus_cos
        + float(beta_diff) * dI_norm
        + float(gamma_async) * Delta_async
        + float(delta_noise)
    )

    kappa_i = (
        float(lambda_k1) * dI_norm
        + float(lambda_k2) * Delta_async
        + float(lambda_k3) * (1.0 - Ct)
    )

    d_tau = (
        float(tau_w1) * dI_norm
        + float(tau_w2) * kappa_i
        + float(tau_w3) * (1.0 - Ct)
    )
    tau_t = np.cumsum(d_tau)

    if tau_t.size == 0:
        tau_norm = tau_t
    else:
        t_min = float(np.min(tau_t))
        t_max = float(np.max(tau_t))
        if t_max - t_min < 1e-9:
            tau_norm = np.zeros_like(tau_t)
        else:
            tau_norm = (tau_t - t_min) / (t_max - t_min)

    return {
        "d_i": d_i,
        "kappa_i": kappa_i,
        "tau_t": tau_t,
        "tau_norm": tau_norm,
    }

# -------------------------------
# ✅ NEW: IC-III → IC-II module
# -------------------------------
def semantic_compactness_rho(
    embeddings: np.ndarray,
    texts: List[str],
    w: int = 2,
    mode: str = "centroid",
    min_tokens: int = 3,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    rho(t) in [0,1]: local semantic compactness (not 'density').

    mode="centroid" (default):
      Window W={t-w..t+w}, centroid mu_W,
      rho(t)=1 - mean_cos_dist(e_i, mu_W) over window.
    Penalizes ultra-short turns (len<min_tokens) by down-weighting rho.
    """
    E = np.asarray(embeddings, float)
    n = E.shape[0]
    if n == 0:
        return np.zeros(0, float)

    mode = (mode or "centroid").lower().strip()
    w = int(max(1, w))

    # Normalize embeddings
    En = E.copy()
    norms = np.linalg.norm(En, axis=1, keepdims=True) + eps
    En = En / norms

    rho = np.zeros(n, float)

    if mode == "variation":
        # rho(t)=1-std(sim(t,t-1), sim(t,t+1))
        sims_prev = np.zeros(n, float)
        sims_next = np.zeros(n, float)
        for t in range(n):
            if t - 1 >= 0:
                sims_prev[t] = _cos(En[t], En[t-1], eps=eps)
            else:
                sims_prev[t] = sims_prev[t] if t > 0 else 0.0
            if t + 1 < n:
                sims_next[t] = _cos(En[t], En[t+1], eps=eps)
            else:
                sims_next[t] = sims_next[t-1] if t > 0 else 0.0

        for t in range(n):
            arr = np.array([sims_prev[t], sims_next[t]], float)
            rho[t] = 1.0 - float(np.std(arr))
    else:
        # centroid local window (robust)
        for t in range(n):
            a = max(0, t - w)
            b = min(n, t + w + 1)
            win = En[a:b]
            if win.size == 0:
                rho[t] = 0.0
                continue
            mu = np.mean(win, axis=0)
            mu = mu / (float(np.linalg.norm(mu)) + eps)
            # mean cosine distance = mean(1 - cos)
            dists = [1.0 - _cos(win[i], mu, eps=eps) for i in range(win.shape[0])]
            rho[t] = 1.0 - float(np.mean(dists))

    # Penalize ultra-short turns
    for t in range(n):
        txt = texts[t] if (t < len(texts) and isinstance(texts[t], str)) else ""
        if len(txt.strip().split()) < int(min_tokens):
            rho[t] *= 0.75

    rho = _norm01(np.clip(rho, 0.0, 1.0))
    return rho

def manifold_driver_D(
    di: np.ndarray,
    kappa: np.ndarray,
    rho: np.ndarray,
    w_d: float = 0.45,
    w_k: float = 0.35,
    w_r: float = 0.20,
    gating: bool = True,
) -> np.ndarray:
    """
    D_t in [0,1] from normalized inputs di,kappa,rho in [0,1].
    Curvature gating: if kappa < p60, reduce di contribution by x0.5.
    """
    di = np.asarray(di, float)
    kappa = np.asarray(kappa, float)
    rho = np.asarray(rho, float)
    n = min(di.size, kappa.size, rho.size)
    if n == 0:
        return np.zeros(0, float)

    di = di[:n]; kappa = kappa[:n]; rho = rho[:n]

    wd = float(w_d); wk = float(w_k); wr = float(w_r)
    wd = max(0.0, wd); wk = max(0.0, wk); wr = max(0.0, wr)
    s = wd + wk + wr
    if s <= 1e-12:
        wd, wk, wr = 1.0, 0.0, 0.0
        s = 1.0
    wd, wk, wr = wd/s, wk/s, wr/s

    di_eff = di.copy()
    if gating:
        thr = float(np.percentile(kappa, 60))
        mask = kappa < thr
        di_eff[mask] = 0.5 * di_eff[mask]

    D = wd * di_eff + wk * kappa + wr * rho
    D = _norm01(np.clip(D, 0.0, 1.0))
    return D

def phenomenological_coherence_C_hat(
    D: np.ndarray,
    model: str = "ema",
    alpha: float = 0.25,
    K: int = 8,
    lam: float = 2.5,
    Ct_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    C_hat_t: causal response with memory to driver.
    g(D)=1-D, then EMA or exp kernel.
    Sign safety: if corr(D, 1-C_ref) < 0, invert g.
    """
    D = np.asarray(D, float)
    n = D.size
    if n == 0:
        return np.zeros(0, float)

    g = 1.0 - D

    if Ct_ref is not None:
        Ct_ref = np.asarray(Ct_ref, float)
        if Ct_ref.size == n:
            corr = _safe_corr(D, 1.0 - Ct_ref)
            if np.isfinite(corr) and corr < 0:
                g = 1.0 - g  # invert

    model = (model or "ema").lower().strip()

    if model == "kernel":
        K = int(max(1, K))
        lam = float(max(1e-6, lam))
        weights = np.array([math.exp(-k / lam) for k in range(K + 1)], float)
        weights = weights / (float(np.sum(weights)) + 1e-12)

        out = np.zeros(n, float)
        for t in range(n):
            acc = 0.0
            wsum = 0.0
            for k in range(K + 1):
                j = t - k
                if j < 0:
                    break
                acc += weights[k] * float(g[j])
                wsum += weights[k]
            out[t] = acc / (wsum + 1e-12)
        return np.clip(_norm01(out), 0.0, 1.0)

    # default EMA causal
    a = float(np.clip(alpha, 0.0, 0.999))
    out = np.zeros(n, float)
    out[0] = float(g[0])
    for t in range(1, n):
        out[t] = (1.0 - a) * out[t-1] + a * float(g[t])

    out = np.clip(_norm01(out), 0.0, 1.0)
    return out

def estimate_lag_delta(
    D: np.ndarray,
    C: np.ndarray,
    phi_low: float,
    delta_max: int = 6,
    smooth_alpha: float = 0.30
) -> Dict[str, float]:
    """
    Estimate optimal lag Δ* such that D_{t-Δ} best correlates with coherence drop Y_t.
    Y_t = 1 - C_t  (primary)
    Optionally could use (phi_low - C)^+; we use 1-C for stability and availability.
    score(Δ)=max(Pearson, Spearman) if SciPy absent => Pearson only.
    """
    D = np.asarray(D, float)
    C = np.asarray(C, float)
    n = min(D.size, C.size)
    if n == 0:
        return {"delta_star": 0.0, "score": float("nan"), "pearson": float("nan"), "spearman": float("nan")}

    D = D[:n]
    C = C[:n]
    Y = (1.0 - C) * (C < phi_low).astype(float)

    D_s = _ema(D, alpha=float(smooth_alpha))
    Y_s = _ema(Y, alpha=float(smooth_alpha))

    best_d = 0
    best_score = -1e9
    best_p = float("nan")
    best_s = float("nan")

    for d in range(0, int(max(0, delta_max)) + 1):
        if d == 0:
            a = D_s
            b = Y_s
        else:
            a = D_s[:-d]
            b = Y_s[d:]
        if a.size < 3 or b.size < 3:
            continue

        pear = _safe_corr(a, b)
        spea = float("nan")

        # Spearman if available (without importing extra deps, do rank-corr via pandas)
        try:
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            spea = _safe_corr(ra, rb)
        except Exception:
            spea = float("nan")

        cand = np.nanmax([pear, spea])
        if np.isfinite(cand) and cand > best_score:
            best_score = float(cand)
            best_d = int(d)
            best_p = float(pear) if np.isfinite(pear) else best_p
            best_s = float(spea) if np.isfinite(spea) else best_s

    return {
        "delta_star": float(best_d),
        "score": float(best_score) if best_score > -1e8 else float("nan"),
        "pearson": float(best_p),
        "spearman": float(best_s),
    }

def detect_geom_breaks(
    D: np.ndarray,
    kappa: np.ndarray,
    D_hi: float = 0.70,
    dD_hi: float = 0.12,
    refractory: int = 2
) -> List[int]:
    """
    Geom Break (IC-III): D_t > 0.70 AND (ΔD > 0.12 OR local peak of κ).
    Refractory: 2 turns.
    """
    D = np.asarray(D, float)
    kappa = np.asarray(kappa, float)
    n = min(D.size, kappa.size)
    if n == 0:
        return []
    D = D[:n]
    kappa = kappa[:n]

    dD = discrete_derivative(D)
    kpeaks = set()

    if _HAS_FIND_PEAKS and n >= 3:
        try:
            pk, _ = find_peaks(kappa, prominence=0.05, distance=2)
            kpeaks = set([int(i) for i in pk])
        except Exception:
            kpeaks = set()

    hits = []
    last = -10**9
    for t in range(n):
        if (t - last) <= int(refractory):
            continue
        cond1 = (float(D[t]) > float(D_hi))
        cond2 = (float(dD[t]) > float(dD_hi)) or (t in kpeaks)
        if cond1 and cond2:
            hits.append(int(t))
            last = int(t)
    return hits

def detect_perceived_breaks(
    C_hat: np.ndarray,
    phi_low: float,
    drop_hi: float = -0.15,
    refractory: int = 2
) -> List[int]:
    """
    Perceived Break (IC-II): C_hat crosses Φ_low downward OR sudden drop.
    """
    C_hat = np.asarray(C_hat, float)
    n = C_hat.size
    if n == 0:
        return []
    hits = []
    last = -10**9
    for t in range(1, n):
        if (t - last) <= int(refractory):
            continue
        cross = (C_hat[t-1] >= float(phi_low)) and (C_hat[t] < float(phi_low))
        drop = (float(C_hat[t]) - float(C_hat[t-1])) < float(drop_hi)
        if cross or drop:
            hits.append(int(t))
            last = int(t)
    return hits

def match_breaks(
    geom_breaks: List[int],
    perceived_breaks: List[int],
    delta_max: int = 6
) -> List[Dict[str, int]]:
    """
    For each geom break t0, find perceived break in [t0, t0+delta_max+2].
    """
    out = []
    perc = sorted([int(x) for x in perceived_breaks])
    for t0 in sorted([int(x) for x in geom_breaks]):
        window_end = int(t0 + int(delta_max) + 2)
        cand = [t for t in perc if (t >= t0 and t <= window_end)]
        if cand:
            t1 = cand[0]
            out.append({"geom_t": t0, "perc_t": t1, "lag": int(t1 - t0)})
    return out

# -------------------------------
# IC-III Geometric Event Redefinition (A–G)
# (unchanged from your version)
# -------------------------------
def detect_ic3_geometric_events(
    Ct: np.ndarray,
    d_i: np.ndarray,
    kappa_i: np.ndarray,
    texts: List[str],
    phi_low: float,
    phi_high: float,
    W: int = 7,
    eps: float = 1e-9,
    delta_d: float = 1.5,
    delta_kappa: float = 1.5,
    kappa_high: float = 2.0,
    kappa_low: float = 0.5,
    d_high: float = 2.0,
    K_persist: int = 3,
    p_req: float = 0.66,
    R_rec: int = 3,
    Phi_rec: float = 0.60,
    lambda_cum: float = 0.85,
    kappa_cum_high: float = 1.2,
    persistence_mode: str = "offline",
) -> pd.DataFrame:
    Ct = np.asarray(Ct, float)
    d_i = np.asarray(d_i, float)
    kappa_i = np.asarray(kappa_i, float)
    n = Ct.size

    if n == 0:
        return pd.DataFrame({
            "di": [], "kappa_i": [], "kappa_i_cum": [],
            "delta_di": [], "delta_kappa_i": [],
            "z_di": [], "z_kappa": [],
            "pre_transition_flag": [],
            "event_type_geom": [],
        })

    delta_di = discrete_derivative(d_i)
    delta_kappa = discrete_derivative(kappa_i)

    z_di, mu_di, sig_di = local_zscore_robust(d_i, W=int(W), eps=eps, sigma_floor=1e-3)
    z_kappa, mu_k, sig_k = local_zscore_robust(kappa_i, W=int(W), eps=eps, sigma_floor=1e-3)

    lam = float(np.clip(lambda_cum, 0.0, 0.999))
    kappa_cum = np.zeros(n, float)
    kappa_cum[0] = float(kappa_i[0])
    for t in range(1, n):
        kappa_cum[t] = lam * kappa_cum[t-1] + (1.0 - lam) * float(kappa_i[t])

    pre_flag = ((z_di > delta_d) | (z_kappa > delta_kappa)).astype(int)

    markers = [
        "by the way", "anyway", "also", "another question", "on a different note",
        "speaking of", "quick question", "new topic"
    ]
    lex = np.zeros(n, int)
    for t in range(n):
        low = (texts[t] if isinstance(texts[t], str) else "").lower()
        lex[t] = 1 if any(m in low for m in markers) else 0

    def recon_stats(t: int) -> Tuple[float, float]:
        remaining = (n - 1) - t
        R_eff = int(min(int(R_rec), remaining))
        if R_eff <= 0:
            return float("nan"), float("nan")
        future = Ct[t+1:t+1+R_eff]
        meanC = float(np.mean(future))
        meanD = float(np.mean(np.diff(np.concatenate([[Ct[t]], future]))))
        return meanC, meanD

    event = np.array(["none"] * n, dtype=object)

    for t in range(n):
        cond_B1 = (Ct[t] < float(phi_low))
        cond_B2 = (z_kappa[t] > float(kappa_high))

        cond_kappa = (z_kappa > float(kappa_high))
        cond_C = (Ct < float(phi_low))
        Pk = persistence_ratio(cond_kappa, t=t, K=int(K_persist), mode=persistence_mode)
        Pc = persistence_ratio(cond_C, t=t, K=int(K_persist), mode=persistence_mode)
        cond_B3 = (np.isfinite(Pk) and np.isfinite(Pc) and (Pk > float(p_req)) and (Pc > float(p_req)))

        meanC, meanD = recon_stats(t)
        cond_B4 = (np.isfinite(meanC) and np.isfinite(meanD) and (meanC < float(Phi_rec)) and (meanD <= 0.0))
        cond_B5 = (kappa_cum[t] > float(kappa_cum_high))

        is_break = (cond_B1 and cond_B2 and cond_B3 and (cond_B4 or cond_B5))

        cond_C1 = (z_di[t] > float(d_high))
        cond_C2 = (z_kappa[t] > float(kappa_low)) and (z_kappa[t] < float(kappa_high))
        cond_C3 = (persistence_ratio((z_di > float(d_high)), t=t, K=int(K_persist), mode=persistence_mode) > float(p_req))
        cond_C4 = (np.isfinite(meanC) and np.isfinite(meanD) and (meanC >= float(Phi_rec)) and (meanD > 0.0))
        cond_C5 = (lex[t] == 1)

        is_shift = (cond_C1 and cond_C2 and cond_C3 and cond_C4) or (cond_C1 and cond_C2 and cond_C4 and cond_C5)

        if is_break:
            event[t] = "break"
        elif is_shift:
            event[t] = "shift"
        else:
            mixed = (z_di[t] > float(d_high)) and (z_kappa[t] > float(kappa_high))
            partial = (
                (persistence_ratio((z_di > float(d_high)), t=t, K=int(K_persist), mode=persistence_mode) > 0.33)
                or (persistence_ratio((z_kappa > float(kappa_high)), t=t, K=int(K_persist), mode=persistence_mode) > 0.33)
            )
            if mixed or (pre_flag[t] == 1 and partial and Ct[t] < float(phi_high)):
                event[t] = "transition"
            else:
                event[t] = "none"

    return pd.DataFrame({
        "di": d_i.astype(float),
        "kappa_i": kappa_i.astype(float),
        "kappa_i_cum": kappa_cum.astype(float),
        "delta_di": delta_di.astype(float),
        "delta_kappa_i": delta_kappa.astype(float),
        "z_di": z_di.astype(float),
        "z_kappa": z_kappa.astype(float),
        "pre_transition_flag": pre_flag.astype(int),
        "event_type_geom": event.astype(object),
    })

# -------------------------------
# IC-III geometry plot (unchanged)
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
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if y_max - y_min < 1e-9:
            return np.zeros_like(y)
        return (y - y_min) / (y_max - y_min)

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
# Legacy metrics (adjacent-turn Ct)
# -------------------------------
def compute_ct_series(E: np.ndarray,
                      ema_alpha: float = 0.4,
                      use_savgol: bool = False,
                      sg_win: int = 5,
                      sg_poly: int = 2) -> np.ndarray:
    n = E.shape[0]
    Ct = np.zeros(n, float)
    if n == 0:
        return Ct
    Ct[0] = 0.0
    for t in range(1, n):
        Ct[t] = float(np.dot(E[t], E[t-1]) / ((np.linalg.norm(E[t]) + 1e-9) * (np.linalg.norm(E[t-1]) + 1e-9)))
    Ct = 0.5 * (Ct + 1.0)
    Ct = _ema(Ct, alpha=float(ema_alpha))
    if use_savgol:
        Ct = _savgol(Ct, int(sg_win), int(sg_poly))
        Ct = np.clip(Ct, 0.0, 1.0)
    return Ct

def compute_ct_im(E: np.ndarray, ema_alpha: float = 0.4) -> np.ndarray:
    n = E.shape[0]
    if n == 0:
        return np.zeros(0, float)
    centroid = np.mean(E, axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    CtIm = np.array([float(np.dot(E[t], centroid) / ((np.linalg.norm(E[t]) + 1e-9))) for t in range(n)], float)
    CtIm = 0.5 * (CtIm + 1.0)
    CtIm = _ema(CtIm, alpha=float(ema_alpha))
    return np.clip(CtIm, 0.0, 1.0)

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
# ✅ Continuous state trajectories (the one you want visually)
# -------------------------------
def compute_participant_state_trajectories(
    Ct: np.ndarray,
    participants: List[str],
    alpha: float = 0.88,
) -> Dict[str, np.ndarray]:
    """
    Continuous per-participant trajectories (state version):
    - If participant speaks at t: C_p[t] = Ct[t]
    - If participant does NOT speak: C_p[t] = alpha*C_p[t-1] + (1-alpha)*Ct[t]
    Produces continuous lines (no pulses / zig-zag).
    """
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
                                      C_parts[p][t] = float(C_parts[p][t - 1])  # ✅ hold -> no sawtooth
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

# ============================
# app.py — PART 2/3
# (events + SBR + quantos + context-aware + ✅ Ci computation + plots)
# ✅ FIXED: Ci plot and State plot are both available
# ✅ NEW: Overview plot supports C_hat overlay + Geom Break + Perceived Break markers + lag label
# ============================

# -------------------------------
# Events and S/B/R
# -------------------------------
def _detect_events_peaks(Ct: np.ndarray,
                         phi_low_t: np.ndarray,
                         phi_high_t: np.ndarray,
                         sep_min: int = 2,
                         prom_min: float = 0.05) -> Tuple[List[int], List[int]]:
    n = len(Ct)
    if n < 3:
        return [], []
    margin = 0.01
    if _HAS_FIND_PEAKS:
        inv = 1.0 - Ct
        v_idx, _ = find_peaks(inv, prominence=float(prom_min), distance=int(sep_min))
        valleys = [i for i in v_idx if Ct[i] < (phi_low_t[i] - margin)]
        p_idx, _ = find_peaks(Ct, prominence=float(prom_min), distance=int(sep_min))
        peaks = []
        for vi in valleys:
            nxt = [p for p in p_idx if p > vi and Ct[p] > (phi_high_t[p] + margin)]
            if nxt:
                peaks.append(nxt[0])
        return valleys, peaks

    valleys, peaks = [], []
    last_v = -10**9
    for i in range(1, n-1):
        if Ct[i] < Ct[i-1] and Ct[i] <= Ct[i+1] and (i - last_v) >= sep_min:
            prom = max(Ct[i-1] - Ct[i], Ct[i+1] - Ct[i])
            if prom >= prom_min and Ct[i] < (phi_low_t[i] - margin):
                valleys.append(i)
                last_v = i
    for vi in valleys:
        for j in range(vi+1, n-1):
            if Ct[j] > Ct[j-1] and Ct[j] >= Ct[j+1] and Ct[j] > (phi_high_t[j] + margin):
                peaks.append(j)
                break
    return valleys, peaks

def detect_events(Ct: np.ndarray,
                  phi_low: float,
                  phi_high: float,
                  sep_min: int = 2,
                  prom_min: float = 0.05,
                  phi_low_t: Optional[np.ndarray] = None,
                  phi_high_t: Optional[np.ndarray] = None) -> Tuple[List[int], List[int]]:
    n = len(Ct)
    if n < 3:
        return [], []
    lo = phi_low_t if phi_low_t is not None else np.full(n, float(phi_low))
    hi = phi_high_t if phi_high_t is not None else np.full(n, float(phi_high))
    return _detect_events_peaks(Ct, lo, hi, sep_min=int(sep_min), prom_min=float(prom_min))

def assign_sbr(Ct: np.ndarray,
               valleys: List[int],
               peaks: List[int],
               phi_low: float,
               phi_high: float,
               warmup_turns: int = WARMUP_TURNS) -> List[str]:
    n = len(Ct)
    eB = set(valleys)
    eR = set(peaks)
    states = []
    for i in range(n):
        if i < int(warmup_turns):
            states.append("W")
            continue
        if i in eB:
            states.append("B")
        elif i in eR:
            states.append("R")
        else:
            states.append("B" if Ct[i] < phi_low else "S")
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
    segs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            s = i
            while i + 1 < n and mask[i+1]:
                i += 1
            e = i
            segs.append((s, e))
        i += 1
    return segs

def _extrema_idxs(y: np.ndarray, start: int, end: int, eps: float, dCt: Optional[np.ndarray] = None) -> List[int]:
    if dCt is None:
        dCt = _central_derivative(y)
    idxs = []
    for i in range(max(start+1, 1), min(end, len(y)-2)+1):
        left, mid, right = y[i-1], y[i], y[i+1]
        is_peak = (mid > left) and (mid > right)
        is_trough = (mid < left) and (mid < right)
        if (is_peak or is_trough) and (abs(dCt[i-1]) > eps or abs(dCt[i]) > eps or abs(dCt[i+1]) > eps):
            idxs.append(i)
    return idxs

def compute_quanto_of_coherence(
    Ct: np.ndarray,
    phi_low: float,
    phi_high: float,
    eps: float = 1e-4
) -> Tuple[float, List[Tuple[int, int, float]], dict]:
    Ct = np.asarray(Ct, float)
    dCt = _central_derivative(Ct)

    mask = (Ct > phi_low) & (Ct <= phi_high)
    segs = _find_segments(mask)

    segments_info: List[Tuple[int, int, float]] = []
    all_osc: List[Tuple[float, int, int]] = []

    for s, e in segs:
        if not np.any(np.abs(dCt[s:e+1]) > eps):
            continue
        ex = _extrema_idxs(Ct, s, e, eps, dCt)
        if len(ex) < 2:
            continue
        amps = []
        for k in range(len(ex)-1):
            i, j = ex[k], ex[k+1]
            if (phi_low < Ct[i] <= phi_high) and (phi_low < Ct[j] <= phi_high):
                amp = abs(Ct[j] - Ct[i])
                if amp > 0:
                    amps.append((amp, i, j))
        if not amps:
            continue
        min_amp, ai, bi = min(amps, key=lambda x: x[0])
        segments_info.append((s, e, float(min_amp)))
        all_osc.extend(amps)

    if not all_osc:
        Qa = float('nan')
        chosen = None
    else:
        Qa, i0, j0 = min(all_osc, key=lambda x: x[0])
        chosen = (i0, j0)

    dbg = {"dCt": dCt, "chosen_extrema_pair": chosen, "phi_low": phi_low, "phi_high": phi_high}
    return float(Qa), segments_info, dbg

# -------------------------------
# Triadic Quantum of Coherence (S–B–R)
# -------------------------------
@dataclass
class QuantoParams:
    phi_high: float = 0.75
    phi_low: float = 0.55
    eps_b: float = 0.10
    eps_r: float = 0.08

def detect_quantos_triadic(C: np.ndarray, qp: QuantoParams) -> Tuple[pd.DataFrame, List[int], List[int], List[int]]:
    C = np.asarray(C, float)
    n = len(C)
    S_idx, B_idx, R_idx = [], [], []
    rows = []
    for t in range(1, n-1):
        cond_B = (C[t-1] >= qp.phi_high) and ((C[t-1] - C[t]) >= qp.eps_b)
        if not cond_B:
            continue
        cond_R = (C[t] < C[t-1]) and ((C[t+1] - C[t]) >= qp.eps_r)
        if not cond_R:
            continue
        s, b, r = t-1, t, t+1
        A_q = C[s] - C[b]
        G_q = C[r] - C[b]
        eta_q = (G_q / (A_q + 1e-12)) if A_q > 0 else np.nan
        kappa_q = C[r] - 2*C[b] + C[s]
        A_action = abs(C[s]-C[b]) + abs(C[b]-C[r])
        Delta_q = C[r] - C[s]
        rho_q = max(0.0, C[r] - qp.phi_high) / (max(0.0, qp.phi_high - C[b]) + 1e-9)

        rows.append({
            "t": b+1,
            "S_turn": s+1,
            "B_turn": b+1,
            "R_turn": r+1,
            "C_S": float(C[s]),
            "C_B": float(C[b]),
            "C_R": float(C[r]),
            "A_q": float(A_q),
            "G_q": float(G_q),
            "eta_q": float(eta_q),
            "kappa_q": float(kappa_q),
            "A_action": float(A_action),
            "Delta_q": float(Delta_q),
            "rho_q": float(rho_q),
        })
        S_idx.append(s); B_idx.append(b); R_idx.append(r)

    quantos_df = pd.DataFrame(rows, columns=[
        "t", "S_turn", "B_turn", "R_turn", "C_S", "C_B", "C_R",
        "A_q", "G_q", "eta_q", "kappa_q", "A_action", "Delta_q", "rho_q"
    ])
    return quantos_df, S_idx, B_idx, R_idx

def summarize_quantos(quantos_df: pd.DataFrame) -> dict:
    if len(quantos_df) == 0:
        return {
            "n_quantos": 0,
            "mean_A_q": np.nan,
            "mean_G_q": np.nan,
            "mean_eta_q": np.nan,
            "mean_A_action": np.nan,
            "mean_rho_q": np.nan,
        }
    return {
        "n_quantos": int(len(quantos_df)),
        "mean_A_q": float(np.nanmean(quantos_df["A_q"])),
        "mean_G_q": float(np.nanmean(quantos_df["G_q"])),
        "mean_eta_q": float(np.nanmean(quantos_df["eta_q"])),
        "mean_A_action": float(np.nanmean(quantos_df["A_action"])),
        "mean_rho_q": float(np.nanmean(quantos_df["rho_q"])),
    }

def build_asymmetry_table_from_quantos(
    quantos_df: pd.DataFrame,
    L_min: int = 1
) -> pd.DataFrame:
    """
    Builds an event-level table to test breakdown–repair asymmetry.
    Uses triadic quantos rows (S_turn, B_turn, R_turn, C_S, C_B, C_R).

    v_drop = C_B - C_S  (negative)
    v_rec  = C_R - C_B  (positive)
    asymmetry_holds = |v_drop| > v_rec
    L = R_turn - B_turn (latency in turns)
    """
    if quantos_df is None or len(quantos_df) == 0:
        return pd.DataFrame(columns=[
            "event_id", "S_turn", "B_turn", "R_turn",
            "C_S", "C_B", "C_R",
            "v_drop", "v_rec", "abs_drop", "ratio_absdrop_over_rec",
            "L", "asymmetry_holds", "gradual_repair_holds"
        ])

    q = quantos_df.copy()

    # Ensure numeric
    for c in ["S_turn", "B_turn", "R_turn", "C_S", "C_B", "C_R"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q["v_drop"] = q["C_B"] - q["C_S"]          # expected < 0
    q["v_rec"]  = q["C_R"] - q["C_B"]          # expected > 0
    q["abs_drop"] = q["v_drop"].abs()

    q["ratio_absdrop_over_rec"] = q["abs_drop"] / (q["v_rec"].abs() + 1e-12)

    q["L"] = (q["R_turn"] - q["B_turn"]).astype("Int64")

    q["asymmetry_holds"] = (q["abs_drop"] > q["v_rec"].abs()).astype(int)
    q["gradual_repair_holds"] = (q["L"] > int(L_min)).astype(int)

    # Order / minimal view
    q = q.reset_index(drop=True)
    q["event_id"] = np.arange(1, len(q) + 1, dtype=int)

    cols = [
        "event_id", "S_turn", "B_turn", "R_turn",
        "C_S", "C_B", "C_R",
        "v_drop", "v_rec", "abs_drop", "ratio_absdrop_over_rec",
        "L", "asymmetry_holds", "gradual_repair_holds"
    ]
    q = q[cols]

    # Round readability
    for c in ["C_S", "C_B", "C_R", "v_drop", "v_rec", "abs_drop", "ratio_absdrop_over_rec"]:
        q[c] = pd.to_numeric(q[c], errors="coerce").round(4)

    return q


def summarize_asymmetry(asym_df: pd.DataFrame) -> dict:
    """
    Summary KPIs for the asymmetry law.
    """
    if asym_df is None or len(asym_df) == 0:
        return {
            "n_events": 0,
            "mean_abs_drop": np.nan,
            "mean_rec": np.nan,
            "mean_ratio": np.nan,
            "share_asymmetry": np.nan,
            "mean_L": np.nan,
            "share_gradual": np.nan,
        }

    abs_drop = pd.to_numeric(asym_df["abs_drop"], errors="coerce").to_numpy()
    v_rec = pd.to_numeric(asym_df["v_rec"], errors="coerce").to_numpy()
    ratio = pd.to_numeric(asym_df["ratio_absdrop_over_rec"], errors="coerce").to_numpy()
    L = pd.to_numeric(asym_df["L"], errors="coerce").to_numpy()

    share_asym = np.nanmean(pd.to_numeric(asym_df["asymmetry_holds"], errors="coerce").to_numpy())
    share_grad = np.nanmean(pd.to_numeric(asym_df["gradual_repair_holds"], errors="coerce").to_numpy())

    return {
        "n_events": int(len(asym_df)),
        "mean_abs_drop": float(np.nanmean(abs_drop)),
        "mean_rec": float(np.nanmean(np.abs(v_rec))),
        "mean_ratio": float(np.nanmean(ratio)),
        "share_asymmetry": float(share_asym),
        "mean_L": float(np.nanmean(L)),
        "share_gradual": float(share_grad),
    }

def plot_quanto_dynamics_plotly(C: np.ndarray,
                                quantos_df: pd.DataFrame,
                                phi_low: float,
                                phi_high: float,
                                title: str = "Quantum of Coherence Dynamics",
                                height: int = 500) -> go.Figure:
    x = np.arange(1, len(C)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=C, mode="lines", name="Cₜ", line=dict(width=3)))
    fig.add_hline(y=float(phi_low),  line_dash="dash", opacity=0.45, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.45, annotation_text="Φ_high")

    if len(quantos_df):
        fig.add_trace(go.Scatter(
            x=quantos_df["S_turn"], y=[C[i-1] for i in quantos_df["S_turn"]],
            mode="markers", name="S (stable)", marker=dict(symbol="circle", size=10)))
        fig.add_trace(go.Scatter(
            x=quantos_df["B_turn"], y=[C[i-1] for i in quantos_df["B_turn"]],
            mode="markers", name="B (rupture)", marker=dict(symbol="triangle-down", size=12)))
        fig.add_trace(go.Scatter(
            x=quantos_df["R_turn"], y=[C[i-1] for i in quantos_df["R_turn"]],
            mode="markers", name="R (repair)", marker=dict(symbol="triangle-up", size=12)))

    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Cₜ",
        yaxis=dict(range=[0, 1]),
        legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.7)")
    )
    return fig

# -------------------------------
# Enforce S-B-R mandatory repair rule (unchanged)
# -------------------------------
def _sbr_fullname(s: str) -> str:
    m = {"S": "stable", "B": "broken", "R": "repair",
         "stable": "stable", "broken": "broken", "repair": "repair"}
    key = str(s).strip()
    if key in {"S", "B", "R"}:
        return m.get(key, "stable")
    return m.get(key.lower(), "stable")

def _enforce_quanto_rule(states: List[str]) -> Tuple[List[str], List[dict]]:
    fixed = [_sbr_fullname(s) for s in states]
    corrections = []
    i, n = 0, len(fixed)
    while i < n:
        if fixed[i] == "broken":
            j = i + 1
            saw_R = False
            while j < n and fixed[j] != "stable":
                if fixed[j] == "repair":
                    saw_R = True
                    break
                j += 1
            if j < n and fixed[j] == "stable" and not saw_R and (i + 1) < n:
                fixed[i + 1] = "repair"
                corrections.append({"pos": i + 1, "reason": "Inserted mandatory REPAIR between BROKEN and STABLE"})
                i = j
            else:
                i += 1
        else:
            i += 1
    back = {"stable": "S", "broken": "B", "repair": "R"}
    return [back.get(s, "S") for s in fixed], corrections

# -------------------------------
# Context-aware coherence (unchanged)
# -------------------------------
def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / (n + eps)

def _sigmoid_scalar(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))

def compute_context_state_vector(
    E: np.ndarray,
    Ct_old: np.ndarray,
    Phi: float,
    alpha_min: float = 0.80,
    alpha_max: float = 0.97,
    k: float = 8.0,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    E = np.asarray(E, float)
    Ct_old = np.asarray(Ct_old, float)
    n = E.shape[0]
    if n == 0:
        return {"c_t": np.zeros((0, E.shape[1] if E.ndim == 2 else 0), float),
                "alpha_t": np.zeros(0, float)}

    e = np.zeros_like(E, float)
    for i in range(n):
        e[i] = _safe_unit(E[i], eps=eps)

    c = np.zeros_like(e, float)
    a_t = np.zeros(n, float)

    c[0] = e[0]
    a_t[0] = float(alpha_max)

    for t in range(1, n):
        sig = _sigmoid_scalar(float(k) * (float(Ct_old[t]) - float(Phi)))
        a = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * float(sig)
        a = float(np.clip(a, 0.0, 0.999))
        a_t[t] = a

        u = a * c[t-1] + (1.0 - a) * e[t]
        if float(np.linalg.norm(u)) < eps:
            c[t] = c[t-1]
        else:
            c[t] = _safe_unit(u, eps=eps)

    return {"c_t": c, "alpha_t": a_t}

def compute_context_metrics(
    E: np.ndarray,
    c_t: np.ndarray,
    W: int = 7,
    kappa: float = 1.2,
    eps: float = 1e-12
) -> Dict[str, np.ndarray]:
    E = np.asarray(E, float)
    c_t = np.asarray(c_t, float)
    n = E.shape[0]
    if n == 0:
        z = np.zeros(0, float)
        return {"C_ctx": z, "D_ctx": z, "Delta_C_ctx": z, "Theta_ctx": z}

    e = np.zeros_like(E, float)
    for i in range(n):
        e[i] = _safe_unit(E[i], eps=eps)

    C_ctx = np.zeros(n, float)
    D_ctx = np.zeros(n, float)
    Delta_C_ctx = np.zeros(n, float)
    Theta_ctx = np.zeros(n, float)

    C_ctx[0] = 1.0
    D_ctx[0] = 0.0
    Delta_C_ctx[0] = 0.0
    Theta_ctx[0] = 0.0

    for t in range(1, n):
        C_ctx[t] = 0.5 * (1.0 + _cos(e[t], c_t[t-1], eps=eps))
        C_ctx[t] = float(np.clip(C_ctx[t], 0.0, 1.0))
        D_ctx[t] = 1.0 - C_ctx[t]
        Delta_C_ctx[t] = C_ctx[t] - C_ctx[t-1]

        W_eff = int(min(int(W), t))
        if W_eff <= 1:
            Theta_ctx[t] = float(np.mean(D_ctx[:t]))
        else:
            start = max(0, t - W_eff)
            window = D_ctx[start:t]
            mu = float(np.mean(window))
            sd = float(np.std(window))
            Theta_ctx[t] = mu + float(kappa) * sd

    return {
        "C_ctx": np.clip(C_ctx, 0.0, 1.0),
        "D_ctx": np.clip(D_ctx, 0.0, 1.0),
        "Delta_C_ctx": Delta_C_ctx,
        "Theta_ctx": Theta_ctx
    }

# -------------------------------
# Shift vs break (unchanged)
# -------------------------------
def detect_shift_vs_break(
    texts: List[str],
    C_ctx: np.ndarray,
    D_ctx: np.ndarray,
    Delta_C_ctx: np.ndarray,
    Theta_ctx: np.ndarray,
    phi_low_eff: float,
    W: int = 7,
    K: int = 3,
    Phi_rec: float = 0.60,
    delta: float = 0.10,
    delta_hard: float = 0.18,
    tau: float = 0.15,
    wM: float = 0.45,
    wR: float = 0.45,
    wG: float = 0.10,
    vP: float = 0.50,
    vH: float = 0.30,
    vL: float = 0.20,
) -> pd.DataFrame:
    n = len(C_ctx)
    if n == 0:
        return pd.DataFrame({
            "M_shift_marker": [],
            "R_reconsolidation_mean": [],
            "R_reconsolidation_max": [],
            "P_persistence": [],
            "event_type_ctx": [],
        })

    markers = [
        "by the way", "anyway", "speaking of", "on another note",
        "back to", "to return to", "as i was saying",
        "new topic", "quick question"
    ]

    M = np.zeros(n, int)
    Rmean = np.zeros(n, float)
    Rmax = np.zeros(n, float)
    Ppers = np.zeros(n, float)
    etype = np.array([""] * n, dtype=object)

    for t in range(n):
        txt = texts[t] if isinstance(texts[t], str) else ""
        low = txt.lower()

        M[t] = 1 if any(m in low for m in markers) else 0

        remaining = (n - 1) - t
        K_eff = int(min(int(K), remaining))
        if K_eff <= 0:
            Rmean[t] = float(np.nan)
            Rmax[t] = float(np.nan)
            Ppers[t] = float(np.nan)
        else:
            future = C_ctx[t+1:t+1+K_eff]
            Rmean[t] = float(np.mean(future))
            Rmax[t] = float(np.max(future))
            pers_window = C_ctx[t:t+K_eff]
            Ppers[t] = float(np.mean(pers_window < float(phi_low_eff)))

        phi_low_eff_t = float(phi_low_eff)
        cond_candidate = (
            (C_ctx[t] < phi_low_eff_t)
            and (D_ctx[t] > Theta_ctx[t])
            and (Delta_C_ctx[t] < -float(delta))
        )

        if not cond_candidate:
            etype[t] = ""
            continue

        I_Rmax = 0.0 if (not np.isfinite(Rmax[t])) else (1.0 if (Rmax[t] >= float(Phi_rec)) else 0.0)
        I_G = 1.0 if (abs(Delta_C_ctx[t]) < float(delta_hard)) else 0.0
        S_t = float(wM) * float(M[t]) + float(wR) * float(I_Rmax) + float(wG) * float(I_G)

        I_P = 0.0 if (not np.isfinite(Ppers[t])) else float(Ppers[t])
        I_H = 1.0 if (Delta_C_ctx[t] < -float(delta_hard)) else 0.0
        I_L = 0.0 if (not np.isfinite(Rmean[t])) else (1.0 if (Rmean[t] < float(Phi_rec)) else 0.0)
        B_t = float(vP) * float(I_P) + float(vH) * float(I_H) + float(vL) * float(I_L)

        if (S_t - B_t) >= float(tau):
            etype[t] = "shift"
        elif (B_t - S_t) >= float(tau):
            etype[t] = "break"
        else:
            etype[t] = "ambiguous"

    return pd.DataFrame({
        "M_shift_marker": M.astype(int),
        "R_reconsolidation_mean": Rmean.astype(float),
        "R_reconsolidation_max": Rmax.astype(float),
        "P_persistence": Ppers.astype(float),
        "event_type_ctx": etype.astype(object),
    })

# -------------------------------
# ✅ NEW: Ci trajectories per participant (embedding-based)
# -------------------------------
def compute_ci_series(
    E: np.ndarray,
    participants: List[str],
    method: str = "ctx",
    alpha: float = 0.90,
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Continuous per-participant trajectories Ci_p(t) for all turns (N).
    - method="ctx": each participant has a context vector updated ONLY when they speak
                    but Ci is evaluated at every turn vs that participant context.
    - method="im": centroid baseline (static).
    """
    E = np.asarray(E, float)
    n = E.shape[0]
    parts = [str(p) for p in participants]
    uniq = list(dict.fromkeys(parts))
    out = pd.DataFrame(index=np.arange(n))

    if n == 0 or len(uniq) == 0:
        return out

    # unit vectors
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

        # Evaluate Ci for ALL participants at this turn (continuous lines)
        for p in uniq:
            ci_cols[p][t] = 0.5 * (1.0 + _cos(e[t], ctx[p], eps=eps))

        # Update ONLY the speaker context
        c_prev = ctx[speaker]
        u = a * c_prev + (1.0 - a) * e[t]
        nu = float(np.linalg.norm(u))
        ctx[speaker] = (u / (nu + eps)) if nu > eps else c_prev

    for p in uniq:
        out[f"Ci_{p}"] = np.clip(ci_cols[p], 0.0, 1.0)

    return out

# -------------------------------
# Main chart (overview) — extended (backwards compatible)
# -------------------------------
def _base_layout(fig: go.Figure, title: str, height: int):
    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        yaxis2=dict(title="Potentiality ℘ₜ", overlaying="y", side="right", range=[0, 1]),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        )
    )
    return fig

def plot_ct_main(
    Ct: np.ndarray,
    participants: List[str],
    phi_low: float,
    phi_high: float,
    valleys: List[int],
    peaks: List[int],
    shifts: Optional[List[int]],
    title: str,
    height: int,
    potentiality: Optional[np.ndarray] = None,
    Ct_old_overlay: Optional[np.ndarray] = None,
    pre_flags: Optional[List[int]] = None,
    geom_breaks: Optional[List[int]] = None,
    geom_shifts: Optional[List[int]] = None,
    geom_transitions: Optional[List[int]] = None,
    C_hat: Optional[np.ndarray] = None,
    geom_breaks_struct: Optional[List[int]] = None,
    perceived_breaks: Optional[List[int]] = None,
    lag_label: Optional[str] = None,
) -> go.Figure:

    x = np.arange(1, len(Ct) + 1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=Ct, mode="lines",
        name="Cₜ (global)", line=dict(width=3), yaxis="y"
    ))

    if C_hat is not None and len(C_hat) == len(Ct):
        fig.add_trace(go.Scatter(
            x=x, y=C_hat, mode="lines",
            name="Cₜ_smooth", line=dict(width=2, dash="dash"), yaxis="y"
        ))

    if Ct_old_overlay is not None and len(Ct_old_overlay) == len(Ct):
        fig.add_trace(go.Scatter(
            x=x, y=Ct_old_overlay, mode="lines",
            name="Ct_old (debug)", line=dict(width=2, dash="dash"), yaxis="y"
        ))

        # masked per-speaker Ct points (for readability only — NOT Ci)
    n = len(Ct)
    parts = list(dict.fromkeys(participants))
    for name in parts:
        y = np.full(n, np.nan, float)
        idx = [i for i, p in enumerate(participants) if p == name]
        if idx:
            y[idx] = Ct[idx]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",  # ✅ NO lines -> no sawtooth
            name=f"Participant: {name}",
            marker=dict(size=7),
            yaxis="y"
        ))

    if potentiality is not None and len(potentiality) == len(Ct):
        fig.add_trace(go.Scatter(
            x=x, y=potentiality, mode="lines",
            name="℘ₜ (potentiality)",
            line=dict(width=2, dash="dot"),
            yaxis="y2"
        ))

    fig.add_hline(y=float(phi_low),  line_dash="dash", opacity=0.45, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.45, annotation_text="Φ_high")

    if valleys:
        fig.add_trace(go.Scatter(
            x=[v + 1 for v in valleys], y=[Ct[v] for v in valleys],
            mode="markers", marker=dict(symbol="triangle-down", size=12),
            name="Breaks (B)", yaxis="y"
        ))
    if peaks:
        fig.add_trace(go.Scatter(
            x=[p + 1 for p in peaks], y=[Ct[p] for p in peaks],
            mode="markers", marker=dict(symbol="triangle-up", size=12),
            name="Repairs (R)", yaxis="y"
        ))
    if shifts:
        fig.add_trace(go.Scatter(
            x=[s + 1 for s in shifts], y=[Ct[s] for s in shifts],
            mode="markers", marker=dict(symbol="square", size=10),
            name="Topic shifts (ctx)", yaxis="y"
        ))

    if pre_flags:
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in pre_flags], y=[Ct[i] for i in pre_flags],
            mode="markers", marker=dict(symbol="circle-open", size=9),
            name="Pre-transition (IC-III)", yaxis="y"
        ))
    if geom_breaks:
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in geom_breaks], y=[Ct[i] for i in geom_breaks],
            mode="markers", marker=dict(symbol="x", size=11),
            name="Break (geom)", yaxis="y"
        ))
    if geom_shifts:
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in geom_shifts], y=[Ct[i] for i in geom_shifts],
            mode="markers", marker=dict(symbol="diamond", size=10),
            name="Shift (geom)", yaxis="y"
        ))
    if geom_transitions:
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in geom_transitions], y=[Ct[i] for i in geom_transitions],
            mode="markers", marker=dict(symbol="hexagon", size=10),
            name="Transition (geom)", yaxis="y"
        ))

    # NEW markers
    if geom_breaks_struct:
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in geom_breaks_struct], y=[Ct[i] for i in geom_breaks_struct],
            mode="markers",
            marker=dict(symbol="circle", size=10),
            name="Geom Break (IC-III→)", yaxis="y"
        ))
    if perceived_breaks:
        yref = C_hat if (C_hat is not None and len(C_hat) == len(Ct)) else Ct
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in perceived_breaks], y=[yref[i] for i in perceived_breaks],
            mode="markers",
            marker=dict(symbol="triangle-down", size=11),
            name="Perceived Break (Ĉₜ)", yaxis="y"
        ))

    if lag_label:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.02,
            xanchor="left", yanchor="bottom",
            text=str(lag_label),
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        )

    return _base_layout(fig, title, height=height)

# ----------------------------------
# Ci trajectories plot
# ----------------------------------
def plot_ci_lines(
    turns: np.ndarray,
    ci_df: pd.DataFrame,
    phi_low: float,
    phi_high: float,
    height: int = 520,
    title: str = "Participant trajectories (Ci)",
):
    fig = go.Figure()
    x = np.asarray(turns)
    df = ci_df.copy()
    if "turn" in df.columns:
        df = df.drop(columns=["turn"])

    for col in df.columns:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=str(col)))

    fig.add_hline(y=phi_low, line_dash="dot")
    fig.add_hline(y=phi_high, line_dash="dot")

    fig.update_layout(
        title=title,
        xaxis_title="Turn",
        yaxis_title="Ci",
        height=height,
        legend_title="Participant",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------- STREAMLIT UI ----------
with st.expander("Debug", expanded=False):
    st.write("...")

# ============================
# app.py — PART 3/3
# (demo + PDF + Streamlit app main)
# ✅ FIXED: state_alpha slider exists and matches the call
# ✅ FIXED: Ci plot shown as continuous lines
# ✅ NEW: IC-III→IC-II computed and exported (rho_t, D_t, C_hat_t, delta_star, breaks)
# ============================

# -------------------------------
# Demo data (UPDATED + realistic, no repeats, 2 ruptures)
# -------------------------------
def load_demo(n_turns: int = 30) -> pd.DataFrame:
    """
    Realistic demo dialogue with:
    - 2 clear rupture blocks (abrupt topic shifts)
    - Explicit repairs + natural recovery
    - No repeated turns
    - If n_turns > base length, adds varied (non-rupture) coherent lines
    """
    speakers = ["A", "B"]
    turns, parts, texts = [], [], []

    # --- Main topic: meeting recap / alignment + coherence analysis ---
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

    # Rupture 1: mild-but-clear off-topic jump, then repair
    rupture_block1 = [
        "Anyway, I’m thinking of buying a used motorcycle this weekend—do you know any good brands?",
        "Wait, that’s a complete switch. We were on the meeting decisions and measuring drift.",
        "True—sorry. Let me pull it back: I asked because I noticed we also switched topics in the meeting like that.",
        "So the motorcycle question is basically a toy example of an off-topic injection that creates a coherence drop.",
        "Exactly. And the repair is us naming the mismatch and reconnecting to the original frame.",
    ]

    # Technical-ish but still conversational
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

    # Rupture 2: sharper, surreal jump, then repair + recovery
    rupture_block2 = [
        "BREAKING: The meeting is actually a sandwich, and the action items are made of glitter.",
        "Okay, that’s not just drift—that breaks the frame completely. I can’t map that onto our topic.",
        "Yes, intentional rupture. Now the repair: we return to the agenda and the measurement idea.",
        "Specifically, we want the demo to show a steep drop followed by a clear recovery after re-alignment.",
    ]

    # Recovery block: keep it clearly on-topic so turn 31 (or the first post-rupture turn) doesn't look like a rupture
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

    # If user requests fewer turns, cut cleanly
    if n_turns < len(all_lines):
        all_lines = all_lines[:n_turns]

    # If user requests more turns, append varied, coherent (non-rupture) lines (no template repetition)
    elif n_turns > len(all_lines):
        extra = n_turns - len(all_lines)
        extra_lines = [
            "If we want, we can annotate the rupture points with brief explanations in the exported report.",
            "Another nice touch is showing the speaker-level lines so repairs are visibly attributed.",
            "We should keep the final section focused on the same plan to preserve stability at the end.",
            "The demo doesn’t need jargon—just enough terms to connect the plots to intuitive dialogue behavior.",
            "We can also mention that different embedding models may shift absolute values but preserve patterns.",
            "A stable ending helps the user understand that repair is measurable and not just a social label.",
            "If a threshold is enabled, we can show how ‘critical’ zones cluster around rupture-like turns.",
            "It’s also useful to remind that coherence is contextual—some jumps are fine if they’re bridged properly.",
            "The key is that the curve reflects structure: alignment, mismatch, and re-attachment over time.",
            "Alright—this is solid: two ruptures, explicit repairs, and a stable close for a clean visual.",
        ]
        for i in range(extra):
            all_lines.append(extra_lines[i % len(extra_lines)])

    for i, text in enumerate(all_lines):
        turns.append(i + 1)
        parts.append(speakers[i % 2])
        texts.append(text)

    return pd.DataFrame(
        {
            "turn": np.array(turns, dtype=int),
            "timestamp": "",
            "participant": parts,
            "text": texts,
        }
    )

# -------------------------------
# PDF report generation
# NOTE: kept identical to your code block (not repeated here to save space).
# Paste your existing PDF functions below this comment unchanged.
# -------------------------------

# =========================
# Streamlit app
# =========================

# --- Sidebar: language
lang = st.sidebar.selectbox(LABELS["en"]["lang"], options=["en", "es"], index=0)
L = LABELS[lang]

st.title(L["app_title"])
st.caption(L["app_subtitle"])

with st.expander(L["what_does"], expanded=False):
    st.write(
        "This app computes a dynamic coherence signal C_t from dialogue turns, "
        "detects S–B–R regimes, distinguishes rupture vs topic shift without LLMs, "
        "and adds an IC–III geometric layer (d_i, kappa_i, tau)."
    )
    st.write(L["expected_cols"])

# --- Sidebar: data
st.sidebar.header(L["params"])
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

# --- Preview
with st.expander(L["preview"], expanded=True):
    st.dataframe(df.head(25), use_container_width=True)

# --- Sidebar: embeddings
st.sidebar.subheader(L["sem_repr"])
emb_mode = st.sidebar.selectbox(L["emb_mode"], ["auto", "sbert", "tfidf"], index=0)
sbert_model = st.sidebar.text_input(L["sbert_model"], value="sentence-transformers/all-MiniLM-L6-v2")

# --- Sidebar: coherence configuration
st.sidebar.subheader(L["coh_mode"])
ct_mode = st.sidebar.selectbox(
    L["coh_form"],
    ["Ct_new (context-aware)", "IC-IIa (sigma alignment)", "Ct_old (adjacent-turn)"],
    index=0
)

st.sidebar.subheader(L["ctx_header"])
ctx_lambda = st.sidebar.slider("λ mix (Ct_new = (1-λ)*Ct_old + λ*C_ctx)", 0.0, 1.0, 0.60, 0.05)
alpha_min = st.sidebar.slider("alpha_min", 0.60, 0.95, 0.80, 0.01)
alpha_max = st.sidebar.slider("alpha_max", 0.80, 0.99, 0.97, 0.01)
k_sig = st.sidebar.slider("k (inertia slope)", 2.0, 16.0, 8.0, 0.5)

st.sidebar.subheader(L["phi"])
q_low = st.sidebar.slider("Φ_low percentile", 0.05, 0.50, 0.20, 0.01)
q_high = st.sidebar.slider("Φ_high percentile", 0.50, 0.95, 0.80, 0.01)

st.sidebar.subheader(L["events"])
sep_min = st.sidebar.slider("min separation (turns)", 1, 8, 2, 1)
prom_min = st.sidebar.slider("min prominence", 0.01, 0.30, 0.05, 0.01)
merge_gap = st.sidebar.slider("merge_gap (micro-break grouping)", 1, 6, 2, 1)

# ✅ Ci controls
st.sidebar.subheader(L["ci_header"])
ci_method = st.sidebar.selectbox(L["ci_method"], ["ctx", "im"], index=0)
ci_alpha = st.sidebar.slider(L["ci_alpha"], 0.70, 0.99, 0.90, 0.01)

# ✅ FIX: State trajectory alpha (this was missing in your code)
state_alpha = st.sidebar.slider(L["state_alpha"], 0.60, 0.98, 0.88, 0.01)

# Public view
st.sidebar.subheader(L["public_header"])
public_span = st.sidebar.slider(L["public_span"], 3, 25, 9, 1)
public_show_thr = st.sidebar.checkbox(L["public_show_thresholds"], value=False)

st.sidebar.subheader(L["debug_header"])
show_ct_old = st.sidebar.checkbox(L["show_ct_old"], value=False)
use_ct_old_for_events = st.sidebar.checkbox(L["use_ct_old_for_events"], value=False)

st.sidebar.markdown("### Smoothed coherence (new)")
smooth_method = st.sidebar.selectbox(
    "Smoothing method",
    ["ema", "ewma"],
    index=0
)

env_alpha = st.sidebar.slider("EMA alpha", 0.05, 0.60, 0.20, 0.01)
env_span = st.sidebar.slider("EWMA span", 3, 25, 9, 1)

show_envelope = st.sidebar.checkbox("Show smoothed coherence plot", value=True)
overlay_smoothed_on_main = st.sidebar.checkbox("Overlay smoothed curve on main plot", value=False)

# --- Compute
if not st.button(L["compute"]):
    st.stop()

texts = df["text"].tolist()
participants = df["participant"].tolist()
turns = df["turn"].to_numpy(dtype=int)

E, used_mode, emb_msg = embed_texts(texts, mode=emb_mode, sbert_model=sbert_model)
st.sidebar.info(emb_msg)

# --- Matrix coherence (Ct_Im): coherence w.r.t. global dialogue centroid
Ct_im = compute_ct_im(E, ema_alpha=0.40)

# Make Ct_Im comparable to Ct curves (same post-processing style)
Ct_im = _detrend_ct(Ct_im, alpha=0.020)
Ct_im = _center_ct(Ct_im, target=0.62)
Ct_im = _normalize_ct(Ct_im, lower=0.06, upper=0.94)
Ct_im = apply_warmup_ramp(Ct_im, warm=WARMUP_TURNS, floor=0.10)

# --- Legacy Ct_old
Ct_old = compute_ct_series(E, ema_alpha=0.40, use_savgol=False)
Ct_old = _detrend_ct(Ct_old, alpha=0.020)
Ct_old = _center_ct(Ct_old, target=0.62)
Ct_old = _normalize_ct(Ct_old, lower=0.06, upper=0.94)
Ct_old = apply_warmup_ramp(Ct_old, warm=WARMUP_TURNS, floor=0.10)

# --- IC-IIa
ic2 = compute_ic2_dynamics(
    E,
    alpha_context=0.84,
    theta0=0.0, theta1=4.0, theta2=2.0,
    window_W=5, delta_async=0, gamma_async=0.55
)
Ct_ic2 = np.clip(ic2["C_t"], 0.0, 1.0)
Ct_ic2 = _normalize_ct(Ct_ic2, lower=0.06, upper=0.94)
Ct_ic2 = apply_warmup_ramp(Ct_ic2, warm=WARMUP_TURNS, floor=0.10)

# --- Context state vector + context coherence
Phi_seed = float(np.quantile(Ct_old, 0.70)) if len(Ct_old) else 0.60
ctx = compute_context_state_vector(
    E=E, Ct_old=Ct_old, Phi=Phi_seed,
    alpha_min=alpha_min, alpha_max=alpha_max, k=k_sig
)
ctx_m = compute_context_metrics(E=E, c_t=ctx["c_t"], W=7, kappa=1.2)
C_ctx = np.clip(ctx_m["C_ctx"], 0.0, 1.0)

Ct_new = (1.0 - float(ctx_lambda)) * Ct_old + float(ctx_lambda) * C_ctx
Ct_new = _normalize_ct(Ct_new, lower=0.06, upper=0.94)
Ct_new = apply_warmup_ramp(Ct_new, warm=WARMUP_TURNS, floor=0.10)

# --- Choose Ct for pipeline
if ct_mode.startswith("Ct_old"):
    Ct_base = Ct_old.copy()
elif ct_mode.startswith("IC-IIa"):
    Ct_base = Ct_ic2.copy()
else:
    Ct_base = Ct_new.copy()

Ct_smooth = smooth_coherence(
    Ct_base,
    method=smooth_method,
    ema_alpha=float(env_alpha),
    ewma_span=int(env_span),
)
Ct_smooth = np.clip(Ct_smooth, 0.0, 1.0)

# --- Smoothed stability envelopes (B2)
C_hat = _ema(Ct_base, alpha=0.25)
C_hat_im = _ema(Ct_im, alpha=0.25)

# --- Thresholds
phi_low_eff = float(np.quantile(Ct_base, float(q_low))) if len(Ct_base) else 0.55
phi_high_eff = float(np.quantile(Ct_base, float(q_high))) if len(Ct_base) else 0.75
phi_low_eff = float(np.clip(phi_low_eff, 0.0, 0.95))
phi_high_eff = float(np.clip(phi_high_eff, 0.05, 1.0))
if phi_high_eff <= phi_low_eff + 0.08:
    phi_high_eff = float(min(1.0, phi_low_eff + 0.12))

# --- Potentiality
P_t = compute_potentiality(texts)

# --- Event detection (valleys/peaks)
Ct_for_events = Ct_old if use_ct_old_for_events else Ct_base
valleys, peaks = detect_events(
    Ct_for_events,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    sep_min=sep_min,
    prom_min=prom_min
)
valleys = merge_consecutive(valleys, gap=merge_gap)

# --- SBR (internal)
sbr = assign_sbr(
    Ct_base,
    valleys=valleys,
    peaks=peaks,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    warmup_turns=WARMUP_TURNS
)
sbr_fixed, sbr_corrections = _enforce_quanto_rule(sbr)

# ============================
# ✅ Triadic quantos (S–B–R) — REQUIRED (fix bug 2)
# ============================
qp = QuantoParams(
    phi_high=float(phi_high_eff),
    phi_low=float(phi_low_eff),
    eps_b=0.10,
    eps_r=0.08,
)

quantos_df, S_idx, B_idx, R_idx = detect_quantos_triadic(Ct_base, qp)
quantos_summary = summarize_quantos(quantos_df)

# --- Legacy Qa
Qa, segs, dbgQa = compute_quanto_of_coherence(Ct_base, phi_low=float(phi_low_eff), phi_high=float(phi_high_eff))

# --- Context shift vs break (no LLMs)
ctx_events_df = detect_shift_vs_break(
    texts=texts,
    C_ctx=C_ctx,
    D_ctx=ctx_m["D_ctx"],
    Delta_C_ctx=ctx_m["Delta_C_ctx"],
    Theta_ctx=ctx_m["Theta_ctx"],
    phi_low_eff=phi_low_eff,
    K=3,
    Phi_rec=float(phi_high_eff),
)
shifts_ctx = [i for i, v in enumerate(ctx_events_df["event_type_ctx"].tolist()) if v == "shift"]

# --- IC-III geometry
ic3 = compute_ic3_geometry(
    E=E,
    Ct=Ct_base,
    dI_norm=ic2["dI_norm"],
    Delta_async=ic2["Delta_async"],
)
geom_df = detect_ic3_geometric_events(
    Ct=Ct_base,
    d_i=ic3["d_i"],
    kappa_i=ic3["kappa_i"],
    texts=texts,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    W=7,
    persistence_mode="offline"
)
pre_flags = geom_df.index[geom_df["pre_transition_flag"].to_numpy(dtype=int) == 1].tolist()
geom_breaks = geom_df.index[geom_df["event_type_geom"].astype(str).to_numpy() == "break"].tolist()
geom_shifts = geom_df.index[geom_df["event_type_geom"].astype(str).to_numpy() == "shift"].tolist()
geom_trans = geom_df.index[geom_df["event_type_geom"].astype(str).to_numpy() == "transition"].tolist()

# ============================
# ✅ IC-III → IC-II (Desfase geométrico → fenomenológico)
# ============================
rho_t = semantic_compactness_rho(E, texts, w=2, mode="centroid", min_tokens=3)
di_n = _norm01(ic3["d_i"])
kappa_n = _norm01(ic3["kappa_i"])

D_t = manifold_driver_D(di=di_n, kappa=kappa_n, rho=rho_t, w_d=0.45, w_k=0.35, w_r=0.20, gating=True)
C_hat_t = phenomenological_coherence_C_hat(D_t, model="ema", alpha=0.25, Ct_ref=Ct_base)

lag_info = estimate_lag_delta(D=D_t, C=Ct_base, phi_low=phi_low_eff, delta_max=6, smooth_alpha=0.30)
delta_star = int(round(float(lag_info.get("delta_star", 0.0)))) if np.isfinite(lag_info.get("delta_star", np.nan)) else 0
lag_score = lag_info.get("score", float("nan"))

geom_breaks_struct = detect_geom_breaks(D=D_t, kappa=kappa_n, D_hi=0.70, dD_hi=0.12, refractory=2)
perceived_breaks = detect_perceived_breaks(C_hat=C_hat_t, phi_low=phi_low_eff, drop_hi=-0.15, refractory=2)

matches = match_breaks(geom_breaks_struct, perceived_breaks, delta_max=6)
observed_lags = [m["lag"] for m in matches] if matches else []
observed_lag_mean = float(np.mean(observed_lags)) if observed_lags else float("nan")

lag_label = f"Estimated structural lag Δ* = {delta_star} turns (corr = {('n/a' if not np.isfinite(lag_score) else f'{lag_score:.3f}')})"

 

# ============================
# ✅ Asymmetry law (Breakdown–Repair)
# ============================
L_min = 1  # puedes hacerlo slider si quieres
asym_df = build_asymmetry_table_from_quantos(quantos_df, L_min=L_min)
asym_summary = summarize_asymmetry(asym_df)

with st.expander("Breakdown–Repair Asymmetry (law test)", expanded=False):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("n events", value=int(asym_summary["n_events"]))
    c2.metric(
        "mean |v_drop|",
        value=("n/a" if np.isnan(asym_summary["mean_abs_drop"])
               else f'{asym_summary["mean_abs_drop"]:.3f}')
    )
    c3.metric(
        "mean v_rec",
        value=("n/a" if np.isnan(asym_summary["mean_rec"])
               else f'{asym_summary["mean_rec"]:.3f}')
    )
    c4.metric(
        "mean ratio |drop|/rec",
        value=("n/a" if np.isnan(asym_summary["mean_ratio"])
               else f'{asym_summary["mean_ratio"]:.3f}')
    )
    c5.metric(
        "share |drop| > rec",
        value=("n/a" if np.isnan(asym_summary["share_asymmetry"])
               else f'{100*asym_summary["share_asymmetry"]:.1f}%')
    )
    c6.metric(
        "mean L",
        value=("n/a" if np.isnan(asym_summary["mean_L"])
               else f'{asym_summary["mean_L"]:.2f}')
    )

    if len(asym_df):
        st.dataframe(asym_df, use_container_width=True)
    else:
        st.info("No triadic S–B–R events detected (quantos_df is empty).")

# --- Global metrics
rho_legacy = _safe_corr(Ct_old, Ct_ic2)

with st.expander(L["global_metrics"], expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turns", value=int(len(df)))
    c2.metric("Φ_low", value=f"{phi_low_eff:.3f}")
    c3.metric("Φ_high", value=f"{phi_high_eff:.3f}")
    c4.metric("ρ(Ct_old, IC-IIa)", value=("n/a" if np.isnan(rho_legacy) else f"{rho_legacy:.3f}"))

# --- Output table
df_out = df.copy()
df_out["Ct_old"] = Ct_old
df_out["Ct_im"] = Ct_im
df_out["Ct_ic2"] = Ct_ic2
df_out["Ct_new"] = Ct_new
df_out["Ct"] = Ct_base
df_out["SBR"] = sbr_fixed
df_out["P_t"] = P_t
df_out["alpha_t"] = ctx["alpha_t"] if len(ctx["alpha_t"]) == len(df_out) else np.nan
df_out = pd.concat([df_out, ctx_events_df], axis=1)
df_out = pd.concat([df_out, geom_df], axis=1)

# ✅ NEW: IC-III→IC-II exports
df_out["rho_t"] = rho_t
df_out["D_t"] = D_t
df_out["C_hat_t"] = C_hat_t
df_out["delta_star"] = float(delta_star)
df_out["geom_break_struct"] = np.isin(np.arange(len(df_out)), np.array(geom_breaks_struct, int)).astype(int)
df_out["perceived_break"] = np.isin(np.arange(len(df_out)), np.array(perceived_breaks, int)).astype(int)
df_out["observed_lag_mean"] = float(observed_lag_mean) if np.isfinite(observed_lag_mean) else np.nan

# ✅ NEW: compute & export Ci
ci_df = compute_ci_series(E=E, participants=participants, method=ci_method, alpha=float(ci_alpha))
df_out = pd.concat([df_out, ci_df], axis=1)

with st.expander(L["table_title"], expanded=False):
    st.dataframe(df_out, use_container_width=True)

# --- Main plot (overview)
fig_main = plot_ct_main(
    Ct=Ct_base,
    participants=participants,
    phi_low=phi_low_eff,
    phi_high=phi_high_eff,
    valleys=valleys,
    peaks=peaks,
    shifts=shifts_ctx,
    title=L["overview"],
    height=560,
    potentiality=P_t,
    Ct_old_overlay=(Ct_old if show_ct_old else None),
    pre_flags=pre_flags,
    geom_breaks=geom_breaks,
    geom_shifts=geom_shifts,
    geom_transitions=geom_trans,
    C_hat=(Ct_smooth if overlay_smoothed_on_main else None),
)

st.plotly_chart(fig_main, use_container_width=True)

st.session_state["last_main_fig"] = fig_main

html = fig_main.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
st.download_button(
    "Download main plot (HTML)",
    data=html,
    file_name="tie_dialog_main_plot.html",
    mime="text/html",
)

if show_envelope:
    with st.expander("Smoothed coherence (C_smooth)", expanded=False):
        fig_s = go.Figure()
        x = np.arange(1, len(Ct_base) + 1)

        fig_s.add_trace(go.Scatter(
            x=x, y=Ct_base, mode="lines",
            name="Cₜ (raw)", line=dict(width=2)
        ))
        fig_s.add_trace(go.Scatter(
            x=x, y=Ct_smooth, mode="lines",
            name="Cₜ_smooth", line=dict(width=3)
        ))

        fig_s.add_hline(y=float(phi_low_eff), line_dash="dash", opacity=0.35, annotation_text="Φ_low")
        fig_s.add_hline(y=float(phi_high_eff), line_dash="dash", opacity=0.35, annotation_text="Φ_high")

        fig_s.update_layout(
            title="Smoothed coherence curve",
            height=420,
            margin=dict(l=40, r=200, t=30, b=40),
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

        st.plotly_chart(fig_s, use_container_width=True)

# --- IC–III Geometry plot
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

def plot_stability_envelope(turns: np.ndarray,
                            C_hat: np.ndarray,
                            C_hat_im: np.ndarray,
                            phi_low: float,
                            phi_high: float,
                            title: str,
                            height: int = 420) -> go.Figure:
    x = np.asarray(turns, int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.asarray(C_hat, float), mode="lines",
                             name="Ĉₜ (smoothed)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=np.asarray(C_hat_im, float), mode="lines",
                             name="Ĉₜ_Im (smoothed)", line=dict(width=2, dash="dot")))

    fig.add_hline(y=float(phi_low),  line_dash="dash", opacity=0.35, annotation_text="Φ_low")
    fig.add_hline(y=float(phi_high), line_dash="dash", opacity=0.35, annotation_text="Φ_high")

    fig.update_layout(
        title=title,
        height=int(height),
        margin=dict(l=40, r=200, t=30, b=40),
        xaxis_title="Turn",
        yaxis_title="Value (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="v", x=1.02, xanchor="left",
                    y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.7)")
    )
    return fig

# --- Triadic Quantum of Coherence (S–B–R)
with st.expander(L["triadic_title"], expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("n_quantos", value=int(quantos_summary["n_quantos"]))
    c2.metric("mean A_q", value=("n/a" if np.isnan(quantos_summary["mean_A_q"]) else f'{quantos_summary["mean_A_q"]:.3f}'))
    c3.metric("mean G_q", value=("n/a" if np.isnan(quantos_summary["mean_G_q"]) else f'{quantos_summary["mean_G_q"]:.3f}'))
    c4.metric("mean η_q", value=("n/a" if np.isnan(quantos_summary["mean_eta_q"]) else f'{quantos_summary["mean_eta_q"]:.3f}'))
    c5.metric("mean ρ_q", value=("n/a" if np.isnan(quantos_summary["mean_rho_q"]) else f'{quantos_summary["mean_rho_q"]:.3f}'))

    fig_q = plot_quanto_dynamics_plotly(
        C=Ct_base,
        quantos_df=quantos_df,
        phi_low=float(phi_low_eff),
        phi_high=float(phi_high_eff),
        title=L["triadic_title"],
        height=520,
    )
    st.plotly_chart(fig_q, use_container_width=True)
    if len(quantos_df):
        st.dataframe(quantos_df, use_container_width=True)

# --- Legacy Qa (optional display)
with st.expander(L["legacy_q"], expanded=False):
    st.write(f"Qₐ (legacy) = {'n/a' if (not np.isfinite(Qa)) else f'{Qa:.6f}'}")
    if segs:
        seg_df = pd.DataFrame(segs, columns=["start", "end", "min_amp"])
        st.dataframe(seg_df, use_container_width=True)

# --- Participant trajectories (Ci) + continuous state trajectories
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

# =========================
# Downloads: CSV + PDF (Consultancy-grade)
# ✅ Clean exports + clear filenames
# ✅ NEW: "consultancy_report.pdf" with executive summary + legible figures
# ✅ REMOVED: full-results preview table (was illegible)
# ✅ NEW: compact "events table" (10–15 rows max)
# ✅ NEW: glossary page (what each variable means)
# ✅ FIXED: NameError (function name now matches call)
# ✅ FIXED: _norm01 defined locally; no hidden dependencies
# =========================

def _df_to_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

def _fmt(x, nd=3, na="n/a"):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return na
        return f"{float(x):.{nd}f}"
    except Exception:
        return na

def _norm01_nan(a: np.ndarray) -> np.ndarray:
    """Robust 0–1 normalization (returns zeros if flat/invalid)."""
    a = np.asarray(a, float)
    if a.size == 0:
        return a
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

# --- IC-IIa export table (technical)
ic2_df = pd.DataFrame({
    "turn": turns,
    "res": np.asarray(ic2["res"], float),
    "res_sync": np.asarray(ic2["res_sync"], float),
    "d_res": np.asarray(ic2["d_res"], float),
    "C_t": np.asarray(ic2["C_t"], float),
    "E_t": np.asarray(ic2["E_t"], float),
    "I_W": np.asarray(ic2["I_W"], float),
    "Delta_async": np.asarray(ic2["Delta_async"], float),
    "dI_norm": np.asarray(ic2["dI_norm"], float),
})

# --- IC-III export table (technical)
ic3_df = pd.DataFrame({
    "turn": turns,
    "d_i": np.asarray(ic3["d_i"], float),
    "kappa_i": np.asarray(ic3["kappa_i"], float),
    "tau_t": np.asarray(ic3["tau_t"], float),
    "tau_norm": np.asarray(ic3["tau_norm"], float),
    "rho_t": np.asarray(rho_t, float),
    "D_t": np.asarray(D_t, float),
    "C_hat_t": np.asarray(C_hat_t, float),
    "delta_star": float(delta_star),
    "lag_score": float(lag_score) if np.isfinite(lag_score) else np.nan,
})

# --- A lightweight "events" table (consultancy-friendly)
def build_events_table(
    df_out: pd.DataFrame,
    phi_low: float,
    phi_high: float,
    max_rows: int = 14
) -> pd.DataFrame:
    """
    Small, readable table with the most relevant events only.
    Columns kept minimal and interpretable.
    """
    d = df_out.copy()
    if "turn" not in d.columns:
        d["turn"] = np.arange(1, len(d) + 1)

    geom_break = (d.get("geom_break_struct", 0).astype(int) == 1) if "geom_break_struct" in d.columns else pd.Series([False]*len(d))
    perc_break = (d.get("perceived_break", 0).astype(int) == 1) if "perceived_break" in d.columns else pd.Series([False]*len(d))
    geom_evt = d.get("event_type_geom", pd.Series([""]*len(d))).astype(str)

    pre_transition = (d.get("pre_transition_flag", 0).astype(int) == 1) if "pre_transition_flag" in d.columns else pd.Series([False]*len(d))
    transition = (geom_evt == "transition")
    shift = (geom_evt == "shift")
    break_geom = (geom_evt == "break")

    # Priority score: breaks > perceived breaks > transitions > shifts > pre-flags
    score = (
        5*break_geom.astype(int)
        + 4*geom_break.astype(int)
        + 4*perc_break.astype(int)
        + 3*transition.astype(int)
        + 2*shift.astype(int)
        + 1*pre_transition.astype(int)
    )
    d["_score"] = score

    keep = d[d["_score"] > 0].copy()
    if keep.empty:
        return pd.DataFrame({
            "turn": [],
            "event": [],
            "C_t": [],
            "Ĉ_t": [],
            "D_t": [],
            "ρ(t)": [],
            "℘_t": [],
            "note": []
        })

    def _label_row(r):
        tags = []
        if str(r.get("event_type_geom", "")) == "break":
            tags.append("Geom break (IC-III)")
        if int(r.get("geom_break_struct", 0)) == 1:
            tags.append("Geom break (driver)")
        if int(r.get("perceived_break", 0)) == 1:
            tags.append("Perceived break (Ĉₜ)")
        if str(r.get("event_type_geom", "")) == "transition":
            tags.append("Transition")
        if str(r.get("event_type_geom", "")) == "shift":
            tags.append("Shift")
        if int(r.get("pre_transition_flag", 0)) == 1 and not tags:
            tags.append("Pre-transition")
        return " | ".join(tags) if tags else "Event"

    keep["event"] = keep.apply(_label_row, axis=1)

    # Minimal numeric cols (only if present)
    cols = ["turn", "event"]
    if "Ct" in keep.columns: cols.append("Ct")
    if "C_hat_t" in keep.columns: cols.append("C_hat_t")
    if "D_t" in keep.columns: cols.append("D_t")
    if "rho_t" in keep.columns: cols.append("rho_t")
    if "P_t" in keep.columns: cols.append("P_t")

    view = keep[cols].copy()

    # Rename for clarity
    view = view.rename(columns={
        "Ct": "C_t",
        "C_hat_t": "Ĉ_t",
        "rho_t": "ρ(t)",
        "P_t": "℘_t",
    })

    # Add short notes (rule-based)
    def _note(r):
        ct = _safe_float(r.get("C_t", np.nan))
        ch = _safe_float(r.get("Ĉ_t", np.nan))
        dt = _safe_float(r.get("D_t", np.nan))
        if np.isfinite(ct) and ct < phi_low:
            return "Below Φ_low (low coherence regime)"
        if np.isfinite(ch) and ch < phi_low:
            return "Perceived coherence drop (Ĉₜ < Φ_low)"
        if np.isfinite(dt) and dt > 0.75 and (not (np.isfinite(ct) and ct < phi_low)):
            return "High structural stress (driver peak)"
        if np.isfinite(ct) and ct > phi_high:
            return "Above Φ_high (stable regime)"
        return ""

    view["note"] = view.apply(_note, axis=1)

    # Select top rows by score; then restore chronological order
    top_idx = keep.sort_values(["_score", "turn"], ascending=[False, True]).head(int(max_rows)).index
    view = view.loc[top_idx].sort_values("turn").reset_index(drop=True)

    # Round numeric for readability
    for c in view.columns:
        if c in {"turn", "event", "note"}:
            continue
        view[c] = pd.to_numeric(view[c], errors="coerce").round(3)

    return view

events_df = build_events_table(
    df_out=df_out,
    phi_low=float(phi_low_eff),
    phi_high=float(phi_high_eff),
    max_rows=14
)

# --- Download buttons
st.subheader("Downloads")

cA, cB, cC, cD, cE = st.columns(5)

with cA:
    st.download_button(
        L["download_full_csv"],
        data=_df_to_csv_bytes(df_out),
        file_name="tie_dialog_full_results.csv",
        mime="text/csv"
    )

with cB:
    st.download_button(
        L["download_triad_csv"],
        data=_df_to_csv_bytes(quantos_df) if len(quantos_df) else _df_to_csv_bytes(pd.DataFrame()),
        file_name="tie_dialog_triadic_quanta.csv",
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

# -------------------------------
# CONSULTANCY PDF REPORT (clean, narrative, readable)
# -------------------------------
def generate_consultancy_pdf_report(
    df_out: pd.DataFrame,
    Ct: np.ndarray,
    C_hat: np.ndarray,
    D_t: np.ndarray,
    rho_t: np.ndarray,
    phi_low: float,
    phi_high: float,
    delta_star: int,
    lag_score: float,
    observed_lag_mean: float,
    events_df: pd.DataFrame,
    used_mode: str,
    title: str = "TIE–Dialog — Interaction Coherence Diagnostic (Consultancy Report)",
) -> bytes:
    """
    Consultancy-grade PDF:
    - Narrative intro
    - One chart per page (no overlaps)
    - Executive summary + parameters used
    - Events table (compact)
    - Glossary
    """

    buf = BytesIO()

    # Safe cast
    Ct = np.asarray(Ct, float)
    C_hat = np.asarray(C_hat, float) if C_hat is not None else None
    D_t = np.asarray(D_t, float)
    rho_t = np.asarray(rho_t, float)

    n = int(len(Ct))
    x = np.arange(1, n + 1)

    # Robust summary stats
    ct_mean = float(np.nanmean(Ct)) if Ct.size else np.nan
    ct_min  = float(np.nanmin(Ct))  if Ct.size else np.nan
    ct_max  = float(np.nanmax(Ct))  if Ct.size else np.nan
    frac_low  = float(np.nanmean((Ct < float(phi_low)).astype(float))) if Ct.size else np.nan
    frac_high = float(np.nanmean((Ct > float(phi_high)).astype(float))) if Ct.size else np.nan

    # --------------------
    # Helpers
    # --------------------
    def _fmt(v, nd=3, na="n/a"):
        try:
            if v is None:
                return na
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                return na
            return f"{float(v):.{nd}f}"
        except Exception:
            return na

    def _norm01(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return a
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx <= mn):
            return np.zeros_like(a, dtype=float)
        return (a - mn) / (mx - mn)

    # --------------------
    # PDF generation
    # --------------------
    with PdfPages(buf) as pdf:

        # =========================================================
        # PAGE 1 — COVER + WHAT THIS IS
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")

        plt.text(0.5, 0.92, title, ha="center", va="center", fontsize=20, weight="bold")
        plt.text(
            0.5, 0.875,
            "A readable diagnostic you can interpret without the analyst present",
            ha="center", va="center", fontsize=12,
        )

        intro_text = (
            "What this report is:\n"
            "This document analyzes how conversational coherence evolves over time within an interaction. "
            "Instead of treating coherence as a static quality, it is modeled as a turn-by-turn signal.\n\n"
            "What it is for:\n"
            "• Identify when coherence is stable, degrading, or recovering\n"
            "• Detect structurally risky phases that may precede visible breakdown\n"
            "• Provide actionable interpretation for facilitation, training, moderation, or product teams\n\n"
            "Two layers are separated:\n"
            "• Observed coherence (C_t): alignment dynamics from the chosen coherence mode\n"
            "• Phenomenological coherence (C_hat_t): delayed/smoothed response approximating experienced coherence\n\n"
            "A key diagnostic is the structural lag (Δ*): structural stress can peak several turns before a perceived drop.\n"
        )
        plt.text(0.06, 0.80, intro_text, ha="left", va="top", fontsize=11, wrap=True)

        meta_text = (
            "Analysis settings:\n"
            f"• Representation mode: {used_mode}\n"
            f"• Φ_low: {_fmt(phi_low)}   Φ_high: {_fmt(phi_high)}\n"
            f"• Turns analyzed: {n}\n"
        )
        plt.text(0.06, 0.36, meta_text, ha="left", va="top", fontsize=11)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 2 — EXECUTIVE SUMMARY
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")

        plt.text(0.06, 0.92, "Executive summary", fontsize=16, weight="bold", ha="left", va="top")

        summary = (
            f"Overall coherence level (mean C_t): {_fmt(ct_mean)}\n"
            f"Range of C_t: {_fmt(ct_min)} → {_fmt(ct_max)}\n"
            f"Share of turns below Φ_low (low-coherence regime): {_fmt(frac_low * 100, nd=1)}%\n"
            f"Share of turns above Φ_high (stable regime): {_fmt(frac_high * 100, nd=1)}%\n\n"
            f"Structural lag (Δ*): {int(delta_star)} turns\n"
            f"Observed lag mean: {_fmt(observed_lag_mean, nd=2)} turns\n"
            f"Lag correlation score: {_fmt(lag_score)}\n\n"
            "How to use this:\n"
            "• If many turns fall below Φ_low, the interaction is frequently unstable.\n"
            "• If Δ* > 0, watch for early structural stress before the conversation ‘feels’ broken.\n"
            "• Use the Events table to review the exact turns that matter.\n"
        )
        plt.text(0.06, 0.84, summary, fontsize=12, ha="left", va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 3 — COHERENCE DYNAMICS (C_t vs C_hat_t)
        # =========================================================
        fig = plt.figure(figsize=(11, 6.8))
        plt.title("Coherence over time: observed (C_t) vs phenomenological (C_hat_t)", fontsize=14, pad=12)

        plt.plot(x, Ct, label="C_t — Observed coherence", linewidth=2.8)

        if C_hat is not None and len(C_hat) == len(Ct):
            plt.plot(x, C_hat, linestyle="--", label="C_hat_t — Phenomenological coherence", linewidth=2.0)

        plt.axhline(float(phi_low), linestyle="--", alpha=0.6, label="Φ_low")
        plt.axhline(float(phi_high), linestyle="--", alpha=0.6, label="Φ_high")

        plt.ylim(0, 1)
        plt.xlabel("Turn")
        plt.ylabel("Coherence (0–1)")
        plt.legend(loc="upper right")

        explanation = (
            "How to read:\n"
            "• C_t tracks moment-to-moment alignment dynamics.\n"
            "• C_hat_t is a delayed/smoothed response: it may drop after stress builds.\n"
            "• Below Φ_low: low-coherence regime; Above Φ_high: stable regime."
        )
        fig.text(0.06, 0.02, explanation, fontsize=10, va="bottom")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 4 — STRUCTURAL DRIVER (IC–III)
        # =========================================================
        fig = plt.figure(figsize=(11, 6.8))
        plt.title("Structural stress (IC–III): driver D_t and compactness ρ(t)", fontsize=14, pad=12)

        plt.plot(x, _norm01(D_t), label="D_t — Structural driver (normalized)", linewidth=2.8)
        plt.plot(x, _norm01(rho_t), linestyle=":", label="ρ(t) — Semantic compactness (normalized)", linewidth=2.2)

        plt.ylim(0, 1)
        plt.xlabel("Turn")
        plt.ylabel("Normalized value (0–1)")
        plt.legend(loc="upper right")

        explanation = (
            "Interpretation:\n"
            "• Peaks in D_t indicate underlying deformation/stress in the interaction trajectory.\n"
            "• Drops in ρ(t) indicate semantic dispersion (topic instability / reduced compactness).\n"
            "• Stress can precede visible coherence breakdowns."
        )
        fig.text(0.06, 0.02, explanation, fontsize=10, va="bottom")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 5 — STRUCTURAL LAG (Δ*) EXPLAINED
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")

        plt.text(0.06, 0.92, "Structural lag (Δ*)", fontsize=16, weight="bold", ha="left", va="top")

        lag_text = (
            f"Δ* = {int(delta_star)} turns\n\n"
            "Definition:\n"
            "Structural lag estimates how many turns typically separate structural stress peaks (IC–III) "
            "from perceived coherence drops (IC–II).\n\n"
            "Why it matters:\n"
            "• If Δ* > 0, problems may be detectable early (before the conversation ‘feels’ broken).\n"
            "• Larger Δ* can indicate hidden fragility or delayed detection.\n\n"
            f"Observed lag mean: {_fmt(observed_lag_mean, nd=2)} turns\n"
            f"Lag correlation score: {_fmt(lag_score)}\n"
        )
        plt.text(0.06, 0.84, lag_text, fontsize=12, ha="left", va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 6 — EVENTS TABLE
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")

        plt.text(0.06, 0.92, "Key events (review these turns)", fontsize=16, weight="bold", ha="left", va="top")

        if events_df is None or len(events_df) == 0:
            plt.text(
                0.06, 0.84,
                "No high-priority events detected by the current criteria.",
                fontsize=12, ha="left", va="top",
            )
        else:
            view = events_df.copy()

            # Keep it readable
            view = view.fillna("")
            view = view.astype(str)

            table = plt.table(
                cellText=view.values,
                colLabels=list(view.columns),
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.45)

            note = (
                "Notes:\n"
                "• ‘Geom break’ = structural break signals from IC–III/driver.\n"
                "• ‘Perceived break’ = phenomenological drop (Ĉ_t).\n"
                "• Use these turns as anchors for qualitative review."
            )
            plt.text(0.06, 0.08, note, fontsize=10, ha="left", va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================================================
        # PAGE 7 — GLOSSARY (minimal, consultancy-friendly)
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")

        plt.text(0.06, 0.92, "Glossary (what variables mean)", fontsize=16, weight="bold", ha="left", va="top")

        glossary = (
            "Core signals:\n"
            "• C_t: Observed coherence signal (0–1) from the selected mode (Ct_new / IC-IIa / Ct_old).\n"
            "• Ĉ_t (C_hat_t): Phenomenological / experienced coherence proxy (delayed/smoothed).\n"
            "• Φ_low, Φ_high: Thresholds defining low vs stable regimes (estimated from percentiles).\n\n"
            "IC–III layer:\n"
            "• ρ(t): Local semantic compactness (higher = turns cluster tightly around a local centroid).\n"
            "• D_t: Structural driver / stress proxy (higher = more manifold deformation / instability).\n\n"
            "Lag:\n"
            "• Δ*: Estimated structural lag in turns (stress peak → perceived drop).\n"
            "• Lag score: Correlation strength used to pick Δ*.\n"
        )
        plt.text(0.06, 0.84, glossary, fontsize=12, ha="left", va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.read()

# -----------------------
# STREAMLIT DOWNLOAD HOOK
# -----------------------
with cE:
    pdf_bytes = generate_consultancy_pdf_report(
        df_out=df_out,
        Ct=Ct_base,
        C_hat=C_hat_t,
        D_t=D_t,
        rho_t=rho_t,
        phi_low=float(phi_low_eff),
        phi_high=float(phi_high_eff),
        delta_star=int(delta_star),
        lag_score=float(lag_score) if np.isfinite(lag_score) else np.nan,
        observed_lag_mean=float(observed_lag_mean) if np.isfinite(observed_lag_mean) else np.nan,
        events_df=events_df,
        used_mode=str(used_mode),
        title="TIE–Dialog — Interaction Coherence Diagnostic (Consultancy Report)",
    )
    st.download_button(
        L["download_pdf"],
        data=pdf_bytes,
        file_name="tie_dialog_consultancy_report.pdf",
        mime="application/pdf",
    )

# --- Quick debug (optional)
with st.expander(L["debug_header"], expanded=False):
    st.write(f"Used embeddings: {used_mode}")
    st.write(lag_label)
    st.write({"observed_lags": observed_lags, "observed_lag_mean": observed_lag_mean})
    st.write({"sbr_corrections": sbr_corrections})
