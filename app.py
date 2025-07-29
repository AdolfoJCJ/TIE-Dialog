from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 🌐 Configuración e idioma
# -------------------------
st.set_page_config(page_title="TIE–Dialog Multilingüe", layout="centered")
language = st.sidebar.selectbox("Choose language / Elegir idioma", options=["English", "Español"], index=1)
lang = "es" if language == "Español" else "en"

# -------------------------
# ⚙️ Controles avanzados (sidebar)
# -------------------------
st.sidebar.markdown("### ⚙️ Parámetros")
win = st.sidebar.slider("Ventana media/std", 3, 10, 5, 1)
alpha = st.sidebar.slider("α (peso media móvil)", 0.1, 1.0, 0.6, 0.05)
beta  = st.sidebar.slider("β (peso std móvil)",   0.1, 2.0, 0.8, 0.05)
gamma = st.sidebar.slider("γ (penalización exponencial)", 0.0, 2.0, 0.5, 0.05)
delta_thr = st.sidebar.slider("Δ ruptura (caída en C_t)", 0.05, 0.8, 0.30, 0.01)
rupt_margin = st.sidebar.slider("Margen extra en Φₜ tras ruptura", 0.05, 0.5, 0.20, 0.01)

# -------------------------
# 📊 Diccionario de textos
# -------------------------
t = {
    "title": {
        "es": "🧰 TIE–Dialog: Coherencia, Umbral y Fases",
        "en": "🧰 TIE–Dialog: Coherence, Threshold and Phases"
    },
    "upload": {
        "es": "📂 Carga un archivo .csv con columna 'texto' (coherencia se calculará automáticamente)",
        "en": "📂 Upload a .csv file with a 'texto' column (coherence will be calculated automatically)"
    },
    "error": {
        "es": "El archivo debe incluir una columna llamada 'texto'.",
        "en": "The file must include a column named 'texto'."
    },
    "plot_title": {
        "es": "🔢 Evolución de C_t y Φ_t (con rupturas)",
        "en": "🔢 Evolution of C_t and Φ_t (with breaks)"
    },
    "report_title": {
        "es": "🔍 Reporte automático",
        "en": "🔍 Automatic report"
    },
    "download_txt": {
        "es": "📄 Descargar reporte (.txt)",
        "en": "📄 Download report (.txt)"
    },
    "download_csv": {
        "es": "📄 Descargar datos enriquecidos (.csv)",
        "en": "📄 Download enriched data (.csv)"
    },
    "preview": {
        "es": "🔍 Vista previa de resultados:",
        "en": "🔍 Results preview:"
    }
}

# -------------------------
# 📂 Carga CSV y procesamiento
# -------------------------
st.title(t["title"][lang])
uploaded_file = st.file_uploader(t["upload"][lang], type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'turno': list(range(1, 7)),
        'participante': ['Ana', 'Luis', 'Ana', 'Luis', 'Ana', 'Luis'],
        'texto': [
            "La coherencia informacional puede medirse en conversaciones.",
            "Exacto, se calcula comparando el sentido entre turnos.",
            "Hoy llovió mucho en la ciudad.",
            "¿Crees que la lluvia afecta a la comunicación entre personas?",
            "El perro de mi vecino ladra todo el día.",
            "Nada que ver, pero sirve como ejemplo de incoherencia contextual."
        ]
    })

if 'texto' not in df.columns:
    st.error(t["error"][lang])
else:
    # -------------------------
    # 🔹 Calcular embeddings (modelo más sensible)
    # -------------------------
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
    except Exception:
        # Fallback por si no está descargado
        model = SentenceTransformer('all-MiniLM-L6-v2')

    embs = model.encode(df['texto'].tolist(), convert_to_tensor=True)

    # Similaridad coseno consecutiva
    similarities = [1.0]
    for i in range(1, len(embs)):
        sim = util.cos_sim(embs[i], embs[i-1]).item()
        similarities.append(float(sim))
    df['similarity'] = similarities

    # -------------------------
    # 🔹 Normalizar C_t de forma robusta
    #    - Usamos cuantiles para evitar que un outlier eleve todo.
    # -------------------------
    sims_series = pd.Series(similarities[1:]) if len(similarities) > 1 else pd.Series([1.0])
    q05, q95 = sims_series.quantile(0.05), sims_series.quantile(0.95)
    min_sim = float(q05)
    max_sim = float(q95) if q95 > q05 else float(sims_series.max())
    rng = (max_sim - min_sim) if (max_sim > min_sim) else 1.0

    df['C_t'] = ((df['similarity'] - min_sim) / rng).clip(0.0, 1.0)

    # -------------------------
    # 🔹 Umbral Φ_t (dinámico + penalización no lineal)
    # -------------------------
    rolling_mean = df['C_t'].rolling(window=win, min_periods=1).mean()
    rolling_std  = df['C_t'].rolling(window=win, min_periods=1).std().fillna(0.0)

    penalizacion = np.exp(-gamma * rolling_std)  # más std => menos penalización
    phi_base = (alpha * rolling_mean - beta * rolling_std * penalizacion).clip(0.0, 1.0)

    # -------------------------
    # 🔹 Detección de rupturas por caída en C_t
    #     - Si ΔC_t < -delta_thr => ruptura
    #     - Forzamos Φ_t >= C_t + rupt_margin en esos puntos
    # -------------------------
    rupt_flag = [0]
    for i in range(1, len(df)):
        delta = df.loc[i, 'C_t'] - df.loc[i-1, 'C_t']
        rupt_flag.append(1 if delta < -delta_thr else 0)
    df['ruptura'] = rupt_flag

    phi_adj = phi_base.copy()
    idx_rupt = df.index[df['ruptura'] == 1]
    if len(idx_rupt) > 0:
        phi_adj.loc[idx_rupt] = np.maximum(phi_adj.loc[idx_rupt], df.loc[idx_rupt, 'C_t'] + rupt_margin)
    df['Phi_t'] = phi_adj.clip(0.0, 1.0)

    # -------------------------
    # 🔹 Clasificación de fases
    # -------------------------
    fases = []
    for c, p in zip(df['C_t'], df['Phi_t']):
        if c > p:
            fases.append('Alta coherencia' if lang == "es" else "High coherence")
        elif c < p - 0.1:
            fases.append('Incoherencia' if lang == "es" else "Incoherence")
        else:
            fases.append('Reconfiguración' if lang == "es" else "Reconfiguration")
    df['fase'] = fases

    # -------------------------
    # 📊 Gráfico
    # -------------------------
    st.subheader(t["plot_title"][lang])
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], label='C_t (normalizado)')
    ax.plot(df.index + 1, df['Phi_t'], label='Phi_t (umbral dinámico)', linestyle='--')
    # Marcas de ruptura
    if df['ruptura'].sum() > 0:
        ax.scatter(df.index[df['ruptura'] == 1] + 1,
                   df.loc[df['ruptura'] == 1, 'C_t'],
                   label='Ruptura (ΔC_t)', marker='x')
    ax.set_xlabel('Turno' if lang == 'es' else 'Turn')
    ax.set_ylabel('Valor' if lang == 'es' else 'Value')
    ax.legend()
    st.pyplot(fig)

    # -------------------------
    # 📋 Reporte
    # -------------------------
    st.subheader(t["report_title"][lang])
    participantes = df['participante'].unique().tolist() if 'participante' in df.columns else []
    porcentaje_supera = (df['C_t'] > df['Phi_t']).mean() * 100
    conteo_fases = df['fase'].value_counts().to_dict()
    num_rupturas = int(df['ruptura'].sum())

    texto = (
        f"Participantes: {', '.join(participantes) if participantes else '—'}\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
        f"Turnos con C_t > Phi_t: {porcentaje_supera:.1f}%\n"
        f"Rupturas detectadas (ΔC_t < -{delta_thr:.2f}): {num_rupturas}\n"
        f"Fases: {conteo_fases}\n"
        f"Parámetros: ventana={win}, α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, Δ={delta_thr:.2f}, margen={rupt_margin:.2f}\n"
    )
    if porcentaje_supera > 90:
        texto += ("⚠️ Advertencia: la mayoría de los turnos supera Φₜ. Revisa parámetros o dataset.\n"
                  if lang == 'es' else "⚠️ Warning: most turns exceed Φₜ. Review parameters or dataset.\n")

    st.markdown(f"```\n{texto}\n```")

    # -------------------------
    # 📄 Descargas
    # -------------------------
    st.download_button(t["download_txt"][lang], data=texto,
                       file_name="reporte_TIE_Dialog.txt", mime="text/plain")
    st.download_button(t["download_csv"][lang], data=df.to_csv(index=False),
                       file_name="datos_TIE_Dialog.csv", mime="text/csv")

    # -------------------------
    # Vista previa final
    # -------------------------
    st.subheader(t["preview"][lang])
    st.dataframe(df)














