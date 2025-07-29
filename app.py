from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 🌐 Configuración e idioma
# -------------------------
st.set_page_config(page_title="TIE–Dialog Multilingüe", layout="centered")
language = st.sidebar.selectbox("Choose language / Elegir idioma", options=["English", "Español"], index=1)
lang = "es" if language == "Español" else "en"

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
        "es": "🔢 Evolución de C_t, C_t_local, C_t_Im y Phi_t",
        "en": "🔢 Evolution of C_t, C_t_local, C_t_Im and Phi_t"
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
    # Calcular embeddings y similitud
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['texto'].tolist(), convert_to_tensor=True)
    similarities_raw = [1.0]  # Primer turno no tiene anterior
    for i in range(1, len(embeddings)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        similarities_raw.append(sim)
    df['C_t_raw'] = similarities_raw

    # Normalizar C_t al rango observado (más realista)
    min_sim, max_sim = 0.5, 0.95
    df['C_t'] = ((df['C_t_raw'] - min_sim) / (max_sim - min_sim)).clip(0, 1)

    # Calcular medias móviles
    df['C_t_local'] = df['C_t'].rolling(3, min_periods=1).mean()
    df['C_t_Im'] = df['C_t'].ewm(span=4, adjust=False).mean()

    # Calcular Phi_t dinámico ajustado
    rolling_mean = df['C_t'].rolling(window=5, min_periods=1).mean()
    rolling_std = df['C_t'].rolling(window=5, min_periods=1).std().fillna(0)
    df['Phi_t'] = (rolling_mean + 0.5 * rolling_std - 0.15 * rolling_mean).clip(0, 1)

    # Clasificación de fases
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
    ax.plot(df.index + 1, df['C_t_local'], label='C_t_local')
    ax.plot(df.index + 1, df['C_t_Im'], label='C_t_Im')
    ax.plot(df.index + 1, df['Phi_t'], label='Phi_t', linestyle='--')
    ax.scatter(df[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')].index + 1,
               df[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')]['C_t'],
               color='red', label='Ruptura', marker='x')
    ax.scatter(df[df['fase'] == ('Alta coherencia' if lang == 'es' else 'High coherence')].index + 1,
               df[df['fase'] == ('Alta coherencia' if lang == 'es' else 'High coherence')]['C_t'],
               color='green', label='Emergencia', marker='o')
    ax.set_xlabel('Turno' if lang == 'es' else 'Turn')
    ax.set_ylabel('Valor' if lang == 'es' else 'Value')
    ax.legend()
    st.pyplot(fig)

    # -------------------------
    # 📋 Reporte
    # -------------------------
    st.subheader(t["report_title"][lang])
    participantes = df['participante'].unique().tolist() if 'participante' in df.columns else []
    coherencia_total = float(df['C_t'].sum())
    porcentaje_supera = (df['C_t'] > df['Phi_t']).mean() * 100
    texto = (
        f"Participantes: {', '.join(participantes) if participantes else '—'}\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
        f"Turnos con C_t > Phi_t: {porcentaje_supera:.1f}%\n"
    )

    if porcentaje_supera > 90:
        texto += ("⚠️ Advertencia: el umbral Φₜ podría estar demasiado bajo o los turnos son muy coherentes.\n" 
                  if lang == 'es' else "⚠️ Warning: Φₜ threshold may be too low or turns are highly coherent.\n")

    st.markdown(f"```\n{texto}\n```")

    # -------------------------
    # 📄 Descargas
    # -------------------------
    st.download_button(t["download_txt"][lang], data=texto, file_name="reporte_TIE_Dialog.txt", mime="text/plain")
    st.download_button(t["download_csv"][lang], data=df.to_csv(index=False), file_name="datos_TIE_Dialog.csv", mime="text/csv")

    # -------------------------
    # Vista previa final
    # -------------------------
    st.subheader(t["preview"][lang])
    st.dataframe(df)










