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
        "es": "🔢 Evolución de C_t y Φ_t (umbral)",
        "en": "🔢 Evolution of C_t and Φ_t (threshold)"
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
    # 🔹 Calcular embeddings
    # -------------------------
    model = SentenceTransformer('all-mpnet-base-v2')
    embs = model.encode(df['texto'].tolist(), convert_to_tensor=True)

    # Similaridades por turno
    similarities = [1.0]
    for i in range(1, len(embs)):
        sim = util.cos_sim(embs[i], embs[i - 1]).item()
        similarities.append(sim)
    df['similarity'] = similarities

    # Normalización robusta
    min_sim = min(similarities[1:])
    max_sim = max(similarities[1:])
    rng = max_sim - min_sim if max_sim > min_sim else 1.0
    df['C_t'] = ((df['similarity'] - min_sim) / rng).clip(0.0, 1.0)

    # -------------------------
    # 🔹 Umbral Φ_t dinámico + sensibilidad ajustable
    # -------------------------
    media_ct = df['C_t'].mean()
    desv_ct = df['C_t'].std()
    sensibilidad = 0.5  # Puedes ajustar este valor entre 0.3 y 0.7 según la rigurosidad deseada
    df['Phi_t'] = (media_ct + sensibilidad * desv_ct).clip(0.0, 1.0)

    # Detectar caídas abruptas
    sim_deltas = [0.0]
    for i in range(1, len(df)):
        delta = df.loc[i, 'similarity'] - df.loc[i - 1, 'similarity']
        sim_deltas.append(delta)
    df['delta_sim'] = sim_deltas

    # Si hay una caída fuerte, ajustar el umbral localmente
    ruptura_mask = df['delta_sim'] < -0.2
    df['ruptura'] = ruptura_mask.astype(int)
    df.loc[ruptura_mask, 'Phi_t'] = (df['C_t'] + 0.15).clip(0.0, 1.0)

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
    ax.plot(df.index + 1, df['Phi_t'], label='Phi_t (umbral)', linestyle='--')
    ax.scatter(df[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')].index + 1,
               df[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')]['C_t'],
               color='red', label='Incoherencia', marker='x')
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
    porcentaje_supera = (df['C_t'] > df['Phi_t']).mean() * 100
    conteo_fases = df['fase'].value_counts().to_dict()
    num_rupturas = int(df['ruptura'].sum())

    texto = (
        f"Participantes: {', '.join(participantes) if participantes else '—'}\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
        f"Turnos con C_t > Phi_t: {porcentaje_supera:.1f}%\n"
        f"Rupturas detectadas: {num_rupturas}\n"
        f"Fases: {conteo_fases}\n"
    )
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

















