import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import numpy as np

st.set_page_config(page_title="TIE–Dialog Avanzado", layout="wide")
st.title("🧠 TIE–Dialog: Análisis Avanzado de Coherencia Dialogal")

uploaded_file = st.file_uploader("📂 Carga un archivo .csv con columnas 'turno', 'participante' y 'texto'", type="csv")
model = SentenceTransformer('all-mpnet-base-v2')

rupture_threshold = st.slider("Umbral de ruptura (Φ)", 0.0, 1.0, 0.65)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'turno': [1, 2, 3, 4],
        'participante': ['A', 'B', 'A', 'B'],
        'texto': [
            "Hola, ¿cómo estás hoy?",
            "Estoy bien, gracias. ¿Y tú?",
            "También estoy bien. ¿Qué planes tienes?",
            "Quiero salir a caminar un rato por el parque."
        ]
    })

if not {'texto', 'participante'}.issubset(df.columns):
    st.error("❌ El archivo debe tener columnas 'texto' y 'participante'.")
else:
    st.success("✅ Archivo cargado correctamente.")

    textos = df['texto'].tolist()
    participantes = df['participante'].tolist()
    embs = model.encode(textos, convert_to_tensor=True)

    # Coherencia global entre turnos
    C_t = [1.0]
    rupturas = [False]
    for i in range(1, len(embs)):
        sim = util.cos_sim(embs[i], embs[i - 1]).item()
        C_t.append(round(sim, 4))
        rupturas.append(sim < rupture_threshold)
    df['C_t'] = C_t
    df['ruptura'] = rupturas

    # Coherencia individual por participante
    coherencia_individual = {}
    for p in set(participantes):
        idxs = df[df['participante'] == p].index
        coh = [1.0]
        for i in range(1, len(idxs)):
            sim = util.cos_sim(embs[idxs[i]], embs[idxs[i - 1]]).item()
            coh.append(round(sim, 4))
        coh_full = [np.nan] * len(df)
        for i, idx in enumerate(idxs[1:]):
            coh_full[idx] = coh[i + 1]
        coherencia_individual[p] = coh_full
        df[f'C_t_{p}'] = coh_full

    # Visualización
    st.subheader("📈 Curva de Coherencia C_t")
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], marker='o', label='C_t (global)')
    for p in coherencia_individual:
        ax.plot(df.index + 1, df[f'C_t_{p}'], marker='x', linestyle='--', label=f'C_t ({p})')
    ax.axhline(y=rupture_threshold, color='r', linestyle=':', label='Umbral de ruptura Φ')
    ax.set_xlabel("Turno")
    ax.set_ylabel("Coherencia")
    ax.set_ylim(0, 1.1)
    ax.legend()
    st.pyplot(fig)

    # Informe de rupturas
    st.subheader("📉 Rupturas detectadas")
    rupt_df = df[df['ruptura']]
    if rupt_df.empty:
        st.success("✅ No se detectaron rupturas informacionales.")
    else:
        st.warning(f"⚠️ Se detectaron {len(rupt_df)} rupturas.")
        st.dataframe(rupt_df[['turno', 'participante', 'texto', 'C_t']])

    # Informe resumen estructural
    st.subheader("📋 Informe estructural")
    resumen = {
        "Coherencia promedio (global)": round(df['C_t'].mean(), 4),
        "Número de rupturas detectadas": int(df['ruptura'].sum()),
        "Turno de menor coherencia": int(df['C_t'].idxmin() + 1),
    }
    for p in coherencia_individual:
        valores = df[f'C_t_{p}'].dropna()
        if not valores.empty:
            resumen[f'Coherencia promedio ({p})'] = round(valores.mean(), 4)
    st.json(resumen)

    # Tabla final
    st.subheader("📄 Tabla completa de análisis")
    st.dataframe(df[['turno', 'participante', 'texto', 'C_t', 'ruptura'] + [f'C_t_{p}' for p in coherencia_individual]])

























