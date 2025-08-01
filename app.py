# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ✅ Verificación del modelo
try:
    model_test = SentenceTransformer('intfloat/e5-large-v2')
    test_embs = model_test.encode(["Uno", "Dos"])
    test_sim = util.cos_sim(test_embs[0], test_embs[1]).item()
    print(f"[Verificación OK] Similaridad: {test_sim:.4f}")
except Exception as e:
    print(f"[Error] No se pudo cargar el modelo: {e}")

# 🌐 Configuración Streamlit
st.set_page_config(page_title="TIE–Dialog Total", layout="centered")
lang = "es" if st.sidebar.selectbox("Idioma / Language", ["Español", "English"], index=0) == "Español" else "en"
st.title("🧰 TIE–Dialog: Coherencia, Resonancia, Dimensionalidad y Qualia")

uploaded_file = st.file_uploader("📂 Carga un archivo .csv con columnas 'texto' y 'participante'", type="csv")

# Datos de prueba
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'turno': list(range(1, 7)),
        'participante': ['Ana', 'Luis'] * 3,
        'texto': [
            "La coherencia informacional puede medirse.",
            "Exacto, se compara el sentido entre turnos.",
            "El cielo está cubierto hoy.",
            "¿Crees que el clima afecta el diálogo?",
            "Mi perro ladra todo el día.",
            "Un buen ejemplo de ruido contextual."
        ]
    })

if 'texto' not in df.columns:
    st.error("❌ El archivo debe tener una columna llamada 'texto'.")
else:
    # 🔹 Cargar modelo y calcular embeddings
    model = SentenceTransformer('intfloat/e5-large-v2')
    embs = model.encode(df['texto'].tolist(), convert_to_tensor=True)

    similarities, resonancias, dimensionalidades = [1.0], [0.0], [1.0]
    for i in range(1, len(embs)):
        context = embs[i-2:i] if i >= 2 else embs[i-1:i]
        sim = util.cos_sim(embs[i], context.mean(dim=0)).item()
        similarities.append(sim)

        # ℛ: resonancia como cambio estructural
        delta = util.cos_sim(embs[i], embs[i-1]).item() - util.cos_sim(embs[i-1], embs[i-2]).item() if i >= 2 else 0
        r = abs(sim * delta)
        resonancias.append(r)

        # 𝒟: magnitud del cambio vectorial
        dif = embs[i] - embs[i-1]
        D = np.linalg.norm(dif.cpu().numpy())
        dimensionalidades.append(D)

    df['similarity'] = similarities
    df['R'] = resonancias
    df['D'] = dimensionalidades

    # 🔹 Coherencia global normalizada (𝒞ₜ)
    min_sim = min(similarities[1:])
    max_sim = max(similarities[1:])
    rng = max_sim - min_sim if max_sim > min_sim else 1.0
    df['C_t'] = ((df['similarity'] - min_sim) / rng).clip(0.0, 1.0)

    # 🔹 Umbral dinámico Φₜ (percentil 80)
    phi_percentil = 80
    phi_t_value = np.percentile(df['C_t'], phi_percentil)
    df['Phi_t'] = phi_t_value

    # 🔹 Rupturas
    sim_deltas = [0.0]
    for i in range(1, len(df)):
        delta = df.loc[i, 'similarity'] - df.loc[i - 1, 'similarity']
        sim_deltas.append(delta)
    df['delta_sim'] = sim_deltas
    df['ruptura'] = (df['delta_sim'] < -0.2).astype(int)
    df.loc[df['ruptura'] == 1, 'Phi_t'] = (df['C_t'] + 0.15).clip(0.0, 1.0)

    # 🔹 Fases
    fases = []
    for c, p in zip(df['C_t'], df['Phi_t']):
        if c > p:
            fases.append('Alta coherencia')
        elif c < p - 0.1:
            fases.append('Incoherencia')
        else:
            fases.append('Reconfiguración')
    df['fase'] = fases

    # 🔹 Coherencia individual (𝒞ᵢ)
    if 'participante' in df.columns:
        coherencias_ind = df.groupby('participante')['C_t'].mean().round(3).to_dict()
        for p in coherencias_ind:
            df.loc[df['participante'] == p, 'C_i'] = coherencias_ind[p]
    else:
        df['C_i'] = df['C_t']

    # 🔹 Métrica qualia (𝒬ₛ)
    df['Q_s'] = (df['C_t'] * df['R'] * df['D']).round(4)

    # 📊 Visualización
    st.subheader("🔢 C_t, Φ_t, ℛ, 𝒟 y 𝒬ₛ")
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], label='C_t')
    ax.plot(df.index + 1, df['Phi_t'], label='Φ_t', linestyle='--')
    ax.plot(df.index + 1, df['R'], label='ℛ', linestyle=':')
    ax.plot(df.index + 1, df['D'], label='D', linestyle='-.')
    ax.plot(df.index + 1, df['Q_s'], label='Qₛ', linestyle='-')
    ax.set_xlabel("Turno")
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

    # 📋 Reporte automático
    participantes = df['participante'].unique().tolist() if 'participante' in df.columns else []
    resumen = (
        f"Participantes: {', '.join(participantes)}\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Φ_t: {df['Phi_t'].mean():.3f}\n"
        f"Promedio ℛ: {df['R'].mean():.3f}\n"
        f"Promedio D: {df['D'].mean():.3f}\n"
        f"Promedio 𝒬ₛ: {df['Q_s'].mean():.4f}\n"
        f"Coherencia individual: {coherencias_ind if 'participante' in df.columns else '—'}\n"
        f"Fases: {df['fase'].value_counts().to_dict()}\n"
        f"Rupturas detectadas: {int(df['ruptura'].sum())}\n"
    )

    st.subheader("🔍 Reporte automático")
    st.markdown(f"```\n{resumen}\n```")
    st.download_button("📄 Descargar reporte", resumen, "reporte_TIE_Dialog.txt")
    st.download_button("📄 Descargar CSV", df.to_csv(index=False), "datos_TIE_Dialog.csv")
    st.subheader("📊 Vista previa")
    st.dataframe(df)





















