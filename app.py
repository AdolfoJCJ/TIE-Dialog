from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="TIE–Dialog Ajustado", layout="centered")
lang = "es" if st.sidebar.selectbox("Idioma / Language", ["Español", "English"], 0) == "Español" else "en"
st.title("🧰 TIE–Dialog: Coherencia calibrada, Resonancia, Dimensionalidad, Qualia")

uploaded_file = st.file_uploader("📂 Carga un .csv con columnas 'texto' y 'participante'", type="csv")

# Modelo base ajustado
model = SentenceTransformer('all-mpnet-base-v2')

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
    embs = model.encode(df['texto'].tolist(), convert_to_tensor=True)

    similarities, resonancias, dimensionalidades = [1.0], [0.0], [1.0]
    for i in range(1, len(embs)):
        sim = util.cos_sim(embs[i], embs[i-1]).item()
        similarities.append(sim)

        # Resonancia local (variación en el flujo de similitud)
        if i >= 2:
            delta = util.cos_sim(embs[i], embs[i-1]).item() - util.cos_sim(embs[i-1], embs[i-2]).item()
        else:
            delta = 0
        resonancias.append(abs(sim * delta))

        # Dimensionalidad informacional (magnitud del cambio vectorial)
        dif = embs[i] - embs[i-1]
        dimensionalidades.append(np.linalg.norm(dif.cpu().numpy()))

    df['similarity'] = similarities
    df['R'] = resonancias
    df['D'] = dimensionalidades

    # 🔹 Normalización centrada de C_t
    mean_sim = np.mean(similarities[1:])
    std_sim = np.std(similarities[1:]) or 1.0
    df['C_t'] = ((df['similarity'] - mean_sim) / (2 * std_sim) + 0.5).clip(0.0, 1.0)

    # 🔹 Umbral dinámico con piso
    alpha = 0.2
    base_phi = 0.5
    df['Phi_t'] = np.maximum(base_phi, df['C_t'].mean() + alpha * df['C_t'].std()).clip(0.0, 1.0)

    # 🔹 Rupturas por caída brusca
    sim_deltas = [0.0] + [df.loc[i, 'similarity'] - df.loc[i - 1, 'similarity'] for i in range(1, len(df))]
    df['delta_sim'] = sim_deltas
    df['ruptura'] = (df['delta_sim'] < -0.2).astype(int)
    df.loc[df['ruptura'] == 1, 'Phi_t'] = (df['C_t'] + 0.15).clip(0.0, 1.0)

    # 🔹 Fases
    def clasificar_fase(c, p):
        if c > p:
            return 'Alta coherencia'
        elif c < p - 0.1:
            return 'Incoherencia'
        else:
            return 'Reconfiguración'
    df['fase'] = [clasificar_fase(c, p) for c, p in zip(df['C_t'], df['Phi_t'])]

    # 🔹 Coherencia individual C_i
    if 'participante' in df.columns:
        coherencias_ind = df.groupby('participante')['C_t'].mean().round(3).to_dict()
        df['C_i'] = df['participante'].map(coherencias_ind)
    else:
        df['C_i'] = df['C_t']

    # 🔹 Qualia Q_s
    df['Q_s'] = (df['C_t'] * df['R'] * df['D']).round(4)

    # 🔹 Diagnóstico
    print("\n>> Similaridades crudas:", similarities)
    print(">> C_t promedio:", df['C_t'].mean())
    print(">> Phi_t dinámico:", df['Phi_t'].mean())

    # 🔹 Visualización
    st.subheader("🔢 Métricas: C_t, Φ_t, ℛ, 𝒟, 𝒬ₛ")
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], label='C_t')
    ax.plot(df.index + 1, df['Phi_t'], label='Φ_t', linestyle='--')
    ax.plot(df.index + 1, df['R'], label='ℛ', linestyle=':')
    ax.plot(df.index + 1, df['D'], label='𝒟', linestyle='-.')
    ax.plot(df.index + 1, df['Q_s'], label='𝒬ₛ', linestyle='-')
    ax.set_xlabel("Turno")
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

    # 🔹 Reporte
    st.subheader("📋 Reporte")
    participantes = df['participante'].unique().tolist() if 'participante' in df.columns else []
    resumen = (
        f"Participantes: {', '.join(participantes)}\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Φ_t: {df['Phi_t'].mean():.3f}\n"
        f"Promedio ℛ: {df['R'].mean():.3f}\n"
        f"Promedio 𝒟: {df['D'].mean():.3f}\n"
        f"Promedio 𝒬ₛ: {df['Q_s'].mean():.4f}\n"
        f"Coherencia individual: {coherencias_ind if 'participante' in df.columns else '—'}\n"
        f"Fases: {df['fase'].value_counts().to_dict()}\n"
        f"Rupturas detectadas: {int(df['ruptura'].sum())}\n"
    )
    st.markdown(f"```\n{resumen}\n```")
    st.download_button("📄 Descargar reporte", resumen, "reporte_TIE_Dialog.txt")
    st.download_button("📄 Descargar CSV", df.to_csv(index=False), "datos_TIE_Dialog.csv")
    st.subheader("🔍 Vista previa")
    st.dataframe(df)























