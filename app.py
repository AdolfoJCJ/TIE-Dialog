import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="TIE–Dialog Básico", layout="centered")
st.title("🧰 TIE–Dialog: Versión Básica (Similitud entre turnos)")

uploaded_file = st.file_uploader("📂 Carga un archivo .csv con columna 'texto'", type="csv")

model = SentenceTransformer('all-mpnet-base-v2')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'turno': [1, 2, 3, 4],
        'texto': [
            "Hola, ¿cómo estás hoy?",
            "Estoy bien, gracias. ¿Y tú?",
            "También estoy bien. ¿Qué planes tienes?",
            "Quiero salir a caminar un rato por el parque."
        ]
    })

if 'texto' not in df.columns:
    st.error("❌ El archivo debe tener una columna llamada 'texto'.")
else:
    st.write("✅ Texto cargado correctamente.")
    
    textos = df['texto'].tolist()
    embs = model.encode(textos, convert_to_tensor=True)

    # Calcular similitudes consecutivas
    similarities = [1.0]
    for i in range(1, len(embs)):
        sim = util.cos_sim(embs[i], embs[i - 1]).item()
        similarities.append(round(sim, 4))
    df['C_t'] = similarities

    # Visualizar
    st.subheader("📊 Coherencia C_t entre turnos consecutivos")
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], marker='o', label='C_t')
    ax.set_xlabel("Turno")
    ax.set_ylabel("C_t (similitud)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    st.pyplot(fig)

    # Mostrar tabla
    st.subheader("🔍 Datos")
    st.dataframe(df[['turno', 'texto', 'C_t']])
























