
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from io import StringIO

# 🌍 Configuración e idioma
st.set_page_config(page_title="TIE–Dialog", layout="wide")
idioma = st.selectbox("Idioma / Language", ["Español", "English"])

textos = {
    "Español": {
        "titulo": "🧠 TIE–Dialog — Análisis de coherencia informacional",
        "sube": "Sube un archivo `.csv` con columnas: `speaker`, `timestamp`, `text`",
        "cargar": "Subir diálogo",
        "vista_previa": "Vista previa:",
        "graf1": "📈 Coherencia local y con Im",
        "graf2": "👥 Evolución por hablante (coherencia con Im)",
        "rupturas": "🧨 Rupturas informacionales detectadas:",
        "descargar": "Descargar resultados como CSV",
        "boton_descarga": "⬇️ Descargar CSV"
    },
    "English": {
        "titulo": "🧠 TIE–Dialog — Informational Coherence Analysis",
        "sube": "Upload a `.csv` file with columns: `speaker`, `timestamp`, `text`",
        "cargar": "Upload dialogue",
        "vista_previa": "Preview:",
        "graf1": "📈 Local coherence and Im coherence",
        "graf2": "👥 Individual evolution (coherence with Im)",
        "rupturas": "🧨 Detected informational ruptures:",
        "descargar": "Download results as CSV",
        "boton_descarga": "⬇️ Download CSV"
    }
}
t = textos[idioma]

# 🧠 Cargar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📥 Cargar archivo
st.title(t["titulo"])
st.markdown(t["sube"])
archivo = st.file_uploader(t["cargar"], type=["csv", "txt"])

if archivo is not None:
    if archivo.type == "text/csv":
        df = pd.read_csv(archivo)
    else:
        contenido = archivo.read().decode("utf-8")
        df = pd.read_csv(StringIO(contenido))

    st.subheader(t["vista_previa"])
    st.dataframe(df.head())

    # 🔁 Embeddings y coherencia
    textos = df["text"].astype(str).tolist()
    embeddings = model.encode(textos, convert_to_tensor=True)

    # Coherencia local
    coherencia_local = [None]
    for i in range(1, len(embeddings)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        coherencia_local.append(sim)
    df["coherencia_local"] = coherencia_local

    # Coherencia con Im
    coherencia_Im = [None]
    for i in range(1, len(embeddings)):
        Im_t = sum(embeddings[:i]) / len(embeddings[:i])
        coherence = util.cos_sim(embeddings[i], Im_t).item()
        coherencia_Im.append(coherence)
    df["coherencia_Im"] = coherencia_Im

    # 📈 Gráfico 1
    st.subheader(t["graf1"])
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(coherencia_local)), coherencia_local, marker='x', label="Coherencia local")
    plt.plot(range(len(coherencia_Im)), coherencia_Im, marker='o', label="Coherencia con Im")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

    # 👥 Tracker por hablante
    st.subheader(t["graf2"])
    plt.figure(figsize=(10, 5))
    for speaker in df["speaker"].unique():
        datos = df[df["speaker"] == speaker]
        plt.plot(datos.index, datos["coherencia_Im"], marker='o', label=f"{speaker}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

    # 🧨 Rupturas
    rupturas = [None]
    for i in range(1, len(df)):
        media_anterior = df["coherencia_Im"][:i].mean()
        actual = df["coherencia_Im"][i]
        if actual < 0.75 * media_anterior:
            rupturas.append(True)
        else:
            rupturas.append(False)
    df["ruptura"] = rupturas

    st.subheader(t["rupturas"])
    rupt_df = df[df["ruptura"] == True][["speaker", "text", "coherencia_Im"]]
    if not rupt_df.empty:
        st.dataframe(rupt_df)
    else:
        st.write("—")

    # ⬇️ Exportar resultados
    st.subheader(t["descargar"])
    if st.button(t["boton_descarga"]):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(t["boton_descarga"], csv, "resultados_tie_dialog.csv", "text/csv")
