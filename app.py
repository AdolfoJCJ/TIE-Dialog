import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from fpdf import FPDF

# Forzar uso de CPU
device = "cpu"
model = SentenceTransformer('all-mpnet-base-v2', device=torch.device(device))

# Multilenguaje
lang = st.selectbox("🌐 Choose language / Elige idioma", ["Español", "English"])

# Textos multilingües
text = {
    "Español": {
        "title": "🧠 TIE–Dialog: Análisis Avanzado de Coherencia Dialogal",
        "uploader": "📂 Carga un archivo .csv con columnas 'turno', 'participante' y 'texto'",
        "slider": "Umbral de ruptura (Φ)",
        "upload_success": "✅ Archivo cargado correctamente.",
        "upload_error": "❌ El archivo debe tener columnas 'texto' y 'participante'.",
        "chart_title": "📈 Curva de Coherencia C_t",
        "ruptures_title": "📉 Rupturas detectadas",
        "no_ruptures": "✅ No se detectaron rupturas informacionales.",
        "ruptures_found": "⚠️ Se detectaron {} rupturas.",
        "report_title": "📋 Informe estructural",
        "table_title": "📄 Tabla completa de análisis",
        "export_csv": "⬇️ Exportar resultados como CSV",
        "download_pdf": "📄 Descargar informe PDF",
        "graph_title": "🧭 Mapa de nodos informacionales"
    },
    "English": {
        "title": "🧠 TIE–Dialog: Advanced Dialog Coherence Analysis",
        "uploader": "📂 Upload a .csv file with columns 'turno', 'participante' and 'texto'",
        "slider": "Rupture Threshold (Φ)",
        "upload_success": "✅ File successfully loaded.",
        "upload_error": "❌ File must have 'texto' and 'participante' columns.",
        "chart_title": "📈 Coherence Curve C_t",
        "ruptures_title": "📉 Detected Ruptures",
        "no_ruptures": "✅ No informational ruptures detected.",
        "ruptures_found": "⚠️ {} ruptures detected.",
        "report_title": "📋 Structural Summary",
        "table_title": "📄 Full Analysis Table",
        "export_csv": "⬇️ Export results as CSV",
        "download_pdf": "📄 Download PDF Report",
        "graph_title": "🧭 Informational Node Map"
    }
}

# UI
st.set_page_config(page_title="TIE–Dialog Avanzado", layout="wide")
st.title(text[lang]["title"])
uploaded_file = st.file_uploader(text[lang]["uploader"], type="csv")
rupture_threshold = st.slider(text[lang]["slider"], 0.0, 1.0, 0.65)

# Datos por defecto si no se sube archivo
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

# Validación
if not {'texto', 'participante'}.issubset(df.columns):
    st.error(text[lang]["upload_error"])
    st.stop()
else:
    st.success(text[lang]["upload_success"])

# Embeddings
textos = df['texto'].tolist()
participantes = df['participante'].tolist()
embs = model.encode(textos, convert_to_tensor=True)

# Coherencia entre turnos consecutivos
C_t = [1.0]
rupturas = [False]
for i in range(1, len(embs)):
    sim = util.cos_sim(embs[i], embs[i - 1]).item()
    C_t.append(round(sim, 4))
    rupturas.append(sim < rupture_threshold)
df['C_t'] = C_t
df['ruptura'] = rupturas

# Coherencia por participante
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

# Visualización curva C_t
st.subheader(text[lang]["chart_title"])
fig, ax = plt.subplots()
ax.plot(df.index + 1, df['C_t'], marker='o', label='C_t (global)')
for p in coherencia_individual:
    ax.plot(df.index + 1, df[f'C_t_{p}'], marker='x', linestyle='--', label=f'C_t ({p})')
ax.axhline(y=rupture_threshold, color='r', linestyle=':', label='Φ')
ax.set_xlabel("Turno")
ax.set_ylabel("Coherencia")
ax.set_ylim(0, 1.1)
ax.legend()
st.pyplot(fig)

# Rupturas
st.subheader(text[lang]["ruptures_title"])
rupt_df = df[df['ruptura']]
if rupt_df.empty:
    st.success(text[lang]["no_ruptures"])
else:
    st.warning(text[lang]["ruptures_found"].format(len(rupt_df)))
    st.dataframe(rupt_df[['turno', 'participante', 'texto', 'C_t']])

# Informe estructural
st.subheader(text[lang]["report_title"])
resumen = {
    "C_t (global)": round(df['C_t'].mean(), 4),
    "Rupturas": int(df['ruptura'].sum()),
    "Turno mínima coherencia": int(df['C_t'].idxmin() + 1),
}
for p in coherencia_individual:
    valores = df[f'C_t_{p}'].dropna()
    if not valores.empty:
        resumen[f'C_t promedio ({p})'] = round(valores.mean(), 4)
st.json(resumen)

# Tabla
st.subheader(text[lang]["table_title"])
st.dataframe(df[['turno', 'participante', 'texto', 'C_t', 'ruptura'] + [f'C_t_{p}' for p in coherencia_individual]])

# Exportar CSV
st.download_button(
    label=text[lang]["export_csv"],
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="TIE-Dialog-resultados.csv",
    mime="text/csv"
)

# Informe PDF
def generar_pdf(resumen, rupt_df, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=text[lang]["report_title"], ln=True)
    for key, value in resumen.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.cell(200, 10, txt="", ln=True)
    pdf.cell(200, 10, txt=text[lang]["ruptures_title"], ln=True)
    for _, row in rupt_df.iterrows():
        pdf.multi_cell(0, 10, f"{row['turno']} - {row['participante']}: {row['texto']} (C_t={row['C_t']})")
    return pdf.output(dest='S').encode('latin-1')

pdf_bytes = generar_pdf(resumen, rupt_df, lang)
st.download_button(
    label=text[lang]["download_pdf"],
    data=pdf_bytes,
    file_name="TIE-Dialog-informe.pdf",
    mime="application/pdf"
)

# Mapa de nodos
st.subheader(text[lang]["graph_title"])
G = nx.Graph()
for i in range(len(df) - 1):
    G.add_node(i + 1, label=df.loc[i, 'texto'][:30] + "...")
    sim = df.loc[i + 1, 'C_t']
    G.add_edge(i + 1, i + 2, weight=sim)

pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight'] for u, v in G.edges()]
plt.figure(figsize=(10, 5))
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
st.pyplot(plt)


























