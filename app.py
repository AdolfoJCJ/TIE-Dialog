import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Cargar modelo y forzar CPU si no hay GPU disponible
model = SentenceTransformer('all-mpnet-base-v2')
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Multilenguaje
lang = st.selectbox("\ud83c\udf10 Choose language / Elige idioma", ["Espa\u00f1ol", "English"])
text = {
    "Espa\u00f1ol": {
        "title": "\ud83e\udde0 TIE–Dialog: An\u00e1lisis Avanzado de Coherencia Dialogal",
        "uploader": "\ud83d\udcc2 Carga un archivo .csv con columnas 'turno', 'participante' y 'texto'",
        "slider": "Umbral de ruptura (Φ)",
        "upload_success": "\u2705 Archivo cargado correctamente.",
        "upload_error": "\u274c El archivo debe tener columnas 'texto' y 'participante'.",
        "chart_title": "\ud83d\udcc8 Curva de Coherencia C_t",
        "ruptures_title": "\ud83d\udcc9 Rupturas detectadas",
        "no_ruptures": "\u2705 No se detectaron rupturas informacionales.",
        "ruptures_found": "\u26a0\ufe0f Se detectaron {} rupturas.",
        "report_title": "\ud83d\udccb Informe estructural",
        "table_title": "\ud83d\udcc4 Tabla completa de an\u00e1lisis",
        "export_csv": "\u2b07\ufe0f Exportar resultados como CSV",
        "download_pdf": "\ud83d\udcc4 Descargar informe PDF (no disponible)",
        "graph_title": "\ud83d\udded Mapa de nodos informacionales"
    },
    "English": {
        "title": "\ud83e\udde0 TIE–Dialog: Advanced Dialog Coherence Analysis",
        "uploader": "\ud83d\udcc2 Upload a .csv file with columns 'turno', 'participante' and 'texto'",
        "slider": "Rupture Threshold (Φ)",
        "upload_success": "\u2705 File successfully loaded.",
        "upload_error": "\u274c File must have 'texto' and 'participante' columns.",
        "chart_title": "\ud83d\udcc8 Coherence Curve C_t",
        "ruptures_title": "\ud83d\udcc9 Detected Ruptures",
        "no_ruptures": "\u2705 No informational ruptures detected.",
        "ruptures_found": "\u26a0\ufe0f {} ruptures detected.",
        "report_title": "\ud83d\udccb Structural Summary",
        "table_title": "\ud83d\udcc4 Full Analysis Table",
        "export_csv": "\u2b07\ufe0f Export results as CSV",
        "download_pdf": "\ud83d\udcc4 Download PDF Report (not available)",
        "graph_title": "\ud83d\udded Informational Node Map"
    }
}

st.set_page_config(page_title="TIE–Dialog", layout="wide")
st.title(text[lang]["title"])

uploaded_file = st.file_uploader(text[lang]["uploader"], type="csv")
rupture_threshold = st.slider(text[lang]["slider"], 0.0, 1.0, 0.65)

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
    st.error(text[lang]["upload_error"])
    st.stop()
else:
    st.success(text[lang]["upload_success"])

textos = df['texto'].tolist()
participantes = df['participante'].tolist()
embs = model.encode(textos, convert_to_tensor=True)

C_t = [1.0]
rupturas = [False]
for i in range(1, len(embs)):
    sim = util.cos_sim(embs[i], embs[i - 1]).item()
    C_t.append(round(sim, 4))
    rupturas.append(sim < rupture_threshold)
df['C_t'] = C_t
df['ruptura'] = rupturas

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

st.subheader(text[lang]["ruptures_title"])
rupt_df = df[df['ruptura']]
if rupt_df.empty:
    st.success(text[lang]["no_ruptures"])
else:
    st.warning(text[lang]["ruptures_found"].format(len(rupt_df)))
    st.dataframe(rupt_df[['turno', 'participante', 'texto', 'C_t']])

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

st.subheader(text[lang]["table_title"])
st.dataframe(df[['turno', 'participante', 'texto', 'C_t', 'ruptura'] + [f'C_t_{p}' for p in coherencia_individual]])

st.download_button(
    label=text[lang]["export_csv"],
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="TIE-Dialog-resultados.csv",
    mime="text/csv"
)

st.info(text[lang]["download_pdf"])

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

# Explicación de resultados
st.markdown("### 🧠 " + ("Interpretación de Resultados" if lang == "Español" else "Interpretation of Results"))
if lang == "Español":
    st.markdown("""
**Coherencia global (`Cₕ`)** representa la continuidad informacional entre turnos consecutivos. Valores cercanos a 1.0 indican que la conversación fluye sin saltos temáticos importantes.

**El umbral Φ** es el punto a partir del cual se considera que hay una **ruptura informacional**. Una ruptura significa que el diálogo pierde coherencia, cambia bruscamente de tema o aparece ruido conceptual.

**Coherencia individual (`Cₕₚ`)** indica cuánto se mantiene consistente cada participante consigo mismo. Comparar estas curvas permite ver quién mantiene su foco conversacional más estable.

**El informe estructural** resume los valores clave: promedio de coherencia, número de rupturas y el turno con menor continuidad. Ayuda a evaluar la calidad del diálogo como sistema coherente.

**El grafo de nodos informacionales** representa cada turno como un nodo y la coherencia como enlaces. Enlaces más fuertes significan más continuidad. Un grafo disperso o débil indica ruptura de sentido.
""")
else:
    st.markdown("""
**Global coherence (`Cₕ`)** reflects the informational continuity between consecutive turns. Values near 1.0 mean the conversation flows smoothly without major thematic jumps.

**The Φ threshold** defines when a **rupture** is detected — a point where the dialogue loses coherence, shifts abruptly in topic, or introduces noise.

**Individual coherence (`Cₕₚ`)** shows how consistent each participant is with their own previous turns. Comparing them reveals who maintains a more stable focus.

**The structural summary** highlights key values: average coherence, rupture count, and the least coherent turn. It helps assess the dialogue as a coherent system.

**The informational node graph** shows each turn as a node, with coherence as edges. Stronger edges indicate smoother flow. A scattered graph means meaning is breaking down.
""")





























