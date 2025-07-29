# TIE-Dialog multilingüe completo con embeddings, sin normalización, y umbral ajustado a coseno real
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 🌐 Selector de idioma
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
        "es": "📂 Carga un archivo .csv con columnas 'texto' y 'participante' (o deja vacío para prueba)",
        "en": "📂 Upload a .csv file with 'texto' and 'participante' columns (or leave empty to test)"
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
# 📂 Carga CSV o datos de prueba
# -------------------------
st.title(t["title"][lang])
uploaded_file = st.file_uploader(t["upload"][lang], type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        'turno': range(1, 11),
        'participante': ['Ana', 'Luis', 'Ana', 'Luis', 'Ana', 'Luis', 'Ana', 'Luis', 'Ana', 'Luis'],
        'texto': [
            "¿Has leído la teoría de la emergencia informacional?",
            "Sí, dice que la consciencia emerge del acoplamiento informacional.",
            "Exacto, y que la coherencia es clave para que surja la perspectiva.",
            "También mencionan I_s e I_m como componentes del sistema.",
            "Eso permite medir cómo evoluciona la coherencia en el tiempo.",
            "Y nos ayuda a detectar cuándo un sistema cruza el umbral Phi_t.",
            "Así podemos diseñar interfaces que mantengan el sentido estable.",
            "Incluso podríamos usarlo en IA para alinear sistemas con el contexto.",
            "Sí, representando el espacio de qualia como topología coherente.",
            "Ese enfoque puede cambiar por completo nuestra comprensión de la mente."
        ]
    })

# -------------------------
# 🔎 Cálculo de coherencia con embeddings (sin normalización)
# -------------------------
if 'texto' not in df.columns:
    st.error(t["error"][lang])
    st.stop()

with st.spinner("Calculando coherencia informacional..."):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    textos = df['texto'].astype(str).tolist()
    embeddings = model.encode(textos, convert_to_tensor=True)
    coherencias = []
    for i in range(len(embeddings) - 1):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
        coherencias.append(sim)
    coherencias.append(coherencias[-1])
    df['coherencia'] = coherencias

# -------------------------
# 🔢 Cálculo de métricas completas
# -------------------------
df['C_t'] = df['coherencia'].astype(float)
df['C_t_local'] = df['C_t'].rolling(5, min_periods=1).mean()
df['C_t_Im'] = df['C_t'].ewm(span=8, adjust=False).mean()

phi_0, alpha, beta = 0.3, 0.4, 0.3
phi_vals = []
for i in range(len(df)):
    if i < 3:
        phi_vals.append(phi_0)
    else:
        w = df['C_t'][i-3:i]
        phi_t = phi_0 + alpha * w.std() - beta * w.mean()
        phi_vals.append(max(-1, min(1, float(phi_t))))
df['Phi_t'] = phi_vals

fases = []
for c, p in zip(df['C_t'], df['Phi_t']):
    if c > p:
        fases.append('Alta coherencia' if lang == "es" else "High coherence")
    elif c < p - 0.1:
        fases.append('Incoherencia' if lang == "es" else "Incoherence")
    else:
        fases.append('Reconfiguración' if lang == "es" else "Reconfiguration")
df['fase'] = fases









