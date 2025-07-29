# TIE-Dialog multilingüe completo con embeddings, métricas, reporte y exportación (con normalización y umbral ajustado)
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
# 🔎 Cálculo de coherencia con embeddings
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
        sim = (sim + 1) / 2  # Normaliza a rango [0, 1]
        coherencias.append(sim)
    coherencias.append(coherencias[-1])
    df['coherencia'] = coherencias

# -------------------------
# 🔢 Cálculo de métricas completas
# -------------------------
df['C_t'] = df['coherencia'].astype(float)
df['C_t_local'] = df['C_t'].rolling(5, min_periods=1).mean()
df['C_t_Im'] = df['C_t'].ewm(span=8, adjust=False).mean()

phi_0, alpha, beta = 0.65, 0.3, 0.1
phi_vals = []
for i in range(len(df)):
    if i < 3:
        phi_vals.append(phi_0)
    else:
        w = df['C_t'][i-3:i]
        phi_t = phi_0 + alpha * w.std() - beta * w.mean()
        phi_vals.append(max(0, min(1, float(phi_t))))
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

# -------------------------
# 📊 Gráfico
# -------------------------
st.subheader(t["plot_title"][lang])
fig, ax = plt.subplots()
ax.plot(df.index + 1, df['C_t'], label='C_t')
ax.plot(df.index + 1, df['C_t_local'], label='C_t_local')
ax.plot(df.index + 1, df['C_t_Im'], label='C_t_Im')
ax.plot(df.index + 1, df['Phi_t'], label='Phi_t', linestyle='--')
ax.legend()
st.pyplot(fig)

# -------------------------
# 📋 Reporte
# -------------------------
st.subheader(t["report_title"][lang])
participantes = df['participante'].unique().tolist() if 'participante' in df.columns else []
coherencia_total = float(df['C_t'].sum())
if 'participante' in df.columns:
    ranking_df = df.groupby('participante').agg(
        C_t_sum=('C_t', 'sum'),
        turnos=('C_t', 'count'),
        C_t_mean=('C_t', 'mean')
    ).sort_values('C_t_sum', ascending=False).reset_index()
    ranking_df['share_pct'] = (ranking_df['C_t_sum'] / coherencia_total * 100.0).round(2)
else:
    ranking_df = pd.DataFrame(columns=['participante', 'C_t_sum', 'turnos', 'C_t_mean', 'share_pct'])

rupturas = df.index[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')].tolist()
rupturas = [int(i+1) for i in rupturas]
emergencias = df.index[df['fase'] == ('Alta coherencia' if lang == 'es' else 'High coherence')].tolist()
emergencias = [int(i+1) for i in emergencias]
estabilidad = 1 / (df['C_t'].std() + 1e-6)

texto = (
    f"Participantes: {', '.join(participantes) if participantes else '—'}\n"
    f"Promedio C_t: {df['C_t'].mean():.3f}\n"
    f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
    f"Coherencia total (suma C_t): {coherencia_total:.3f}\n"
    f"Índice de estabilidad: {estabilidad:.3f}\n"
    f"Rupturas detectadas: {rupturas}\n"
    f"Turnos de alta coherencia: {emergencias}\n"
)

# Mostrar texto
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
if not ranking_df.empty:
    st.dataframe(ranking_df)








