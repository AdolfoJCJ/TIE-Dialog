# TIE-Dialog multilingüe con selector de idioma y secciones con emojis
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
        "es": "📂 Carga un archivo .csv con al menos 'coherencia'",
        "en": "📂 Upload a .csv file with at least 'coherencia' column"
    },
    "error": {
        "es": "El archivo debe incluir una columna llamada 'coherencia'.",
        "en": "The file must include a column named 'coherencia'."
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
# 📂 Carga CSV y procesamiento
# -------------------------
st.title(t["title"][lang])
uploaded_file = st.file_uploader(t["upload"][lang], type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'coherencia' not in df.columns:
        st.error(t["error"][lang])
    else:
        df['C_t'] = df['coherencia']
        df['C_t_local'] = df['C_t'].rolling(5, min_periods=1).mean()
        df['C_t_Im'] = df['C_t'].ewm(span=8, adjust=False).mean()

        # Calculo umbral dinámico
        phi_0, alpha, beta = 0.75, 0.3, 0.2
        phi_vals = []
        for i in range(len(df)):
            if i < 5:
                phi_vals.append(phi_0)
            else:
                w = df['C_t'][i-5:i]
                phi_t = phi_0 + alpha * w.std() - beta * w.mean()
                phi_vals.append(max(0, min(1, phi_t)))
        df['Phi_t'] = phi_vals

        # Etiquetado de fases
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
        # 🌈 Gráfica
        # -------------------------
        st.subheader(t["plot_title"][lang])
        fig, ax = plt.subplots()
        ax.plot(df['C_t'], label='C_t')
        ax.plot(df['C_t_local'], label='C_t_local')
        ax.plot(df['C_t_Im'], label='C_t_Im')
        ax.plot(df['Phi_t'], label='Phi_t', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # -------------------------
       # -------------------------
# 🔍 Reporte
# -------------------------
st.subheader(t["report_title"][lang])

# Asegurar columna 'turno'
if 'turno' not in df.columns:
    df['turno'] = range(1, len(df) + 1)

# Texto base del reporte (bilingüe)
if lang == "es":
    texto = (
        f"TIE–Dialog Report (Español)\n\n"
        f"Promedio C_t: {df['C_t'].mean():.3f}\n"
        f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
        f"Turnos con C_t > Phi_t: {(df['C_t'] > df['Phi_t']).mean() * 100:.1f}%\n"
        f"Fases encontradas:\n"
    )
else:
    texto = (
        f"TIE–Dialog Report (English)\n\n"
        f"Average C_t: {df['C_t'].mean():.3f}\n"
        f"Average Phi_t: {df['Phi_t'].mean():.3f}\n"
        f"Turns with C_t > Phi_t: {(df['C_t'] > df['Phi_t']).mean() * 100:.1f}%\n"
        f"Detected phases:\n"
    )

# Cambios de fase compactados (robusto, sin reset_index conflictivo)
cambios = df['fase'].ne(df['fase'].shift()).cumsum()
resumen = df.groupby(cambios, as_index=False).first()  # ✅ evita error de reset_index

# Ordenar por turno por si el groupby altera el orden
resumen = resumen.sort_values(by='turno')

# Añadir las líneas al reporte
for _, row in resumen.iterrows():
    if lang == "es":
        texto += f"Turno {int(row['turno'])}: {row['fase']}\n"
    else:
        texto += f"Turn {int(row['turno'])}: {row['fase']}\n"

# Mostrar reporte
st.markdown(f"```\n{texto}\n```")

# ✅ Botón de descarga del reporte
st.download_button(
    label=t["download_txt"][lang],
    data=texto,
    file_name="reporte_TIE_Dialog.txt" if lang == "es" else "report_TIE_Dialog.txt",
    mime="text/plain"
)


        # -------------------------
        # 🔍 Vista previa
        # -------------------------
        st.subheader(t["preview"][lang])
        st.dataframe(df.head())



