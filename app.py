# TIE-Dialog multilingüe completo con selector de idioma, reporte y descarga CSV/TXT con mejoras sugeridas
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
        "es": "📂 Carga un archivo .csv con al menos 'coherencia' (o deja vacío para cálculo automático)",
        "en": "📂 Upload a .csv file with at least 'coherencia' column (or leave empty for auto-calculation)"
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
    "download_rank": {
        "es": "🏅 Descargar ranking por participante (.csv)",
        "en": "🏅 Download per-participant ranking (.csv)"
    },
    "preview": {
        "es": "🔍 Vista previa de resultados:",
        "en": "🔍 Results preview:"
    },
    "ranking_title": {
        "es": "🏆 Ranking de coherencia por participante (ordenable)",
        "en": "🏆 Coherence contribution ranking (sortable)"
    }
}

# -------------------------
# 📂 Carga CSV y procesamiento
# -------------------------
st.title(t["title"][lang])
uploaded_file = st.file_uploader(t["upload"][lang], type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Simulación de coherencia automática si no hay archivo cargado
    df = pd.DataFrame({
        'turno': range(1, 21),
        'participante': ['Alice', 'Bob', 'Carla', 'Bob', 'Alice', 'Carla', 'Bob', 'Alice', 'Carla', 'Bob',
                         'Alice', 'Carla', 'Bob', 'Alice', 'Carla', 'Bob', 'Alice', 'Carla', 'Bob', 'Alice'],
        'texto': [f"Mensaje {i}" for i in range(1, 21)],
        'coherencia': [0.62, 0.68, 0.75, 0.73, 0.77, 0.80, 0.65, 0.60, 0.70, 0.72,
                       0.66, 0.71, 0.74, 0.78, 0.82, 0.85, 0.88, 0.87, 0.69, 0.65]
    })

if 'coherencia' not in df.columns:
    st.error(t["error"][lang])
else:
    # Crear columnas base
    df['C_t'] = df['coherencia'].astype(float)
    df['C_t_local'] = df['C_t'].rolling(5, min_periods=1).mean()
    df['C_t_Im'] = df['C_t'].ewm(span=8, adjust=False).mean()

    # Calcular Phi_t
    phi_0, alpha, beta = 0.75, 0.3, 0.2
    phi_vals = []
    for i in range(len(df)):
        if i < 5:
            phi_vals.append(phi_0)
        else:
            w = df['C_t'][i-5:i]
            phi_t = phi_0 + alpha * w.std() - beta * w.mean()
            phi_vals.append(max(0, min(1, float(phi_t))))
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
    # 🌈 Gráfica con marcadores
    # -------------------------
    st.subheader(t["plot_title"][lang])
    fig, ax = plt.subplots()
    ax.plot(df.index + 1, df['C_t'], label='C_t')
    ax.plot(df.index + 1, df['C_t_local'], label='C_t_local')
    ax.plot(df.index + 1, df['C_t_Im'], label='C_t_Im')
    ax.plot(df.index + 1, df['Phi_t'], label='Phi_t', linestyle='--')
    # Marcadores de rupturas y emergencias
    rupturas_idx = df.index[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')]
    emergencias_idx = df.index[df['fase'] == ('Alta coherencia' if lang == 'es' else 'High coherence')]
    ax.scatter(rupturas_idx + 1, df.loc[rupturas_idx, 'C_t'], color='red', label='Ruptura', marker='x')
    ax.scatter(emergencias_idx + 1, df.loc[emergencias_idx, 'C_t'], color='green', label='Emergencia', marker='o')
    ax.set_xlabel('Turno' if lang == 'es' else 'Turn')
    ax.set_ylabel('Valor' if lang == 'es' else 'Value')
    ax.legend()
    st.pyplot(fig)

    # -------------------------
    # 🔍 Reporte
    # -------------------------
    st.subheader(t["report_title"][lang])
    if 'turno' not in df.columns:
        df['turno'] = range(1, len(df) + 1)

    # Estadísticas adicionales
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

    # Rupturas = turnos marcados como Incoherencia
    rupturas = df.index[df['fase'] == ('Incoherencia' if lang == 'es' else 'Incoherence')].tolist()
    rupturas = [int(i+1) for i in rupturas]
    # Perspectiva emergente = turnos de Alta coherencia
    emergencias = df.index[df['fase'] == ('Alta coherencia' if lang == 'es' else 'High coherence')].tolist()
    emergencias = [int(i+1) for i in emergencias]

    # Índice de estabilidad (desviación estándar inversa)
    estabilidad = 1 / (df['C_t'].std() + 1e-6)

    # Texto del reporte
    if lang == 'es':
        texto = (
            "TIE–Dialog Report (Español)\n\n"
            f"Participantes: {', '.join(participantes) if participantes else '—'}\n"
            f"Promedio C_t: {df['C_t'].mean():.3f}\n"
            f"Promedio Phi_t: {df['Phi_t'].mean():.3f}\n"
            f"Coherencia total (suma C_t): {coherencia_total:.3f}\n"
            f"Índice de estabilidad: {estabilidad:.3f}\n"
            f"Turnos con C_t > Phi_t: {(df['C_t'] > df['Phi_t']).mean() * 100:.1f}%\n"
            f"Rupturas detectadas (Incoherencia): {rupturas}\n"
            f"Turnos de perspectiva emergente (Alta coherencia): {emergencias}\n\n"
            "Ranking de coherencia aportada:\n"
        )
    else:
        texto = (
            "TIE–Dialog Report (English)\n\n"
            f"Participants: {', '.join(participantes) if participantes else '—'}\n"
            f"Average C_t: {df['C_t'].mean():.3f}\n"
            f"Average Phi_t: {df['Phi_t'].mean():.3f}\n"
            f"Total coherence (sum C_t): {coherencia_total:.3f}\n"
            f"Stability index: {estabilidad:.3f}\n"
            f"Turns with C_t > Phi_t: {(df['C_t'] > df['Phi_t']).mean() * 100:.1f}%\n"
            f"Detected ruptures (Incoherence): {rupturas}\n"
            f"Emergent perspective turns (High coherence): {emergencias}\n\n"
            "Coherence contribution ranking:\n"
        )

    # Añadir ranking al texto
    if not ranking_df.empty:
        for i, row in enumerate(ranking_df.itertuples(index=False), start=1):
            texto += (f"{i}. {row.participante}: suma={row.C_t_sum:.3f}, media={row.C_t_mean:.3f}, turnos={row.turnos}, {row.share_pct:.2f}%\n")
    else:
        texto += "—\n"

    # Fases compactadas
    cambios = df['fase'].ne(df['fase'].shift()).cumsum()
    resumen = df.groupby(cambios, as_index=False).first()
    resumen = resumen.sort_values(by='turno')
    texto += "\n" + ("Fases encontradas:" if lang == 'es' else "Detected phases:") + "\n"
    for _, row in resumen.iterrows():
        texto += (f"Turno {int(row['turno'])}: {row['fase']}\n" if lang == 'es' else f"Turn {int(row['turno'])}: {row['fase']}\n")

    # Mostrar reporte
    st.markdown(f"```\n{texto}\n```")

    # -------------------------
    # 📄 Descargas
    # -------------------------
    st.download_button(
        t["download_txt"][lang],
        data=texto,
        file_name=("reporte_TIE_Dialog.txt" if lang == "es" else "report_TIE_Dialog.txt"),
        mime="text/plain"
    )

    st.download_button(
        t["download_csv"][lang],
        data=df.to_csv(index=False),
        file_name=("datos_TIE_Dialog.csv" if lang == "es" else "TIE_Dialog_data.csv"),
        mime="text/csv"
    )

    if not ranking_df.empty:
        st.download_button(
            t["download_rank"][lang],
            data=ranking_df.to_csv(index=False),
            file_name=("ranking_participantes.csv" if lang == "es" else "participants_ranking.csv"),
            mime="text/csv"
        )

    # -------------------------
    # 🏆 Tabla de ranking ordenable
    # -------------------------
    if not ranking_df.empty:
        st.subheader(t["ranking_title"][lang])
        st.dataframe(ranking_df)

    # -------------------------
    # 🔍 Vista previa de resultados
    # -------------------------
    st.subheader(t["preview"][lang])
    st.dataframe(df.head())






