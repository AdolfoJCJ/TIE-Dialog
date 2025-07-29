import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ---------- 1. Función para calcular el umbral dinámico ----------
def calcular_phi_dinamico(df, window_size=5, phi_0=0.75, alpha=0.3, beta=0.2):
    phi_dinamico = []
    for i in range(len(df)):
        if i < window_size:
            phi_dinamico.append(phi_0)
        else:
            ventana = df['coherencia'][i-window_size:i]
            std_c = ventana.std()
            mean_c = ventana.mean()
            phi_t = phi_0 + alpha * std_c - beta * mean_c
            phi_dinamico.append(max(0, min(1, phi_t)))
    df['phi_dinamico'] = phi_dinamico
    return df

# ---------- 2. Función para generar el reporte automático ----------
def generar_reporte(df):
    coherencia = df['coherencia']
    phi = df['phi_dinamico']
    n = len(df)

    media_c = coherencia.mean()
    media_phi = phi.mean()
    porcentaje_superado = (coherencia > phi).sum() / n * 100

    cruce_indices = (coherencia > phi).astype(int).diff().fillna(0)
    primer_cruce = cruce_indices[cruce_indices == 1].index.min()
    pico_max = coherencia.idxmax()
    desfase_max = (phi - coherencia).idxmax()

    fases = []
    fase_actual = None
    for i in range(n):
        if coherencia[i] > phi[i]:
            if fase_actual != 'Alta coherencia':
                fases.append((i, 'Alta coherencia'))
                fase_actual = 'Alta coherencia'
        elif coherencia[i] < phi[i] - 0.1:
            if fase_actual != 'Incoherencia':
                fases.append((i, 'Incoherencia'))
                fase_actual = 'Incoherencia'
        else:
            if fase_actual != 'Reconfiguración':
                fases.append((i, 'Reconfiguración'))
                fase_actual = 'Reconfiguración'

    texto = f"""🔎 REPORTE AUTOMÁTICO DE COHERENCIA INFORMACIONAL

• Coherencia promedio: {media_c:.3f}
• Umbral dinámico promedio: {media_phi:.3f}
• Porcentaje de turnos con coherencia > umbral: {porcentaje_superado:.2f}%

⏱️ Momentos clave:
• Primer cruce del umbral: Turno {primer_cruce if pd.notna(primer_cruce) else 'No detectado'}
• Máximo de coherencia: Turno {pico_max} (𝒞 = {coherencia[pico_max]:.3f})
• Mayor desfase negativo: Turno {desfase_max} (𝒞 = {coherencia[desfase_max]:.3f}, Φ = {phi[desfase_max]:.3f})

🌀 Fases detectadas:
"""
    for idx, fase in fases:
        texto += f"• Turno {idx}: {fase}\n"

    return texto

# ---------- 3. Interfaz Streamlit ----------
st.title("TIE–Dialog: Coherencia y Umbral Dinámico")

# Cargar CSV
uploaded_file = st.file_uploader("Carga un archivo .csv con columna 'coherencia'", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'coherencia' not in df.columns:
        st.error("❌ El archivo debe tener una columna llamada 'coherencia'.")
    else:
        # Calcular umbral dinámico
        df = calcular_phi_dinamico(df)

        # Gráfico
        st.subheader("Evolución de la coherencia y del umbral dinámico")
        fig, ax = plt.subplots()
        ax.plot(df['coherencia'], label='𝒞(t)')
        ax.plot(df['phi_dinamico'], label='Φ(t)', linestyle='--')
        ax.set_xlabel("Turno")
        ax.set_ylabel("Valor")
        ax.set_title("Coherencia y Umbral Dinámico")
        ax.legend()
        st.pyplot(fig)

        # Reporte
        st.subheader("Reporte automático")
        reporte_texto = generar_reporte(df)
        st.markdown(f"```\n{reporte_texto}\n```")

        # Botón de descarga
        buffer = StringIO()
        buffer.write(reporte_texto)
        buffer.seek(0)
        st.download_button(
            label="📥 Descargar reporte como .txt",
            data=buffer,
            file_name="reporte_coherencia_TIE_Dialog.txt",
            mime="text/plain"
        )

