
# 🧠 TIE–Dialog — Public Demo

TIE–Dialog es una herramienta basada en la **Teoría de la Emergencia Informacional (TEI)** que permite analizar la coherencia estructural de un diálogo humano.  
**TIE–Dialog is a tool based on the Theory of Informational Emergence (TIE) that analyzes the structural coherence of human dialogues.**

---

## 🇪🇸 Español

### 🧩 ¿Qué hace TIE–Dialog?

- Calcula la **coherencia local** entre turnos de diálogo (𝒞ₜ)  
- Evalúa la **coherencia con el campo estructurante** Im (𝒞ₜ_Im)  
- Muestra la **evolución por hablante** (𝒞ᵢ)  
- Detecta **rupturas informacionales**  
- Genera gráficos interactivos y resultados exportables

### 📁 ¿Cómo usarlo?

1. Sube un archivo `.csv` con las siguientes columnas:
   - `speaker`: nombre o código del hablante
   - `timestamp`: marca temporal (opcional)
   - `text`: contenido de cada intervención

2. Espera el análisis automático y explora los gráficos, métricas y rupturas detectadas.

3. Descarga los resultados como `.csv` para análisis posterior.

### 💻 Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🇬🇧 English

### 🧩 What does TIE–Dialog do?

- Computes **local coherence** between dialogue turns (𝒞ₜ)  
- Evaluates **coherence with the structuring field** Im (𝒞ₜ_Im)  
- Displays **individual speaker evolution** (𝒞ᵢ)  
- Detects **informational ruptures**  
- Generates interactive plots and exportable results

### 📁 How to use it?

1. Upload a `.csv` file with the following columns:
   - `speaker`: speaker name or label
   - `timestamp`: time mark (optional)
   - `text`: content of each utterance

2. The tool will automatically process and visualize:
   - Coherence over time
   - Speaker-specific trends
   - Detected ruptures

3. Download results as `.csv` for further analysis.

### 💻 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ✨ Créditos | Credits

Creado por / Created by **Adolfo J. Céspedes Jiménez**  
Basado en la / Based on the **Theory of Informational Emergence (TEI)**  
🔗 [ResearchGate profile](https://www.researchgate.net/profile/Adolfo-Cespedes)  
🔗 [Zenodo preprints](https://zenodo.org/search?page=1&size=20&q=adolfo%20cespedes)
