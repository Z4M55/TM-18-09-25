# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="🧠 Image Recognition | Tech Mode",
    page_icon="📸",
    layout="centered"
)

# =========================
# Estilos Tech (oscuro + neón)
# =========================
st.markdown("""
<style>
  :root {
    --bg:#0b1220;
    --panel:#0f182b;
    --text:#e6f7ff;
    --muted:#9fb3c8;
    --accent:#00e5ff;
    --accent2:#00ffa3;
  }
  html, body, .stApp {
    background: radial-gradient(1000px 600px at 10% 0%, #0f1a30 0%, var(--bg) 60%);
    color: var(--text) !important;
  }
  [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0e1628 0%, #0b1220 100%) !important;
    color: var(--text) !important;
    border-right: 1px solid rgba(0,229,255,.15);
  }
  h1, h2, h3, h4, h5, h6 {
    color: var(--accent);
    font-family: "JetBrains Mono", monospace;
  }
  p, label, span, div, .stMarkdown {
    color: var(--text) !important;
    font-family: "Inter", sans-serif;
  }
  .stButton>button {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #00121a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: transform .08s ease-in-out, box-shadow .2s ease-in-out;
    box-shadow: 0 0 12px rgba(0,229,255,.5);
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(0,229,255,.75);
  }
  textarea, input {
    background: #0f182b !important;
    color: var(--text) !important;
    border: 1px solid rgba(0,229,255,.3) !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)

# =========================
# Información del entorno
# =========================
st.caption(f"💻 Python versión: `{platform.python_version()}`")
st.title("🧠 Reconocimiento de Imágenes | Tech Mode")
st.markdown("""
Sistema de **visión computacional** basado en un modelo de **Keras** entrenado con **Teachable Machine**.  
Analiza imágenes capturadas desde tu cámara 📸 y predice su **clasificación en tiempo real**. ⚙️
""")

# =========================
# Carga del modelo
# =========================
try:
    model = load_model('keras_model.h5')
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# =========================
# Imagen de presentación
# =========================
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=350, caption="🧠 Sistema de Reconocimiento Visual")
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# =========================
# Sidebar (informativo)
# =========================
with st.sidebar:
