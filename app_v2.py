# Python In-built packages
from pathlib import Path
import PIL
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

from PIL import Image, UnidentifiedImageError
import os
import requests
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No ccache found")

# Función para cargar imagen desde archivo
def load_image_from_file(file_path):
    try:
        img = Image.open(file_path)
        return img
    except UnidentifiedImageError as e:
        st.error(f"Error al cargar la imagen desde el archivo: {e}")
        return None

# Ruta del logo de la institución académica
logo_path = "images/logo_unfv.png"

# Ruta de la segunda imagen
second_image_path = "ejemplo_1_detection_violencia.png"

# Ajusta el ancho de la imagen según tus necesidades
img_logo = load_image_from_file(logo_path)
img_second = load_image_from_file(second_image_path)

# Setting page layout
st.set_page_config(
    page_title="RECONOCIMIENTO DE ACCIONES DE VIOLENCIA CON INTELIGENCIA ARTIFICIAL",
    page_icon=img_logo if img_logo else "🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for font and background
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main page heading with two columns
col1, col2 = st.columns([1, 6])
with col1:
    if img_logo:
        st.image(img_logo, width=200)

with col2:
    st.title("RECONOCIMIENTO DE ACCIONES DE VIOLENCIA CON INTELIGENCIA ARTIFICIAL")

# Banner Image
banner_path = "images/banner_v2.png"
try:
    banner_img = Image.open(banner_path)
    st.image(banner_img, use_container_width=True)
except Exception as e:
    st.error(f"Error loading banner image: {e}")

# Simplified Description
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        <h3>Proyecto de Doctorado</h3>
        <p style="font-size: 18px;">Sistema de Inteligencia Artificial para la detección de violencia en tiempo real.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Configuración del modelo de Aprendizaje Profundo")

# Model Options
model_type = st.sidebar.radio(
    "Seleccionar algoritmo:", ['YOLOv8n'])

confidence = float(st.sidebar.slider(
    "Seleccionar Confianza del Modelo:", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'YOLOv8n':
    model_path = Path(settings.DETECTION_MODEL_Y8)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {model_path}")
        st.error(ex)
        
elif model_type == 'Pico_detl640':
    model_path = Path(settings.DETECTION_MODEL_PICO_DETL640)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {model_path}")
        st.error(ex)
    
elif model_type == 'rtdetr_r18vd_6x':
    model_path = Path(settings.DETECTION_MODEL_RTDETR_R18_6X)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {model_path}")
        st.error(ex)

st.sidebar.header("Configuración de Video")
source_radio = st.sidebar.radio(
    "Seleccionar origen:", settings.SOURCES_LIST)

# Si se selecciona imagen
if source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, model_type, "video")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, model_type, "real_time")

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("¡Por favor selecciona un tipo de origen válido!")

# Footer
st.markdown(
    """
    <style>
    .footer {
        font-size: 16px;
        font-weight: bold;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #a56317;
        color: white;
        text-align: center;
        padding: 15px;
        z-index: 100;
    }
    .footer a {
        color: white;
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p style="margin: 0;">Universidad Nacional Federico Villarreal - Proyecto de Doctorado / ©2024 <a href="https://ctivitae.concytec.gob.pe/appDirectorioCTI/VerDatosInvestigador.do?id_investigador=31125" target="_blank">García Díaz José Edgar</a>. Todos los derechos reservados.</p>
    </div>
    """,
    unsafe_allow_html=True
)
