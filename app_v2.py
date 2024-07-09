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

# Función para cargar imagen desde archivo
def load_image_from_file(file_path):
    try:
        img = Image.open(file_path)
        return img
    except UnidentifiedImageError as e:
        st.error(f"Error al cargar la imagen desde el archivo: {e}")
        return None

# Ruta del logo de la institución académica
logo_path = "png_logo_unap.png"

# Ruta de la segunda imagen
second_image_path = "ejemplo_1_detection_violencia.png"

# Ajusta el ancho de la imagen según tus necesidades
img_logo = load_image_from_file(logo_path)
img_second = load_image_from_file(second_image_path)

# Setting page layout
st.set_page_config(
    page_title="Detección de violencia usando algoritmos de Aprendizaje Profundo",
    page_icon=img_logo if img_logo else "🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for font and background
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
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
    st.title("Detección de violencia usando algoritmos de Aprendizaje Profundo")

# Nuevo row con dos columnas para la descripción del proyecto y la segunda imagen
col3, col4 = st.columns([4, 2])
with col3:
    st.markdown(
        """
        ## 📋 Descripción del Proyecto

        El proyecto "Detección de Violencia usando Algoritmos de Aprendizaje Profundo" tiene como objetivo desarrollar una aplicación web avanzada que utilice técnicas de inteligencia artificial para identificar automáticamente actos de violencia en vídeos, ya sea en tiempo real o mediante la carga de archivos de video pregrabados. Este sistema está diseñado para mejorar la seguridad pública y proporcionar herramientas eficaces para la vigilancia y prevención de incidentes violentos en diversos entornos.
        """
    )

with col4:
    if img_second:
        st.image(img_second, width=500)

# Descripción extendida del proyecto en la Página Principal
st.markdown(
    """
    ### 🎯 Objetivos del Proyecto

    1. **Desarrollar un Sistema de Detección de Violencia:** Implementar una aplicación web que emplee algoritmos de aprendizaje profundo, específicamente el YOLOv8n, para analizar y detectar actos violentos en secuencias de video.
    2. **Procesamiento en Tiempo Real y Almacenado:** Permitir a los usuarios cargar videos o utilizar transmisiones en vivo para la detección de violencia.
    3. **Interfaz de Usuario Intuitiva:** Proporcionar una plataforma fácil de usar donde los resultados de la detección se presenten de manera clara y precisa.

    ### 🛠️ Metodología

    El sistema se basa en el uso de redes neuronales convolucionales (CNNs) que han demostrado ser altamente efectivas en la detección de patrones visuales complejos. El modelo YOLOv8n, conocido por su capacidad para realizar detecciones rápidas y precisas, se entrenará y ajustará utilizando grandes conjuntos de datos etiquetados con instancias de comportamientos violentos. La confianza del modelo se ajustará para equilibrar la precisión y la sensibilidad de las detecciones.

    ### 🌐 Aplicaciones

    Este sistema tiene múltiples aplicaciones prácticas:
    - **Seguridad Pública:** Monitoreo de espacios públicos para identificar y responder rápidamente a incidentes violentos.
    - **Instituciones Educativas:** Vigilancia en escuelas y universidades para prevenir y manejar situaciones de violencia.
    - **Lugares de Trabajo:** Implementación en entornos laborales para garantizar la seguridad de los empleados.
    - **Hogares:** Uso en sistemas de seguridad doméstica para proteger a los residentes.

    ### 📈 Resultados Esperados

    Se espera que la aplicación proporcione:
    - **Detección Precisa:** Identificación confiable de comportamientos violentos con un alto nivel de precisión.
    - **Notificaciones en Tiempo Real:** Alertas instantáneas a los usuarios cuando se detecte violencia en tiempo real.
    - **Informes Detallados:** Resúmenes de las detecciones, incluyendo los momentos exactos y la naturaleza de las actividades violentas identificadas.
    """
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
        font-size: 14px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        © 2024 <a href="https://ctivitae.concytec.gob.pe/appDirectorioCTI/VerDatosInvestigador.do?id_investigador=31125" target="_blank">García Díaz José Edgar</a>. Todos los derechos reservados.
    </div>
    """,
    unsafe_allow_html=True
)
