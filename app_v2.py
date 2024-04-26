# Python In-built packages
from pathlib import Path
import PIL
import numpy as np


# External packages
import streamlit as st


# Local Modules
import settings
import helper



def custom_print(data, data_name, salto_linea_tipo1=False, salto_linea_tipo2=False, display_data=True, has_len=True, wanna_exit=False):
    if salto_linea_tipo1:
        print(f"")
    if salto_linea_tipo2:
        print(f"\n")
    if has_len:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)} | len: {len(data)}")
        else:
            print(f"{data_name}: | type: {type(data)} | len: {len(data)}")
    else:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)}")
        else:
            print(f"{data_name}: | type: {type(data)}")
    if wanna_exit:
        exit()


# Setting page layout
st.set_page_config(
    page_title="Detección de violencia usando algoritmos de Aprendizaje Profundo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Detección de violencia usando algoritmos de Aprendizaje Profundo")

# Sidebar
st.sidebar.header("Configuración del modelo Aprendizaje Profundo")

# Model Options
model_type = st.sidebar.radio(
    "Seleccionar algoritmo:", ['YOLOv8n', 'Pico_detl640', 'rtdetr_r18vd_6x'])

confidence = float(st.sidebar.slider(
    "Seleccionar Confianza del Modelo:", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'YOLOv8n':
    model_path = Path(settings.DETECTION_MODEL_Y8)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        
elif model_type == 'Pico_detl640':
    model_path = Path(settings.DETECTION_MODEL_PICO_DETL640)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    
elif model_type == 'rtdetr_r18vd_6x':
    model_path = Path(settings.DETECTION_MODEL_RTDETR_R18_6X)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

st.sidebar.header("Configuración de Video")
source_radio = st.sidebar.radio(
    "Seleccionar origen:", settings.SOURCES_LIST)

source_img = None
# If image is selected


if source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, model_type, "video")

elif source_radio == settings.WEBCAM:
    
    helper.play_webcam(confidence, model, model_type, "real_time")

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
