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
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection YOLOv8', 'Detection Pico_detl640', 'Detection rtdetr_r18vd_6x', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection YOLOv8':
    model_path = Path(settings.DETECTION_MODEL_Y8)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        
elif model_type == 'Detection Pico_detl640':
    model_path = Path(settings.DETECTION_MODEL_PICO_DETL640)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    
elif model_type == 'Detection rtdetr_r18vd_6x':
    model_path = Path(settings.DETECTION_MODEL_RTDETR_R18_6X)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_paddle_model(model_path, confidence_threshold=confidence)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)
    
    with col2:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col1:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                
                custom_print("botonnnn_gaaaaa", f"botonnnn_gaaaaa", has_len=False, wanna_exit=False)
                
                if model_type == 'Detection YOLOv8':
                    
                    custom_print(uploaded_image, f"uploaded_image", has_len=False, wanna_exit=False)
                    
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    
                    print(f"res: {res}")
                    
                    print(f"boxes: {boxes}")
                    
                    # exit()
                    
                    res_plotted = res[0].plot()[:, :, ::-1]
                    
                    
                    
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")
                    
                elif model_type == 'Detection Pico_detl640' or model_type == 'Detection rtdetr_r18vd_6x':
                    
                    # Convierte la imagen a un arreglo de numpy
                    uploaded_image = np.array(uploaded_image)
                    
                    custom_print(uploaded_image, f"uploaded_image", has_len=False, wanna_exit=False)
                    
                    res_plotted, res = model.custom_predict_image(uploaded_image, return_results=True, return_result_image_plotted=True)
                    
                    
                    # res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            st.write(res)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")

                    # res = model.custom_predict_image(uploaded_image, return_results=True)
                                        


elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, model_type, "video")

elif source_radio == settings.WEBCAM:
    
    helper.play_webcam(confidence, model, model_type, "real_time")

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
