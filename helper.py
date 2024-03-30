from ultralytics import YOLO

from custom_OD_prediction_tools import Custom_Paddle_Detector

import time
import streamlit as st
import cv2
from pytube import YouTube

import numpy as np

import settings

import threading
from datetime import datetime
import winsound


# Charts
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Bar
import random
import streamlit as st
import pyecharts.options as opts


import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QMutex, QMutexLocker

bool_start_init_time = False


initialTime = 0
finalTime = 0
bool_init_time = False

minutes_output = 0
seconds_output = 0


estado_violencia = False


class SoundPlayer(threading.Thread):
    def __init__(self):
        super(SoundPlayer, self).__init__()
        self.is_sound_playing = False
        self.should_stop = False
        self.array = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500]
        self.minutes_output = 0
        self.seconds_output = 0

    def run(self):
        self.is_sound_playing = True
        while not self.should_stop:
            
            if self.seconds_output != -1:
                    
                winsound.Beep(frequency=self.array[self.seconds_output], duration=500)
                print("SOUNDING... (", str(self.seconds_output), ")")
        self.is_sound_playing = False
        
    def set_stop(self):
        print(f"Stopping thread... 1")
        self.should_stop = True # terminate() ==> función que finaliza el proceso


class PredictionThread(QThread):
    # imageUpdated = pyqtSignal(QtGui.QImage)
    countSecondChanged=pyqtSignal(int)
    sendpredictedresults = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

    def run(self):
        self.start_prediction_images()
        
    def start_prediction_images(self):
        print(f"Hello world")
        
        scond = 10
        
        for data in range(4):
            scond = random.randint(0, 20)
            print(f"Hello world {scond}")
            self.countSecondChanged.emit(scond)
        
        
            
    def set_stop(self):

        print(f"Stopping thread... 3")
        self.terminate() # terminate() ==> función que finaliza el proceso

def set_progress_bar_changed_state(value):

    print(f"valor actual: {value}")


sound_player = SoundPlayer()


# app = QApplication(sys.argv)
# prediction_thread = PredictionThread()
# prediction_thread.countSecondChanged.connect(set_progress_bar_changed_state)




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

def custom_render_random_bar_chart(custom_data_0, custom_data_1, custom_data_2, custom_data_3):
    b = (
        Bar()
        .add_xaxis(["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "10s", "11s"])
        .add_yaxis("patada", custom_data_0)
        .add_yaxis("trompon", custom_data_1)
        .add_yaxis("forcegeo", custom_data_2)
        .add_yaxis("estrangulamiento", custom_data_3)
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Predicción", subtitle=""
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
    )
    st_pyecharts(
        b, key="echarts", height="500px"
    )  # Add key argument to not remount component at every Streamlit run

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def load_paddle_model(model_path, confidence_threshold):
    
    import paddle
    paddle.enable_static()
    
    model = Custom_Paddle_Detector(model_dir=model_path, device='CPU', run_mode='paddle', batch_size=1, threshold=confidence_threshold)
    
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, model_type, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    print(f"model_type: {model_type}")
    
    
    res_plotted, res = None, None

    coco_results = []
    
    if model_type == 'Detection YOLOv8':
        
        # Resize the image to a standard size
        # image = cv2.resize(image, (640, int(640*(9/16))))

        # Display object tracking, if specified
        if is_display_tracking:
            res = model.track(image, conf=conf, persist=True, tracker=tracker)
        else:
            # Predict the objects in the image using the YOLOv8 model
            # res = model.predict(image, conf=conf, device="cpu")
            res = model.predict(image, conf=conf, device="cuda:0")

        # # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        
        for i_idx, each_results in enumerate(res):
            
            boxes_data = each_results.boxes.data.cpu().tolist()
        
            print(f"boxes_data: {boxes_data} len: {len(boxes_data)}")
            
            if len(boxes_data) != 0:
                print(f"boxes_data: {boxes_data}")
                for j_idx in range(len(boxes_data)):
                    box_data = boxes_data[j_idx]

                    x_min = int(box_data[0])
                    y_min = int(box_data[1])
                    x_max = int(box_data[2])
                    y_max = int(box_data[3])
                    score = float(box_data[4])
                    category_id = int(box_data[5])
                    
                    coco_annotation = {
                        "category_id": category_id,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "segmentation": [],  # Puedes ajustar esta parte dependiendo de tus necesidades
                        "score": score,
                        "iscrowd": 0  # Por defecto no es una multianotación (crowd)
                    }
                    
                    # Agrega coco_annotation a la lista de anotaciones
                    coco_results.append(coco_annotation)
                 
                    
        
    elif model_type == 'Detection Pico_detl640':
        
        print(f"image: {image} | type: {type(image)}")
        
        # Convierte la imagen a un arreglo de numpy
        image = np.array(image)

        res_plotted, res = model.custom_predict_image(image, return_results=True, return_result_image_plotted=True)
    
    elif model_type == 'Detection rtdetr_r18vd_6x':
        
        print(f"image: {image} | type: {type(image)}")
        
        # Convierte la imagen a un arreglo de numpy
        image = np.array(image)

        res_plotted, res = model.custom_predict_image(image, return_results=True, return_result_image_plotted=True)
        
    
    return res_plotted, coco_results


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    res_plotted, res = _display_detected_frames(conf,
                                             model,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                    
                    st_frame.image(res_plotted,
                        caption='Detected Video',
                        channels="RGB",
                        use_column_width=True
                        )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    res_plotted, res = _display_detected_frames(conf,
                                             model,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                    st_frame.image(res_plotted,
                        caption='Detected Video',
                        channels="RGB",
                        use_column_width=True
                        )
                    
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model, model_type):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    
    # res_plotted, res = model.custom_predict_image(uploaded_image, return_results=True, return_result_image_plotted=True)
    
    if st.sidebar.button('Detectar Objetos'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    res_plotted, res = _display_detected_frames(conf,
                                             model,
                                             model_type,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                    st_frame.image(res_plotted,
                        caption='Detected Video',
                        channels="RGB",
                        use_column_width=True
                        )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def clean_all_values():
    global initialTime, finalTime, bool_init_time
    initialTime = 0
    finalTime = 0
    bool_init_time = False

    print("######### CLEANING VALUES... #########")

def play_sound_2(seconds_output):
    import winsound
    global is_sound_playing, array, minutes_output
    # array = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500] # 1,2,3,4,5,6,7,8,9,10 seconds
    # array = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500]
    # array = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000]
    array = [1000 * (i + 1) for i in range(20)]

    is_sound_playing = True
    # winsound.PlaySound("alarma_deteccion_emergencia.mp3", winsound.SND_FILENAME)
    winsound.Beep(frequency=array[seconds_output], duration=500)
    print("SOUNDING... (",str(seconds_output), ")")
    is_sound_playing = False
    
    
def format_timedelta(delta):
    seconds = int(delta.total_seconds())
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)
    time_fmt = f"{hours:02d} hrs {minutes:02d} min {seconds:02d} s"

    return time_fmt, minutes, seconds
    
def time_difference(initialTime, finalTime):
    # subtract two variables of type datetime in python
    resta_time = finalTime - initialTime
    

    #resta_time = str(resta_time)
    #resta_time = datetime.strptime(resta_time, "%H:%M:%S")
    total_second = resta_time
    time_fmt_output, minutes_output, seconds_output = format_timedelta(total_second)
    #print("\n",time_fmt_output,"\n")
    #print(resta_time)
    # print(type(resta_time))

    # print("hours: ", hours)
    # print("minutes: ", minutes)
    # print("seconds: ", seconds)
    

    return time_fmt_output, minutes_output, seconds_output

def play_stored_video(conf, model, model_type):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    
    global estado_violencia, bool_init_time, sound_player
    
    
    print(f"empezando a predecir video...")
    
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    

    class_names = ["patada", "trompon", "forcegeo", "estrangulamiento"]
    
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    

    is_display_tracker, tracker = display_tracker_options()
    
    
    classes_data_list = [[0.0 for _ in range(20)], [0.0 for _ in range(20)], [0.0 for _ in range(20)], [0.0 for _ in range(20)]]
    
    custom_print(classes_data_list, f"classes_data_list")

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
        
    # if video_bytes:
    #     st.video(video_bytes)
    
    
    col1, col2 = st.columns(2)
    
    with col1:

        if st.sidebar.button('Detect Video Objects'):
            
            try:
                vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    copia_image = image.copy()
                    
                    if success:
                        
                        if estado_violencia:
                            # se mantiene el conteo del tiempo
                            pass
                        else:
                            clean_all_values()
                        
                        res_plotted, res = _display_detected_frames(conf,
                                                model,
                                                model_type,
                                                image,
                                                is_display_tracker,
                                                tracker
                                                )
                                                
                        if len(res) != 0:
                            print("\n ############# ",res,"\n #############")
                            print("SI HAY VIOLENCIA")
                            print("\n ############# ",res,"\n #############")
                            
                            estado_violencia = True
                        else:
                            print("NO HAY VIOLENCIA")
                            estado_violencia = False
                            
                            
                        if estado_violencia:
                                                        
                            for annotation in res:
                                                    
                                # Obtiene la información de las anotaciones de cada imagen
                                id_categoria_imagen = int(annotation["category_id"])
                                iscrowd_imagen = int(annotation["iscrowd"])
                                
                                box_imagen = annotation["bbox"]
                                
                                x, y, w, h = box_imagen[0], box_imagen[1], box_imagen[2], box_imagen[3]
                                
                                class_name = class_names[id_categoria_imagen]
                                
                                color = COLORS[int(id_categoria_imagen) % len(COLORS)]
                                                            
                                if bool_init_time == False:
                                    initialTime = datetime.now()
                                    bool_init_time = True
                            
                                finalTime = datetime.now()
                                
                                time_difference_output, minutes_output, seconds_output = time_difference(initialTime, finalTime)
                                
                                custom_print(id_categoria_imagen, f"id_categoria_imagen - seconds_output: {seconds_output}", has_len=False)
                                
                                classes_data_list[id_categoria_imagen][seconds_output-1] = annotation["score"]
                                
                                play_sound_2(seconds_output)
                                
                                cv2.putText(copia_image,time_difference_output,(1123,1030), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),2)
                                
                                cv2.putText(copia_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                                cv2.rectangle(copia_image, (x, y), (x + w, y + h), color, 3)
                                                
                        st_frame.image(copia_image,
                            caption='Detected Video',
                            channels="RGB",
                            use_column_width=True
                            )
                        
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
 
            # if prediction_thread.isRunning():
            #     print(f"ESTA VIVO EL HILO 3...")
                
            #     prediction_thread.set_stop()
                
            #     prediction_thread.countSecondChanged.connect(set_progress_bar_changed_state)
                
            # else:
            #     print(f"ESTA MUERTO EL HILO 3...")
            
            # # Iniciar el hilo de la cámara
            # prediction_thread.start()
        
            # sys.exit(app.exec_())
            
            # pass


    with col2:
        
        # custom_print()
        
        empty_text = st.empty()
        
        # while estado_violencia:
            
        #     custom_render_random_bar_chart(classes_data_list[0], classes_data_list[1], classes_data_list[2], classes_data_list[3])
            
        
        empty_text.text("Procesando... Por favor, espere...")
        
        
        # Botón para personalizar los datos
        custom_data_0 = [round(random.uniform(0.0, 1.0), 2) for _ in range(11)]
        custom_data_1 = [round(random.uniform(0.0, 1.0), 2) for _ in range(11)]
        custom_data_2 = [round(random.uniform(0.0, 1.0), 2) for _ in range(11)]
        custom_data_3 = [round(random.uniform(0.0, 1.0), 2) for _ in range(11)]
        
        custom_render_random_bar_chart(custom_data_0, custom_data_1, custom_data_2, custom_data_3)

    