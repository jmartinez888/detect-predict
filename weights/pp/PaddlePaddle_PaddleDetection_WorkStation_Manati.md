
[[Acceso_al_servidor_WorkStation_Manati]]

```shell
screen -S train_dataset_violencia_one_Picodet_paddle_paddle
```
```shell
screen -ls
```
```shell
screen -r 5709.train_dataset_violencia_one_Picodet_paddle_paddle
```

```shell
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
```

or

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection"
```


[[Acceso_al_servidor_SuperComputador_Manati]]

#### 2. Crear tu Entorno Local Portable o Conda (Python 3.7)

```shell
Instalar python 3.7 y virtualenv

sudo yum update

sudo yum install python3 python3-venv python3-devel -y

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

sudo update-alternatives --config python3

sudo apt install python3-pip

sudo apt-get install python3.7 python3.7-venv python3.7-distutils -y

python --version
```
```shell
ls /usr/bin/python*
```
```shell
pip3 cache purge
```


```shell
python3.7 -m venv venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375
```
```shell
.\venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375\Scripts\activate
```
```shell
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
```
```shell
python --version
```
```shell
python -m pip install --upgrade pip
```

```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/"
```

```shell
sudo nano requirements.txt
```

```shell
pip install -r requirements.txt
```

```shell
pip uninstall torch torchvision torchaudio -y
```

```shell
pip uninstall paddlepaddle-gpu -y
```

```shell
python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html 
```

```shell
pip install pyyaml tqdm opencv-python==4.5.5.64 pycocotools imgaug scikit-learn pandas lapx numba==0.56.4 motmetrics wandb paddlelite visualdl terminaltables
```

```shell
python -c "import wandb; wandb.login();" ## wandadb.ai: 7a956f54076501f53f648d6cd03e688d7b4bb361
```

```shell
python -c "import paddle; print(f'version of paddle: {paddle.__version__}')" # version of paddle: 2.3.2
```

```shell
export PYTHONPATH='pwd':$PYTHONPATH
```

```shell
python ppdet/modeling/tests/test_architectures.py
```

```shell
export CUDA_VISIBLE_DEVICES=0
```

```shell
python dataset/coco/download_coco.py
```
```shell
python dataset/voc/download_voc.py
```

```shell
screen -S train_dataset_violencia_one_Picodet_paddle_paddle
```
```shell
screen -ls
```
```shell
screen -r 4809.train_dataset_violencia_one_Picodet_paddle_paddle
```
```shell
screen -X -S 4809.train_dataset_violencia_one_Picodet_paddle_paddle quit
```




## Picodet paddle - violencia_one

[[Acceso_al_servidor_WorkStation_Manati]]

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/datasets_ppr/"
```

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/datasets_ppr/dataset_violencia_one_yolo_crude_all_in_one_v1"
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/
clear
```

```shell
python tools/infer.py -c configs/picodet/picodet_l_640_coco_lcnet.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/picodet_l_640_coco_lcnet.pdparams --infer_img=demo/000000014439.jpg
```

```shell
python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg
```

```shell
python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets
```

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted"
```

```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/datasets_ppr/dataset_violencia_one_COCO_v5_splitted" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset"
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/picodet_l_640_coco_lcnet.yml" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection.yml" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
sudo nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted
```

```shell
sudo nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml"
```

[[Acceso_al_servidor_WorkStation_Manati]]

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/"
clear
```

```shell
ls /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection"
```

```shell
python tools/train.py --config configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml --use_wandb True -o wandb-project=MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO wandb-entity=MyTeam wandb-save_dir=./logs_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO --eval
```

```shell
python tools/train.py --config configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar --amp
```

```shell
"/home/manati/.cache/paddle/weights/PPLCNet_x2_0_pretrained.pdparams"
```
```shell
rm -rf "/home/manati/.cache/paddle/weights/PPLCNet_x2_0_pretrained.pdparams"
rm -rf "/home/manati/.cache/paddle/weights/picodet_l_640_coco_lcnet.pdparams"
```
```shell
"/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/"
```
```shell
"/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/"
```

```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/"
```
```shell
"D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\valid"
```
```shell
"D:\My Proyects D\resultados_algoritmos_violencia_one\datasets\dataset_violencia_one_yolo_crude_all_in_one_v1"
```

```shell

```

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json"
```
```shell
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdparams
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdema
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdopt
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.jpg
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/best_model
clear
```

```shell
ls output/
```

```shell

```

```shell
init: 18:40:15
end: 06:51:50
```

```shell
tiempo tomado de entrenamiento (PP-PicoDet-det-L): 12 h 11 min y 35 s
```

![[Pasted image 20240315185239.png]]

View the change curve in real time through the visualdl command:

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir"
```

```shell
visualdl --logdir ./vdl_dir/scalar/ --port 8080
```
```shell
visualdl --logdir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/vdl_dir/scalar/ --port=8080
```
```shell
visualdl --logdir=./vdl_dir/scalar/ --port=8080
```
```shell
http://localhost:8080/
```

```shell
ls "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir/scalar/"
```
```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted"
```
```shell
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted"
```
```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```
```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/vdl_dir/"
```

```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```

```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```

```shell
rm -rf "/home/manati/Downloads/bbox-mAP_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_bbox_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_dfl_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png"
```

```shell
cp "/home/manati/Downloads/bbox-mAP_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```
```shell
cp "/home/manati/Downloads/loss_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```
```shell
cp "/home/manati/Downloads/loss_bbox_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```
```shell
cp "/home/manati/Downloads/loss_dfl_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```
```shell
cp "/home/manati/Downloads/loss_vfl_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```

```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```

```shell
python tools/eval.py -c configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml -o weights="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output/best_model.pdparams"
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets
```

```shell
d:
```
```shell
cd "D:\My Proyects D\para_plotear_datasets"
```
```shell
conda activate env_yolov8_ultralytics_Jetson_v1_Manati_py38
```

```shell
python generar_nuevo_bbox_annotations_coco_json_and_split_in_train_val_segun_yolov8_annotations_de_bbox_train_val_test_v2.py --ruta_origen "D:\My Proyects D\resultados_algoritmos_violencia_one\datasets\dataset_violencia_one_yolo_crude_all_in_one_v1" --ruta_origen_labels_entrada "D:\My Proyects D\resultados_algoritmos_violencia_one\datasets\dataset_violencia_one_yolo_crude_all_in_one_v1\_1classes.txt" --ruta_salida_json_images "D:\My Proyects D\resultados_algoritmos_violencia_one\datasets\dataset_violencia_one_COCO_v6_splitted" --nombre_json_salida "_annotations_coco.json" --train_ratio 0.75 --valid_ratio 0.15 --test_ratio 0.10
```

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/test/AJ00185.png"
```

```shell
python "/mnt/disk2tb/LITAAP/my_proyects_ppr/para_plotear_datasets/leer_archivos_corruptos_pares_v1.py" --ruta_origen "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/test"
```

![[Pasted image 20240315190127.png]]

[[Acceso_al_servidor_WorkStation_Manati]]

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox_pr_curve"
```
```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/bbox_pr_curve"
```


```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/_valid_bbox_picodetl640cocolcnet.json"
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/_test_bbox_picodetl640cocolcnet.json"
```

```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json"
```

```shell
mv "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/_valid_bbox_picodetl640cocolcnet.json"
```
```shell
mv "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/_test_bbox_picodetl640cocolcnet.json"
```

```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/"
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets
```

```shell
cat "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
d:
```
```shell
cd "D:\My Proyects D\para_plotear_datasets"
```
```shell
conda activate env_yolov8_ultralytics_Jetson_v1_Manati_py38
```

```shell
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/metricas_valid_coco_eval"
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/metricas_test_coco_eval"
```
```shell
ls "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/metricas_test_coco_eval"
```
```shell

```
```shell

```
```shell

```
```shell

```


### valid

linux:

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection"
```
```shell
nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py"
```

```shell
python "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py" --ruta_origen "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/valid" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted" --nombre_dt_json_entrada "_valid_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/metricas_valid_coco_eval"
```

windows:

```shell
python matrix_confusion_gt_dt_jsons_v2.py --ruta_origen "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\valid" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\valid" --nombre_dt_json_entrada "_valid_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "D:\My Proyects D\resultados_algoritmos_violencia_one\algoritmo_2\metricas_valid_coco_eval"
```

### test

linux:

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection"
```
```shell
nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py"
```

```shell
python "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py" --ruta_origen "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/test" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted" --nombre_dt_json_entrada "_test_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/metricas_test_coco_eval"
```

windows:

```shell
python matrix_confusion_gt_dt_jsons_v2.py --ruta_origen "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\test" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\test" --nombre_dt_json_entrada "_test_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "D:\My Proyects D\resultados_algoritmos_violencia_one\algoritmo_2\metricas_test_coco_eval"
```


pendiente

[[Acceso_al_servidor_WorkStation_Manati]]

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/tools/
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/ppdet/metrics/
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/ppdet/metrics/coco_utils.py
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/valid/_annotations_coco.json
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/valid/_annotations_coco.json
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/
clear
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/
clear
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/valid/"
```

[[Acceso_al_servidor_WorkStation_Manati]]

```shell
screen -r 14938.runnning_paddle
```

```shell
python tools/infer.py -c "configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml" -o use_gpu=true weights="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output/best_model" --infer_img="demo/AG00199.png" --output_dir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO"
```


### Deployment

1. Export model

```shell
python tools/export_model.py -c configs/picodet/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted.yml -o weights="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output/best_model.pdparams" --output_dir="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output_export_model"
```

```shell
python deploy/python/infer.py --model_dir="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output_export_model/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted" --image_file=demo/AG00199.png --device=GPU --output_dir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO"
```


2. Convert to PaddleLite (click to expand)

```shell
# FP32
paddle_lite_opt --model_dir="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output_export_model/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted" --valid_targets=arm --optimize_out="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/ds_violencia_one_PaddleLite_model_fp32"
```
```shell
# FP16
paddle_lite_opt --model_dir="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/output_export_model/picodet_l_640_coco_lcnet_dataset_violencia_one_COCO_v5_splitted" --valid_targets=arm --optimize_out="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorpicodetl640cocolcnet_dataset_violencia_one_COCO_v5_splitted/ds_violencia_one_PaddleLite_model_fp16" --enable_fp16=true
```

[[Acceso_al_servidor_WorkStation_Manati]]

## rt_detr paddle - violencia_one


```shell
screen -S train_dataset_violencia_one_rt_detr_paddle_paddle
```
```shell
screen -ls
```
```shell
screen -r 11136.train_dataset_violencia_one_rt_detr_paddle_paddle
```

```shell
ggaa
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/
clear
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/picodet/
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets
```

```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/datasets_ppr/dataset_violencia_one_COCO_v5_splitted" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset"
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/rtdetr/rtdetr_r18vd_6x_coco.yml" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection.yml" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
sudo nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
sudo nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
source venv_PaddlePaddle_PaddleDetection_WorkStation_Manati_py375/bin/activate
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/"
cd /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/
clear
```

```shell
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdparams
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdema
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.pdopt
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.jpg
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/*.png
rm -rf /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output/best_model
clear
```
```shell
ls output/
```
```shell
ls /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted
```

```shell
cat "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
python tools/train.py --config configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar -amp
```

```shell
cp -r /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/copia_dataset_violencia_one_COCO_v5_splitted
```

```shell

```

```shell
init: 21:43:48
end: 23:47:24
```

```shell
tiempo tomado de entrenamiento (pp rtdetr_r18vd_6x_coco): 01 d 02 h 03 min y 36 s
```

![[Pasted image 20240317215901.png]]



```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO"
```
```shell
ls "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir/scalar/"
```
```shell
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO"
```
```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/vdl_dir" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO"
```
```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/vdl_dir/scalar/"
```


View the change curve in real time through the visualdl command:

```shell
visualdl --logdir ./vdl_dir/scalar/ --port 8080
```
```shell
visualdl --logdir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/vdl_dir/scalar/ --port=8080
```
```shell
visualdl --logdir=./vdl_dir/scalar/ --port=8080
```
```shell
http://localhost:8080/
```


```shell
rm -rf "/home/manati/Downloads/bbox-mAP_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_bbox_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png"
rm -rf "/home/manati/Downloads/loss_dfl_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png"
```


```shell
cp "/home/manati/Downloads/bbox-mAP_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_bbox_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_bbox_aux_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_bbox_aux_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_bbox_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_class_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_class_aux_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_class_aux_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_class_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_giou_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_giou_aux_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_giou_aux_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```
```shell
cp "/home/manati/Downloads/loss_giou_dn_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.png" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```

```shell
cp "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```

```shell
cp -r "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/output" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```

```shell
python tools/eval.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml -o weights="/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output/best_model.pdparams"
```


```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json"
```
```shell
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/_valid_bbox_detr_r18vd_6x_coco.json"
rm -rf "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/_test_bbox_detr_r18vd_6x_coco.json"
```

```shell
mv "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/_valid_bbox_detr_r18vd_6x_coco.json"
```
```shell
mv "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/bbox.json" "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/_test_bbox_detr_r18vd_6x_coco.json"
```

```shell
ls -la "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/"
```

```shell
/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets
```

```shell
cat "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/configs/datasets/coco_detection_dataset_violencia_one_COCO_v5_splitted.yml"
```

```shell
d:
```
```shell
cd "D:\My Proyects D\para_plotear_datasets"
```
```shell
conda activate env_yolov8_ultralytics_Jetson_v1_Manati_py38
```

```shell

```



```shell
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/metricas_valid_coco_eval"
mkdir "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/metricas_test_coco_eval"
```
```shell
ls "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/metricas_test_coco_eval"
```

### valid

linux:

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection"
```
```shell
nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py"
```

```shell
python "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py" --ruta_origen "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/valid" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO" --nombre_dt_json_entrada "_valid_bbox_detr_r18vd_6x_coco.json" --ruta_salida_metrics_cm_etc "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/metricas_valid_coco_eval"
```

windows:

```shell
python matrix_confusion_gt_dt_jsons_v2.py --ruta_origen "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\valid" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\valid" --nombre_dt_json_entrada "_valid_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "D:\My Proyects D\resultados_algoritmos_violencia_one\algoritmo_3\metricas_valid_coco_eval"
```

### test

linux:

```shell
cd "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection"
```
```shell
nano "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py"
```

```shell
python "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/matrix_confusion_gt_dt_jsons_v2.py" --ruta_origen "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/dataset/dataset_violencia_one_COCO_v5_splitted/test" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO" --nombre_dt_json_entrada "_test_bbox_detr_r18vd_6x_coco.json" --ruta_salida_metrics_cm_etc "/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/metricas_test_coco_eval"
```

windows:

```shell
python matrix_confusion_gt_dt_jsons_v2.py --ruta_origen "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\test" --nombre_gt_json_entrada "_annotations_coco.json" --ruta_origen_dt "D:\My Proyects D\resultados_algoritmos_violencia_one\imagenes_testing_violencia_one\dataset_violencia_one_COCO_v5_splitted\test" --nombre_dt_json_entrada "_test_bbox_picodetl640cocolcnet.json" --ruta_salida_metrics_cm_etc "D:\My Proyects D\resultados_algoritmos_violencia_one\algoritmo_3\metricas_test_coco_eval"
```


(env_yolov8_ultralytics_Jetson_v1_Manati_py38) D:\My Proyects D\para_plotear_datasets>

[[Acceso_al_servidor_WorkStation_Manati]]

### Deployment

1. Export model

```shell
python tools/export_model.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted.yml -o weights=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output/model_final.pdparams --output_dir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output_export_model
```

```shell
python deploy/python/infer.py --model_dir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output_export_model/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted --image_file=demo/AG00199.png --device=GPU --output_dir /mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO
```


2. Convert to PaddleLite (click to expand)

```shell
# FP32
paddle_lite_opt --model_dir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output_export_model/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted --valid_targets=arm --optimize_out=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/ds_violencia_one_PaddleLite_model_fp32
```
```shell
# FP16
paddle_lite_opt --model_dir=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/output_export_model/rtdetr_r18vd_6x_coco_dataset_violencia_one_COCO_v5_splitted --valid_targets=arm --optimize_out=/mnt/disk2tb/LITAAP/my_proyects_ppr/PaddlePaddle_PaddleDetection/PaddleDetection/logs_plot_map_loss_etc_MyDetectorrtdetr_r18vd_6x_coco_dataset_violencia_one_COCO/ds_violencia_one_PaddleLite_model_fp16 --enable_fp16=true
```

[[Acceso_al_servidor_WorkStation_Manati]]






referencia: https://colab.research.google.com/drive/1m5XiQigd6usI3GLeGMDwpArHsWcCNxCO?usp=sharing#scrollTo=xUzb63HDtuHF


https://docs.wandb.ai/guides/integrations/paddledetection

https://www.paddlepaddle.org.cn/documentation/docs/en/guides/advanced/visualdl_en.html

https://aistudio.baidu.com/modelsoverview?lang=en (very interesting)

https://github.com/PaddlePaddle/Paddle-Lite-Demo (very important)

https://www.paddlepaddle.org.cn/documentation/docs/en/guides/advanced/visualdl_usage_en.html (very very important for Dynamically display scalar data, such as loss, accuracy, etc.)
https://github.com/PaddlePaddle/VisualDL (very important)

https://aistudio.baidu.com/projectdetail/4187344

https://paddledetection.readthedocs.io/advanced_tutorials/TRANSFER_LEARNING.html (very interesting)

https://paddledetection.readthedocs.io/tutorials/GETTING_STARTED.html
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/docs/tutorials/QUICK_STARTED.md

https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/picodet (very very important)

Android Demo with Paddle Lite model:
https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/android/app/cxx/picodet_detection_demo (very very important)

https://github.com/PaddlePaddle/PaddleSlim (very very important)

https://github.com/PaddlePaddle/Paddle-Lite-Demo (very very important)

https://github.com/JiweiMaster/MobileDetBenchmark (very very important)

https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/deploy/lite (very very important)


https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing

https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing

https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-With-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw

https://colab.research.google.com/drive/1m5XiQigd6usI3GLeGMDwpArHsWcCNxCO?usp=sharing#scrollTo=oTuCXBF3IU12

https://github.com/PaddlePaddle/PaddleDetection

https://medium.com/@gary.tsai.advantest/how-to-run-pp-yolo-in-google-colab-2b17701c401a

https://medium.com/@alifyafebriana/how-to-install-baidu-paddlepaddle-framework-on-google-colab-2b5ae30a13c2

https://www.youtube.com/watch?v=TU6mBJ6yuX0

https://www.paddlepaddle.org.cn/

https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-Weights-Biases--VmlldzoyMDU4MjY0

https://github.com/datanomica/Convert-SynthText-to-PaddleOCR-format-Train-SVTR/blob/main/PaddleOCR_train_SVTR-T.ipynb

https://www.google.com/search?q=train+paddle+paddle%2C+object+detection%2C+colab&rlz=1C1VDKB_esPE991PE991&oq=train+paddle+paddle%2C+object+detection%2C+colab&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIKCAEQABiABBiiBNIBCTEzMTk3ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8#ip=1

https://analyticsindiamag.com/guide-to-pp-yolo-an-effective-and-efficient-implementation-of-object-detector/




