# sensors
Code for the innovation in my paper

## first of all, download the YOLOv5 from ultralytics:

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt # install
```
## then,replace the yolo.py with the yolo.py in our repository and create a folder "modules" in models 
## replace utils/metric.py with the metric.py in our repository
## add '__init_.py,shufflenetv2.py,split-DLKA.py' to modules.
## add 'yolov5s_shufflenet.yaml' to models

## train 
```
'python train.py --cfg models/yolov5s_shufflenet.yaml --data seaships7000.yaml --hyp hyp.yaml --epochs 100'
