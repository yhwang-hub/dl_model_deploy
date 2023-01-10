# Introduction

yolov5团队在最近一段时间在官方网站又有了新动作，即将发布yolov8算法，虽然目前算法源码并未开源，但是发布了预训练模型及推理和模型导出教程，所以并不妨碍我们对yolov8算法进行部署

# How to deploy

## yolov8算法的环境部署
### 官方教程
https://colab.research.google.com/github/glenn-jocher/glenn-jocher.github.io/blob/main/tutorial.ipynb#scrollTo=nPZZeNrLCQG6
### docker环境
```C++
docker pull longxiaowyh/yolov5:v1.0
```
## 预训练模型下载
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
## onnx模型导出
```C++
root@3e9b6779a283:/home/wyh/ultralytics-0.0.59# yolo mode=export model=weights/yolov8s.pt format=onnx simplify=True opset=11 imgsz=[480,640]
Fusing layers... 
YOLOv8s summary: 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs
Ultralytics YOLO 🚀 0.0.59 Python-3.8.13 torch-1.12.1+cu113 CPU
Fusing layers... 
YOLOv8s summary: 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from weights/yolov8s.pt with output shape (1, 84, 6300) (21.5 MB)
requirements: YOLOv5 requirement "onnx>=1.12.0" not found, attempting AutoUpdate...
requirements: ❌ AutoUpdate skipped (offline)

ONNX: starting export with onnx 1.11.0...
ONNX: simplifying with onnx-simplifier 0.4.8...
ONNX: export success ✅ 8.1s, saved as weights/yolov8s.onnx (42.7 MB)

Export complete (8.6s)
Results saved to /home/wyh/ultralytics-0.0.59/weights
Predict:         yolo task=detect mode=predict model=weights/yolov8s.onnx -WARNING ⚠️ not yet supported for YOLOv8 exported models
Validate:        yolo task=detect mode=val model=weights/yolov8s.onnx -WARNING ⚠️ not yet supported for YOLOv8 exported models
Visualize:       https://netron.app
```
## onnx转TensorRT
```C++
 ./trtexec --onnx=/path_to/ultralytics-0.0.59/weights/yolov8s.onnx --saveEngine=/path_to/yolov8_TensorRT/yolov8s.engine
```

## 运行
```C++
mkdir build && cd build
cmake ..
make
./demo
```
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov8_TneosrRT/yolov8.jpg)
