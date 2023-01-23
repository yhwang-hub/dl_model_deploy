# LapDepth-TensorRT
LapDepth TensorRT示例项目（使用基于拉普拉斯金字塔的深度残差进行单眼深度估计)

# Target Environment, How to Build, How to Run
## clone project code
```C++
git clone git@github.com:tjqansthd/LapDepth-release.git
```
## download pretrained model
```C++
wget https://drive.google.com/file/d/10Fsw3KbhiKj-rRkoIesghSPCn84TYY5P/view?usp=sharing
```
## convert pytorch to onnx
```C++
python pytorch_pkl_2_onnx_LapDepth.py
```
## onnx infernce
```C++
python onnx_inference.py
```
## convert onnx modek to TensorRT
```C++
./trtexec --onnx=/home/uisee/disk/dl_model_deploy/ldrn_TensorRT/ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx --saveEngine=/home/uisee/disk/dl_model_deploy/ldrn_TensorRT/ldrn_kitti_resnext101_pretrained_data_grad_256x512.engine
```

![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/ldrn_TensorRT/ufld.jpg)
