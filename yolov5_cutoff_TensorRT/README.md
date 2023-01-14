# Introduction
因为yolov5算法anchor部分带有大量的五维算子，如下图所示，并且根据之前的部署经验，诸如transpose这类的算子，在很多硬件设备(比如高通8155、地平线J5)上并不支持这些算子运行在其加速硬件部分(比如高通8155的NPU、地平线J5的BPU)上，因此为了做到适配所有硬件，我们需要在导出onnx时将这些算子截断并自己手动实现，这里给出了手动实现的grid部分代码。
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov5_cutoff_TensorRT/yolov5-grid.png)

# 1.export onnx

修改yolov5源码中的grid部分如下图所示：
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov5_cutoff_TensorRT/yolov5-grid.png)

使用如下指令导出截断后处理的onnx
```C++
python onnx_cutoff_export.py --weights weights/yolov5s.pt --simplify --opset 11 --include onnx
```
修改后的onnx输出如下所示：
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov5_cutoff_TensorRT/yolov5-cutoff-output0.png)
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov5_cutoff_TensorRT/yolov5-cutoff-output1.png)
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov5_cutoff_TensorRT/yolov5-cutoff-output2.png)

# 2.onnx转TensorRT
```C++
 ./trtexec --onnx=/home/uisee/yolov5/weights/yolov5s_cutoff_640x640.onnx --saveEngine=/home/uisee/disk/dl_model_deploy/yolov5_cutoff_TensorRT/yolov5s_cutoff_640x640.engine
```

# 3.运行
```C++
mkdir build && cd build
cmake ..
make
./demo
```
