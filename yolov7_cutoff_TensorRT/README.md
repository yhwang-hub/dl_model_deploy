# Introduction
因为yolov7算法anchor部分带有大量的五维算子，并且根据之前的部署经验，诸如transpose这类的算子，在很多硬件设备(比如高通8155、地平线J5)上并不支持这些算子运行在其加速硬件部分(比如高通8155的NPU、地平线J5的BPU)上，因此为了做到适配所有硬件，我们需要在导出onnx时将这些算子截断并自己手动实现，这里给出了手动实现的grid部分代码。

# 1.export onnx

修改yolov7源码中的grid部分如下图所示：
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolov7_cutoff_TensorRT/yolov7_cutoff.png)
使用如下指令导出截断后处理的onnx
```C++
python onnx_cutoff_export.py --weights checkpoints/yolov7.pt --simplify
```

# 2.onnx转TensorRT
```C++
 ./trtexec --onnx=/path_to/yolov7.onnx --saveEngine=/path_to/yolov7.engine
```

# 3.运行
```C++
mkdir build && cd build
cmake ..
make
./demo
```
