# Ultra-Fast-Lane-Detection_TensorRT
这是一个基于TensorRT加速UFLD的repo，对源码和论文感兴趣的请参见：https://github.com/cfzd/Ultra-Fast-Lane-Detection

# onnx模型下载
Download the model using the following script

https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/download.sh

# onnx转TensorRT
```C++
./trtexec --onnx=path_to/ultra_falst_lane_detection_culane_288x800.onnx --saveEngine=path_to/ultra_falst_lane_detection_culane_288x800.engine
```

# 运行
```C++
mkdir build && cd build
cmake ..
make
./ufld
```

![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/Ultra-Fast-Lane-Detection_TensorRT/ufld.jpg)
