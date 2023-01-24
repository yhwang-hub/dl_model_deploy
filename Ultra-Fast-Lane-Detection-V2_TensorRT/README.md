# Ultra-Fast-Lane-Detection-V2_TensorRT
这是一个基于TensorRT加速UFLDV2的repo，对源码和论文感兴趣的请参见：https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

# onnx模型下载
Download the model using the following script

https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh

# onnx转TensorRT
```C++
./trtexec --onnx=path_to/ufldv2_culane_res18_320x1600.onnx --saveEngine=path_to/ufldv2_culane_res18_320x1600.engine
```

# 运行
```C++
mkdir build && cd build
cmake ..
make
./ufldv2
```

![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/Ultra-Fast-Lane-Detection-V2_TensorRT/ufldv2.jpg)

