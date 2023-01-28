# PointPillar_TensorRT
这是一个基于TensorRT加速PointPillar的repo

# 导出onnx模型

按照https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars项目导出onnx模型

将onnx模型放在model文件夹下

# 运行
```C++
mkdir build && cd build
cmake ..
make
./pointpillar
```

最终检测到的结果会存储在pred_velo文件夹下