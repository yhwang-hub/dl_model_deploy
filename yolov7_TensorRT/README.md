# 1.export onnx

下载yolov7源码(https://github.com/WongKinYiu/yolov7)
完成环境搭建，建议使用docker(docker pull nvcr.io/nvidia/pytorch:21.08-py3)
```C++
python export.py --weights yolov7.pt --simplify --grid --img-size 640 640
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
