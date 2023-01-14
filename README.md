![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/onnx.png)
# dl_model_deploy
# Introduction
这个项目是为了记录经典深度学习模型在不同框架(x86)中的部署

# 环境
```C
Ubuntu
CUDA
OpenCV
TensorRT
ONNXRuntime
OpenVino
```

# TensorRT安装
下载与cuda版本相对应TensorRT(建议下载tar版本)，下载地址： https://developer.nvidia.com/nvidia-tensorrt-download
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/TensorRT.png)
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/TensorRT-tar.png)

直接解压，在~/.bashrc(或者/etc/profile)文件中添加环境变量：
```C
export LD_LIBRARY_PATH=path_to/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH
source ~/.bshrc
```

# ONNXRuntime C++安装
参考 https://blog.csdn.net/weixin_48592526/article/details/128023674
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/onnxruntime.png)
# OpenVino docker配置
```C
1.下载Ubuntu18.04 docker
docker pull openvino/ubuntu18_dev

2.启动docker
docker run -itu root:root --name openvino -v /home/path/:/home/docker_path/ -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY --shm-size=64g openvino/ubuntu18_dev /bin/bash

3.模型转换
python3 /opt/intel/openvino_2021.4.689/deployment_tools/model_optimizer/mo_onnx.py --input_model yolox_s_sim_modify.onnx --input_shape [1,3,640,640] --output_dir ./
```
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/openvino.png)
# 运行demo
将使用的部署框架安装好之后按照下面的流程即可完成运行
```C
mkdir build && cd build
cmake ..
make
./demo
```
