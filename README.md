# dl_model_deploy
# Introduction
这个项目是为了记录经典深度学习模型在不同框架(x86)中的部署
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/deployment-framework.jpg)

# TensorRT安装
下载与cuda版本相对应TensorRT(建议下载tar版本)，下载地址：https://developer.nvidia.com/nvidia-tensorrt-download
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/TensorRT-tar.jpg)

直接解压，在~/.bashrc(或者/etc/profile)文件中添加环境变量：
```C
export LD_LIBRARY_PATH=path_to/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH
source ~/.bshrc
```

# 运行
将使用的部署框架安装好之后按照下面的流程即可完成运行
```C
mkdir build && cd build
cmake ..
make
./executable_file
```
