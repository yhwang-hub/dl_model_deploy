# dl_model_deploy
# Introduction
这个项目是为了记录经典深度学习模型在不同框架(x86)中的部署
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/image/deployment-framework.jpg)

# TensorRT部署
下载与cuda版本相对应TensorRT版本（建议下载tar版本），下载地址：https://developer.nvidia.com/nvidia-tensorrt-download


# 运行
将使用的部署框架安装好之后按照下面的流程即可完成运行

mkdir build && cd build

cmake ..

make

./executable_file
