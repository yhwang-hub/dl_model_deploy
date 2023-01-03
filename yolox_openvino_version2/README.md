# docker配置方法

1.下载Ubuntu18.04 docker

docker pull openvino/ubuntu18_dev

2.启动docker

docker run -itu root:root --name openvino -v /home/uisee/:/home/wyh/ -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY  --shm-size=64g openvino/ubuntu18_dev /bin/bash

3.模型转换

python3 /opt/intel/openvino_2021.4.689/deployment_tools/model_optimizer/mo_onnx.py --input_model yolox_s_sim_modify.onnx --input_shape [1,3,640,640] --output_dir ./
