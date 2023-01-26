# Road Segmentation Adas with TensorRT Lite in C++
Click the image to open in YouTube. https://youtu.be/k7upHMvalWI

# onnx模型下载
Download the model using the following script

https://github.com/PINTO0309/PINTO_model_zoo/blob/main/136_road-segmentation-adas-0001/download.sh

# onnxsim model
```C++
python onnx-sim.py
```

# onnx转TensorRT
```C++
./trtexec --onnx=path_to/road-segmentation-adas_float32_sim.onnx --saveEngine=path_to/road-segmentation-adas_float32_sim.engine
```

# 运行
```C++
mkdir build && cd build
cmake ..
make
./ufldv2
```

![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/road-segmentation-adas_TensorRT/roadSeg1.jpg)
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/road-segmentation-adas_TensorRT/roadSeg2.jpg)

# Acknowledgements

https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_ss_road-segmentation-adas-0001

https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001

https://motchallenge.net/