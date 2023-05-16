# YOLO_NAS_TensorRT
This repo is used to record the deployment of yolo-nas in TensorRT

# necessary information
```C
git clone git@github.com:Deci-AI/super-gradients.git
pip install super-gradients
```

# export onnx file
copy this code in 
```C
python onnx_export.py
```

# export trt file
```C
/home/path_to/TensorRT-8.5.1.7/bin/trtexec --onnx=./yolo-nas-s.onnx --saveEngine=yolo-nas-s_fp16.onnx --fp16
```

# Run this demo
```C
mkdir build && cd build
cmake ..
make
./yolo-nas
```

# Result
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolo-nas_TensorRT/pred_0.jpg)
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/yolo-nas_TensorRT/yolo_nas_result.jpg)