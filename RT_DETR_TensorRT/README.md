# RT_DETR_TensorRT
This repo is used to record the deployment of RT_DETR in TensorRT

# necessary information
```C
# 1. tensorrt环境准备，请自行安装，选择版本大于或等于trt-v8.5.1
tensorrt版本选择v8.6.0.xx # 因为RT-DETR里有GridSample算子，这个在trt-v8.5.1版本之后进行了支持

# 2. 选择带有cuda的服务器，常用的带有显卡的x86服务器，或者Jetson系列的arm板子等
RTX3090 # 作者测试服务器所使用的显卡

# 3. 训练自己的rt-detr模型并导出，不熟悉paddleDetection框架的，也没关系，只需要按照官方README操作即可
     # 这里给出官方的模型 https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
 
# 4. 测试一下自己的paddledetection环境是否正确，运行下面的命令，不报错即可
python tools/infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -o weights=weights/rtdetr_r50vd_6x_coco.pdparams \ # 改成你自己的模型
    --infer_img=scripts/dog.jpg # 改成你自己的文件

# 5. 一般而言rt-detr默认是导出前后处理的，这会使得导出的onnx会有多个输入，影响部署，所以要关闭前后处理的导出
# 5.1 修改 configs/rtdetr/_base_/rtdetr_r50vd.yml中的DETR类
DETR:
  backbone: ResNet
  neck: HybridEncoder
  transformer: RTDETRTransformer
  detr_head: DINOHead
  post_process: DETRPostProcess
  exclude_post_process: True # 增加该行，并将其改为True

# 6. 【可选】我在使用cuda后处理的过程中，发现onnx导出的是两个分支[box,class]，这对部署并不友好。
     # 而且，box.shape=[batch,300,4] class.shape=[batch,300,80],其实是可以这两个输出合并的
     # 修改ppdet/modeling/architectures/detr.py的DETR类中的_forward方法
     # 将return output修改为：
import paddle.nn.functional as F # 直接输出class为[0,1],这样就不需要再对score进行后处理了
return paddle.concat([bbox,F.sigmoid(bbox_num)],axis=-1) # 这样，就能保证是一个输出，且output.shape=[batch,300,84]
```

# export onnx & trt
```C
# 1. 按照paddledetection里提供的rtdetr的README.md，按照其步骤导出onnx

# 2. 导出的onnx先不要转trt engine，可以使用onnxsim对onnx模型进行简化去除一些不必要的op算子
pip install onnxsim # 安装onnxsim库，可以直接将复杂的onnx转为简单的onnx模型，且不改变其推理精度
onnxsim input_onnx_model output_onnx_model # 通过该命令得到RT-DERT的rtdetr_r50vd_6x_coco_sim.onnx

# 3. 导出静态engine模型，该模型只支持输入固定图片尺寸[1,3,640,640]
trtexec --onnx=./rtdetr_r50vd_6x_coco_sim.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetr_r50vd_6x_coco_static_fp16.trt \
        --avgRuns=100 \
        --fp16
```

# run the demo
```C
cd RT_DETR_TensorRT
mkdir build && cd build
cmake ..
make
./rtdetr
```

# Result
![image](https://github.com/yhwang-hub/dl_model_deploy/blob/master/RT_DETR_TensorRT/rt_detr_result.jpg)

# related repo
https://github.com/AiQuantPro/AiInfer

https://zhuanlan.zhihu.com/p/623794029
