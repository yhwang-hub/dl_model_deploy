import cv2 as cv
from matplotlib.colors import to_rgb
import numpy as np
import onnx
import torch
import math
import onnxruntime as rt
import onnx.helper as helper
from mmdet.apis import init_detector

def path_check(file_path):
    import os
    if os.path.exists(file_path):
        os.remove(file_path)
        print(file_path + " has been deleted sucessfully!")

config = '../configs/yolox/yolox_s_8x8_300e_coco.py'
checkpoint = '../checkpoints/yolox_s.pth'
output_file = 'yolox_s.onnx'

device = 'cuda:0'

input_img = '../demo/demo.jpg'
input_shape = (1, 3, 640, 640)
one_img = cv.imread(input_img)
print("one_img.shape:", one_img.shape)

ratio = min(input_shape[2] / one_img.shape[0], input_shape[3] / one_img.shape[1])
print("ratio:", ratio)
padded_img = np.ones((input_shape[2], input_shape[3], 3), dtype=np.uint8) * 114
# padded_img[:,:,1:] = 0
resized_img = cv.resize(
    one_img,
    (math.ceil(one_img.shape[1] * ratio), math.ceil(one_img.shape[0] * ratio)),
    interpolation=cv.INTER_LINEAR
).astype(np.uint8)
print("resized_img.shape:", resized_img.shape)
padded_img[: math.ceil(one_img.shape[0] * ratio), : math.ceil(one_img.shape[1] * ratio)] = resized_img
one_img = padded_img
one_img = one_img.transpose(2, 0, 1)
one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

if 0:
    print("------start preprocesse-------------")
    onnx_preprocess_txt = "onnx_preprocess.txt"
    path_check(onnx_preprocess_txt)
    print("one_img.shape:",one_img.shape)
    onnx_preprocess_data = one_img.detach().numpy().flatten()
    for i in range(len(onnx_preprocess_data)):
        with open(onnx_preprocess_txt,'a+') as f:
            f.write(str(onnx_preprocess_data[i]) + "\n")
    pytorch_inference_preprocess_txt = "pytorch_inference_preprocess.txt"
    pytorch_inference_preprocess_data = []
    for line in open(pytorch_inference_preprocess_txt,'r'):
        data = float(line.split('\n')[0])
        pytorch_inference_preprocess_data.append(data)
    print("len_pytorch_inference_preprocess_data:",len(pytorch_inference_preprocess_data))
    max_diff = 0
    diff_all = 0
    for i in range(len(onnx_preprocess_data)):
        diff = abs(onnx_preprocess_data[i] - pytorch_inference_preprocess_data[i])
        diff_all += diff
        if diff > max_diff:
            max_diff = diff
            print(str(i) + ": " + str(onnx_preprocess_data[i]) + ", " + str(pytorch_inference_preprocess_data[i]))
    print("begin compare bettween " + pytorch_inference_preprocess_txt + " and " + onnx_preprocess_txt)
    print("preprocess max diff:",max_diff)
    print("preprocess average diff:",diff_all / len(onnx_preprocess_data))
    print("------end preprocesse----------------")


one_img = one_img.to(device)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config, checkpoint, device = device)
        print("self.model:\n", self.model)

    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.neck(x)
        x = self.model.bbox_head(x)
        return x 

model = MyModel().eval()

torch.onnx.export(
    model, 
    one_img,
    output_file,
    input_names=['input'],
    export_params=True,
    keep_initializers_as_inputs=True,
    verbose=True,
    opset_version=11)
print(f'Successfully exported ONNX model: {output_file}')
print("end exchange pytorch model to ONNX model.....")

from onnxsim import simplify
print("output_file:",output_file)
onnx_model = onnx.load(output_file)# load onnx model
onnx_model_sim_file = output_file.split('.')[0] + "_sim.onnx"
model_simp, check_ok = simplify(onnx_model)
if check_ok:
    print("check_ok:", check_ok)
    onnx.save(model_simp, onnx_model_sim_file)
    print(f'Successfully simplified ONNX model: {onnx_model_sim_file}')


print("#################### start get pytorch inference result ####################")
print("start pytorch inference .......")
pytorch_result = model(one_img)
print("end pytorch inference .......")
pytorch_results = []
for i in range(len(pytorch_result)):
    for j in range(len(pytorch_result[i])):
        print(str(i) + "," + str(j) + ":" + str(pytorch_result[i][j].shape))
        data = pytorch_result[i][j].cpu().detach().numpy().flatten()
        pytorch_results.extend(data)
print("#################### end get pytorch inference result ####################")


print("#################### start onnxruntime inference ####################")
onnx_model = onnx.load(onnx_model_sim_file)
input_all = [node.name for node in onnx_model.graph.input]
output_all = [node.name for node in onnx_model.graph.output]
print("input_all:", input_all)
print("ouput_all:\n",output_all)
input_initializer = [
    node.name for node in onnx_model.graph.initializer
]
print("input_initializer:\n", input_initializer)
net_feed_input = list(set(input_all) - set(input_initializer))
print("net_feed_input:", net_feed_input)
sess = rt.InferenceSession(onnx_model_sim_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_data = one_img.cpu().detach().numpy()
onnx_result = sess.run(
    None, {net_feed_input[0]: input_data})
onnx_inference_results = []
for i in range(len(onnx_result)):
    onnx_inference_results.extend(onnx_result[i].flatten())
    print("onnx_inference_results["+str(i) + "].shape:", onnx_result[i].shape)
print("#################### end onnxruntime inference ####################")

print("len_onnx_results:",len(onnx_inference_results))
print("len_pytorch_results:", len(pytorch_results))
assert len(pytorch_results) == len(onnx_inference_results),'len(pytorch_results) != len(onnx_results)'

print("#################### start compare  bettween pytorch inference result and onnx inference result ####################")
if 0:
    diff = 0.0
    maxdiff = 0.0
    onnx_result_txt = "onnx_result.txt"
    path_check(onnx_result_txt)
    pytorch_result_txt = "pytorch_result.txt"
    path_check(pytorch_result_txt)
    for i in range(len(onnx_inference_results)):
        diff += abs(onnx_inference_results[i] - pytorch_results[i])
        if abs(onnx_inference_results[i] - pytorch_results[i]) > maxdiff:
            maxdiff = abs(onnx_inference_results[i] - pytorch_results[i])
        with open(onnx_result_txt,'a+') as f:
            f.write(str(onnx_inference_results[i]) + "\n")
        with open(pytorch_result_txt,'a+') as f:
            f.write(str(pytorch_results[i]) + "\n")

    print("diff bettween onnx and pytorch:",diff)
    print("average_diff bettween onnx and pytorch:",diff/len(onnx_inference_results))
    print("maxdiff bettween onnx and pytorch:",maxdiff)

    if diff / len(onnx_inference_results) < 1e-04:
        print('The numerical values are same between Pytorch and ONNX')
    else:
        print('The outputs are different between Pytorch and ONNX')
print("#################### end compare  bettween pytorch inference result and onnx inference result ####################")

cls_name  = ["Conv_266", "Conv_281", "Conv_296"]
bbox_name = ["Conv_267", "Conv_282", "Conv_297"]
obj_name  = ["Conv_268", "Conv_283", "Conv_298"]

cls_output_name  = ["det_cls0", "det_cls1", "det_cls2"]
bbox_output_name = ["det_bbox0", "det_bbox1", "det_bbox2"]
obj_output_name  = ["det_obj0", "det_obj1", "det_obj2"]

onnx_modify_path = onnx_model_sim_file.split('.')[0] + "_modify.onnx"
onnx_model = onnx.load(onnx_model_sim_file)
node = onnx_model.graph.node

for i in range(3):
    onnx_model.graph.output[i].name     = cls_output_name[i]
    onnx_model.graph.output[i + 3].name = bbox_output_name[i]
    onnx_model.graph.output[i + 6].name = obj_output_name[i]

for i in range(len(node)):
    # modify cls output name
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == cls_name[0]:
            node_rise.output[0] = cls_output_name[0]
            node_rise.name = cls_output_name[0]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == cls_name[1]:
            node_rise.output[0] = cls_output_name[1]
            node_rise.name = cls_output_name[1]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == cls_name[2]:
            node_rise.output[0] = cls_output_name[2]
            node_rise.name = cls_output_name[2]
    # modify bbox output name
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == bbox_name[0]:
            node_rise.output[0] = bbox_output_name[0]
            node_rise.name = bbox_output_name[0]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == bbox_name[1]:
            node_rise.output[0] = bbox_output_name[1]
            node_rise.name = bbox_output_name[1]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == bbox_name[2]:
            node_rise.output[0] = bbox_output_name[2]
            node_rise.name = bbox_output_name[2]
    # modify obj output name
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == obj_name[0]:
            node_rise.output[0] = obj_output_name[0]
            node_rise.name = obj_output_name[0]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == obj_name[1]:
            node_rise.output[0] = obj_output_name[1]
            node_rise.name = obj_output_name[1]
    if node[i].op_type == 'Conv':
        node_rise = node[i]
        if node_rise.name == obj_name[2]:
            node_rise.output[0] = obj_output_name[2]
            node_rise.name = obj_output_name[2]

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, onnx_modify_path)
print(f'Successfully modify sim_onnx model: {onnx_modify_path}')