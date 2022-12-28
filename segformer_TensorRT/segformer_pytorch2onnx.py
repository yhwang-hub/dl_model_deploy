import cv2 as cv
import numpy as np
import onnx
import math
import onnxruntime as rt
import onnx.helper as helper

from mmseg.apis import init_segmentor
from mmseg.ops import resize

import torch
import torch.nn as nn
import torch.nn.functional as F

def path_check(file_path):
    import os
    if os.path.exists(file_path):
        os.remove(file_path)
        print(file_path + " has been deleted sucessfully!")

def imnormalize_(img, mean, std, to_rgb=True):
    img = img.copy().astype(np.float32)
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv.cvtColor(img, cv.COLOR_BGR2RGB, img)  # inplace
    cv.subtract(img, mean, img)  # inplace
    cv.multiply(img, stdinv, img)  # inplace
    return img

config = '/home/wyh/mmsegmentation/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py'
checkpoint = '/home/wyh/mmsegmentation/checkpoints/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.pth'
output_file = 'segformer_mit-b0_8x1_1024x1024_160k_cityscapes.onnx'

device = 'cuda:0'

input_img = '/home/wyh/mmsegmentation/demo/demo.png'

input_shape = (1, 3, 1024, 2048)
img = cv.imread(input_img)
print("origin img.shape:", img.shape)

one_img = cv.resize(
    img,
    (input_shape[3], input_shape[2]),
    interpolation=cv.INTER_LINEAR
).astype(np.uint8)
print("after resize one_img.shape:", one_img.shape)
cv.imwrite("resize_img.jpg", one_img)

mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]
to_rgb = True
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)

one_img = imnormalize_(one_img, mean, std, to_rgb=True)
one_img = one_img.transpose(2, 0, 1)
one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

if 1:
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
        self.mode = 'slide'
        self.crop_size = (1024, 1024)
        self.stride = (768, 768)
        self.out_channels = 19
        self.model = init_segmentor(config, checkpoint, device = device)
    
    def forward(self, input_tensor):
        print("input_tensor shape: {}".format(input_tensor.shape))
        h_stride, w_stride = self.stride
        h_crop, w_crop = self.crop_size
        batch_size, _, h_img, w_img = input_tensor.shape
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        print("batch_size: {}, h_img: {}, w_img: {}, h_grids: {}, w_grids: {}".format(batch_size, h_img, w_img, h_grids, w_grids))
        preds = input_tensor.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = input_tensor.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = input_tensor[:, :, y1:y2, x1:x2]
                backbone_out = self.model.backbone(crop_img)
                decode_head_out = self.model.decode_head(backbone_out)
                crop_seg_logit = resize(
                    input = decode_head_out,
                    size = self.crop_size,
                    mode = 'bilinear',
                    align_corners = False
                )
                print("crop_seg_logit shape: {}".format(crop_seg_logit.shape))
                print("preds shape: {}".format(preds.shape))
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=input_tensor.device)
        preds = preds / count_mat
        # remove padding area
        resize_shape = input_tensor.size()[2:]
        preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
        seg_logit = resize(
            preds,
            size = input_tensor.size()[2:],
            mode ='bilinear',
            align_corners = False,
            warning = False)

        return seg_logit # [1, 19, 1024, 2048]

model = MyModel().eval()

torch.onnx.export(
    model, 
    one_img,
    output_file,
    input_names=['input'],
    export_params=True,
    keep_initializers_as_inputs=True,
    verbose=True,
    opset_version=11
)
print(f'Successfully exported ONNX model: {output_file}')
print("end exchange pytorch model to ONNX model.....")

from onnxsim import simplify
print("output_file:",output_file)
onnx_model = onnx.load(output_file)# load onnx model
onnx_model_sim_file = output_file.split('.')[0] + "_sim.onnx"
model_simp, check_ok = simplify(onnx_model)
if check_ok:
    print("check_ok:",check_ok)
    onnx.save(model_simp, onnx_model_sim_file)
    print(f'Successfully simplified ONNX model: {onnx_model_sim_file}')
    

print("#################### start get pytorch inference result ####################")
print("start pytorch inference .......")
pytorch_result = model(one_img)
print("pytorch result shape:", pytorch_result.shape)
print("end pytorch inference .......")
pytorch_results = []
data = pytorch_result.cpu().detach().numpy().flatten()
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

print("len_onnx_results:", len(onnx_inference_results))
print("len_pytorch_results:", len(pytorch_results))
assert len(pytorch_results) == len(onnx_inference_results),'len(pytorch_results) != len(onnx_results)'

print("#################### start compare  bettween pytorch inference result and onnx inference result ####################")
if 1:
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
    print("average_diff bettween onnx and pytorch:", diff / len(onnx_inference_results))
    print("maxdiff bettween onnx and pytorch:",maxdiff)

    if diff / len(onnx_inference_results) < 1e-04:
        print('The numerical values are same between Pytorch and ONNX')
    else:
        print('The outputs are different between Pytorch and ONNX')
print("#################### end compare  bettween pytorch inference result and onnx inference result ####################")