import cv2 as cv
from matplotlib.colors import to_rgb
import numpy as np
import onnx
import torch
import math
import onnxruntime as rt
import onnx.helper as helper
from mmseg.apis import init_segmentor
from mmseg.ops import resize

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

config = '/home/wyh/TopFormer/local_configs/topformer/topformer_small_512x512_160k_2x8_ade20k.py'
checkpoint = '/home/wyh/TopFormer/weights/TopFormer-S_512x512_2x8_160k-36.5.pth'
output_file = 'TopFormer-S_512x512_2x8_160k.onnx'

device = 'cuda:0'

input_img = '/home/wyh/TopFormer/demo/demo.jpg'

input_shape = (1, 3, 512, 512)
input_img = cv.imread(input_img)
print("origin one_img.shape:", input_img.shape[0:2])

one_img = cv.resize(
    input_img,
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
        self.model = init_segmentor(config, checkpoint, device = device)

    def forward(self, input_tensor):
        print("input_tensor shape: {}".format(input_tensor.shape))
        backbone_out = self.model.backbone(input_tensor)
        aspp_out = self.model.decode_head(backbone_out)

        seg_logit = resize(
            input = aspp_out,
            size = input_tensor.shape[2:],
            mode = 'bilinear',
            align_corners = False
        )

        seg_logit = resize(
            input = seg_logit,
            size = input_tensor.shape[2:],
            mode = 'bilinear',
            align_corners = False
        )
        print("seg_logit shape: {}".format(seg_logit.shape))

        return seg_logit

model = MyModel().eval()

torch.onnx.export(
    model, 
    one_img,
    output_file,
    input_names=['input'],
    output_names=['output'],
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
            print("{}: {}, {}".format(i, pytorch_results[i], onnx_inference_results[i]))
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