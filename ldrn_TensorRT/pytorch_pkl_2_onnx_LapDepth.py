WIDTH = 512
HEIGHT = 256
ONNX_FILENAME = "./weights/LDRN_KITTI_ResNext101_" + str(HEIGHT) + "_" + str(WIDTH) + ".onnx"
ONNX_SIMPLE_FILENAME = "./weights/LDRN_KITTI_ResNext101_" + str(HEIGHT) + "_" + str(WIDTH) + "_sim.onnx"

import argparse
# from torchinfo import summary
import torch
import torch.onnx
from model import LDRN

class args():
    def __init__(self):
        self.model_dir = './weights/LDRN_KITTI_ResNext101_pretrained_data.pkl'
        self.encoder = 'ResNext101'
        self.pretrained = 'KITTI'
        self.norm = 'BN'
        self.n_Group = 32
        self.reduction = 16
        self.act = 'ReLU'
        self.max_depth = 80.0
        self.lv6 = 'store_true'
        self.rank = 0
args = args()

model = LDRN(args)
# model = model.cpu()
model = model.cuda()
# model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load(args.model_dir))
# https://qiita.com/tand826/items/fd11f84e1b015b88642e
from collections import OrderedDict
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict
model.load_state_dict(fix_key(torch.load(args.model_dir)))

model.eval()

# summary(model, input_size=(1, 3, HEIGHT, WIDTH))

dummy_input = torch.randn(1, 3, HEIGHT, WIDTH, requires_grad=True)
dummy_input = dummy_input.cuda()
torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILENAME,
    opset_version=11)

from onnxsim import simplify
import onnx
print("output_file:",ONNX_FILENAME)
onnx_model = onnx.load(ONNX_FILENAME)# load onnx model
onnx_model_sim_file = ONNX_SIMPLE_FILENAME
model_simp, check_ok = simplify(onnx_model)
if check_ok:
    print("check_ok:", check_ok)
    onnx.save(model_simp, onnx_model_sim_file)
    print(f'Successfully simplified ONNX model: {onnx_model_sim_file}')