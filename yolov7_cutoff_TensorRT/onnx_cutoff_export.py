import argparse
import sys
import time
import warnings
import cv2
import numpy as np
import math

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS

def path_check(file_path):
    import os
    if os.path.exists(file_path):
        os.remove(file_path)
        print(file_path + " has been deleted sucessfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand

    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export

    # ONNX export
    try:
        import onnx
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '_cutoff.onnx')  # filename
        model.eval()
        output_names = ['det_out0', 'det_out1', 'det_out2']

        model.model[-1].concat = True

        torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                          output_names=output_names)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        onnx.save(onnx_model,f)
        print('ONNX export success, saved as %s' % f)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    import onnxruntime as rt
    onnx_model_sim_file = f
    print("#################### start get pytorch inference result ####################")
    print("start pytorch inference .......")
    pytorch_result = model(img)
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
    input_data = img.cpu().detach().numpy()
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
        print("average_diff bettween onnx and pytorch:",diff/len(onnx_inference_results))
        print("maxdiff bettween onnx and pytorch:",maxdiff)

        if diff / len(onnx_inference_results) < 1e-04:
            print('The numerical values are same between Pytorch and ONNX')
        else:
            print('The outputs are different between Pytorch and ONNX')
    print("#################### end compare  bettween pytorch inference result and onnx inference result ####################")
