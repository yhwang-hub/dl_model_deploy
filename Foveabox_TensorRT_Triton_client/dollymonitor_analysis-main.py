from pyparsing import os
from client import Triton
import dma
import sys
from common import path_check
import client

if __name__ == '__main__':
    file_path = sys.argv[1]
    # output_img_dir = sys.argv[2]
    # path_check(output_img_dir)
    DM = dma.DollyMonitor(file_path)
    DM.data_prepare()
    output_img_dir = DM.WORK_PATH + '_infer_result'
    path_check(output_img_dir)
    Triton_client = client.Triton(output_img_dir)
    input4_img_dir = os.path.join(DM.DM_PATH, "human_dump_cap4")
    input5_img_dir = os.path.join(DM.DM_PATH, "human_dump_cap5")
    if os.path.exists(input4_img_dir):
        Triton_client.triton_inference(input4_img_dir)
    else:
        print(input4_img_dir + " is not exist!")
    if os.path.exists(input5_img_dir):
        Triton_client.triton_inference(input5_img_dir)
    else:
        print(input5_img_dir + " is not exist!")