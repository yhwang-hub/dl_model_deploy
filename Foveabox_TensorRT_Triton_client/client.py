import argparse

import time
import os

import cv2

from render import imshow_det_bboxes
from process import postprocess

from triton import triton_preprocess
from triton import triton_infer


class Triton():
    def __init__(self, 
                output_img_dir):
        self.num_stage = 4
        self.input_shape = (1, 3, 672, 384)
        self.mean = [104.0, 117.0, 123.0]
        self.std = [125.0, 125.0, 125.0]
        self.img_shape = (384, 672, 3)
        self.scale_factor = [0.525, 0.53333336, 0.525, 0.53333336]
        self.score_thr = 0.05
        self.nms_pre = 1000
        self.max_per_img = 100
        self.iou_threshold = 0.5
        self.strides = [8, 16, 32, 64]
        self.base_edge_list = [16, 32, 64, 128]
        self.cls_out_channels = 3
        self.background_label = 2
        self.num_classes = 2
        self.class_names = ['person', 'car']
        self.show_score_thr = 0.5
        self.mt_reg_output_blob_name  = ["module.bbox_head.conv_reg.0", "module.bbox_head.conv_reg.1", \
                                        "module.bbox_head.conv_reg.2", "module.bbox_head.conv_reg.3"]
        self.mt_cls_output_blob_name  = ["cls_s8", "cls_s16", "cls_s32", "cls_s64"]
        self.url = '10.0.89.174:8001'
        self.model = 'dollymonitor_trt'
        self.output_img_dir = output_img_dir
    
    def triton_inference(self,
                         input_img_dir):
        triton_client, model_info, client_timeout = triton_preprocess(url = self.url,
                                                                model = self.model)
        img_list = os.listdir(input_img_dir)
        print("len(img_list):", len(img_list))
        for img_name in img_list:
            start_time = time.time()
            input_img = os.path.join(input_img_dir, img_name)
            print("start process ", input_img)
            cls_score_list, bbox_preds_list = triton_infer(model = self.model,
                                                        client_timeout = client_timeout,
                                                        model_info = model_info,
                                                        num_stage = self.num_stage,
                                                        input_img = input_img, 
                                                        input_shape = self.input_shape,
                                                        mean = self.mean,
                                                        std = self.std,
                                                        triton_client = triton_client,
                                                        mt_cls_output_blob_name = self.mt_cls_output_blob_name,
                                                        mt_reg_output_blob_name = self.mt_reg_output_blob_name)
            result = postprocess(cls_scores = cls_score_list, 
                                bbox_preds = bbox_preds_list,
                                scale_factor = self.scale_factor,
                                img_shape = self.img_shape,
                                score_thr = self.score_thr,
                                nms_pre = self.nms_pre,
                                max_per_img = self.max_per_img,
                                iou_threshold = self.iou_threshold,
                                strides = self.strides,
                                base_edge_list = self.base_edge_list,
                                cls_out_channels = self.cls_out_channels,
                                background_label = self.background_label,
                                num_classes = self.num_classes)

            img_result = imshow_det_bboxes(input_img, 
                                           result, 
                                           score_thr = self.show_score_thr,
                                           bbox_color='green',
                                           text_color='green',
                                           thickness=1,
                                           font_scale=0.5,
                                           class_names = self.class_names)
            output_img_path = os.path.join(self.output_img_dir, img_name)
            img_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(img_name.split('.')[0])))
            print("img_date:", img_date)
            cv2.putText(img_result, img_date, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imwrite(output_img_path, img_result)
            end_time = time.time()
            print("process time:", end_time - start_time)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     num_stage = 4
#     input_shape = (1, 3, 672, 384)
#     mean = [104.0, 117.0, 123.0]
#     std = [125.0, 125.0, 125.0]
#     img_shape = (384, 672, 3)
#     scale_factor = [0.525, 0.53333336, 0.525, 0.53333336]
#     score_thr = 0.05
#     nms_pre = 1000
#     max_per_img = 100
#     iou_threshold = 0.5
#     strides = [8, 16, 32, 64]
#     base_edge_list = [16, 32, 64, 128]
#     cls_out_channels = 3
#     background_label = 2
#     num_classes = 2
#     class_names = ['person', 'car']
#     show_score_thr = 0.5

#     mt_reg_output_blob_name  = ["module.bbox_head.conv_reg.0", "module.bbox_head.conv_reg.1", \
#                                     "module.bbox_head.conv_reg.2", "module.bbox_head.conv_reg.3"]
#     mt_cls_output_blob_name  = ["cls_s8", "cls_s16", "cls_s32", "cls_s64"]
#     url = '10.0.89.174:8001'
#     model = 'dollymonitor_trt'
#     triton_client, model_info, client_timeout = triton_preprocess(url = url,
#                                                                 model = model)
#     img_dir = "/home/uisee/Downloads/key"
#     img_list = os.listdir(img_dir)
#     print("len(img_list):", len(img_list))
#     for img_name in img_list:
#         start_time = time.time()
#         input_img = os.path.join(img_dir, img_name)
#         print("start process ", input_img)
#         cls_score_list, bbox_preds_list = triton_infer(model = model,
#                                                        client_timeout = client_timeout,
#                                                        model_info = model_info,
#                                                        num_stage = num_stage,
#                                                        input_img = input_img, 
#                                                        input_shape = input_shape,
#                                                        mean = mean,
#                                                        std = std,
#                                                        triton_client = triton_client,
#                                                        mt_cls_output_blob_name = mt_cls_output_blob_name,
#                                                        mt_reg_output_blob_name = mt_reg_output_blob_name)
#         result = postprocess(cls_scores = cls_score_list, 
#                              bbox_preds = bbox_preds_list,
#                              scale_factor = scale_factor,
#                              img_shape = img_shape,
#                              score_thr = score_thr,
#                              nms_pre = nms_pre,
#                              max_per_img = max_per_img,
#                              iou_threshold = iou_threshold,
#                              strides = strides,
#                              base_edge_list = base_edge_list,
#                              cls_out_channels = cls_out_channels,
#                              background_label = background_label,
#                              num_classes = num_classes)

#         imshow_det_bboxes(input_img, 
#                           result, 
#                           score_thr = show_score_thr,
#                           bbox_color='green',
#                           text_color='green',
#                           thickness=1,
#                           font_scale=0.5,
#                           class_names = class_names)
#         end_time = time.time()
#         print("process time:", end_time - start_time)
