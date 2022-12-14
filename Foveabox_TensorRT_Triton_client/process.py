import cv2
import numpy as np

from render import get_bboxes
from render import bbox2result

def imnormalize_(img, mean, std, to_rgb=True):
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def imnormalize(img, mean, std, to_rgb):
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def preprocess(input_img = None,
            input_shape = None,
            mean = None,
            std = None):
    one_img = cv2.imread(input_img)
    one_img = cv2.resize(one_img,input_shape[2:])
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    one_img = imnormalize(one_img, mean, std, to_rgb = False)
    one_img = one_img.transpose(2, 0, 1)
    return one_img

def postprocess(cls_scores = None, 
                bbox_preds = None,
                scale_factor = None,
                img_shape = None,
                score_thr = None,
                nms_pre = None,
                max_per_img = None,
                iou_threshold = None,
                strides = None,
                base_edge_list = None,
                cls_out_channels = None,
                background_label = None,
                num_classes = None):
    bbox_list = get_bboxes(cls_scores = cls_scores, 
                           bbox_preds = bbox_preds,
                           scale_factor = scale_factor,
                           img_shape = img_shape,
                           score_thr = score_thr,
                           nms_pre = nms_pre,
                           max_per_img = max_per_img,
                           iou_threshold = iou_threshold,
                           strides = strides,
                           base_edge_list = base_edge_list,
                           cls_out_channels = cls_out_channels,
                           background_label = background_label)
        
    bbox_results = [
        bbox2result(det_bboxes, det_labels, num_classes)
            for det_bboxes, det_labels in bbox_list
    ]

    result = bbox_results[0]

    return result