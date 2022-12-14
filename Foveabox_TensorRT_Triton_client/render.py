import cv2
import numpy as np
from labels import color_val

def imshow_det_bboxes(input_img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    class_names = ['person', 'car']):
    img = cv2.imread(input_img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        # cv2.imwrite(img_name, img)
    
    return img


def _get_points_single(featmap_size, dtype, flatten=False):
    """Get points of a single scale level."""
    h, w = featmap_size
    y_range = np.arange(w, dtype=dtype)
    x_range = np.arange(h, dtype=dtype)
    x, y = np.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    return y + 0.5000, x + 0.5000

def get_points(featmap_sizes, dtype, flatten=False):
    mlvl_points = []
    for i in range(len(featmap_sizes)):
        mlvl_points.append(
            _get_points_single(featmap_sizes[i], dtype, flatten))
    return mlvl_points


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def nms_(boxes, box_confidences, nms_threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # areas = width * height
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[ordered[1:]])
        yy1 = np.maximum(y1[i], y1[ordered[1:]])
        xx2 = np.minimum(x2[i], x2[ordered[1:]])
        yy2 = np.minimum(y2[i], y2[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep


def _get_bboxes_single(cls_scores,
                    bbox_preds,
                    featmap_sizes,
                    point_list,
                    img_shape,
                    scale_factor,
                    score_thr,
                    nms_pre,
                    max_per_img,
                    iou_threshold,
                    strides,
                    base_edge_list,
                    cls_out_channels,
                    background_label):
    assert len(cls_scores) == len(bbox_preds) == len(point_list)
    det_bboxes = []
    det_scores = []
    
    for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) \
            in zip(cls_scores, bbox_preds, featmap_sizes, strides,
                    base_edge_list, point_list):
        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

        # get cls scores
        scores = cls_score.transpose(1, 2, 0).reshape(-1, cls_out_channels)
        if background_label == 0:
            scores = scores[:, 1:]
        else:
            scores = scores[:, :-1]
        bbox_pred = bbox_pred.transpose(1, 2, 0).reshape(-1, 4)
        bbox_pred = np.exp(bbox_pred)

        if (nms_pre > 0) and (scores.shape[0] > nms_pre):
            max_scores = np.max(scores, axis=1)
            _, topk_inds = find_topk(max_scores, nms_pre)
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            y = y[topk_inds]
            x = x[topk_inds]
        x1 = (stride * x - base_len * bbox_pred[:, 0]).\
            clip(min=0, max=img_shape[1] - 1)
        y1 = (stride * y - base_len * bbox_pred[:, 1]).\
            clip(min=0, max=img_shape[0] - 1)
        x2 = (stride * x + base_len * bbox_pred[:, 2]).\
            clip(min=0, max=img_shape[1] - 1)
        y2 = (stride * y + base_len * bbox_pred[:, 3]).\
            clip(min=0, max=img_shape[0] - 1)
        bboxes = np.stack([x1, y1, x2, y2], -1)
        det_bboxes.append(bboxes)
        det_scores.append(scores)
    det_bboxes = np.concatenate(det_bboxes)
    det_bboxes /= scale_factor
    det_scores = np.concatenate(det_scores)

    # uyse cross-class nms instead of multi-class nms
    det_max_scores = np.max(det_scores, axis=1)
    det_labels = np.argmax(det_scores, axis=1)
    # filter bboxes using score threshold
    
    keep_inds = (det_max_scores > score_thr).nonzero()
    det_bboxes = det_bboxes[keep_inds, :]
    det_max_scores = det_max_scores[keep_inds]
    det_labels = det_labels[keep_inds]
    
    inds = nms_(det_bboxes[0], det_max_scores, iou_threshold)
    det_bboxes = det_bboxes[0][inds]
    det_labels = det_labels[inds]
    det_max_scores = det_max_scores[inds]
    det_bboxes = np.concatenate((det_bboxes, det_max_scores.reshape(-1, 1)), axis=1)
    
    if max_per_img < det_bboxes.shape[0]:
        det_bboxes = det_bboxes[:max_per_img, :]
        det_labels = det_labels[:max_per_img]

    return det_bboxes, det_labels


def get_bboxes(cls_scores = None, 
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
               background_label = None):
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
    points = get_points(featmap_sizes,
                        np.float32, 
                        flatten=True)
    result_list = []
    cls_score_list = [
        cls_scores[i][0] for i in range(num_levels)
    ]
    bbox_pred_list = [
        bbox_preds[i][0] for i in range(num_levels)
    ]
    
    scale_factor = np.array(scale_factor, dtype=np.float32)
    det_bboxes = _get_bboxes_single(cls_score_list,
                                    bbox_pred_list, featmap_sizes,
                                    points, img_shape,
                                    scale_factor,
                                    score_thr,
                                    nms_pre,
                                    max_per_img,
                                    iou_threshold,
                                    strides,
                                    base_edge_list,
                                    cls_out_channels,
                                    background_label)
    result_list.append(det_bboxes)
    return result_list


def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes)]


