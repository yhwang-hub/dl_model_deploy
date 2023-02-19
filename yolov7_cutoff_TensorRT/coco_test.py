from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

# accumulate predictions from all images
# 载入coco2017验证集标注文件
coco_true = COCO(annotation_file="/home/wyh/disk/coco/annotations/instances_val2017.json")
# 载入网络在coco2017验证集上预测的结果
json_name = sys.argv[-1]
coco_pre = coco_true.loadRes(json_name)

coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()