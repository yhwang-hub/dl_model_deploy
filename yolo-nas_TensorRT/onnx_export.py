# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

model.predict("bus.jpg").save("yolo-nas_result")

models.convert_to_onnx(model=model, input_shape=(3, 640, 640), out_path='yolo-nas-s.onnx')