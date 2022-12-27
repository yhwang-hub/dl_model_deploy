from matplotlib.pyplot import show
import mmdet
import mmcv
print(mmdet.__version__)
print(mmcv.__version__)
import cv2 as cv

from mmdet.apis import inference_detector,init_detector,show_result_pyplot

config = "../configs/yolox/yolox_s_8x8_300e_coco.py"
checkpoint = "../checkpoints/yolox_s.pth"

# initialize the detector
model=init_detector(config,checkpoint,device='cuda:0')

# Use the detector to do inference
# img='../demo/demo.jpg'
img = '20220809-113739.png'
# img = 'demo/1642318073.379330_00498868.jpg'
# img = "/home/uisee/disk/log_key_pic/OOG_data_2022-01-03/log_key_src_2022_1_3/1640164297.267819_00715009.jpg"
results=inference_detector(model,img)

# Plot the result
img = show_result_pyplot(model,img,results,score_thr=0.5)
cv.imwrite("result.jpg",img)
