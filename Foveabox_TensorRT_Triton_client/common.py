import os

def iou_calculate(box1, box2):

    '''
    calculate iou between bbox1 an bbox2.
    @param[IN]:bbox1 in [x1,x2,y1,y2] format
    @param[IN]:bbox2 in [x1,x2,y1,y2] format
    @param[OUT]:iou between bbox1 an bbox2
    '''
    [box1_x1,box1_y1,box1_x2,box1_y2] = box1[:4]
    [box2_x1,box2_y1,box2_x2,box2_y2] = box2[:4]
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    return iou

def path_check(file_path):

    '''
    Check if the path exists, create it if it doesn't.
    @param[IN]：file path
    '''
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        if os.path.exists(file_path):
            print(file_path + " has been created sucessfully!")
        else:
            print(file_path + " created failed!")
            return False
    else:
        print(file_path + " has existed!")
    return True