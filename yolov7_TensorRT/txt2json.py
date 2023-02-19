'''
@Author : ZJH
@Brief  : Convert coco txt results which from C++ program creating to pycoco-api format json file.
@Date   : 2021-03-30
'''
import json
import os
import sys

# import pycocotools.coco as coco
# from   pycocotools.cocoeval import COCOeval

'''coco_valid_ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,\
                  34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,\
                  62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]'''

coco_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]

if __name__ == "__main__":
    txt_name = sys.argv[-1]
    coco_txt_res_path = txt_name
    json_name = "result_" + txt_name.split('.')[0] + ".json"
    coco_js_file_path = json_name
    coco_format_list = []
    if os.path.exists(coco_js_file_path):
        os.remove(coco_js_file_path)
        if os.path.exists(coco_js_file_path) == 0:
            print(coco_js_file_path + " has been deleted succefuly!")
    print('length of valid ids:', len(coco_valid_ids))
    with open(coco_txt_res_path, 'r') as f:
        data = f.readlines()
        mp = dict()
        img_name_mp = dict()
        for i, line in enumerate(data):
            line = line.strip('\n')
            img_path, cat_id, x, y, w, h, score = line.split(',')
            img_id = img_path.split('/')[-1]
            # print("img_id: ", img_id)
            # count cat_id
            if mp.get(int(cat_id)) != None:
                mp[int(cat_id)] += 1
            else:
                mp[int(cat_id)] = 1

            img_name_mp[int(img_id)] = 1

            # dic = {"image_id": int(img_id), "category_id": coco_valid_ids[int(cat_id)], "bbox": [float(x), float(y), float(w), float(h)], "score": float(score)}
            # dic = {"image_id": int(img_id), "category_id": int(cat_id),
            #        "bbox": [float(x), float(y), float(w), float(h)], "score": float(score)}
            dic = {"image_id": int(img_id), "category_id": coco_valid_ids[int(cat_id)],
                   "bbox": [float(x), float(y), float(w), float(h)], "score": float(score)}
            coco_format_list.append(dic)

    json.dump(coco_format_list, open(coco_js_file_path, 'w'), indent=4)

    new_mp = sorted(mp.items(), key=lambda d: d[1], reverse=False)
    print(new_mp)
    print('img_name_mp.length:', len(img_name_mp))