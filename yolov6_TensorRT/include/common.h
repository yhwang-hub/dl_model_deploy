#ifndef _COMMON_H
#define _COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include "NvInfer.h"
#include <sstream>
#include <iostream>

const int NUM_CLASSES = 80;
const float NMS_THRESH = 0.45;
const float BBOX_CONF_THRESH = 0.5;

#define CHECK_CUDA_FATAL(call)                                  \
     do {                                                        \
          cudaError_t result_ = (call);                           \
          if ( result_ != cudaSuccess )                           \
          {                                                       \
             printf(#call "failed (@loc: %d): %s \n",   \
                  __LINE__, cudaGetErrorString(result_));     \
             abort();                                            \
          }                                                       \
     } while (0)

#define CHECK_CUDA_ERROR(res_ok)                                \
    do {                                                        \
         cudaError_t status_ = cudaGetLastError();               \
         if ( status_ != cudaSuccess  )                          \
         {                                                       \
              printf("Cuda failure (@loc: %d): %s\n",    \
              __LINE__, cudaGetErrorString(status_));     \
              res_ok = false;                                     \
         }                                                       \
    } while (0)

static Logger gLogger;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

void generate_yolox_proposals(int num_anchors, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    std::cout << num_anchors << std::endl;
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        int basic_pos = anchor_idx * (NUM_CLASSES + 5);
        float x_center = feat_blob[basic_pos + 0];
        float y_center = feat_blob[basic_pos + 1];
        float w = feat_blob[basic_pos + 2];
        float h = feat_blob[basic_pos + 3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos + 4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;
                objects.push_back(obj);
            }
            

        } // class loop

    } // point anchor loop
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void decode_outputs(float* prob, std::vector<Object>& objects, int num_anchors, float scale, const int img_w, const int img_h) {
    std::vector<Object> proposals;
    generate_yolox_proposals(num_anchors, prob, BBOX_CONF_THRESH, proposals);
    std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);


    int count = picked.size();

    std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

void draw_objects(const cv::Mat& image, const std::vector<Object>& objects)
{
     const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
     };
     // cv::Mat image = bgr.clone();
     for (size_t i = 0; i < objects.size(); i++)
     {
          const Object& obj = objects[i];

          //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

          cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
          float c_mean = cv::mean(color)[0];
          cv::Scalar txt_color;
          if (c_mean > 0.5) {
               txt_color = cv::Scalar(0, 0, 0);
          }
          else {
               txt_color = cv::Scalar(255, 255, 255);
          }

          cv::rectangle(image, obj.rect, color * 255, 2);

          char text[256];
          sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

          int baseLine = 0;
          cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

          cv::Scalar txt_bk_color = color * 0.7 * 255;

          int x = obj.rect.x;
          int y = obj.rect.y + 1;
          //int y = obj.rect.y - label_size.height - baseLine;
          if (y > image.rows)
               y = image.rows;
          //if (x + label_size.width > image.cols)
               //x = image.cols - label_size.width;

          cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
               txt_bk_color, -1);

          cv::putText(image, text, cv::Point(x, y + label_size.height),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
     }

     // cv::imwrite("det_res.jpg", image);
     fprintf(stderr, "save vis file\n");
     /* cv::imshow("image", image); */
     /* cv::waitKey(0); */
}


#endif