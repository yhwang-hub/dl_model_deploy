#ifndef _YOLOV8_H
#define _YOLOV8_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <unistd.h>
#include <cmath>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

struct affine_matrix  //前处理仿射变换矩阵和逆矩阵
{
    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

};

class Yolov8_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    int output_bbox_num;
    int output_c;
    int num_classes = 80;

    float conf_thresh = 0.25f;
    float nms_thresh = 0.5f;
    int output_objects_width = 7; // 7:left, top, right, bottom, confidence, class, keepflag;
    int topK = 1000;

    const std::string input_blob_name  = "images";
    const std::string output_blob_name = "output0";

    Yolov8_detector(const std::string& _engine_file);
    virtual ~Yolov8_detector();
    virtual void do_detection(cv::Mat& img);

private:
    int input_buffer_size;
    int output_buffer_size;
    int input_index;
    int output_index;
    float* host_input;
    float* host_output;

    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

    void* device_buffers[2];
    float* device_output_transpose;
    float* device_output_objects;
    int* device_output_idx;
    float* device_output_conf;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;

    void init_context();
    void destroy_context();
    void pre_process(cv::Mat img);
    void post_process(cv::Mat& img);
    void post_process_gpu(cv::Mat& img);
    void get_affine_martrix(affine_matrix &afmt,cv::Size &to,cv::Size &from);  //计算放射变换的正矩阵和逆矩阵
    void nms_sorted_bboxes(const std::vector<Object>& proposals,
                        std::vector<int>& picked,
                        float nms_threshold);
};

#endif