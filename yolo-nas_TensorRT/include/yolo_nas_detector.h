#ifndef YOLO_NAS_DETECTOR_H
#define YOLO_NAS_DETECTOR_H

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

class yolo_nas_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_h;
    int input_w;
    int output_bbox_num;
    int det_bbox_len;
    int det_cls_len;

    float conf_thresh = 0.5f;
    float nms_thresh = 0.5f;
    int output_objects_width = 7;// 7:left, top, right, bottom, confidence, class, keepflag;
    
    const std::string input_name = "input.1";
    const std::string output_cls_name = "1167";
    const std::string output_bbox_name = "1175";

    yolo_nas_detector(const std::string& _engine_file);
    ~yolo_nas_detector();
    void do_detection(cv::Mat& img);

private:
    int input_buffer_size;
    int output_cls_buffer_size;
    int output_bbox_buffer_size;
    int input_index;
    int output_cls_index;
    int output_bbox_index;
    float* host_input;
    float* host_cls_output;
    float* host_bbox_output;

    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

    void* device_buffers[3];

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;

    void init_context();
    void destroy_context();
    void pre_process_cpu(cv::Mat img);
    void post_process_cpu(cv::Mat& img);

};

#endif