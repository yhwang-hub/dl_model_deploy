#ifndef _YOLOV5_TENORRT_H
#define _YOLOV5_TENORRT_H

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
// #include "logging.h"
#include <unistd.h>
#include <cmath>

class Yolov5_Detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    int seg_width;
    int seg_height;
    int seg_channels;
    int num_bbox;
    int det_output_channels;
    int classes_num = 80;

    const int nms_max_output = 100;
    const float confThreshold = 0.35;
    const float nmsThreshold = 0.5;
    const float maskThreshold = 0.5;
    const std::string input_blob_name = "images";
    const std::string detect_output_name = "output0";
    const std::string seg_output_name = "output1";

    Yolov5_Detector(const std::string& _engine_file);
    virtual ~Yolov5_Detector();
    virtual void do_detection(cv::Mat& img);

private:
    int input_buffer_size;
    int det_output_buffer_size;
    int seg_output_buffer_size;
    int input_index;
    int det_output_index;
    int seg_output_index;
    float* host_input;
    float* det_output_cpu;
    float* seg_output_cpu;
    void* device_buffers[3];
    std::vector<int> padsize;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;

    void init_context();
    void destroy_context();
    void pre_process(cv::Mat& img);
    void post_process(cv::Mat& img);
};

#endif