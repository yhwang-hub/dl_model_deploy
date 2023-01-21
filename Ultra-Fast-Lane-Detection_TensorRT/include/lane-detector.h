#ifndef _LANEDET_H
#define _LANEDET_H

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

class Lane_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    int output_c;
    int output_h;
    int output_w;

    const std::string input_blob_name = "input.1";
    const std::string output_blob_name = "200";

    const float norm_mean[3] = {0.485f, 0.450f, 0.406f};
    const float norm_std[3]  = {0.229f, 0.224f, 0.225f};
    std::vector<int> tusimple_row_anchor
            { 64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284 };
    int culane_row_anchor[18] = 
            {121, 131, 141, 150, 160, 
            170, 180, 189, 199, 209, 219, 
            228, 238, 248, 258, 267, 277, 287};
    int cuLaneGriding_num = 200;

    Lane_detector(const std::string& _engine_file);
    virtual ~Lane_detector();
    virtual void do_detection(cv::Mat& img);

private:
    float kCropRatio = 0.6;
    int input_buffer_size;
    int output_buffer_size;

    int input_index;
    int output_index;

    float* host_input;
    float* host_output;

    void* device_buffers[2];

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
};

#endif