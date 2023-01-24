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

    std::vector<int32_t> loc_row_dims;
    std::vector<int32_t> loc_col_dims;
    std::vector<int32_t> exist_row_dims;
    std::vector<int32_t> exist_col_dims;

    const std::string input_blob_name = "input";
    const std::string exist_row_name  = "exist_row";
    const std::string exist_col_name  = "exist_col";
    const std::string loc_row_name    = "loc_row";
    const std::string loc_col_name    = "loc_col";

    const int kNumRow = 72;
    const int kNumCol = 81;
    const float kCropRatio = 0.6;

    const float norm_mean[3] = {0.485f, 0.450f, 0.406f};
    const float norm_std[3]  = {0.229f, 0.224f, 0.225f};

    Lane_detector(const std::string& _engine_file);
    virtual ~Lane_detector();
    virtual void do_detection(cv::Mat& img);

private:
    int crop_x;
    int crop_y;
    int crop_w;
    int crop_h;

    int input_buffer_size;
    int exist_row_buffer_size;
    int exist_col_buffer_size;
    int loc_row_buffer_size;
    int loc_col_buffer_size;

    int input_index;
    int exist_row_index;
    int exist_col_index;
    int loc_row_index;
    int loc_col_index;

    float* host_input;
    float* exist_row_host_output;
    float* exist_col_host_output;
    float* loc_row_host_output;
    float* loc_col_host_output;

    void* device_buffers[5];

    std::vector<float> row_anchor_;
    std::vector<float> col_anchor_;
    template <typename T> 
    using Line = std::vector<std::pair<T, T>>;

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