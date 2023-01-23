#ifndef _LDRN_H
#define _LDRN_H

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

class Ldrn_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    static const int num_output = 6;

    const std::string input_blob_name = "input.1";
    const std::string output_blob_names[num_output] = {"2489", "2491", "2493", "2495", "2497", "2499"};

    float norm_mean[3] = {0.485, 0.456, 0.406};
    float norm_std[3]  = {0.229, 0.224, 0.225};

    const int strides[num_output] = {16, 8, 4, 2, 1, 1};

    Ldrn_detector(const std::string& _engine_file);
    virtual ~Ldrn_detector();
    virtual void do_detection(cv::Mat& img);

private:
    int input_buffer_size;
    int output_buffer_size[num_output];

    int input_index;
    int output_index[num_output];

    float* host_input;
    float* host_output[num_output];

    void* device_buffers[num_output + 1];

    int output_dims[num_output][4];

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