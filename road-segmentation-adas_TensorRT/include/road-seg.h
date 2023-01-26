#ifndef _ROAD_SEG_H
#define _ROAD_SEG_H

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

class RoadSeg_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    int output_c;
    int output_w;
    int output_h;

    const std::string input_blob_name  = "data";
    const std::string output_blob_name = "tf.identity";

    RoadSeg_detector(const std::string& _engine_file);
    virtual ~RoadSeg_detector();
    virtual void do_detection(cv::Mat& img);

private:
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