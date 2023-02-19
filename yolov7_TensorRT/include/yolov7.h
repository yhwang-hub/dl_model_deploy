#ifndef _YOLOV7_TENORRT_H
#define _YOLOV7_TENORRT_H

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
#include "logging.h"
#include <unistd.h>
#include <cmath>
#include "common.h"


class yolov7_trt
{
public:
    const int batchsize = 1;
    const char* input = "images";
    const char* output = "output";

    yolov7_trt(const std::string& _engine_file);
    ~yolov7_trt();

    virtual void do_inference(cv::Mat& image,
                            cv::Mat& dst,
                            std::string input_image_path);
    std::vector<Object> getDetectResults() { return objects; };

private:
    /* data */
    const int input_dim0 = 1;
    const int input_dim1 = 3;
    const int input_dim2 = -1;
    const int input_dim3 = -1;
    const size_t input_bufsize = -1;

    const int output_dim0 = 1;
    const int output_dim1 = -1;
    const int output_dim2 = -1;
    const size_t output_bufsize = -1;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;

    std::map<std::string, int> input_layers {
        /* name,         bufferIndex */
        { input,              -1      }
    };
    std::map<std::string, int> output_layers {
        /* name,         bufferIndex */
        { output,             -1      }
    };

    std::vector<Object> objects;

    bool init_done  =false;

    void init_context();

    void destroy_context();

    void pre_process(cv::Mat& img, float* host_input_blob);

    void post_process(cv::Mat& img, float* host_output_data);

    void* get_infer_bufptr(const std::string& bufname, bool is_device);

    int create_binding_memory(const std::string& bufname);
};



#endif