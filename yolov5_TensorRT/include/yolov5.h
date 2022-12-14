#ifndef _YOLO_V5_H
#define _YOLO_V5_H

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

class YOLOV5{
public:
    const int batchsize = 1;
    const char* input = "images";
    const char* output = "output0";
    YOLOV5(const int INPUT_H, 
           const int INPUT_W,
           const std::string& _engine_file);
    ~YOLOV5();
    virtual void do_inference(cv::Mat& image);
private:
    const int input_dim0 = 1;
    const int input_dim1 = 3;
    const int input_dim2 = -1;
    const int input_dim3 = -1;
    const size_t input_bufsize = -1;

    const int output_dim0 = 1;
    // const int output_dim1 = 25200;
    const int output_dim1 = 18900;
    const int output_dim2 = 85;
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
        { output,        -1      }
    };

    bool init_done  = false;

    void init_context();

    void destroy_context();

    void pre_process(cv::Mat& img, float* host_input_blob);

    void post_process(cv::Mat& img, float* host_output_data);

    void* get_infer_bufptr(const std::string& bufname, bool is_device);

    int create_binding_memory(const std::string& bufname);
};

#endif