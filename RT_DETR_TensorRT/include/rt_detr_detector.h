#ifndef RT_DETR_DETECTOR_H
#define RT_DETR_DETECTOR_H


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


class rt_detr_detector
{
public:
    int batchsize = 1;
    int input_c;
    int input_h;
    int input_w;
    int output_bbox_num;
    int feature_len;
    const int det_bbox_len = 4;
    const int det_cls_len = 80;
    
    float conf_thresh = 0.5f;
    int output_objects_width = 7;// 7:left, top, right, bottom, confidence, class, keepflag;
    
    const std::string input_blob_name = "image";
    const std::string output_blob_name = "concat_5.tmp_0";

    rt_detr_detector(const std::string& _engine_file);
    virtual ~rt_detr_detector();
    void do_detection(cv::Mat& img);

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
    void pre_process_cpu(cv::Mat img);
    void post_process_cpu(cv::Mat& img);
};


#endif