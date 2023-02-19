#ifndef YOLOX_TRT_H
#define YOLOX_TRT_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
// #include "cuda_runtime_api.h"
#include "logging.h"
#include <unistd.h>
#include <cmath>
#include "../src/cuda_kernel/plugin_cuda_function.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
};

class Yolox_Detector
{
public:
    const int batchsize = 1;
    const int input_w = 640;
    const int input_h = 640;
    const int input_c = 3;
    const int classes_num = 80;
    const int nms_max_output = 100;
    const float confThreshold = 0.5;
    const float nmsThreshold = 0.5;
    static const int num_stages = 3;
    const char* input = "input";
    const std::string cls_output_name[num_stages] = {"det_cls0", "det_cls1", "det_cls2"};
    const std::string obj_output_name[num_stages] = {"det_obj0", "det_obj1", "det_obj2"};
    const std::string bbox_output_name[num_stages] = {"det_bbox0", "det_bbox1", "det_bbox2"};

    const int strides[num_stages] = {8, 16, 32};
    const int det_obj_len = 1;
    const int det_bbox_len = 4;
    const int det_cls_len = 80;
    const int input_index = 0;
    const int max_out_obj = 1000;
    const int NUM_BOX_ELEMENT = 7;// left, top, right, bottom, confidence, class, keepflag

    const float mean_rgb[3] = {123.67500305, 116.27999878, 103.52999878};
    const float std_rgb[3]  = {58.395, 57.12, 57.375};

    Yolox_Detector(const std::string& _engine_file);
    virtual ~Yolox_Detector();
    virtual void do_detection(cv::Mat& img);
    std::vector<Object> getDetectResults() { return objects; };

private:
    float* host_input;
    int input_buffer_size;
    int det_output_cls_buffer_size[num_stages];
    int det_output_obj_buffer_size[num_stages];
    int det_output_bbox_buffer_size[num_stages];
    int det_output_cls_index[num_stages];
    int det_output_obj_index[num_stages];
    int det_output_bbox_index[num_stages];
    float* det_output_cls_cpu[num_stages];
    float* det_output_obj_cpu[num_stages];
    float* det_output_bbox_cpu[num_stages];
    void* device_buffers[num_stages * 3 + 1];
    void* device_output_buffer;
    void* host_output_buffer;
    uint8_t* psrc_device = nullptr;
    int featmap_sizes[num_stages][2];
    bool init_done = false;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;
    cv::cuda::Stream _gpu_stream;
    cv::cuda::GpuMat _img_gpu;

    std::vector<Object> objects;

    void init_context();
    void destroy_context();
    void pre_process_cpu(cv::Mat &img);
    void pre_process_gpu(cv::Mat &img);
    void nms_sorted_bboxes(const std::vector<Object>& objects,
                        std::vector<int>& picked,
                        float nms_threshold);
    void post_process_cpu(cv::Mat& img);
    void post_process_gpu1(cv::Mat& img);
    void post_process_gpu2(cv::Mat& img);
};
#endif