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
#include "logging.h"
#include <unistd.h>
#include <cmath>

class Yolov5_Detector
{
public:
    const int batchsize = 1;
    // const int input_w = 640;
    const int input_w = 480;
    const int input_h = 640;
    const int input_c = 3;
    const int classes_num = 80;
    const int nms_max_output = 100;
    const float confThreshold = 0.35;
    const float nmsThreshold = 0.5;
    static const int num_stages = 3;
    const char* input = "images";
    const std::string output_blob_name[num_stages] = {"output0", "output1", "output2"};

    const int strides[num_stages] = {8, 16, 32};
    const int det_obj_len = 1;
    const int det_bbox_len = 4;
    const int det_cls_len = 80;
    const int det_len = (det_cls_len + det_bbox_len + det_obj_len) * 3;
    const int input_index = 0;

    const int netAnchors[3][6] = {{10,13, 16,30, 33,23},
                                  {30,61, 62,45, 59,119},
                                  {116,90, 156,198, 373,326}};

    Yolov5_Detector(const std::string& _engine_file);
    virtual ~Yolov5_Detector();
    virtual void do_detection(cv::Mat& img);

private:
    float* host_input;
    int input_buffer_size;
    int det_output_buffer_size[num_stages];
    int det_output_index[num_stages];
    float* det_output_cpu[num_stages];
    void* device_buffers[num_stages + 1];
    int featmap_sizes[num_stages][2];
    bool init_done = false;

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
    float sigmoid_x(float x)
	{
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}
};



#endif