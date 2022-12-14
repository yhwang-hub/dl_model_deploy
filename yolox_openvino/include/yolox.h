#ifndef _YOLOX_H
#define _YOLOX_H

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <fstream> 
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
};

class Yolox_Detector
{
public:
    int batchsize;
    int input_w;
    int input_h;
    int input_c;
    const int classes_num = 80;
    const int nms_max_output = 100;
    const float confThreshold = 0.35;
    const float nmsThreshold = 0.5;
    static const int num_stages = 3;
    std::string input_name = "input";
    const std::string cls_output_name[num_stages] = {"det_cls0", "det_cls1", "det_cls2"};
    const std::string obj_output_name[num_stages] = {"det_obj0", "det_obj1", "det_obj2"};
    const std::string bbox_output_name[num_stages] = {"det_bbox0", "det_bbox1", "det_bbox2"};

    const int strides[num_stages] = {8, 16, 32};
    const int det_obj_len = 1;
    const int det_bbox_len = 4;
    const int det_cls_len = 80;
    const int input_index = 0;
    const int max_out_obj = 1000;

    Yolox_Detector(const std::string& _engine_file, const std::string& _runtime);
    virtual ~Yolox_Detector();
    virtual void do_detection(cv::Mat& img);

private:
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork cnnNetwork;
    InferenceEngine::InferRequest infer_request;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InputsDataMap inputInfo;
    InferenceEngine::OutputsDataMap _outputinfo;
    InferenceEngine::Blob::Ptr imgBlob;

    std::string engine_file;
    std::string runtime;
    std::string _input_name;

    void init_context();
    void destroy_context();
    void pre_process_cpu(cv::Mat& img);
    void nms_sorted_bboxes(const std::vector<Object>& objects,
                        std::vector<int>& picked,
                        float nms_threshold);
    void post_process_cpu(cv::Mat& img);
    const float* get_output_buffer(const std::string output_name);
};

#endif