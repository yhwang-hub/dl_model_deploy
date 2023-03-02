#ifndef __SMOKE_DETECTOR_H
#define __SMOKE_DETECTOR_H

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
#include "smoke_utils.h"

struct BboxDim {
    float x;
    float y;
    float z;
};

class Logger : public nvinfer1::ILogger {
  public:
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportable_severity(severity) {}

    void log(Severity severity, const char* msg) noexcept {
        if (severity > reportable_severity) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportable_severity;
};


class Smoke_detector
{
public:
    int batchsize;
    int input_c;
    int input_h;
    int input_w;

    int output_h;
    int output_w;

    int topk = 100;
    float score_thresh = 0.3;

    std::string input_name = "input.1";

    float mean_rgb[3] = {123.675f, 116.280f, 103.530f};
    float std_rgb[3] = {58.395f, 57.120f, 57.375f};

    Smoke_detector();
    Smoke_detector(const std::string& engine_path, const cv::Mat& intrinsic);
    ~Smoke_detector();

    void do_detection(cv::Mat& img, const cv::Mat& intrinsic);
    void LoadOnnx(const std::string& onnx_path);
    void init_context(const std::string& engine_path);
    // void LoadEngine(const std::string& engine_path);
private:
    int input_index;
    int output_index[3];
    int buffer_size[4];

    cv::Mat intrinsic_;
    cv::Mat input_image;

    Logger gLogger;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::IHostMemory* plan_;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;

    void* device_buffers[4];
    float* host_input;
    float* output_bbox_preds;
    float* output_topk_scores;
    float* output_topk_indices;
    int input_buffer_size;
    int bbox_preds_buffer_size;
    int topk_scores_buffer_size;
    int topk_indices_buffer_size;

    std::vector<float> base_depth;
    std::vector<BboxDim> base_dims;

    void preprocess(cv::Mat image, const cv::Mat& intrinsic);
    void postprocess(cv::Mat& img);
    void destroy_context();
};

#endif