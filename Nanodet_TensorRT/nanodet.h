//
// Create by RangiLyu
// 2020 / 10 / 2
//

#ifndef NANODET_H
#define NANODET_H
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "NvInfer.h"

struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct CenterPrior {
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet
{
public:
    NanoDet(const std::string &pathStr, int numOfThread);

    ~NanoDet();

    static NanoDet* detector;
    int numThread = 0;
    static bool hasGPU;
    // modify these parameters to the same with your config if you want to use your own model
    // int input_size[2] = {320, 320}; // input height and width
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 80; // number of classes. 4 for kitti
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };

private:
    std::string onnx_file = "../nanodet.onnx";
    std::string engine_file = "../nanodet.engine";
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    void *buffers[2];
    cudaStream_t stream;
    int outSize;
    int batch_size = 1;

    std::vector<int64_t> bufferSize;
    std::vector<float> preprocess(cv::Mat &src);
    void decode_infer(float *floatArray, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    bool readTrtFile(const std::string &engineFile, nvinfer1::ICudaEngine *&engine);
    void onnxToTRTModel(const std::string &onnxModelFile,
                    const std::string &trtFilename,
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE);
    int outputShape;
    float total_inference_time = 0;
    int total_frame_count = 0;
};


#endif //NANODET_H
