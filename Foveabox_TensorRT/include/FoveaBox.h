#ifndef _FOVEABOX_H
#define _FOVEABOX_H

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

class FoveaBox
{
public:
    const int batchsize = 1;
    const int mt_batch_size = 1;
    const int mt_input_w = 672;
    const int mt_input_h = 384;
    const int mt_input_c = 3;
    const int mt_nms_pre = 200;
    const int mt_nms_max_out = 100;
    const int mt_background_label = 2;
    const float mt_iou_thr = 0.5;
    const float mt_score_thr = 0.5;
    static const int num_stages = 4;
    const char* input = "data";
    const std::string mt_cls_output_name[num_stages] = {"cls_s8", "cls_s16", \
                                                    "cls_s32", "cls_s64"};
    const std::string mt_reg_output_name[num_stages] = {"module.bbox_head.conv_reg.0", "module.bbox_head.conv_reg.1", \
                                                    "module.bbox_head.conv_reg.2", "module.bbox_head.conv_reg.3"};
    const int strides[num_stages]     = {8, 16, 32, 64};
    const int mt_base_len[num_stages] = {16, 32, 64, 128};
    const int mt_det_cls = 3;
    const int mt_det_reg = 4;
    const int input_index = 0;
    std::vector<float> mean_std = {104.0, 117.0, 123.0, 125.0};

    FoveaBox(const std::string& _engine_file);
    ~FoveaBox();

    virtual void do_inference(cv::Mat& image, cv::Mat& dst);

private:
    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;
    cv::cuda::Stream _gpu_stream;
    cv::cuda::GpuMat _img_gpu;

    int input_buffer_size;
    int featmap_sizes[num_stages][2];
    int det_output_cls_buffer_size[num_stages];
    int det_output_reg_buffer_size[num_stages];
    int det_output_cls_index[num_stages];
    int det_output_reg_index[num_stages];
    float* det_output_cls_cpu[num_stages];
    float* det_output_reg_cpu[num_stages];
    void* device_buffers[num_stages * 2 + 1];
    cv::Mat _grid_x[num_stages], _grid_y[num_stages];

    /* temp data */
    float* _det_bboxes_tmp; /* max elem num: nms_pre*pyramid_level*4  HWC format */
    float* _det_scores_tmp; /* max elem num: nms_pre*pyramid_level*20 HWC */
    int* _det_labels_tmp; /* max elem num: nms_pre*pyramid_level*1 */
    /* output */
    float* _det_bboxes; /* max elem num: nms_max_out*4  HWC */
    float* _det_scores; /* max elem num: nms_max_out */
    int* _det_labels; /* max elem num: nms_max_out */
    // int* _kept_bbox_num_per_cls; /* kept bboxes num on each class */
    int _kept_num; /* kept bboxes num finally */
    float* _det_results; /* detection results, stored in specific format for tracking use */
    std::vector<std::string> _object_classes;

    void init_context();

    void get_points();

    void meshgrid(const int x_start, const int x_end,
                const int y_start, const int y_end,
                cv::Mat &X, cv::Mat &Y, const float grid_size);

    void destroy_context();

    void pre_process(cv::Mat& img);

    void topk(int* topk_inds, const float* src, const int len, const unsigned int k);

    void nms(const int bbox_num, std::vector<int>& kept_idx);

    void postprocess();

    void printResultsDet(cv::Mat& img);

};

#endif