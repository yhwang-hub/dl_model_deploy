#ifndef _POLUGIIN_H
#define _POLUGIIN_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define num_threads 512

typedef unsigned char uint8_t;

struct Size{
    int width = 0, height = 0;

    Size() = default;
    Size(int w, int h)
    :width(w), height(h){}
};

struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

#define CUDA_NUM_THREADS        256
#define DIV_THEN_CEIL(x, y)     (((x) + (y) - 1) / (y))

typedef float PDtype;

/* CHECK the state of CUDA */
#define PERCEPTION_CUDA_CHECK(status)                                   \
    {                                                                   \
        if (status != 0)                                                \
        {                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            abort();                                                    \
        }                                                               \
    }

/* For infrared detection */
static constexpr int locations = 4;
struct alignas(float) Detection
{
    float bbox[locations]; //x, y, w, h
    float score;
    float cls_id;
};

extern "C" void get_det_output(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int origin_h, const int origin_w, const int max_out_obj,
                        const int input_h, const int input_w, const int stride, float scale,
                        const float confThreshold, const float nmsThreshold, const int NUM_BOX_ELEMENT,
                        float* output, cudaStream_t stream);

extern "C" void gpu_decode_nms(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int origin_h, const int origin_w, const int max_out_obj,
                        const int input_h, const int input_w, const int stride, float scale,
                        const float confThreshold, const float nmsThreshold, const int NUM_BOX_ELEMENT,
                        float* output, cudaStream_t stream);

extern "C" void warp_affine_bilinear(
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhre7fV/
     */
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
);

#endif