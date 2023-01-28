#ifndef _POINTPILLAR_H
#define _POINTPILLAR_H

#include <memory>

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "./params.h"
#include "./preprocess.h"
#include "./postprocess.h"

#define PERFORMANCE_LOG 1

class TRT
{
private:
    Params params_;
    
    cudaEvent_t start_, stop_;
    
    Logger gLogger_;
    nvinfer1::IExecutionContext *context_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    
    cudaStream_t stream_;

public:
    TRT(std::string modelFile, cudaStream_t stream);
    ~TRT();

    int doinfer(void** buffers);
};

class PointPillar
{
private:
    Params params_;
    
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;

    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<TRT> trt_;
    std::shared_ptr<PostProcessCuda>post_;

    //input of pre-process
    float* voxel_features_    = nullptr;
    unsigned int* voxel_num_  = nullptr;
    unsigned int* voxel_idxs_ = nullptr;
    unsigned int* pillar_num_ = nullptr;

    unsigned int voxel_features_size_ = 0;
    unsigned int voxel_num_size_      = 0;
    unsigned int voxel_idxs_size_     = 0;

    //TRT-input
    float* features_input_            = nullptr;
    unsigned int* params_input_       = nullptr;
    unsigned int features_input_size_ = 0;

    //output of TRT -- input of post-process
    float* cls_output_      = nullptr;
    float* box_output_      = nullptr;
    float* dir_cls_output_  = nullptr;

    unsigned int cls_size_;
    unsigned int box_size_;
    unsigned int dir_cls_size_;

    //output of post-process
    float* bndbox_output_     = nullptr;
    unsigned int bndbox_size_ = 0;

    std::vector<Bndbox> res_;

public:
    PointPillar(std::string modelFile, cudaStream_t stream);
    ~PointPillar();
    int doinfer(void* points, unsigned int point_size, std::vector<Bndbox>& res);
};

#endif