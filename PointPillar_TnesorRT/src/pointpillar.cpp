#include "../include/pointpillar.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

TRT::TRT(std::string modelFile, cudaStream_t stream):stream_(stream)
{
    std::string trtmodel = modelFile + ".engine";
    std::fstream trtCache(trtmodel, std::ifstream::in);
    checkCudaErrors(cudaEventCreate(&start_));
    checkCudaErrors(cudaEventCreate(&stop_));
    if (!trtCache.is_open())
    {
        std::cout << "Building TRT engine." << std::endl;
        // define builder
        auto builder = nvinfer1::createInferBuilder(gLogger_);

        // define network
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);

        // define onnxpaser
        auto parser = nvonnxparser::createParser(*network, gLogger_);
        if (!parser->parseFromFile(modelFile.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                    << std::endl;
            exit(-1);
        }
        
        // define config
        auto networkConfig = builder->createBuilderConfig();

        networkConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Enable fp16!" << std::endl;

        // set max batch size
        builder->setMaxBatchSize(1);
        // set max workspace
        networkConfig->setMaxWorkspaceSize(size_t(1) << 30);

        engine_ = builder->buildEngineWithConfig(*network, *networkConfig);

        if (engine_ == nullptr)
        {
            std::cerr << ": engine init null!" << std::endl;
            exit(-1);
        }
        
        // serialize the engine, then close everything down
        auto trtModuleStream = engine_->serialize();
        std::fstream trtOut(trtmodel, std::ifstream::out);
        if (!trtOut.is_open())
        {
            std::cerr << "Can't store trt cache." << std::endl;
            exit(-1);
        }
        
        trtOut.write((char*)trtModuleStream->data(), trtModuleStream->size());
        trtOut.close();
        trtModuleStream->destroy();

        networkConfig->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
    }
    else
    {
        std::cout << "load TRT cache." << std::endl;
        char* data;
        unsigned int length;

        //get length of file;
        trtCache.seekg(0, trtCache.end);
        length = trtCache.tellg();
        trtCache.seekg(0, trtCache.beg);

        data = (char*)malloc(length);
        if (data == NULL)
        {
            std::cout << "Can't malloc data" << std::endl;
            exit(-1);
        }
        
        trtCache.read(data, length);
        // create context
        auto runtime = nvinfer1::createInferRuntime(gLogger_);

        if (engine_ == nullptr)
        {
            std::cout << "load TRT cache0."<<std::endl;
            std::cerr << ":engine null!" << std::endl;
            exit(-1);
        }
        free(data);
        trtCache.close();
    }

    context_ = engine_->createExecutionContext();
    return;
}

TRT::~TRT()
{
    delete(context_);
    delete(engine_);
    checkCudaErrors(cudaEventDestroy(start_));
    checkCudaErrors(cudaEventDestroy(stop_));
    return;
}

int TRT::doinfer(void** buffers)
{
    int status;
    status = context_->enqueueV2(buffers, stream_, &start_);
    if (!status)
    {
        return -1;
    }
    return 0;
}

PointPillar::PointPillar(std::string modelFile, cudaStream_t stream):stream_(stream)
{
    checkCudaErrors(cudaEventCreate(&start_));
    checkCudaErrors(cudaEventCreate(&stop_));

    pre_.reset(new PreProcessCuda(stream_));
    trt_.reset(new TRT(modelFile, stream_));
    post_.reset(new PostProcessCuda(stream_));

    // point cloud to voxels
    voxel_features_size_ = MAX_VOXELS * params_.max_num_points_per_pillar * 4 * sizeof(float);
    voxel_num_size_      = MAX_VOXELS * sizeof(float);
    voxel_idxs_size_     = MAX_VOXELS * 4 * sizeof(float);

    checkCudaErrors(cudaMallocManaged((void**)& voxel_features_, voxel_features_size_));
    checkCudaErrors(cudaMallocManaged((void**)& voxel_num_, voxel_num_size_));
    checkCudaErrors(cudaMallocManaged((void**)& voxel_idxs_, voxel_idxs_size_));

    checkCudaErrors(cudaMemsetAsync(voxel_features_, 0, voxel_features_size_, stream_));
    checkCudaErrors(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, stream_));
    checkCudaErrors(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, stream_));

    // TRT-input
    features_input_size_ = MAX_VOXELS * params_.max_num_points_per_pillar * 10 * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&features_input_, features_input_size_));
    checkCudaErrors(cudaMallocManaged((void **)&params_input_, sizeof(unsigned int)));

    checkCudaErrors(cudaMemsetAsync(features_input_, 0, features_input_size_, stream_));
    checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

    //output of TRT -- input of post-process
    cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_classes * params_.num_anchors * sizeof(float);
    box_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_box_values * params_.num_anchors * sizeof(float);
    dir_cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_dir_bins * params_.num_anchors * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&cls_output_, cls_size_));
    checkCudaErrors(cudaMallocManaged((void **)&box_output_, box_size_));
    checkCudaErrors(cudaMallocManaged((void **)&dir_cls_output_, dir_cls_size_));

    //output of post-process
    bndbox_size_ = (params_.feature_x_size * params_.feature_y_size * params_.num_anchors * 9 + 1) * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&bndbox_output_, bndbox_size_));

    res_.reserve(100);
    return;
}

PointPillar::~PointPillar()
{
  pre_.reset();
  trt_.reset();
  post_.reset();

  checkCudaErrors(cudaFree(voxel_features_));
  checkCudaErrors(cudaFree(voxel_num_));
  checkCudaErrors(cudaFree(voxel_idxs_));

  checkCudaErrors(cudaFree(features_input_));
  checkCudaErrors(cudaFree(params_input_));

  checkCudaErrors(cudaFree(cls_output_));
  checkCudaErrors(cudaFree(box_output_));
  checkCudaErrors(cudaFree(dir_cls_output_));

  checkCudaErrors(cudaFree(bndbox_output_));

  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

int PointPillar::doinfer(void* points_data, unsigned int points_size, std::vector<Bndbox>& nms_pred)
{
    float generateVoxelsTime = 0.0f;
    checkCudaErrors(cudaEventRecord(start_, stream_));

    pre_->generateVoxels((float*)points_data, points_size,
            params_input_,
            voxel_features_, 
            voxel_num_,
            voxel_idxs_);

    checkCudaErrors(cudaEventRecord(stop_, stream_));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&generateVoxelsTime, start_, stop_));
    unsigned int params_input_cpu;
    checkCudaErrors(cudaMemcpy(&params_input_cpu, params_input_, sizeof(unsigned int), cudaMemcpyDefault));
    std::cout<<"find pillar_num: "<< params_input_cpu <<std::endl;

    float generateFeaturesTime = 0.0f;
    checkCudaErrors(cudaEventRecord(start_, stream_));

    pre_->generateFeatures(voxel_features_,
        voxel_num_,
        voxel_idxs_,
        params_input_,
        features_input_);

    checkCudaErrors(cudaEventRecord(stop_, stream_));
    checkCudaErrors(cudaEventSynchronize(stop_));
    checkCudaErrors(cudaEventElapsedTime(&generateFeaturesTime, start_, stop_));

    float doinferTime = 0.0f;
    checkCudaErrors(cudaEventRecord(start_, stream_));

    void *buffers[] = {features_input_, voxel_idxs_, params_input_, cls_output_, box_output_, dir_cls_output_};
    trt_->doinfer(buffers);
    checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

    checkCudaErrors(cudaEventRecord(stop_, stream_));
    checkCudaErrors(cudaEventSynchronize(stop_));
    checkCudaErrors(cudaEventElapsedTime(&doinferTime, start_, stop_));

    float doPostprocessCudaTime = 0.0f;
    checkCudaErrors(cudaEventRecord(start_, stream_));

    post_->doPostprocessCuda(cls_output_, box_output_, dir_cls_output_,
                            bndbox_output_);
    checkCudaErrors(cudaDeviceSynchronize());
    float obj_count = bndbox_output_[0];

    int num_obj = static_cast<int>(obj_count);
    auto output = bndbox_output_ + 1;

    for (int i = 0; i < num_obj; i++) {
        auto Bb = Bndbox(output[i * 9],
                        output[i * 9 + 1], output[i * 9 + 2], output[i * 9 + 3],
                        output[i * 9 + 4], output[i * 9 + 5], output[i * 9 + 6],
                        static_cast<int>(output[i * 9 + 7]),
                        output[i * 9 + 8]);
        res_.push_back(Bb);
    }


    nms_cpu(res_, params_.nms_thresh, nms_pred);
    res_.clear();

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(stop_, stream_));
    checkCudaErrors(cudaEventSynchronize(stop_));
    checkCudaErrors(cudaEventElapsedTime(&doPostprocessCudaTime, start_, stop_));
    std::cout<<"TIME: generateVoxels: "<< generateVoxelsTime <<" ms." <<std::endl;
    std::cout<<"TIME: generateFeatures: "<< generateFeaturesTime <<" ms." <<std::endl;
    std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
    std::cout<<"TIME: doPostprocessCuda: "<< doPostprocessCudaTime <<" ms." <<std::endl;

    return 0;
}