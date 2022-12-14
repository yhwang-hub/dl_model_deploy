#ifndef UNET_H
#define UNET_H

#include "common.h"

class UNet_Detector
{
public:
    const int batchsize = 1;
    int input_h;
    int input_w;
    int input_c;
    int out_channels;
    int output_height;
    int output_width;
    const std::string input_name  = "images";
    const std::string output_name = "output";
    UNet_Detector(const std::string& _engine_file);
    virtual ~UNet_Detector();
    virtual void do_detection(cv::Mat& img);
private:
    int input_buffer_size;
    int output_buffer_size;
    int input_index;
    int output_index;
    float* host_input;
    float* host_output;
    void* device_buffers[2];
    void* device_output_buffer;

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;
    cv::cuda::Stream _gpu_stream;
    cv::cuda::GpuMat _img_gpu;

    void init_context();
    void destroy_context();
    void pre_process_cpu(cv::Mat &img);
    void post_process_cpu(cv::Mat& img);
};

#endif