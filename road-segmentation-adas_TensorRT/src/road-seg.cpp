#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/road-seg.h"
#include "../include/common.h"

static const int DEVICE  = 0;

RoadSeg_detector::RoadSeg_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
#ifdef IS_NCHW
    std::cout << "Inference det ["
            << input_c
            << " x "
            << input_h
            << " x "
            << input_w
            << "] constructed"<<std::endl;
#else
    std::cout<<"Inference det ["
            <<input_h
            <<" x "
            <<input_w
            <<" x "
            <<input_c
            <<"] constructed"<<std::endl;
#endif
}

void RoadSeg_detector::init_context()
{
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    TRTLogger Logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(Logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    input_index = engine->getBindingIndex(input_blob_name.c_str());
    auto input_dims = engine->getBindingDimensions(input_index);
#ifdef IS_NCHW
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    input_h = input_dims.d[1];
    input_w = input_dims.d[2];
    input_c = input_dims.d[3];
#endif

    input_buffer_size = batchsize * input_c * input_h * input_w * sizeof(float);
    std::cout<<"input shape: "
            << batchsize
            << " x "<<input_c
            << " x "<<input_h
            << " x "<<input_w
            << ", input_buf_size: "<<input_buffer_size
            << std::endl;
    CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    output_index = engine->getBindingIndex(output_blob_name.c_str());
    auto output_dims = engine->getBindingDimensions(output_index);
    output_h = output_dims.d[1];
    output_w = output_dims.d[2];
    output_c = output_dims.d[3];
    output_buffer_size = batchsize * output_c * output_h *output_w * sizeof(float);
    std::cout<<"output shape: "
            << batchsize
            << " x " << output_h
            << " x " << output_w
            << " x " << output_c
            << ", output_buf_size: " << output_buffer_size
            << std::endl;
    CHECK(cudaHostAlloc((void**)&host_output, output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void RoadSeg_detector::destroy_context()
{
    bool cudart_ok = true;

    /* Release TensorRT */
    if(context)
    {
        context->destroy();
        context = nullptr;
    }
    if(engine)
    {
        engine->destroy();
        engine = nullptr;
    }
    for(int i = 0; i < 2; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }

    if(host_input)
        CHECK(cudaFreeHost(host_input));
    if(host_output)
        CHECK(cudaFreeHost(host_output));
}

RoadSeg_detector::~RoadSeg_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["
            <<output_h
            <<" x "
            <<output_w
            <<" x "
            <<output_c
            <<"]"
            <<std::endl;
}

void RoadSeg_detector::pre_process(cv::Mat image)
{
    cv::Mat input_image = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    cv::resize(image, input_image, input_image.size(), 0, 0, cv::INTER_LINEAR);

    std::cout<< "input_image shape: [" << input_image.rows << ", " << input_image.cols << "]"<<std::endl;

#ifdef IS_NCHW
    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        *phost_b++ = pimage[0];
        *phost_g++ = pimage[1];
        *phost_r++ = pimage[2];
    }
#else
    int image_area = input_image.cols * input_image.rows;
    unsigned char* p_image = input_image.data;
    float* p_input = host_input;
    for (int i = 0; i < image_area; i++, p_image += 3)
    {
        *p_input++ = p_image[0];
        *p_input++ = p_image[1];
        *p_input++ = p_image[2];
    }
#endif

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void RoadSeg_detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout<<"preprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    post_process(img);
    std::cout<<"Post-process done!"<<std::endl;
}

void RoadSeg_detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_output, device_buffers[output_index],
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<cv::Mat> image_fp32_list;
    for (int i = 0; i < output_c; i++)
    {
        image_fp32_list.push_back(cv::Mat(output_h, output_w, CV_32FC1));
    }

    for (int y = 0; y < output_h; y++)
    {
        for (int x = 0; x < output_w; x++)
        {
            for (int c = 0; c < output_c; c++)
            {
                image_fp32_list[c].at<float>(cv::Point(x,y)) = host_output[y * (output_w * output_c) + x * output_c + c];
            }
        }
    }

    for (int i = 0; i < image_fp32_list.size(); i++)
    {
        cv::cvtColor(image_fp32_list[i], image_fp32_list[i], cv::COLOR_GRAY2BGR); /* 1channel -> 3 channel */
        cv::resize(image_fp32_list[i], image_fp32_list[i], img.size());
        cv::multiply(image_fp32_list[i], GetColor(i), image_fp32_list[i]);
        image_fp32_list[i].convertTo(image_fp32_list[i], CV_8UC3, 1, 0);
    }

    for (int i = 0; i < image_fp32_list.size(); i++)
    {
        cv::add(img, image_fp32_list[i], img);
    }
}