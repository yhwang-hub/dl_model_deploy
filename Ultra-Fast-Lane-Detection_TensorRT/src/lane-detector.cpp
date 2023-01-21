#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/lane-detector.h"
#include "../include/common.h"

static const int DEVICE  = 0;

Lane_detector::Lane_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void Lane_detector::init_context()
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

    TRTLogger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
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
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    input_buffer_size = batchsize * input_c * input_h * input_w * sizeof(float);
    std::cout<<"input shape: "
            <<batchsize
            <<" x "<<input_c
            <<" x "<<input_h
            <<" x "<<input_w
            <<", input_buf_size: "<<input_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    output_index = engine->getBindingIndex(output_blob_name.c_str());
    auto exit_dims = engine->getBindingDimensions(output_index);
    output_c = exit_dims.d[1];
    output_h = exit_dims.d[2];
    output_w = exit_dims.d[3];
    output_buffer_size = batchsize * output_c * output_h * output_w * sizeof(float);
    std::cout << "output shape: "
        << batchsize
        <<" x "<< output_c
        <<" x "<< output_h
        <<" x "<< output_w
        <<", input_buf_size: "<< output_buffer_size
        <<std::endl;
    CHECK(cudaHostAlloc((void**)&host_output, output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void Lane_detector::destroy_context()
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

    if (host_input)
    {
        CHECK(cudaFreeHost(host_input));
    }

    if (host_output)
    {
        CHECK(cudaFreeHost(host_output));
    }
}

Lane_detector::~Lane_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Lane_detector::pre_process(cv::Mat image)
{
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(input_w, input_h));

    std::cout<< "input_image shape: [" << input_image.cols << ", " << input_image.rows << "]"<<std::endl;

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - norm_mean[0]) / norm_std[0];
        *phost_g++ = (pimage[1] / 255.0f - norm_mean[1]) / norm_std[1];
        *phost_b++ = (pimage[2] / 255.0f - norm_mean[2]) / norm_std[2];
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Lane_detector::do_detection(cv::Mat& img)
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

void Lane_detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_output, device_buffers[output_index],
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float max_ind[batchsize * output_h * output_w];
    float prob_reverse[batchsize * (output_buffer_size / sizeof(float))];
    /* do out_j = out_j[:, ::-1, :] in python list*/
    float expect[batchsize * output_h * output_w];
    for (int k = 0, wh = output_w * output_h; k < output_c; k++)
    {
        for(int j = 0; j < output_h; j ++)
        {
            for(int l = 0; l < output_w; l++)
            {
                prob_reverse[k * wh + (output_h - 1 - j) * output_w + l] =
                    host_output[k * wh + j * output_w + l];
            }
        }
    }

    argmax(prob_reverse, max_ind, output_h, output_w, output_c);
    /* calculate softmax and Expect */
    softmax_mul(prob_reverse, expect, output_h, output_w, output_c);
    for(int k = 0; k < output_h; k++)
    {
        for(int j = 0; j < output_w; j++) 
        {
            if (max_ind[k * output_w + j] == cuLaneGriding_num)
            {
                expect[k * output_w + j] = 0;
            }
            else
            {
                expect[k * output_w + j] = expect[k * output_w + j];
            }
        }
    }
    std::vector<int> i_ind;
    for(int k = 0; k < output_w; k++)
    {
        int ii = 0;
        for(int g = 0; g < output_h; g++)
        {
            if(expect[g * output_w + k] != 0)
                ii++;
        }
        if(ii > 2)
        {
            i_ind.push_back(k);
        }
    }

    int img_h = img.rows;
    int img_w = img.cols;
    // Logic
    std::vector<double> linSpaceVector = linspace(0, input_w - 1, cuLaneGriding_num);
    double linSpace = linSpaceVector[1] - linSpaceVector[0];
    for(int k = 0; k < output_h; k++)
    {
        for(int ll = 0; ll < i_ind.size(); ll++)
        {
            if(expect[output_w * k + i_ind[ll]] > 0)
            {
                cv::Point pp =
                    { int(expect[output_w * k + i_ind[ll]] * linSpace * img_w / input_w) - 1,
                        int( img_h * culane_row_anchor[output_h - 1 - k] / input_h) - 1 };
                cv::circle(img, pp, 8, CV_RGB(0, 255 ,0), 2);
            }
        }
    }
}