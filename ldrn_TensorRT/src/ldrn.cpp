#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/ldrn.h"
#include "../include/common.h"

static const int DEVICE  = 0;


Ldrn_detector::Ldrn_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void Ldrn_detector::init_context()
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

    for (int i = 0; i < num_output; i++)
    {
        output_index[i] = engine->getBindingIndex(output_blob_names[i].c_str());
        auto output_dim = engine->getBindingDimensions(output_index[i]);
        int batch_size = output_dim.d[0];
        int output_c   = output_dim.d[1];
        int output_h   = output_dim.d[2];
        int output_w   = output_dim.d[3];
        output_dims[i][0] = batch_size;
        output_dims[i][1] = output_c;
        output_dims[i][2] = output_h;
        output_dims[i][3] = output_w;
        output_buffer_size[i] = batch_size * output_c * output_h * output_w * sizeof(float);
        std::cout << "output shape: "
        << batch_size
        << " x " << output_c
        << " x " << output_h
        << " x " << output_w
        << ", output_buf_size: " << output_buffer_size[i]
        << std::endl;
        CHECK(cudaHostAlloc((void**)&host_output[i], output_buffer_size[i], cudaHostAllocDefault));
        CHECK(cudaMalloc(&device_buffers[output_index[i]], output_buffer_size[i]));
    }

    CHECK(cudaStreamCreate(&stream));
}

void Ldrn_detector::destroy_context()
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
    for(int i = 0; i < num_output + 1; ++i)
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

    for (int i = 0; i < num_output; i++)
    {
        if (host_output[i])
        {
            CHECK(cudaFreeHost(host_output[i]));
        }
    }
}

Ldrn_detector::~Ldrn_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Ldrn_detector::pre_process(cv::Mat image)
{
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(input_w, input_h), 0.5, 0.5);

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

void Ldrn_detector::do_detection(cv::Mat& img)
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

void Ldrn_detector::post_process(cv::Mat& img)
{
    for (int i = 0; i < num_output; i++)
    {
        CHECK(cudaMemcpyAsync(host_output[i], device_buffers[output_index[i]],
                    output_buffer_size[i], cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    float* values = host_output[num_output - 1];
    int output_h = output_dims[num_output - 1][2];
    int output_w = output_dims[num_output - 1][3];

    cv::Mat mat_out = cv::Mat(output_h, output_w, CV_32FC1, values);  /* value has no specific range */

    mat_out.convertTo(mat_out, CV_8UC1, -4, 255);   /* experimentally deterined */
    mat_out = mat_out(cv::Rect(0, static_cast<int32_t>(mat_out.rows * 0.18), mat_out.cols, static_cast<int32_t>(mat_out.rows * (1.0 - 0.18))));
    std::cout << "mat_out height: " << mat_out.rows << ", mat_out width: " << mat_out.cols << std::endl;// 209 * 512

    cv::Mat mat_depth;
    cv::applyColorMap(mat_out, mat_depth, cv::COLORMAP_PLASMA);

    /* Create result image */
    //mat = mat_depth;
    double scale = static_cast<double>(img.cols) / mat_depth.cols;
    cv::resize(mat_depth, mat_depth, cv::Size(), scale, scale);
    std::cout << "mat_depth height: " << mat_depth.rows << ", mat_depth width: " << mat_depth.cols << std::endl;
    cv::vconcat(img, mat_depth, img);
}
