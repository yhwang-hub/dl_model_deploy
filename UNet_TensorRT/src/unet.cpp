#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/unet.h"

static const int DEVICE = 0;

#define INPUTDEBUG
// #define INFERDEBUG
// #define ARGMAXDEBUG

UNet_Detector::UNet_Detector(const std::string& _engine_file):
                engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
}

void UNet_Detector::init_context()
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
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    input_index  = engine->getBindingIndex(input_name.c_str());
    output_index = engine->getBindingIndex(output_name.c_str());

    auto input_dims = engine->getBindingDimensions(input_index);
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    std::cout<<"input dim:["
            <<batchsize
            <<" x "<<input_c
            <<" x "<<input_h
            <<" x "<<input_w
            <<"]"<<std::endl;
    input_buffer_size = batchsize * input_c * input_w * input_h * sizeof(float);
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    auto output_dims = engine->getBindingDimensions(output_index);
    out_channels  = output_dims.d[1];
    output_height = output_dims.d[2];
    output_width  = output_dims.d[3];
    std::cout<<"output dim:["
            <<batchsize
            <<" x "<<out_channels
            <<" x "<<output_height
            <<" x "<<output_width
            <<"]"<<std::endl;
    output_buffer_size = batchsize * out_channels * output_height * output_width * sizeof(float);
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_output, output_buffer_size, cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));

    PERCEPTION_CUDA_CHECK(cudaStreamCreate(&stream));
}

UNet_Detector::~UNet_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void UNet_Detector::destroy_context()
{
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
    if(stream)
    {
        cudaStreamDestroy(stream);
    }
    if(host_input)
    {
        PERCEPTION_CUDA_CHECK(cudaFreeHost(host_input));
    }
    if(host_output)
    {
        PERCEPTION_CUDA_CHECK(cudaFreeHost(host_output));
    }
    for(int i = 0; i < 2; ++i)
    {
        if(device_buffers[i])
        {
            PERCEPTION_CUDA_CHECK(cudaFree(device_buffers[i]));
        }
    }
}

void UNet_Detector::pre_process_cpu(cv::Mat& img)
{
    cv::Mat image = img.clone();
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_w + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_h + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_h, input_w, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), \
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(128));
    cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void UNet_Detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process_cpu(img);
    // pre_process_gpu(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout << "preprocess take: " << preprocess_time/1000 << " s." << std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

#ifdef INPUTDEBUG
    std::string pytorch_preprocess_txt = "pytorch_preprocess.txt";
    std::vector<float>pytorch_preprocess;
    std::ifstream pytorch_preprocess_data(pytorch_preprocess_txt);
    float data;
    while (pytorch_preprocess_data >> data)
    {
        pytorch_preprocess.push_back(data);
    }

    std::string Tensorrt_preprocess_txt_name = "Tensorrt_preprocess.txt";
    if (access(Tensorrt_preprocess_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_preprocess_txt_name.c_str()) == 0){
            std::cout<<Tensorrt_preprocess_txt_name<<" has been deleted successfuly!"<<std::endl;
        }
    }
    std::ofstream Tensorrt_prerpocess;
    Tensorrt_prerpocess.open(Tensorrt_preprocess_txt_name);

    float max_diff = 0.0;
    for (size_t i = 0; i < input_buffer_size / sizeof(float); i++)
    {
        float trt_data = static_cast<float*>(host_input)[i];
        float pytorch_data = pytorch_preprocess[i];
        Tensorrt_prerpocess<<trt_data<<std::endl;
        float diff = std::abs(trt_data - pytorch_data);
        if (diff > max_diff)
        {
            std::cout<<i
                    <<", trt_data: "<<trt_data
                    <<", pytorch_data: "<<pytorch_data
                    <<std::endl;
            max_diff = diff;
        }
    }
    std::cout<<"preprocess between Tensorrt and pytorch, max diff is: "<<max_diff<<std::endl;
    Tensorrt_prerpocess.close();
#endif

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;
    post_process_cpu(img);
}

void UNet_Detector::post_process_cpu(cv::Mat& img)
{
    PERCEPTION_CUDA_CHECK(cudaMemcpyAsync(host_output, device_buffers[output_index],
                            output_buffer_size, cudaMemcpyDeviceToHost, stream));
    PERCEPTION_CUDA_CHECK(cudaStreamSynchronize(stream));

    float scale_x = input_w / (float)img.cols;
    float scale_y = input_h / (float)img.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * img.cols + input_w + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * img.rows + input_h + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat prob, iclass;
    std::tie(prob, iclass) = post_process(host_output, output_width, output_height, out_channels);
    cv::warpAffine(prob, prob, m2x3_d2i, img.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, m2x3_d2i, img.size(), cv::INTER_NEAREST);
    render(img, prob, iclass);

    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("image-draw.jpg", img);
}