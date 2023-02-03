#include <iostream>
#include "../include/seaformer.h"
#include "../include/common.h"

static const int DEVICE  = 0;

seaformer_Detector::seaformer_Detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
    std::cout<<"Inference segmentation ["<<output_h<<"x"<<output_w<<"] constructed"<<std::endl;
}

void seaformer_Detector::init_context()
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

    output_index = engine->getBindingIndex(output_name.c_str());
    auto seg_output_dims = engine->getBindingDimensions(output_index);
    output_c = seg_output_dims.d[1];
    output_h = seg_output_dims.d[2];
    output_w = seg_output_dims.d[3];
    output_buffer_size = batchsize * output_c * output_h * output_w * sizeof(float);
    std::cout<<"output shape: "
            <<batchsize
            <<" x "<<output_c
            <<" x "<<output_h
            <<" x "<<output_w
            <<", output_buffer_size: "<<output_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&host_output, output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void seaformer_Detector::destroy_context()
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

seaformer_Detector::~seaformer_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void seaformer_Detector::pre_process(cv::Mat& img)
{
    cv::Mat image = img.clone();
    cv::Size input_geometry(input_w, input_h);
    cv::Mat input_image;
    cv::resize(image, input_image, input_geometry);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] - mean_rgb[2]) / std_rgb[2];
        *phost_g++ = (pimage[1] - mean_rgb[1]) / std_rgb[1];
        *phost_b++ = (pimage[2] - mean_rgb[0]) / std_rgb[0];
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void seaformer_Detector::do_detection(cv::Mat& img)
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

void seaformer_Detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_output, device_buffers[output_index],
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    int img_h = img.rows;
    int img_w = img.cols;
    float scale_x = input_w / (float)img.cols;
    float scale_y = input_h / (float)img.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 0;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 0;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat output_prob(output_h, output_w, CV_32F);
    cv::Mat output_index(output_h, output_w, CV_8U);
    float* pnet   = host_output;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);
    int wh = output_h * output_w;
    for(int index = 0; index < output_prob.cols * output_prob.rows; ++index, ++prob, ++pidx)
    {
        float max = -10000000;
        int max_ind = -1;
        float sum_c = 0.0;
        for(int k = 0; k < output_c; k++)
        {
            float data = host_output[k * wh + index];
            if(data > max)
            {
                max = data;
                max_ind = k;
            }
            sum_c += expf(data);
        }
        // *prob  = 1. / (1. + expf(-pnet[max_ind]));
        *prob = expf(pnet[max_ind]) / sum_c;
        *pidx  = max_ind;
    }

    cv::warpAffine(output_prob, output_prob, m2x3_d2i, img.size(), cv::INTER_LINEAR);
    cv::warpAffine(output_index, output_index, m2x3_d2i, img.size(), cv::INTER_LINEAR);

    for (int i = 0; i < img_h; i++)
    {
        for (int j = 0; j < img_w; j++)
        {
            int max_ind = output_index.at<uint8_t>(i, j);
            img.at<cv::Vec3b>(i, j)[0] = (uint8_t)(img.at<cv::Vec3b>(i, j)[0] * 0.5 +  _classes_colors[max_ind][2] * 0.5);
            img.at<cv::Vec3b>(i, j)[1] = (uint8_t)(img.at<cv::Vec3b>(i, j)[1] * 0.5 +  _classes_colors[max_ind][1] * 0.5);
            img.at<cv::Vec3b>(i, j)[2] = (uint8_t)(img.at<cv::Vec3b>(i, j)[2] * 0.5 +  _classes_colors[max_ind][0] * 0.5);
        }
    }
    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("image-draw.jpg", img);
}