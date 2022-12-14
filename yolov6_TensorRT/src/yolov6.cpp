#include "../include/yolov6.h"
#include <chrono>
#include <iostream>
#include "../include/common.h"
#include <fstream>
#include <unistd.h>

YOLOV6::YOLOV6(const int INPUT_H, 
           const int INPUT_W,
           const std::string& _engine_file):
    input_dim2(INPUT_H),
    input_dim3(INPUT_W),
    input_bufsize(input_dim0 * input_dim1 *
                input_dim2 * input_dim3 *
                sizeof(float)),
    output_bufsize(output_dim0 * 
                output_dim1 *
                output_dim2 *
                sizeof(float)),
    engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_dim2<<"x"<<input_dim3<<"] constructed"<<std::endl;
    init_done = true;
}

YOLOV6::~YOLOV6(){
    if(init_done){
        destroy_context();
        std::cout<<"Context destroyed for ["<<input_dim2<<"x"<<input_dim3<<"]"<<std::endl;
    }
}

void YOLOV6::init_context(){
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

    bool cudart_ok = true;
    cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(cudart_ok);

    /* Allocate memory for inference */
    const size_t buf_num = input_layers.size() + output_layers.size();
    std::cout<<"buf_num:"<<buf_num<<std::endl;
    cudaOutputBuffer = std::vector<void *>(buf_num, nullptr);
    hostOutputBuffer = std::vector<void *>(buf_num, nullptr);

    for(auto & layer : input_layers)
    {
        const std::string & name = layer.first;
        int index = create_binding_memory(name);
        input_layers.at(name) = index; /* init binding index */
    }

    for(auto & layer : output_layers)
    {
        const std::string & name = layer.first;
        int index = create_binding_memory(name);
        output_layers.at(name) = index; /* init binding index */
    }
}

void YOLOV6::destroy_context(){
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

    if(stream) cudaStreamDestroy(stream);

    CHECK_CUDA_ERROR(cudart_ok);

    /* Release memory for inference */
    for(int i=0; i < (int)cudaOutputBuffer.size(); i++)
    {
        if(cudaOutputBuffer[i])
        {
            cudaFree(cudaOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            cudaOutputBuffer[i] = nullptr;
        }
    }
    for(int i=0; i < (int)hostOutputBuffer.size(); i++)
    {
        if(hostOutputBuffer[i])
        {
            free(hostOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            hostOutputBuffer[i] = nullptr;
        }
    }
}

int YOLOV6::create_binding_memory(const std::string& bufname){
    assert(engine != nullptr);

    const int devbuf_num  = static_cast<int>(cudaOutputBuffer.size());
    const int hostbuf_num = static_cast<int>(hostOutputBuffer.size());

    int index = engine->getBindingIndex(bufname.c_str());

    size_t elem_size = 0;
    switch (engine->getBindingDataType(index)){
        case nvinfer1::DataType::kFLOAT:
            elem_size = sizeof(float); break;
        case nvinfer1::DataType::kHALF:
            elem_size = sizeof(float) >> 1; break;
        case nvinfer1::DataType::kINT8:
            elem_size = sizeof(int8_t); break;
        case nvinfer1::DataType::kINT32:
            elem_size = sizeof(int32_t); break;
        default:
            ; /* fallback */
    }
    assert(elem_size != 0);

    size_t elem_count = 0;
    nvinfer1::Dims dims = engine->getBindingDimensions(index);
    for (int i = 0; i < dims.nbDims; i++){
        if (0 == elem_count){
            elem_count = dims.d[i];
        }
        else{
            elem_count *= dims.d[i];
        }
    }

    size_t buf_size = elem_count * elem_size;
    assert(buf_size != 0);

    void * device_mem;
    bool cudart_ok = true;

    cudaMalloc(&device_mem, buf_size);
    CHECK_CUDA_ERROR(cudart_ok);
    assert(cudart_ok);

    cudaOutputBuffer[index] = device_mem;

    void * host_mem = malloc( buf_size );
    assert(host_mem != nullptr);

    hostOutputBuffer[index] = host_mem;

    printf("Created host and device buffer for %s "   \
        "with bindingIndex[%d] and size %lu bytes.\n", \
        bufname.c_str(), index, buf_size );

    return index;
}

void* YOLOV6::get_infer_bufptr(const std::string& bufname, bool is_device){
    assert(init_done);

    int index = -1;

    if (bufname == input)
    {
        index = input_layers.at(bufname);
    }
    else
    {
        index = output_layers.at(bufname);
    }

    return (is_device ? cudaOutputBuffer.at(index) : hostOutputBuffer.at(index));
}

void YOLOV6::do_inference(cv::Mat& image, cv::Mat& dst){
    assert(context != nullptr);

    void* host_input = get_infer_bufptr(input, false);
    void* device_input = get_infer_bufptr(input, true);

    void* host_output = get_infer_bufptr(output, false);
    void* device_output = get_infer_bufptr(output, true);
    
    pre_process(image, input_dim2, input_dim3, static_cast<float *>(host_input));

    cudaMemcpyAsync(device_input, host_input, input_bufsize,
                        cudaMemcpyHostToDevice, stream);
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();

    context->enqueue(batchsize, &cudaOutputBuffer[0], stream, nullptr);

    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    cudaMemcpyAsync(host_output, device_output, output_bufsize,
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(res_ok);
    assert(res_ok);
    std::cout<<"Model enqueue done!"<<std::endl;

    post_process(dst, static_cast<float *>(host_output));
    std::cout<<"Post-process done!"<<std::endl;
}

void YOLOV6::pre_process(cv::Mat& img, 
                    int input_height,
                    int input_width,
                    float* input_blob){
    float r = std::min(input_dim2 / (img.rows * 1.0), input_dim3 / (img.cols * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_dim2, input_dim3, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    int channels = 3;
    int img_h = out.rows;
    int img_w = out.cols;
    std::cout<<channels<<","<<img_h<<","<<img_w<<","<<img.total()*3<<std::endl;
    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                input_blob[c * img_w * img_h + h * img_w + w] = 
                        (((float)out.at<cv::Vec3b>(h, w)[c])) / 255.0;
            }
        }
    }
}

void YOLOV6::post_process(cv::Mat& img,
                        float* host_output){
    float scale = std::min(input_dim3 / (img.cols*1.0), input_dim2 / (img.rows*1.0));
    int img_w = img.cols;
    int img_h = img.rows;
    std::vector<Object> objects;
    decode_outputs(host_output, objects, output_dim1, scale, img_w, img_h);
    draw_objects(img, objects);
}