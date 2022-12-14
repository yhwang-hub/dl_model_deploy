#include "../include/yolov7.h"
// #include <chrono>
#include <iostream>
// #include <fstream> 

// #define DEBUG

static const int DEVICE  = 1;

static const int OUTPUT_H = 25200;
static const int OUTPUT_W = 85;

yolov7_trt::yolov7_trt(const std::string& _engine_file):
        input_dim2(INPUT_H),
        input_dim3(INPUT_W),
        input_bufsize(input_dim0 * input_dim1 * 
                      input_dim2 * input_dim3 * 
                      sizeof(float)),
        output_dim1(OUTPUT_H),
        output_dim2(OUTPUT_W),
        output_bufsize(output_dim0 * output_dim1 * 
                      output_dim2 * 
                      sizeof(float)),
        engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<INPUT_H<<"x"<<INPUT_W<<"] constructed"<<std::endl;
    std::cout<<"input_bufsize: "<<input_bufsize<<", output_bufsize: "<<output_bufsize<<std::endl;
    init_done = true;
}

yolov7_trt::~yolov7_trt(){
     if(init_done){
        destroy_context();
        std::cout<<"Context destroyed for ["<<INPUT_H<<"x"<<INPUT_W<<"]"<<std::endl;
    }
}

void yolov7_trt::init_context(){
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
        std::cout<<"input_layers name: "<<name<<std::endl;
        int index = create_binding_memory(name);
        input_layers.at(name) = index; /* init binding index */
    }

    for(auto & layer : output_layers)
    {
        const std::string & name = layer.first;
        std::cout<<"output_layers name: "<<name<<std::endl;
        int index = create_binding_memory(name);
        output_layers.at(name) = index; /* init binding index */
    }
}

void yolov7_trt::destroy_context(){
    bool cudart_ok = true;

    /* Release TensorRT */
    if(context)
    {
        context->destroy();
        context = nullptr;
        std::cout<<"context destroy!"<<std::endl;
    }

    if(engine)
    {
        engine->destroy();
        engine = nullptr;
        std::cout<<"engine destroy!"<<std::endl;
    }

    if(stream) cudaStreamDestroy(stream);

    CHECK_CUDA_ERROR(cudart_ok);

    /* Release memory for inference */
    for(int i=0; i < (int)cudaOutputBuffer.size(); i++)
    {
        if(cudaOutputBuffer[i])
        {
            std::cout<<"cudaOutputBuffer["<<i<<"] destroy!"<<std::endl;
            cudaFree(cudaOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            cudaOutputBuffer[i] = nullptr;
        }
    }

    for(int i=0; i < (int)hostOutputBuffer.size(); i++)
    {
        if(hostOutputBuffer[i] != nullptr)
        {
            std::cout<<"hostOutputBuffer["<<i<<"] destroy!"<<std::endl;
            free(hostOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            hostOutputBuffer[i] = nullptr;
        }
    }
}

int yolov7_trt::create_binding_memory(const std::string& bufname){
    assert(engine != nullptr);

    const int devbuf_num  = static_cast<int>(cudaOutputBuffer.size());
    const int hostbuf_num = static_cast<int>(hostOutputBuffer.size());

    std::cout<<"devbuf_num: "<<devbuf_num<<", hostbuf_num: "<<hostbuf_num<<std::endl;
    int index = engine->getBindingIndex(bufname.c_str());
    std::cout<<"bufname: "<<bufname<<", index: "<<index<<std::endl;

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

inline void* yolov7_trt::get_infer_bufptr(const std::string& bufname, bool is_device){
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
    std::cout<<bufname<<":"<<index<<std::endl;
    return (is_device ? cudaOutputBuffer.at(index) : hostOutputBuffer.at(index));
}

void yolov7_trt::pre_process(cv::Mat& img, float* host_input){
// #ifdef DEBUG    
//     std::string Tensorrt_preprocess_txt_name = "Tensorrt_preprocess.txt";
//     if (access(Tensorrt_preprocess_txt_name.c_str(),0) == 0){
//         if(remove(Tensorrt_preprocess_txt_name.c_str()) == 0){
//             std::cout<<Tensorrt_preprocess_txt_name<<" has been deleted successfuly!"<<std::endl;
//         }
//     }
//     std::ofstream Tensorrt_preprocess;
//     Tensorrt_preprocess.open(Tensorrt_preprocess_txt_name);
// #endif

//     float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
//     int unpad_w = r * img.cols;
//     int unpad_h = r * img.rows + 0.5;
//     std::cout<<"unpad_w:"<<unpad_w<<","<<"unpad_h:"<<unpad_h<<std::endl;
//     cv::Mat re(unpad_h, unpad_w, CV_8UC3);
//     cv::resize(img, re, re.size());
//     cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
//     re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
//     cv::cvtColor(out, img, cv::COLOR_BGR2RGB);

//     // cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
//     // cv::resize(img, img, re.size());

//     int channels = 3;
//     int img_h = img.rows;
//     int img_w = img.cols;
//     std::cout<<channels<<","<<img_h<<","<<img_w<<","<<img.total()*3<<std::endl;
//     for (size_t c = 0; c < channels; c++) 
//     {
//         for (size_t  h = 0; h < img_h; h++) 
//         {
//             for (size_t w = 0; w < img_w; w++) 
//             {
//                 float data = ((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0;
//                 blob[c * img_w * img_h + h * img_w + w] = data;
// #ifdef DEBUG
//                 Tensorrt_preprocess<<data<<"\n";
// #endif          
//             }
//         }
//     }
// #ifdef DEBUG
//     Tensorrt_preprocess.close();
// #endif

    // letter box
    // 通过双线性插值对图像进行resize
    cv::Mat image = img.clone();
    float scale_x = input_dim3 / (float)image.cols;
    float scale_y = input_dim2 / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_dim3 + scale - 1) * 0;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_dim2 + scale - 1) * 0;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    cv::Mat input_image(input_dim2, input_dim3, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), \
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
}

void yolov7_trt::do_inference(cv::Mat& image,
                            cv::Mat& dst,
                            std::string input_image_path){
    assert(context != nullptr);

    void* host_input = get_infer_bufptr(input, false); /* is_device */
    void* device_input = get_infer_bufptr(input, true);

    void* host_output = get_infer_bufptr(output, false);
    void* device_output = get_infer_bufptr(output, true);

    pre_process(image, static_cast<float*>(host_input));
    
    cudaMemcpyAsync(device_input, host_input, input_bufsize,
                    cudaMemcpyHostToDevice, stream);

    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
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


#ifdef DEBUG    
    std::string Tensorrt_output_txt_name = "Tensorrt_output.txt";
    if (access(Tensorrt_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_output_txt_name.c_str()) == 0){
            std::cout<<Tensorrt_output_txt_name<<" has been deleted successfuly!"<<std::endl;
        }
    }
    std::ofstream Tensorrt_output;
    Tensorrt_output.open(Tensorrt_output_txt_name);

    for (size_t i = 0; i < output_bufsize / sizeof(float); i++)
    {
        Tensorrt_output<<static_cast<float*>(host_output)[i]<<std::endl;
    }
    
    Tensorrt_output.close();
#endif

    std::vector<Object> objects;
    float scale = std::min(INPUT_W / (dst.cols*1.0), INPUT_H / (dst.rows*1.0));
    int img_w = dst.cols;
    int img_h = dst.rows;
    decode_outputs(static_cast<float*>(host_output), output_bufsize / sizeof(float), objects, scale, img_w, img_h);
    draw_objects(dst, objects, input_image_path);
}