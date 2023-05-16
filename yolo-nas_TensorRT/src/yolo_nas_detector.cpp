#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/yolo_nas_detector.h"
#include "../include/common.h"

static const int DEVICE = 0;

yolo_nas_detector::yolo_nas_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void yolo_nas_detector::init_context()
{
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    TRTLogger glogger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(glogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    input_index = engine->getBindingIndex(input_name.c_str());
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

    output_cls_index = engine->getBindingIndex(output_cls_name.c_str());
    auto output_cls_dims = engine->getBindingDimensions(output_cls_index);
    output_bbox_num = output_cls_dims.d[1];
    det_cls_len =  output_cls_dims.d[2];
    output_cls_buffer_size = batchsize * output_bbox_num * det_cls_len * sizeof(float);
    std::cout << "output shape: "
            << batchsize
            << " x "<< output_bbox_num
            << " x "<< det_cls_len
            << ", output_cls_buffer_size: " << output_cls_buffer_size
            << std::endl;
    CHECK(cudaHostAlloc((void**)&host_cls_output, output_cls_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_cls_index], output_cls_buffer_size));
    
    output_bbox_index = engine->getBindingIndex(output_bbox_name.c_str());
    auto output_bbox_dims = engine->getBindingDimensions(output_bbox_index);
    det_bbox_len =  output_bbox_dims.d[2];
    output_bbox_buffer_size = batchsize * output_bbox_num * det_bbox_len * sizeof(float);
    std::cout << "output shape: "
            << batchsize
            << " x "<< output_bbox_num
            << " x "<< det_bbox_len
            << ", output_bbox_buffer_size: " << output_bbox_buffer_size
            << std::endl;
    CHECK(cudaHostAlloc((void**)&host_bbox_output, output_bbox_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_bbox_index], output_bbox_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void yolo_nas_detector::destroy_context()
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

    for(int i = 0; i < 3; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }

    if(stream)
        CHECK(cudaStreamDestroy(stream));
    if(host_input)
        CHECK(cudaFreeHost(host_input));
    if(host_cls_output)
        CHECK(cudaFreeHost(host_cls_output));
    if(host_bbox_output)
        CHECK(cudaFreeHost(host_bbox_output));
}

yolo_nas_detector::~yolo_nas_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void yolo_nas_detector::pre_process_cpu(cv::Mat image)
{
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_w + scale - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_h + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    cv::Mat input_image(input_h, input_w, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), \
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 114));  // 对图像做平移缩放旋转变换,可逆

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0;
        *phost_g++ = pimage[1] / 255.0;
        *phost_b++ = pimage[2] / 255.0;
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void yolo_nas_detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process_cpu(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout<<"preprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    // context->enqueue(batchsize, device_buffers, stream, nullptr);
    context->enqueueV2(&device_buffers[0], stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    auto start_postprocess = std::chrono::high_resolution_clock::now();
    post_process_cpu(img);
    auto end_postprocess = std::chrono::high_resolution_clock::now();
    float postprocess_time = std::chrono::duration<float, std::milli>(end_postprocess - start_postprocess).count();
    std::cout<<"postprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Post-process done!"<<std::endl;
}

void yolo_nas_detector::post_process_cpu(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_bbox_output, device_buffers[output_bbox_index],
                    output_bbox_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(host_cls_output, device_buffers[output_cls_index],
                    output_cls_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float width = img.cols;
    float height = img.rows;
    float ratio_h = (input_h * 1.0f) / height;
    float ratio_w = (input_w * 1.0f) / width;
    float scale = std::min(input_w / (width*1.0), input_h / (height*1.0));

    std::vector<Object> proposals;
    std::vector<Object> objects;

    for (int index = 0; index < output_bbox_num; index++)
    {
        float* ptr = host_bbox_output + index * det_bbox_len;
        float* pclass = host_cls_output + index * det_cls_len;

        int label = std::max_element(pclass, pclass + det_cls_len) - pclass;
        float confidence = pclass[label];

        if (confidence < conf_thresh) continue;;

        float image_base_left = ptr[0];
        float image_base_top = ptr[1];
        float image_base_right = ptr[2];
        float image_base_bottom = ptr[3];

        /* clip */
        image_base_left = std::min(std::max(0.0f, image_base_left), float(img.cols - 1));
        image_base_top = std::min(std::max(0.0f, image_base_top), float(img.rows - 1));
        image_base_right = std::min(std::max(0.0f, image_base_right), float(img.cols - 1));
        image_base_bottom = std::min(std::max(0.0f, image_base_bottom), float(img.rows - 1));

        Object obj;
        obj.rect = cv::Rect_<float>(image_base_left, image_base_top,\
                            image_base_right - image_base_left,  image_base_bottom - image_base_top);
        obj.label = label;
        obj.score = confidence;
        proposals.push_back(obj);
    }

    qsort_descent_inplace(proposals);

    auto start_nms = std::chrono::system_clock::now();
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);
    auto end_nms = std::chrono::system_clock::now();
    std::cout<<"nms time:"<<std::chrono::duration_cast<std::chrono::microseconds>(end_nms - start_nms).count()<<" us"<<std::endl;

    int count = picked.size();

    std::cout << "num of boxes: " << count << std::endl;

    auto start_analysis = std::chrono::system_clock::now();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = objects[i].rect.x * d2i[0] + d2i[2];
        float y0 = objects[i].rect.y * d2i[0] + d2i[5];
        float x1 = (objects[i].rect.x + objects[i].rect.width) * d2i[0] + d2i[2];
        float y1 = (objects[i].rect.y + objects[i].rect.height) * d2i[0] + d2i[5];

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    auto end_analysis = std::chrono::system_clock::now();
    std::cout<<"analysis time:"<<std::chrono::duration_cast<std::chrono::microseconds>(end_analysis - start_analysis).count()<<" us"<<std::endl;

    auto start_draw = std::chrono::system_clock::now();
    draw_objects(img, objects);
    auto end_draw = std::chrono::system_clock::now();
    std::cout<<"draw_time:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_draw - start_draw).count()<<" ms"<<std::endl;
}