#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/yolov8.h"
#include "../include/common.h"
#include "../include/cuda_kernel.h"

#define USE_CUDA

static const int DEVICE  = 0;

Yolov8_detector::Yolov8_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void Yolov8_detector::init_context()
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

    output_index = engine->getBindingIndex(output_blob_name.c_str());
    auto output_dims = engine->getBindingDimensions(output_index);
    output_c = output_dims.d[1];
    output_bbox_num = output_dims.d[2];
    output_buffer_size = batchsize * output_c * output_bbox_num * sizeof(float);
    std::cout<<"output shape: "
            <<batchsize
            <<" x "<< output_c
            <<" x "<< output_bbox_num
            <<", output_buffer_size: "<<output_buffer_size
            <<std::endl;
#ifdef USE_CUDA
    CHECK(cudaMalloc(&device_output_transpose, output_buffer_size));
    CHECK(cudaMalloc(&device_output_objects, (topK * output_objects_width + 1) * sizeof(float)));
    CHECK(cudaMalloc(&device_output_idx, batchsize * topK * sizeof(int)));
    CHECK(cudaMalloc(&device_output_conf, batchsize * topK * sizeof(float)));
    CHECK(cudaHostAlloc((void**)&host_output, (topK * output_objects_width + 1) * sizeof(float), cudaHostAllocDefault));
#else
    CHECK(cudaHostAlloc((void**)&host_output, output_buffer_size, cudaHostAllocDefault));
#endif
    CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));
    

    CHECK(cudaStreamCreate(&stream));
}

void Yolov8_detector::destroy_context()
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

#ifdef USE_CUDA
    if(device_output_transpose)
        CHECK(cudaFree(device_output_transpose));
    if (device_output_objects)
        CHECK(cudaFree(device_output_objects));
    if (device_output_idx)
        CHECK(cudaFree(device_output_idx));
    if (device_output_conf)
        CHECK(cudaFree(device_output_conf));
#endif

    if(stream)
        CHECK(cudaStreamDestroy(stream));
    if(host_input)
        CHECK(cudaFreeHost(host_input));
    if(host_output)
        CHECK(cudaFreeHost(host_output));
}

Yolov8_detector::~Yolov8_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Yolov8_detector::pre_process(cv::Mat image)
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
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
    // cv::imwrite("input_image.jpg", input_image);

    std::cout<< "input_image shape: [" << input_image.cols << ", " << input_image.rows << "]"<<std::endl;

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

void Yolov8_detector::do_detection(cv::Mat& img)
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

    auto start_postprocess = std::chrono::high_resolution_clock::now();
#ifdef USE_CUDA
    post_process_gpu(img);
#else
    post_process(img);
#endif
    auto end_postprocess = std::chrono::high_resolution_clock::now();
    float postprocess_time = std::chrono::duration<float, std::milli>(end_postprocess - start_postprocess).count();
    std::cout<<"postprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Post-process done!"<<std::endl;
}

void Yolov8_detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_output, device_buffers[output_index],
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float width = img.cols;
    float height = img.rows;
    float x_scale = input_w / (img.cols*1.0);
    float y_scale = input_h / (img.rows*1.0);
    float scale = std::min(input_w / (width*1.0), input_h / (height*1.0));
    std::vector<Object> proposals;
    std::vector<Object> objects;
    for (int index = 0; index < output_bbox_num; index++)
    {
        float max_confidence = INT_MIN;
        int label = 0;
        for (int i = 0; i < num_classes; i++)
        {
            int class_confidence_index = 4 * output_bbox_num + i * output_bbox_num + index;
            float class_confidence = host_output[class_confidence_index];
            if (class_confidence > max_confidence)
            {
                max_confidence = class_confidence;
                label = i;
            }
        }
        
        // std::cout<< "max_confidence: " << max_confidence <<std::endl;
        if (max_confidence < conf_thresh) continue;

        int cx_index = index + output_bbox_num * 0;
        int cy_index = index + output_bbox_num * 1;
        int w_index  = index + output_bbox_num * 2;
        int h_index  = index + output_bbox_num * 3;

        float x_center = host_output[cx_index];
        float y_center = host_output[cy_index];
        float w = host_output[w_index];
        float h = host_output[h_index];

        float left   = x_center - w * 0.5f;
        float top    = y_center - h * 0.5f;
        float right  = x_center + w * 0.5f;
        float bottom = y_center + h * 0.5f;
        float image_base_left   = d2i[0] * left   + d2i[2];
        float image_base_right  = d2i[0] * right  + d2i[2];
        float image_base_top    = d2i[0] * top    + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];

        /* clip */
        image_base_left = std::min(std::max(0.0f, image_base_left), float(img.cols - 1));
        image_base_top = std::min(std::max(0.0f, image_base_top), float(img.rows - 1));
        image_base_right = std::min(std::max(0.0f, image_base_right), float(img.cols - 1));
        image_base_bottom = std::min(std::max(0.0f, image_base_bottom), float(img.rows - 1));

        Object obj;
        obj.rect = cv::Rect_<float>(image_base_left, image_base_top,\
                            image_base_right - image_base_left,  image_base_bottom - image_base_top);
        obj.label = label;
        obj.score = max_confidence;
        proposals.push_back(obj);
    }

    std::cout<<"Num of proposals: "<< proposals.size() <<std::endl;

    /* Perform non maximum suppression */
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    int count = picked.size();

    std::cout<<"after nms Num of boxes: "<< count <<std::endl;

    /* Draw & show res */
    objects.resize(count);
    for(int i = 0; i < count; ++i)
    {
        objects[i] = proposals[picked[i]];
        objects[i].rect.x = objects[i].rect.x;
        objects[i].rect.y = objects[i].rect.y;
        objects[i].rect.width  = objects[i].rect.width;
        objects[i].rect.height = objects[i].rect.height;

        const Object& obj = proposals[picked[i]];
        float x1 = obj.rect.x;
        float y1 = obj.rect.y;
        float x2 = x1 + obj.rect.width;
        float y2 = y1 + obj.rect.height;

        std::cout<<"x1:"<<x1
                <<", y1:"<<y1
                <<", x2:"<<x2
                <<", y2:"<<y2
                <<", score:"<<obj.score
                <<", label:"<<obj.label
                <<std::endl;
        // std::string label = _object_classes[obj.label];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2, 8, 0);
        // cv::putText(image, label, cv::Point2d(x1 + 5, y1 + 5),
        //                 cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    draw_objects(img, objects);
}

void Yolov8_detector::nms_sorted_bboxes(const std::vector<Object>& proposals,
                        std::vector<int>& picked,
                        float nms_threshold)
{
    picked.clear();

    const int n = proposals.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = proposals[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = proposals[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = proposals[picked[j]];
            cv::Rect_<float> inter = a.rect & b.rect;
            float inter_area = inter.area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void Yolov8_detector::post_process_gpu(cv::Mat& img)
{
    /* transpose */
    transposeDevice(batchsize,
        static_cast<float*>(device_buffers[output_index]), output_bbox_num, output_c, output_c * output_bbox_num,
        device_output_transpose, output_c, output_bbox_num);
    /* decode */
    decodeDevice(batchsize, num_classes, topK, conf_thresh,
        device_output_transpose, output_c, output_bbox_num, output_c * output_bbox_num,
        device_output_objects, output_objects_width, topK);

#if 0 /* valid */
    {
        float* phost = new float[batchsize * (1 + output_objects_width * topK)];
        float* pdevice = device_output_objects;
        CHECK(cudaMemcpy(phost, pdevice,
            sizeof(float) * (1 + output_objects_width * topK), cudaMemcpyDeviceToHost));
        int num_candidates = phost[0];
        cv::Mat prediction(topK, output_objects_width, CV_32FC1, phost + 1);
        delete[] phost;
    }
#endif

    /* nms */
    // nmsDeviceV1(batchsize, topK, nms_thresh,
    //     device_output_objects, output_objects_width, topK, topK * output_objects_width + 1);

    /* nms with sort */
    nmsDeviceV2(batchsize, topK, nms_thresh,
        device_output_objects, output_objects_width, topK, topK * output_objects_width + 1,
        device_output_idx, device_output_conf);

    /* copy result from gpu to cpu */
    CHECK(cudaMemcpyAsync(host_output, device_output_objects,
                batchsize * sizeof(float) * (1 + output_objects_width * topK),
                cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    float width = img.cols;
    float height = img.rows;
    float x_scale = input_w / (img.cols*1.0);
    float y_scale = input_h / (img.rows*1.0);
    float scale = std::min(input_w / (width*1.0), input_h / (height*1.0));
    std::vector<Object> objects;
    /* transform to source image coordinate */
    int num_boxes = std::min((int)(host_output + 0 * (topK * output_objects_width + 1))[0], topK);
    std::cout << "num_boxes: " << num_boxes << std::endl;
    for (size_t i = 0; i < num_boxes; i++)
    {
        float* ptr = host_output + 0 * (topK * output_objects_width + 1) + output_objects_width * i + 1;
        int keep_flag = ptr[6];
        if (keep_flag)
        {
            float x_lt = d2i[0] * ptr[0] + d2i[1] * ptr[1] + d2i[2]; // left & top
            float y_lt = d2i[3] * ptr[0] + d2i[4] * ptr[1] + d2i[5];
            float x_rb = d2i[0] * ptr[2] + d2i[1] * ptr[3] + d2i[2]; // right & bottom
            float y_rb = d2i[3] * ptr[2] + d2i[4] * ptr[3] + d2i[5];

            Object obj;
            obj.rect.x = x_lt;
            obj.rect.y = y_lt;
            obj.rect.width  = x_rb - x_lt;
            obj.rect.height = y_rb - y_lt;
            obj.score = ptr[4];
            obj.label = ptr[5];
            objects.push_back(obj);

            std::cout << "x1:" << x_lt
                    << ", y1:" << y_lt
                    << ", x2:" << x_rb
                    << ", y2:" << y_rb
                    << ", score:" << obj.score
                    << ", label:" << obj.label
                    << std::endl;
            // std::string label = _object_classes[obj.label];
            cv::rectangle(img, cv::Point(x_lt, y_lt), cv::Point(x_rb, y_rb),
                        cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }

    draw_objects(img, objects);
}