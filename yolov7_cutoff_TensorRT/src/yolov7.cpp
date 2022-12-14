#include "../include/yolov7.h"
#include <iostream>

static const int DEVICE  = 0;

Yolov7_Detector::Yolov7_Detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
}

void Yolov7_Detector::init_context()
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

    input_buffer_size = batchsize * input_c * input_w * input_h * sizeof(float);
    std::cout<<"input_buffer_size:"<<input_buffer_size<<std::endl;
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    for(int i = 0; i < num_stages; ++i)
    {
        featmap_sizes[i][0] = int(input_h / float(strides[i]) + 0.5);
        featmap_sizes[i][1] = int(input_w / float(strides[i]) + 0.5);
        
        det_output_buffer_size[i] = batchsize * det_len *
                featmap_sizes[i][0] * featmap_sizes[i][1] *
                sizeof(float);

        det_output_index[i] = engine->getBindingIndex(output_blob_name[i].c_str());

        std::cout<<"featmap_sizes["<<i<<"][0]:"
                <<featmap_sizes[i][0]
                <<", featmap_sizes["<<i<<"][1]:"
                <<featmap_sizes[i][1]
                <<", det_output_buffer_size:"
                <<det_output_buffer_size[i]
                <<", det_output_index["<<i<<"]:"
                <<det_output_index[i]
                <<std::endl;

        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_cpu[i],\
                                            det_output_buffer_size[i],\
                                            cudaHostAllocDefault));

        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_index[i]],\
                                    det_output_buffer_size[i]));
    }
    PERCEPTION_CUDA_CHECK(cudaStreamCreate(&stream));
}

void Yolov7_Detector::destroy_context()
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
    if(stream) cudaStreamDestroy(stream);
    if(host_input)
        PERCEPTION_CUDA_CHECK(cudaFreeHost(host_input));
    for (int i = 0; i < num_stages; i++)
    {
        if(det_output_cpu[i])
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_cpu[i]));
    }
    for(int i = 0; i < num_stages + 1; ++i)
    {
        if(device_buffers[i])
            PERCEPTION_CUDA_CHECK(cudaFree(device_buffers[i]));
    }
}

Yolov7_Detector::~Yolov7_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Yolov7_Detector::pre_process(cv::Mat image)
{    
    // letter box
    // 通过双线性插值对图像进行resize
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_w + scale - 1) * 0;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_h + scale - 1) * 0;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    cv::Mat input_image(input_h, input_w, CV_8UC3);
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

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Yolov7_Detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout << "preprocess time: " << preprocess_time<< " ms." << std::endl;
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

void Yolov7_Detector::post_process(cv::Mat& img)
{
    objects.clear();
    /* Decode detections */
    auto start_generate = std::chrono::system_clock::now();
    float ratio_h = (float)img.rows / input_h;
	float ratio_w = (float)img.cols / input_w;
    std::vector<Object> proposals;

    for(int stride = 0; stride < num_stages; ++stride)
    {
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync(det_output_cpu[stride], device_buffers[det_output_index[stride]],
                    det_output_buffer_size[stride], cudaMemcpyDeviceToHost, stream));

        int num_grid_x = (int)(input_w / strides[stride]);
        int num_grid_y = (int)(input_h / strides[stride]);
        std::cout<<"num_grid_x:"<<num_grid_x<<", num_grid_y:"<<num_grid_y<<std::endl;

        for(int anchor = 0; anchor < 3; ++anchor)
        {
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];

            for(int i = 0; i < num_grid_x * num_grid_y; ++i)
            {
                int obj_index = i + 4 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float objness = sigmoid_x(det_output_cpu[stride][obj_index]);

                if(objness < BBOX_CONF_THRESH)
                    continue;

                int label = 0;
                float prob = 0.0;
                for (int index = 5; index < 85; index++)
                {
                    int class_index = i + index * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                    if (sigmoid_x(det_output_cpu[stride][class_index]) > prob)
                    {
                        label = index - 5;
                        prob = sigmoid_x(det_output_cpu[stride][class_index]);
                    }
                }

                float confidence = prob * objness;
                if(confidence < BBOX_CONF_THRESH)
                    continue;

                int grid_y = i / num_grid_x;
                int grid_x = i - grid_y * num_grid_x;

                int x_index = i + 0 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float x_data = sigmoid_x(det_output_cpu[stride][x_index]);
                x_data = x_data * 2.0f * strides[stride] + strides[stride] * (grid_x- 0.5);

                int y_index = i + 1 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float y_data = sigmoid_x(det_output_cpu[stride][y_index]);
                y_data = y_data * 2.0f * strides[stride] + strides[stride] * (grid_y - 0.5);

                int w_index = i + 2 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float w_data = sigmoid_x(det_output_cpu[stride][w_index]);
                w_data = w_data * w_data * (4 * anchor_w);

                int h_index = i + 3 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float h_data = sigmoid_x(det_output_cpu[stride][h_index]);
                h_data = h_data * h_data * (4 * anchor_h);

                float x     = x_data;
                float y     = y_data;
                float width  = w_data;
                float height = h_data;
                float x0   = x - width * 0.5;
                float y0    = y - height * 0.5;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = width;
                obj.rect.height = height;
                obj.label = label;
                obj.prob = confidence;

                proposals.push_back(obj);
            }
        }
    }
    cudaStreamSynchronize(stream);

    auto end_generate = std::chrono::system_clock::now();
    std::cout<<"generate proposal time:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_generate - start_generate).count()<<" ms"<<std::endl;
    std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    auto start_qsort = std::chrono::system_clock::now();
    qsort_descent_inplace(proposals);
    auto end_qsort = std::chrono::system_clock::now();
    std::cout<<"qsort time:"<<std::chrono::duration_cast<std::chrono::microseconds>(end_qsort - start_qsort).count()<<" us"<<std::endl;

    auto start_nms = std::chrono::system_clock::now();
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    auto end_nms = std::chrono::system_clock::now();
    std::cout<<"nms time:"<<std::chrono::duration_cast<std::chrono::microseconds>(end_nms - start_nms).count()<<" us"<<std::endl;

    int count = picked.size();

    std::cout << "num of boxes: " << count << std::endl;

    auto start_analysis = std::chrono::system_clock::now();
    objects.resize(count);
    float scale = std::min(input_w / (img.cols*1.0), input_h / (img.rows*1.0));
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

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