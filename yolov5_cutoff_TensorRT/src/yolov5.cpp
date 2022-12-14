#include <iostream>
#include "../include/yolov5.h"
#include "../include/common.h"

// #define OUTPUT_DEBUG
// #define INPUT_DEBUG

static const int DEVICE  = 0;

Yolov5_Detector::Yolov5_Detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
}

void Yolov5_Detector::init_context()
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

    input_buffer_size = batchsize * input_c * input_w * input_h * sizeof(float);
    std::cout<<"input_buffer_size:"<<input_buffer_size<<std::endl;
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    for(int i = 0; i < num_stages; ++i)
    {
        featmap_sizes[i][0] = int(input_h / strides[i]);
        featmap_sizes[i][1] = int(input_w / strides[i]);
        
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
                <<std::endl;

        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_cpu[i],\
                                            det_output_buffer_size[i],\
                                            cudaHostAllocDefault));

        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_index[i]],\
                                    det_output_buffer_size[i]));
    }
    PERCEPTION_CUDA_CHECK(cudaStreamCreate(&stream));
}

void Yolov5_Detector::destroy_context()
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

Yolov5_Detector::~Yolov5_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Yolov5_Detector::pre_process(cv::Mat& image)
{    
#ifdef INPUT_DEBUG    
    std::string Tensorrt_preprocess_txt_name = "Tensorrt_preprocess.txt";
    if (access(Tensorrt_preprocess_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_preprocess_txt_name.c_str()) == 0){
            std::cout<<Tensorrt_preprocess_txt_name<<" has been deleted successfuly!"<<std::endl;
        }
    }
    std::ofstream Tensorrt_preprocess;
    Tensorrt_preprocess.open(Tensorrt_preprocess_txt_name);
#endif

    cv::Mat img = image.clone();
    cv::Size new_shape = cv::Size(640, 640);
    float width = img.cols;
    float height = img.rows;
    float r = std::min(new_shape.width / width, new_shape.height / height);
    r = std::min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    std::cout<<"new_unpadW:"<<new_unpadW<<", new_unpadH:"<<new_unpadH<<std::endl;
    int dw = (new_shape.width - new_unpadW) % 32;
    int dh = (new_shape.height - new_unpadH) % 32;
    dw /= 2, dh /= 2;
    std::cout<<"dw:"<<dw<<", dh:"<<dh<<std::endl;
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, \
                    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    int channels = 3;
    int img_h = dst.rows;
    int img_w = dst.cols;
    std::cout<<channels<<", "<<img_h<<", "<<img_w<<", "<<dst.total()*3<<std::endl;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                float data = ((float)dst.at<cv::Vec3b>(h, w)[c]) / 255.0;
                host_input[c * img_w * img_h + h * img_w + w] = data;   
#ifdef DEBUG  
                Tensorrt_preprocess<<data<<"\n";
#endif 
            }
        }
    }

#ifdef DEBUG  
    Tensorrt_preprocess.close();
#endif
    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Yolov5_Detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout << "preprocess time: " << preprocess_time<< " ms." << std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

#ifdef INPUT_DEBUG
    float max_diff = 0.0;
    std::string onnx_preprocess_txt = "onnx_preprocess.txt";
    std::vector<float>onnx_preprocess;
    std::ifstream onnx_preprocess_data(onnx_preprocess_txt);
    float d;
    while (onnx_preprocess_data >> d)
    {
        onnx_preprocess.push_back(d);
    }
    for(int i = 0; i < input_buffer_size / sizeof(float); ++i)
    {
        float diff = std::abs(static_cast<float *>(host_input)[i] - onnx_preprocess[i]);
        if(diff > max_diff)
        {
            max_diff = diff;
        }
    }
    std::cout<<"preproces max diff:"<<max_diff<<std::endl;
#endif

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueueV2(&device_buffers[input_index], stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    post_process(img);
    std::cout<<"Post-process done!"<<std::endl;
}

void Yolov5_Detector::post_process(cv::Mat& img)
{
#ifdef OUTPUT_DEBUG    
    std::string Tensorrt_output_txt_name = "Tensorrt_output.txt";
    if (access(Tensorrt_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_output_txt_name.c_str()) == 0){
            std::cout<<Tensorrt_output_txt_name<<" has been deleted successfuly!"<<std::endl;
        }
    }
    std::ofstream Tensorrt_output;
    Tensorrt_output.open(Tensorrt_output_txt_name);
#endif

    float x_scale = (float)(input_h / (img.cols*1.0));
    float y_scale = (float)(input_w / (img.rows*1.0));
    std::vector<Object> objects;

    std::cout<<"img.rows:"
            <<img.rows
            <<", img.cols:"
            <<img.cols
            <<std::endl;
    /* Decode detections */
    auto start_generate = std::chrono::system_clock::now();
    float ratio_h = (float)img.rows / input_h;
	float ratio_w = (float)img.cols / input_w;
    std::cout<<"ratio_h:"<<ratio_h
            <<", ratio_w:"<<ratio_w<<std::endl;
    std::vector<Object> proposals;

    for(int stride = 0; stride < num_stages; ++stride)
    {
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync(det_output_cpu[stride], device_buffers[det_output_index[stride]],
                    det_output_buffer_size[stride], cudaMemcpyDeviceToHost, stream));

#ifdef OUTPUT_DEBUG
        for (int i = 0; i < det_output_buffer_size[stride] / sizeof(float); i++)
        {
            float data = det_output_cpu[stride][i];
            Tensorrt_output<<data<<"\n";
        }
#endif

        int num_grid_x = (int)(input_w / strides[stride]);
        int num_grid_y = (int)(input_h / strides[stride]);
        std::cout<<"num_grid_x:"<<num_grid_x<<", num_grid_y:"<<num_grid_y<<std::endl;

        for(int anchor = 0; anchor < 3; ++anchor)
        {
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            std::cout<<"anchor_w:"<<anchor_w<<", anchor_h:"<<anchor_h<<std::endl;

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

                int grid_y = (i / num_grid_x) % num_grid_x;
                int grid_x = i - grid_y * num_grid_x;

                int x_index = i + 0 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float x_data = sigmoid_x(det_output_cpu[stride][x_index]);
                x_data = (x_data * 2.0f + grid_x - 0.5f) * strides[stride];

                int y_index = i + 1 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float y_data = sigmoid_x(det_output_cpu[stride][y_index]);
                y_data = (y_data * 2.0f + grid_y - 0.5f) * strides[stride];

                int w_index = i + 2 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float w_data = sigmoid_x(det_output_cpu[stride][w_index]);
                w_data = (w_data * 2.0f) * (w_data * 2.0f) * anchor_w;

                int h_index = i + 3 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                float h_data = sigmoid_x(det_output_cpu[stride][h_index]);
                h_data = (h_data * 2.0f) * (h_data * 2.0f) * anchor_h;

                float x     = x_data;
                float y     = y_data;
                float width  = w_data;
                float height = h_data;
                float x0   = x - width * 0.5;
                float y0   = y - height * 0.5;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = width;
                obj.rect.height = height;
                obj.label = label;
                obj.prob = confidence;
                // std::cout<<"label:"<<label<<", score:"<<confidence
                //         <<", left:"<<x0<<", top:"<<y0
                //         <<", width:"<<width<<", height:"<<height
                //         <<std::endl;

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

        // float x0 = (objects[i].rect.x) / x_scale;
        // float y0 = (objects[i].rect.y) / y_scale;
        // float x1 = (objects[i].rect.x + objects[i].rect.width) / x_scale;
        // float y1 = (objects[i].rect.y + objects[i].rect.height) / y_scale;
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

#ifdef OUTPUT_DEBUG  
    Tensorrt_output.close();
#endif
}