#include "../include/yolox.h"
#include <chrono>
#include <iostream>
#include <fstream> 

static const int DEVICE  = 0;

Yolox_Detector::Yolox_Detector(const std::string& _engine_file):
                engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
}

void Yolox_Detector::init_context()
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
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));
    for(int i = 0;i < num_stages; ++i)
    {
        featmap_sizes[i][0] = int(input_h / float(strides[i]) + 0.5);
        featmap_sizes[i][1] = int(input_w / float(strides[i]) + 0.5);

        det_output_cls_buffer_size[i] = batchsize *
                featmap_sizes[i][0] * featmap_sizes[i][1] *
                det_cls_len * sizeof(float);
        det_output_obj_buffer_size[i] = batchsize *
                featmap_sizes[i][0] * featmap_sizes[i][1] *
                det_obj_len * sizeof(float);
        det_output_bbox_buffer_size[i] = batchsize *
                featmap_sizes[i][0] * featmap_sizes[i][1] *
                det_bbox_len * sizeof(float);

        det_output_cls_index[i] = engine->getBindingIndex(cls_output_name[i].c_str());
        det_output_obj_index[i] = engine->getBindingIndex(obj_output_name[i].c_str());
        det_output_bbox_index[i] = engine->getBindingIndex(bbox_output_name[i].c_str());

        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_cls_cpu[i], \
                                            det_output_cls_buffer_size[i],\
                                            cudaHostAllocDefault));
        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_bbox_cpu[i], \
                                            det_output_bbox_buffer_size[i],\
                                            cudaHostAllocDefault));
        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_obj_cpu[i], \
                                            det_output_obj_buffer_size[i],\
                                            cudaHostAllocDefault));

        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_cls_index[i]],\
                                    det_output_cls_buffer_size[i]));
        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_obj_index[i]],\
                                    det_output_obj_buffer_size[i]));
        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_bbox_index[i]],\
                                    det_output_bbox_buffer_size[i]));
    }
    PERCEPTION_CUDA_CHECK(cudaStreamCreate(&stream));
}

void Yolox_Detector::destroy_context()
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
        if(det_output_cls_cpu[i])
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_cls_cpu[i]));
        if(det_output_obj_cpu[i])
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_obj_cpu[i]));
        if(det_output_bbox_cpu[i]) 
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_bbox_cpu[i]));
    }
    for(int i = 0; i < num_stages * 3 + 1; ++i)
    {
        if(device_buffers[i])
            PERCEPTION_CUDA_CHECK(cudaFree(device_buffers[i]));
    }
}

Yolox_Detector::~Yolox_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Yolox_Detector::pre_process(cv::Mat &img)
{
    cv::Mat img_cpu = img.clone();
    float r = std::min(input_w / (img_cpu.cols*1.0), input_h / (img_cpu.rows*1.0));
    int unpad_w = r * img_cpu.cols;
    int unpad_h = r * img_cpu.rows + 0.5;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img_cpu, re, re.size());
    if (re.channels() == 1)
    {
        cv::cvtColor(re, re, cv::COLOR_GRAY2BGR);
    }
    cv::Mat out(input_w, input_h, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    cv::cvtColor(out, img_cpu, cv::COLOR_BGR2RGB);
    int channels = 3;
    int img_h = img_cpu.rows;
    int img_w = img_cpu.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                host_input[c * img_w * img_h + h * img_w + w] = (float)img_cpu.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Yolox_Detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    pre_process(img);
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

void Yolox_Detector::nms_sorted_bboxes(const std::vector<Object>& proposals,
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

void Yolox_Detector::post_process(cv::Mat& image)
{
    for (int i = 0; i < num_stages; i++)
    {
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_cls_cpu[i]), device_buffers[det_output_cls_index[i]],
                                        det_output_cls_buffer_size[i], cudaMemcpyDeviceToHost, stream));
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_obj_cpu[i]), device_buffers[det_output_obj_index[i]],
                                        det_output_obj_buffer_size[i], cudaMemcpyDeviceToHost, stream));
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_bbox_cpu[i]), device_buffers[det_output_bbox_index[i]],
                                        det_output_bbox_buffer_size[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    float scale = std::min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));

    /* Decode detections */
    objects.clear();
    std::vector<Object> proposals;
    std::vector<float*> cls_vec;
    std::vector<float*> obj_vec;
    std::vector<float*> bbox_vec;
    for (int i = 0; i < num_stages; i++)
    {
        cls_vec.push_back(det_output_cls_cpu[i]);
        obj_vec.push_back(det_output_obj_cpu[i]);
        bbox_vec.push_back(det_output_bbox_cpu[i]);
    }

    for(int n = 0; n < num_stages; n++) // strides
    {
        int num_grid_x = (int) (input_h) / strides[n];
        int num_grid_y = (int) (input_w) / strides[n];
        float* cls_buffer = cls_vec[n];
        float* obj_buffer = obj_vec[n];
        float* bbox_buffer = bbox_vec[n];

        for(int i = 0; i < num_grid_y; i++) // grids
        {
            for(int j = 0; j < num_grid_x; j++)
            {
                float obj = obj_buffer[i * num_grid_x + j];
                obj = 1 / (1 + expf(-obj));
                
                if(obj < 0.1) continue; // FIXME : to parameterize
                
                float x_feat = bbox_buffer[i * num_grid_x + j];
                float y_feat = bbox_buffer[num_grid_y * num_grid_x + (i * num_grid_x + j)];
                float w_feat = bbox_buffer[num_grid_y * num_grid_x * 2 + (i * num_grid_x + j)];
                float h_feat = bbox_buffer[num_grid_y * num_grid_x * 3 + (i * num_grid_x + j)];
                
                float x_center = (x_feat + j) * strides[n];
                float y_center = (y_feat + i) * strides[n];
                float w = expf(w_feat) * strides[n];
                float h = expf(h_feat) * strides[n];

                for(int class_idx = 0; class_idx < classes_num; class_idx++)
                {
                    float cls = cls_buffer[class_idx * num_grid_x * num_grid_y + (i * num_grid_x + j)];
                    cls = 1 / (1 + expf(-cls));
                    float score = cls * obj;
                    
                    if(score > confThreshold)
                    {
                        int left = (x_center - 0.5 * w) / scale;
                        int top = (y_center - 0.5 * h ) / scale;
                        int ww = (int)(w / scale);
                        int hh = (int)(h / scale);

                        int right = left + ww;
                        int bottom = top + hh;
                        
                        /* clip */
                        left = std::min(std::max(0, left), image.cols);
                        top = std::min(std::max(0, top), image.rows);
                        right = std::min(std::max(0, right), image.cols);
                        bottom = std::min(std::max(0, bottom), image.rows);
                            
                        Object obj;
                        obj.rect = cv::Rect_<float>(left, top, right - left,  bottom - top);
                        obj.label = class_idx;
                        obj.score = score;
                        proposals.push_back(obj);
                    }
                }
            }
        }
    }
    std::cout<<"Num of proposals: "<< proposals.size() <<std::endl;
    
    /* Perform non maximum suppression */
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nmsThreshold);
    
    int count = picked.size();
    
    /* Draw & show res */
    objects.resize(count);
    for(int i = 0; i < count; ++i)
    {
        objects[i] = proposals[picked[i]];
        objects[i].rect.x = objects[i].rect.x;
        objects[i].rect.y = objects[i].rect.y;
        objects[i].rect.width  = objects[i].rect.width;
        objects[i].rect.height = objects[i].rect.height;
    }
}
