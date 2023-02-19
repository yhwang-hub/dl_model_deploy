#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/yolox.h"
#include "../include/common.h"

static const int DEVICE  = 0;
#define INPUTDEBUG

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
    if (device_output_buffer)
    {
         PERCEPTION_CUDA_CHECK(cudaFree(device_output_buffer));
    }
    if (host_output_buffer)
    {
         PERCEPTION_CUDA_CHECK(cudaFreeHost(host_output_buffer));
    }
    if (psrc_device)
    {
        PERCEPTION_CUDA_CHECK(cudaFree(psrc_device));
    }
}

Yolox_Detector::~Yolox_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<" x "<<input_w<<"]"<<std::endl;
}

void Yolox_Detector::pre_process_cpu(cv::Mat &img)
{
    /*******************************************************************************/
    /*及其不推荐YOLOX官方的TensorRT部署代码，又繁琐又臃肿，效率还低！*/
    // cv::Mat img_cpu = img.clone();
    // float r = min(input_w / (img_cpu.cols*1.0), input_h / (img_cpu.rows*1.0));
    // int unpad_w = r * img_cpu.cols;
    // int unpad_h = r * img_cpu.rows + 0.5;
    // cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    // cv::resize(img_cpu, re, re.size());
    // if (re.channels() == 1)
    // {
    //     cv::cvtColor(re, re, cv::COLOR_GRAY2BGR);
    // }
    // cv::Mat out(input_w, input_h, CV_8UC3, cv::Scalar(114, 114, 114));
    // re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    // cv::cvtColor(out, img_cpu, cv::COLOR_BGR2RGB);
    // int channels = 3;
    // int img_h = img_cpu.rows;
    // int img_w = img_cpu.cols;
    // for (size_t c = 0; c < channels; c++) 
    // {
    //     for (size_t  h = 0; h < img_h; h++) 
    //     {
    //         for (size_t w = 0; w < img_w; w++) 
    //         {
    //             host_input[c * img_w * img_h + h * img_w + w] = (float)img_cpu.at<cv::Vec3b>(h, w)[c];
    //         }
    //     }
    // }
    /*******************************************************************************/

    // letter box
    // 通过双线性插值对图像进行resize
    cv::Mat image = img.clone();
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = min(scale_x, scale_y);
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
    // cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0];
        *phost_g++ = pimage[1];
        *phost_b++ = pimage[2];
    }

#ifdef INPUTDEBUG
    std::string Tensorrt_preprocess_txt_name = "Tensorrt_preprocess.txt";
    if (access(Tensorrt_preprocess_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_preprocess_txt_name.c_str()) == 0){
            std::cout<<Tensorrt_preprocess_txt_name<<" has been deleted successfuly!"<<std::endl;
        }
    }
    std::ofstream Tensorrt_prerpocess;
    Tensorrt_prerpocess.open(Tensorrt_preprocess_txt_name);

    for (size_t i = 0; i < input_c * input_h * input_w; i++)
    {
        float trt_data = static_cast<float*>(host_input)[i];
        Tensorrt_prerpocess<<trt_data<<std::endl;
    }
    Tensorrt_prerpocess.close();
#endif

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Yolox_Detector::pre_process_gpu(cv::Mat &img)
{
    cv::Mat image = img.clone();
    cv::Size input_size = cv::Size(input_h, input_w);
    cv::Mat output(input_size, CV_8UC3);
    size_t src_size = image.cols * image.rows * 3;
    PERCEPTION_CUDA_CHECK(cudaMalloc(&psrc_device, src_size));
    // image.data搬运数据到GPU上
    PERCEPTION_CUDA_CHECK(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice));

    // 在CPU上的仿射变换,除了变换图片的尺寸（通过仿射变换），还有减均值除方差、bgrbgrbgr->bbbgggrrr
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        static_cast<float* >(device_buffers[input_index]),
        input_w * 3, input_w, input_h, 114
    );
    // 由于cuda核函数的执行无论stream是否为nullptr，都将会是异步执行（即调用核函数，立马返回），这就需要加个同步等待
    cudaDeviceSynchronize();
    // 检查核函数执行是否存在错误
    PERCEPTION_CUDA_CHECK(cudaPeekAtLastError());
}

void Yolox_Detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process_cpu(img);
    // pre_process_gpu(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout << "preprocess take: " << preprocess_time/1000 << " s." << std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    // post_process_cpu(img);
    // post_process_gpu1(img);
    post_process_gpu2(img);
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

void Yolox_Detector::post_process_cpu(cv::Mat& image)
{
    /* Store detection results */
    std::vector<Box> det_objs;

    float scale = min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));
    std::vector<Object> proposals;
    /* Decode detections */
    for (int stride = 0; stride < num_stages; stride++)
    {
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_cls_cpu[stride]), device_buffers[det_output_cls_index[stride]],
                                        det_output_cls_buffer_size[stride], cudaMemcpyDeviceToHost, stream));
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_obj_cpu[stride]), device_buffers[det_output_obj_index[stride]],
                                        det_output_obj_buffer_size[stride], cudaMemcpyDeviceToHost, stream));
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_bbox_cpu[stride]), device_buffers[det_output_bbox_index[stride]],
                                        det_output_bbox_buffer_size[stride], cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        int num_grid_x = (int) (input_h) / strides[stride];
        int num_grid_y = (int) (input_w) / strides[stride];
        float* cls_buffer = det_output_cls_cpu[stride];
        float* obj_buffer = det_output_obj_cpu[stride];
        float* bbox_buffer = det_output_bbox_cpu[stride];

        for (int index = 0; index < num_grid_x * num_grid_y; index++)
        {
            int i = index / num_grid_x;
            int j = index - i * num_grid_x;

            float obj = obj_buffer[i * num_grid_x + j];
            obj = 1 / (1 + expf(-obj));
            if(obj < 0.1)
                continue; // FIXME : to parameterize

            for(int class_idx = 0; class_idx < classes_num; class_idx++)
            {
                float cls = cls_buffer[class_idx * num_grid_x * num_grid_y + (i * num_grid_x + j)];
                cls = 1 / (1 + expf(-cls));
                float score = cls * obj;
                
                if(score < confThreshold)
                    continue;

                float x_feat = bbox_buffer[i * num_grid_x + j];
                float y_feat = bbox_buffer[num_grid_y * num_grid_x + (i * num_grid_x + j)];
                float w_feat = bbox_buffer[num_grid_y * num_grid_x * 2 + (i * num_grid_x + j)];
                float h_feat = bbox_buffer[num_grid_y * num_grid_x * 3 + (i * num_grid_x + j)];
                
                float x_center = (x_feat + j) * strides[stride];
                float y_center = (y_feat + i) * strides[stride];
                float w = expf(w_feat) * strides[stride];
                float h = expf(h_feat) * strides[stride];

                int left = (x_center - 0.5 * w) / scale;
                int top = (y_center - 0.5 * h ) / scale;
                int ww = (int)(w / scale);
                int hh = (int)(h / scale);

                int right = left + ww;
                int bottom = top + hh;

                /* clip */
                left = min(std::max(0, left), image.cols);
                top = min(std::max(0, top), image.rows);
                right = min(std::max(0, right), image.cols);
                bottom = min(std::max(0, bottom), image.rows);
                    
                Object obj;
                obj.rect = cv::Rect_<float>(left, top, right - left,  bottom - top);
                obj.label = class_idx;
                obj.score = score;
                proposals.push_back(obj);
            }
        }
    }

    std::cout<<"Num of proposals: "<< proposals.size() <<std::endl;
    
    /* Perform non maximum suppression */
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nmsThreshold);
    
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
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2, 8, 0);
        // cv::putText(image, label, cv::Point2d(x1 + 5, y1 + 5),
        //                 cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    // draw_objects(image, objects);
}

void Yolox_Detector::post_process_gpu1(cv::Mat& image)
{
    /* Store detection results */
    std::vector<Box> det_objs;

    float scale = min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));

    std::vector<Object> proposals;
    /* Decode detections */
    int det_output_bufsize = 1 + max_out_obj * sizeof(Detection) / sizeof(float);
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_output_buffer, det_output_bufsize));
    PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&host_output_buffer, \
                                    det_output_bufsize, \
                                    cudaHostAllocDefault));
    PERCEPTION_CUDA_CHECK(cudaMemset(device_output_buffer, 0, sizeof(float)));
    for (int stride = 0; stride < num_stages; stride++)
    {
        get_det_output(static_cast<float* >(device_buffers[det_output_cls_index[stride]]),
                    static_cast<float* >(device_buffers[det_output_obj_index[stride]]),
                    static_cast<float* >(device_buffers[det_output_bbox_index[stride]]),
                    batchsize, det_obj_len, det_bbox_len, det_cls_len,
                    image.rows, image.cols, max_out_obj,
                    input_h, input_w, strides[stride], scale,
                    confThreshold, nmsThreshold, NUM_BOX_ELEMENT,
                    static_cast<float *>(device_output_buffer), stream);
    }
    PERCEPTION_CUDA_CHECK(cudaMemcpyAsync(host_output_buffer, device_output_buffer, det_output_bufsize,
                        cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    float* host_det_output_c = static_cast<float* >(host_output_buffer);
    int num_det = static_cast<int>(host_det_output_c[0]);
    std::cout<<"num_det: "<<num_det<<std::endl;
    det_objs.clear();
    det_objs.resize(num_det);
    memcpy(det_objs.data(), &host_det_output_c[1], num_det * sizeof(Box));

    for (int i = 0; i < num_det; i++)
    {
        Box& det = det_objs[i];
        float x = det.left;
        float y = det.top;
        float w = det.right - x;
        float h = det.bottom - y;
        float score = det.confidence;
        int label = det.label;
        // printf("cls_id: %d, score: %f\n", label, score);
        std::cout<<"x:"<<x
                <<", y:"<<y
                <<", w:"<<w
                <<", h:"<<h
                <<", score:"<<score
                <<", label:"<<label
                <<std::endl;
        if (score < confThreshold) continue;
        
        Object obj;
        obj.rect.x = x;
        obj.rect.y = y;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = label;
        obj.score = score;
        proposals.push_back(obj);
    }

    std::cout<<"Num of proposals: "<< proposals.size() <<std::endl;
    
    /* Perform non maximum suppression */
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nmsThreshold);
    
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
    }
    // draw_objects(image, objects);
}

void Yolox_Detector::post_process_gpu2(cv::Mat& image)
{
    float scale = min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));
    std::vector<Box>box_result;
    float* output_device = nullptr;
    float* output_host = nullptr;

    PERCEPTION_CUDA_CHECK(cudaMalloc(&output_device, sizeof(float) + max_out_obj * NUM_BOX_ELEMENT * sizeof(float)));
    PERCEPTION_CUDA_CHECK(cudaMallocHost(&output_host, sizeof(float) + max_out_obj * NUM_BOX_ELEMENT * sizeof(float)));

    for(int stride = 0; stride < num_stages; ++stride)
    {
        gpu_decode_nms(static_cast<float* >(device_buffers[det_output_cls_index[stride]]),
                    static_cast<float* >(device_buffers[det_output_obj_index[stride]]),
                    static_cast<float* >(device_buffers[det_output_bbox_index[stride]]),
                    batchsize, det_obj_len, det_bbox_len, det_cls_len,
                    image.rows, image.cols, max_out_obj,
                    input_h, input_w, strides[stride], scale,
                    confThreshold, nmsThreshold, NUM_BOX_ELEMENT,
                    static_cast<float *>(output_device), stream);
    }   
    PERCEPTION_CUDA_CHECK(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_out_obj * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    PERCEPTION_CUDA_CHECK(cudaStreamSynchronize(stream));
    int num_boxes = min((int)output_host[0], max_out_obj);
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]
            );
        }
    }
    std::cout<<"after nms Num of boxes: "<< box_result.size() <<std::endl;
    PERCEPTION_CUDA_CHECK(cudaStreamDestroy(stream));

    PERCEPTION_CUDA_CHECK(cudaFree(output_device));
    PERCEPTION_CUDA_CHECK(cudaFreeHost(output_host));

    for(auto& box : box_result){
        // cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        // cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
        Object obj;
        obj.rect.x = box.left;
        obj.rect.y = box.top;
        obj.rect.width = box.right - box.left;
        obj.rect.height = box.bottom - box.top;
        obj.label = box.label;
        obj.score = box.confidence;
        objects.push_back(obj);
    }
    // cv::imwrite("det_res.jpg", image);
    // draw_objects(image, objects);
}