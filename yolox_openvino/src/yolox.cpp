#include "../include/yolox.h"
#include "../include/common.h"

#define INPUTDEBUG

Yolox_Detector::Yolox_Detector(const std::string& _engine_file, const std::string& _runtime):
    engine_file(_engine_file),
    runtime(_runtime)
{
    init_context();
}

void Yolox_Detector::init_context()
{
    std::vector<std::string> availableDevices = core.GetAvailableDevices();
    for (int i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
    }

    //从IR加载检测模型
    cnnNetwork = core.ReadNetwork(engine_file);
    cnnNetwork.setBatchSize(1);
    // 网络输入头参数设置
    inputInfo = InferenceEngine::InputsDataMap(cnnNetwork.getInputsInfo());
    InferenceEngine::InputInfo::Ptr input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;//输入头的名称
    std::cout<<"input name: "<<_input_name<<std::endl;
    input->setPrecision(InferenceEngine::Precision::FP32);
    input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);

    _outputinfo = InferenceEngine::OutputsDataMap(cnnNetwork.getOutputsInfo());
    std::cout<<"output name: ";
    for(auto& output : _outputinfo){
        std::cout<<output.first<<", ";    
        output.second->setPrecision(InferenceEngine::Precision::FP32);
    }
    std::cout<<std::endl;

    executable_network = core.LoadNetwork(engine_file, runtime);
    infer_request = executable_network.CreateInferRequest();
}

void Yolox_Detector::destroy_context()
{

}

Yolox_Detector::~Yolox_Detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h
            <<"x"<<input_w<<"]"
            <<std::endl;
}

void Yolox_Detector::pre_process_cpu(cv::Mat& img)
{
    imgBlob = infer_request.GetBlob(_input_name);
    batchsize = imgBlob->getTensorDesc().getDims()[0];
    input_c = imgBlob->getTensorDesc().getDims()[1];
    input_h = imgBlob->getTensorDesc().getDims()[2];
    input_w = imgBlob->getTensorDesc().getDims()[3];
    std::cout<<"input shape: ["<<batchsize
            <<" x "<<input_c
            <<" x "<<input_h
            <<" x "<<input_w
            <<" ]"
            <<std::endl;

    // letter box
    // 通过双线性插值对图像进行resize
    cv::Mat image = img.clone();
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

    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob)->wmap();
    float* host_input = blobMapped.as<float*>();

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0];
        *phost_g++ = pimage[1];
        *phost_b++ = pimage[2];
    }

#ifdef INPUTDEBUG
    std::string TensorRT_preprocess_txt = "/home/wyh/disk/DL_Model_Deploy/yolox_openvino/Tensorrt_preprocess.txt";
    std::vector<float>TensorRT_preprocess;
    std::ifstream TensorRT_preprocess_data(TensorRT_preprocess_txt);
    float data;
    while (TensorRT_preprocess_data >> data)
    {
        TensorRT_preprocess.push_back(data);
    }

    float max_diff = 0.0;
    for (int i = 0; i < input_c * input_h * input_w; i++)
    {
        float openvino_data = static_cast<float*>(host_input)[i];
        float trt_data = TensorRT_preprocess[i];
        float diff = std::abs(trt_data - openvino_data);
        if (diff > max_diff)
        {
            std::cout<<i
                    <<", trt_data: "<<trt_data
                    <<", pytorch_data: "<<openvino_data
                    <<std::endl;
            max_diff = diff;
        }
    }
    std::cout<<"preprocess between Tensorrt and openvino, max diff is: "<<max_diff<<std::endl;
#endif
}

void Yolox_Detector::do_detection(cv::Mat& img)
{
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process_cpu(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout << "preprocess take: " << preprocess_time/1000 << " s." << std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();

    infer_request.Infer();

    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    post_process_cpu(img);
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

const float* Yolox_Detector::get_output_buffer(const std::string output_name)
{
    const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    const int batch_size = static_cast<int>(output_blob->getTensorDesc().getDims()[0]);    // batch_size
    const int anchor_num = static_cast<int>(output_blob->getTensorDesc().getDims()[1]);    // anchor_num
    const int net_grid_h = static_cast<int>(output_blob->getTensorDesc().getDims()[2]);    // 80/40/20
    const int net_grid_w = static_cast<int>(output_blob->getTensorDesc().getDims()[3]);    // 80/40/20
    std::cout<< output_name
            << " shape: [" <<batch_size
            << " x " <<anchor_num
            << " x " <<net_grid_h
            << " x " <<net_grid_w
            << " ]"
            << std::endl;
    // InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob)->rmap();
    // const float *net_pred = blobMapped.as<float *>();
    
    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    if (!moutput) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                "but by fact we were not able to cast output to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto moutputHolder = moutput->rmap();
    const float* net_pred = moutputHolder.as<const PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    return net_pred;
}

void Yolox_Detector::post_process_cpu(cv::Mat& image)
{
    /* Store detection results */
    std::vector<Object> objects;

    float scale = std::min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));
    std::vector<Object> proposals;
    /* Decode detections */
    for (int stride = 0; stride < num_stages; stride++)
    {
        const float* cls_buffer  = get_output_buffer(cls_output_name[stride]);
        const float* obj_buffer  = get_output_buffer(obj_output_name[stride]);
        const float* bbox_buffer = get_output_buffer(bbox_output_name[stride]);

        int num_grid_x = (int) (input_h) / strides[stride];
        int num_grid_y = (int) (input_w) / strides[stride];
        std::cout<< "num_grid_x: " << num_grid_x
                << ", num_grid_y: " << num_grid_y
                << std::endl;

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

        // std::string label = _object_classes[obj.label];
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 0, 255), 2, 8, 0);
        // cv::putText(image, label, cv::Point2d(x1 + 5, y1 + 5),
        //                 cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    draw_objects(image, objects);
}