#include "../include/yolox.h"
#include "../include/common.h"

#define INPUTDEBUG

Yolox_Detector::Yolox_Detector(const std::string& _model_path, const std::string& _runtime):
    model_path(_model_path),
    runtime(_runtime)
{
    init_context();
}

void Yolox_Detector::init_context()
{
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Read a model --------
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // -------- Step 3. Loading a model to the device --------
    ov::CompiledModel compiled_model = core.compile_model(model, runtime);

    // Get input port for model with one input
    auto input_port = compiled_model.input();

    // -------- Step 4. Create an infer request --------
    infer_request = compiled_model.create_infer_request();
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
    // -------- Step 5. Prepare input --------
    ov::Tensor input_tensor = infer_request.get_tensor(input_name);
    batchsize = input_tensor.get_shape()[0];
    input_c = input_tensor.get_shape()[1];
    input_h = input_tensor.get_shape()[2];
    input_w = input_tensor.get_shape()[3];
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

    float* host_input  = input_tensor.data<float>();

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

    // // -------- Step 5. Prepare input --------
    // ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), host_input);
    // infer_request.set_input_tensor(input_tensor);
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

    // -------- Step 6. Start inference --------
    infer_request.infer();

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

void Yolox_Detector::post_process_cpu(cv::Mat& image)
{
    /* Store detection results */
    std::vector<Object> objects;

    float scale = std::min(input_w / (image.cols*1.0), input_h / (image.rows*1.0));
    std::vector<Object> proposals;
    /* Decode detections */
    for (int stride = 0; stride < num_stages; stride++)
    {
        // -------- Step 7. Process output --------
        ov::Tensor cls_outut_tensor = infer_request.get_tensor(cls_output_name[stride]);
        const float* cls_buffer  = cls_outut_tensor.data<const float>();
        int cls_batchsize = cls_outut_tensor.get_shape()[0];
        int cls_c = cls_outut_tensor.get_shape()[1];
        int cls_h = cls_outut_tensor.get_shape()[2];
        int cls_w = cls_outut_tensor.get_shape()[3];
        std::cout<<"cls output  shape: ["<<cls_batchsize
            <<" x "<<cls_c
            <<" x "<<cls_h
            <<" x "<<cls_w
            <<" ]"
            <<std::endl;

        ov::Tensor obj_output_tensor = infer_request.get_tensor(obj_output_name[stride]);
        const float* obj_buffer  = obj_output_tensor.data<const float>();
        int obj_batchsize = obj_output_tensor.get_shape()[0];
        int obj_c = obj_output_tensor.get_shape()[1];
        int obj_h = obj_output_tensor.get_shape()[2];
        int obj_w = obj_output_tensor.get_shape()[3];
        std::cout<<"obj output  shape: ["<<obj_batchsize
            <<" x "<<obj_c
            <<" x "<<obj_h
            <<" x "<<obj_w
            <<" ]"
            <<std::endl;

        ov::Tensor bbox_output_tensor = infer_request.get_tensor(bbox_output_name[stride]);
        const float* bbox_buffer = bbox_output_tensor.data<const float>();
        int bbox_batchsize = bbox_output_tensor.get_shape()[0];
        int bbox_c = bbox_output_tensor.get_shape()[1];
        int bbox_h = bbox_output_tensor.get_shape()[2];
        int bbox_w = bbox_output_tensor.get_shape()[3];
        std::cout<<"bbox output shape: ["<<bbox_batchsize
            <<" x "<<bbox_c
            <<" x "<<bbox_h
            <<" x "<<bbox_w
            <<" ]"
            <<std::endl;

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