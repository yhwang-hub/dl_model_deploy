#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/lane-detector.h"
#include "../include/common.h"

static const int DEVICE  = 0;

Lane_detector::Lane_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void Lane_detector::init_context()
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

    TRTLogger Logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(Logger);
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
            << batchsize
            << " x "<<input_c
            << " x "<<input_h
            << " x "<<input_w
            << ", input_buf_size: "<<input_buffer_size
            << std::endl;
    CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    exist_row_index = engine->getBindingIndex(exist_row_name.c_str());
    auto exist_row_dim = engine->getBindingDimensions(exist_row_index);
    exist_row_dims = {exist_row_dim.d[0], exist_row_dim.d[1], exist_row_dim.d[2], exist_row_dim.d[3]};
    exist_row_buffer_size = exist_row_dims[0] * exist_row_dims[1] * exist_row_dims[2] * exist_row_dims[3] * sizeof(float);
    std::cout<<"exist_row_dims: "
        << exist_row_dims[0]
        << " x "<< exist_row_dims[1]
        << " x "<< exist_row_dims[2]
        << " x "<< exist_row_dims[3]
        << ", exist_row_buffer_size: "<< exist_row_buffer_size
        << std::endl;
    CHECK(cudaHostAlloc((void**)&exist_row_host_output, exist_row_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[exist_row_index], exist_row_buffer_size));

    exist_col_index = engine->getBindingIndex(exist_col_name.c_str());
    auto exist_col_dim = engine->getBindingDimensions(exist_col_index);
    exist_col_dims = {exist_col_dim.d[0], exist_col_dim.d[1], exist_col_dim.d[2], exist_col_dim.d[3]};
    exist_col_buffer_size = exist_col_dims[0] * exist_col_dims[1] * exist_col_dims[2] * exist_col_dims[3] * sizeof(float);
    std::cout<<"exist_col_dims: "
        << exist_col_dims[0]
        << " x "<< exist_col_dims[1]
        << " x "<< exist_col_dims[2]
        << " x "<< exist_col_dims[3]
        << ", exist_row_buffer_size: "<< exist_col_buffer_size
        << std::endl;
    CHECK(cudaHostAlloc((void**)&exist_col_host_output, exist_col_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[exist_col_index], exist_col_buffer_size));

    loc_row_index = engine->getBindingIndex(loc_row_name.c_str());
    auto loc_row_dim = engine->getBindingDimensions(loc_row_index);
    loc_row_dims = {loc_row_dim.d[0], loc_row_dim.d[1], loc_row_dim.d[2], loc_row_dim.d[3]};
    loc_row_buffer_size = loc_row_dims[0] * loc_row_dims[1] * loc_row_dims[2] * loc_row_dims[3] * sizeof(float);
    std::cout<<"loc_row_dims: "
        << loc_row_dims[0]
        << " x "<< loc_row_dims[1]
        << " x "<< loc_row_dims[2]
        << " x "<< loc_row_dims[3]
        << ", exist_row_buffer_size: "<< loc_row_buffer_size
        << std::endl;
    CHECK(cudaHostAlloc((void**)&loc_row_host_output, loc_row_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[loc_row_index], loc_row_buffer_size));

    loc_col_index = engine->getBindingIndex(loc_col_name.c_str());
    auto loc_col_dim = engine->getBindingDimensions(loc_col_index);
    loc_col_dims = {loc_col_dim.d[0], loc_col_dim.d[1], loc_col_dim.d[2], loc_col_dim.d[3]};
    loc_col_buffer_size = loc_col_dims[0] * loc_col_dims[1] * loc_col_dims[2] * loc_col_dims[3] * sizeof(float);
    std::cout<<"loc_col_dims: "
        << loc_col_dims[0]
        << " x "<< loc_col_dims[1]
        << " x "<< loc_col_dims[2]
        << " x "<< loc_col_dims[3]
        << ", exist_row_buffer_size: "<< loc_col_buffer_size
        << std::endl;
    CHECK(cudaHostAlloc((void**)&loc_col_host_output, loc_col_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[loc_col_index], loc_col_buffer_size));

    for (int i = 0; i < kNumRow; i++)
    {
        row_anchor_.push_back(0.42 + i * (1.0 - 0.42) / (kNumRow - 1));
    }
    for (int i = 0; i < kNumCol; i++)
    {
        col_anchor_.push_back(0.0 + i * (1.0 - 0.0) / (kNumCol - 1));
    }

    CHECK(cudaStreamCreate(&stream));
}

void Lane_detector::destroy_context()
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
    for(int i = 0; i < 5; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }

    CHECK(cudaFreeHost(exist_row_host_output));
    CHECK(cudaFreeHost(exist_col_host_output));
    CHECK(cudaFreeHost(loc_row_host_output));
    CHECK(cudaFreeHost(loc_col_host_output));
}

Lane_detector::~Lane_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Lane_detector::pre_process(cv::Mat image)
{
    crop_x = 0;
    crop_y = image.rows * 0.4;
    crop_w = image.cols;
    crop_h = image.rows * 0.5;
    cv::Mat input_image = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    cv::Mat src = image(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    cv::resize(src, input_image, input_image.size(), 0, 0, cv::INTER_LINEAR);

    std::cout<< "input_image shape: [" << input_image.cols << ", " << input_image.rows << "]"<<std::endl;

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - norm_mean[0]) / norm_std[0];
        *phost_g++ = (pimage[1] / 255.0f - norm_mean[1]) / norm_std[1];
        *phost_b++ = (pimage[2] / 255.0f - norm_mean[2]) / norm_std[2];
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Lane_detector::do_detection(cv::Mat& img)
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

    post_process(img);
    std::cout<<"Post-process done!"<<std::endl;
}

void Lane_detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(exist_row_host_output, device_buffers[exist_row_index],
                    exist_row_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(exist_col_host_output, device_buffers[exist_col_index],
                    exist_col_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(loc_row_host_output, device_buffers[loc_row_index],
                    loc_row_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(loc_col_host_output, device_buffers[loc_col_index],
                    loc_col_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<float> loc_row(loc_row_host_output, loc_row_host_output + loc_row_buffer_size / sizeof(float) + 1);
    std::vector<float> loc_col(loc_col_host_output, loc_col_host_output + loc_col_buffer_size / sizeof(float) + 1);
    std::vector<float> exist_row(exist_row_host_output, exist_row_host_output + exist_row_buffer_size / sizeof(float) + 1);
    std::vector<float> exist_col(exist_col_host_output, exist_col_host_output + exist_col_buffer_size / sizeof(float) + 1);

    std::vector<Line<float>> line_list(4);

    int32_t num_grid_row = loc_row_dims[1]; /* 200 */
    int32_t num_cls_row = loc_row_dims[2];  /* 72 */
    int32_t num_lane_row = loc_row_dims[3]; /* 4 */
    int32_t num_grid_col = loc_col_dims[1];
    int32_t num_cls_col = loc_col_dims[2];
    int32_t num_lane_col = loc_col_dims[3];

    auto max_indices_row = argmax_1(loc_row, loc_row_dims);  /* 1x200x72x4 -> 1x72x4 */
    auto valid_row = argmax_1(exist_row, exist_row_dims);
    auto max_indices_col = argmax_1(loc_col, loc_col_dims);
    auto valid_col = argmax_1(exist_col, exist_col_dims);

    for (int32_t i : { 1, 2 })
    {
        if (sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row / 2)
        {
            for (int32_t k = 0; k < num_cls_row; k++)
            {
                int32_t index = k * num_lane_row + i;
                if (valid_row[index] != 0)
                {
                    /* all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1))) */
                    std::vector<float> pred_all_list;
                    std::vector<int32_t> all_ind_list;
                    for (int32_t all_ind = std::max(0, max_indices_row[index] - 1); all_ind <= std::min(num_grid_row - 1, max_indices_row[index] + 1); all_ind++)
                    {
                        pred_all_list.push_back(loc_row[all_ind * num_cls_row * num_lane_row + index]);
                        all_ind_list.push_back(all_ind);
                    }
                    std::vector<float> pred_all_list_softmax(pred_all_list.size());
                    SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
                    float out_temp = 0;
                    for (int32_t l = 0; l < pred_all_list.size(); l++)
                    {
                        out_temp += pred_all_list_softmax[l] * all_ind_list[l];
                    }
                    float x = (out_temp + 0.5) / (num_grid_row - 1.0);
                    float y = row_anchor_[k];
                    line_list[i].push_back(std::pair<float, float>(x, y));
                }
            }
        }
    }

    for (int32_t i : {0, 3})
    {
        if (sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 8)
        {
            for (int32_t k = 0; k < num_cls_col; k++)
            {
                int32_t index = k * num_lane_col + i;
                if (valid_col[index] != 0)
                {
                    std::vector<float> pred_all_list;
                    std::vector<int32_t> all_ind_list;
                    for (int32_t all_ind = std::max(0, max_indices_col[index] - 1); all_ind <= std::min(num_grid_col - 1, max_indices_col[index] + 1); all_ind++)
                    {
                        pred_all_list.push_back(loc_col[all_ind * num_cls_col * num_lane_col + index]);
                        all_ind_list.push_back(all_ind);
                    }
                    std::vector<float> pred_all_list_softmax(pred_all_list.size());
                    SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
                    float out_temp = 0;
                    for (int32_t l = 0; l < pred_all_list.size(); l++)
                    {
                        out_temp += pred_all_list_softmax[l] * all_ind_list[l];
                    }
                    float y = (out_temp + 0.5) / (num_grid_col - 1.0);
                    float x = col_anchor_[k];
                    line_list[i].push_back(std::pair<float, float>(x, y));
                }
            }
        }
    }

    /* todo: I'm not sure the following code correct */
    /* Adjust height scale : https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c80276bc2fd67d02579b6eeb57a76cb5a905aa3d/demo.py#L88 */
    /* It looks the demo code run inference with height = model_input_height / 0.6 . but our code cannot do this. so after running inference with height = model_input_height, adjust y position */
    const float kInferenceHeight = input_h / kCropRatio;
    for (auto& line : line_list)
    {
        for (auto& p : line)
        {
            p.second = ((p.second * kInferenceHeight) - (kInferenceHeight - input_h)) / input_h;
        }
    }

    std::vector<Line <int32_t>> line_ret_list;
    for (auto& line : line_list)
    {
        Line <int32_t> line_ret;
        for (auto& p : line)
        {
            std::pair<int32_t, int32_t> p_ret;
            p_ret.first = p.first * crop_w + crop_x;
            p_ret.second = p.second * crop_h + crop_y;
            line_ret.push_back(p_ret);
        }
        line_ret_list.push_back(line_ret);
    }

    crop_x = (std::max)(0, crop_x);
    crop_y = (std::max)(0, crop_y);
    crop_w = (std::min)(crop_w, img.cols - crop_x);
    crop_h = (std::min)(crop_h, img.rows - crop_y);

    /* Display target area  */
    cv::rectangle(img, cv::Rect(crop_x, crop_y, crop_w, crop_h),cv::Scalar(0, 0, 0), 2);

    /* Draw line */
    NiceColorGenerator s_nice_color_generator(4);
    for (int32_t lane_index = 0; lane_index < line_ret_list.size(); lane_index++)
    {
        const auto& line = line_ret_list[lane_index];
        std::cout << "line num: " << line.size() << std::endl;
        for (const auto& p : line)
        {
            cv::circle(img, cv::Point(p.first, p.second), 4, s_nice_color_generator.Get((lane_index == 0 || lane_index == 3) ? 0 : 1), -1);
        }
    }
}