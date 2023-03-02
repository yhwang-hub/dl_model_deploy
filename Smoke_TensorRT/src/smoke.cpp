#include <math.h>
#include <cmath>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include "../include/smoke.h"

static const int DEVICE  = 0;

void Smoke_detector::LoadOnnx(const std::string& onnx_path)
{   
    initLibNvInferPlugins(&gLogger, "");
    
    // create runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

    // create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));

    // create config from builder
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // fp16精度的模型类别预测错误,会造成后处理崩溃.
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    size_t workspace_size = (1ULL << 30);
    #if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    #else
        config->setMaxWorkspaceSize(workspace_size);
    #endif

    // create network
    // const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const auto flag = 1U;
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // create parser to fufill network
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(
        onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

    // Build engine
    std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    plan_ = builder->buildSerializedNetwork(*network, *config);
    if (!plan_) {
        std::cout << "Fail to create serialized network" << std::endl;
        return;
    }
    engine = runtime->deserializeCudaEngine(plan_->data(), plan_->size());


    //save engine file
    std::string engine_path = "../smoke_dla34.engine";
    std::cout << "Writing to " << engine_path << "..." << std::endl;
    std::ofstream file(engine_path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

Smoke_detector::Smoke_detector()
{

}

Smoke_detector::Smoke_detector(const std::string& engine_path, const cv::Mat& intrinsic):
                        intrinsic_(intrinsic)
{
    std::cout << "Inference [" 
            << input_h
            << "x" << input_w
            << "] constructed"
            <<std::endl;
}

void Smoke_detector::init_context(const std::string& engine_path)
{
    cudaSetDevice(DEVICE);
    size_t size{0};
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    // getPluginCreator could not find plugin: MMCVModulatedDeformConv2d version: 1
    initLibNvInferPlugins(&gLogger, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trt_model_stream.get(), length);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete runtime;
    runtime = nullptr;
    std::cout << "deserialize done" << std::endl;

    input_index = engine->getBindingIndex(input_name.c_str());
    auto input_dims = engine->getBindingDimensions(0);
    batchsize = input_dims.d[0];
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    input_buffer_size = batchsize * input_c * input_h * input_w * sizeof(float);
    std::cout<< "input shape: "
            << batchsize
            << " x "<< input_c
            << " x "<< input_h
            << " x "<< input_w
            << ", input_buf_size: " << input_buffer_size
            << std::endl;
    std::cout << "start malloc host memory of input......" << std::endl;
    CHECK(cudaHostAlloc((void**)&host_input, input_buffer_size, cudaHostAllocDefault));
    std::cout << "start malloc device memory of input......" << std::endl;
    CHECK(cudaMalloc(&device_buffers[0], input_buffer_size));

    output_h = input_h / 4;
    output_w = input_w / 4;

    std::cout << "start malloc memory of output......" << std::endl;
    bbox_preds_buffer_size = topk * 8 * sizeof(float);
    topk_scores_buffer_size = topk * sizeof(float);
    topk_indices_buffer_size = topk * 1 * sizeof(float);
    CHECK(cudaHostAlloc((void**)&output_bbox_preds, bbox_preds_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[1], bbox_preds_buffer_size));
    CHECK(cudaHostAlloc((void**)&output_topk_scores, topk_scores_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[2], topk_scores_buffer_size));
    CHECK(cudaHostAlloc((void**)&output_topk_indices, topk_indices_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[3], topk_indices_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void Smoke_detector::destroy_context()
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
    for(int i = 0; i < 4; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }
    if(host_input)
        CHECK(cudaFreeHost(host_input));
    if (output_bbox_preds)
        CHECK(cudaFreeHost(output_bbox_preds));
    if (output_topk_scores)
        CHECK(cudaFreeHost(output_topk_scores));
    if (output_topk_indices)
        CHECK(cudaFreeHost(output_topk_indices));
}

Smoke_detector::~Smoke_detector()
{
    destroy_context();
    std::cout << "Context destroyed for ["
            << input_h
            << "x"
            << input_w
            << "]"
            << std::endl;
}

void Smoke_detector::preprocess(cv::Mat image, const cv::Mat& intrinsic)
{
    // https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/smoke.py#L41
    intrinsic_ = intrinsic.clone();
    base_depth = {28.01f, 16.32f};
    base_dims.resize(3);  //pedestrian, cyclist, car
    base_dims[0].x = 0.88f;
    base_dims[0].y = 1.73f;
    base_dims[0].z = 0.67f;
    base_dims[1].x = 1.78f;
    base_dims[1].y = 1.70f;
    base_dims[1].z = 0.58f;
    base_dims[2].x = 3.88f;
    base_dims[2].y = 1.63f;
    base_dims[2].z = 1.53f;
    // Modify camera intrinsics due to scaling
    int image_h = image.rows;
    int image_w = image.cols;
    intrinsic_.at<float>(0, 0) *= static_cast<float>(input_w) / image_w;
    intrinsic_.at<float>(0, 2) *= static_cast<float>(input_w) / image_w;
    intrinsic_.at<float>(1, 1) *= static_cast<float>(input_h) / image_h;
    intrinsic_.at<float>(1, 2) *= static_cast<float>(input_h) / image_h;

    cv::resize(image, input_image, cv::Size(input_w, input_h), cv::INTER_LINEAR);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] - mean_rgb[2]) / std_rgb[2];
        *phost_g++ = (pimage[1] - mean_rgb[1]) / std_rgb[1];
        *phost_b++ = (pimage[2] - mean_rgb[0]) / std_rgb[0];
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[0], host_input, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Smoke_detector::do_detection(cv::Mat& img, const cv::Mat& intrinsic)
{
    std::cout << "start preprocess......" << std::endl;
    preprocess(img, intrinsic);
    std::cout << "start inference......" << std::endl;
    // context->enqueue(batchsize, device_buffers, stream, nullptr);
    context->executeV2(&device_buffers[0]);
    std::cout << "start postprocess......" << std::endl;
    postprocess(img);
    std::cout << "end postprocess......" << std::endl;
}

void Smoke_detector::postprocess(cv::Mat& img)
{
    cudaMemcpyAsync(output_bbox_preds, device_buffers[1], bbox_preds_buffer_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output_topk_scores, device_buffers[2], topk_scores_buffer_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output_topk_indices, device_buffers[3], topk_indices_buffer_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float scale_x = input_w / (float)img.cols;
    float scale_y = input_h / (float)img.rows;

    for (int i = 0; i < topk; i++)
    {
        float score = output_topk_scores[i];
        if (score < score_thresh)
            continue;
        
        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
        int class_id = static_cast<int>(output_topk_indices[i] / output_h / output_w);
        int location = static_cast<int>(output_topk_indices[i]) % (output_h * output_w);
        int img_x = location % output_w;
        int img_y = location / output_w;

        // Depth  bbox_preds_预测的是相对偏移.
        float z = base_depth[0] + output_bbox_preds[8 * i] * base_depth[1];

        // location
        cv::Mat img_point(3, 1, CV_32FC1);
        img_point.at<float>(0) = 4.0f * (static_cast<float>(img_x) + output_bbox_preds[8 * i + 1]);
        img_point.at<float>(1) = 4.0f * (static_cast<float>(img_y) + output_bbox_preds[8 * i + 2]);
        img_point.at<float>(2) = 1.0f;
        cv::Mat cam_point = intrinsic_.inv() * img_point * z;
        float x = cam_point.at<float>(0);
        float y = cam_point.at<float>(1);

        // Dimension
        // std::cout<<"class_id:"<<class_id<<std::endl;
        // std::cout<<"w_offset:"<<bbox_preds_[8*i + 3]<<std::endl;
        float w = base_dims[class_id].x * expf(Sigmoid(output_bbox_preds[8 * i + 3]) - 0.5f);
        float l = base_dims[class_id].y * expf(Sigmoid(output_bbox_preds[8 * i + 4]) - 0.5f);
        float h = base_dims[class_id].z * expf(Sigmoid(output_bbox_preds[8 * i + 5]) - 0.5f);

        // Orientation
        float ori_norm = sqrtf(powf(output_bbox_preds[8 * i + 6], 2.0f) + powf(output_bbox_preds[8 * i + 7], 2.0f));
        output_bbox_preds[8 * i + 6] /= ori_norm;  //sin(alpha)
        output_bbox_preds[8 * i + 7] /= ori_norm;  //cos(alpha)
        float ray = atan(x / (z + 1e-7f));
        float alpha = atan(output_bbox_preds[8 * i + 6] / (output_bbox_preds[8 * i + 7] + 1e-7f));
        if (output_bbox_preds[8 * i + 7] > 0.0f)
        {
            alpha -= M_PI / 2.0f;
        }
        else
        {
            alpha += M_PI / 2.0f;
        }
        float angle = alpha + ray;
        if (angle > M_PI)
        {
            angle -= 2.0f * M_PI;
        }
        else if (angle < -M_PI)
        {
            angle += 2.0f * M_PI;
        }

        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
        //              front z
        //                   /
        //                  /
        //    (x0, y0, z1) + -----------  + (x1, y0, z1)
        //                /|            / |
        //               / |           /  |
        // (x0, y0, z0) + ----------- +   + (x1, y1, z1)
        //              |  /      .   |  /
        //              | / origin    | /
        // (x0, y1, z0) + ----------- + -------> x right
        //              |             (x1, y1, z0)
        //              |
        //              v
        //         down y
        cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << 
            -w, -l, -h,     // (x0, y0, z0)
            -w, -l,  h,     // (x0, y0, z1)
            -w,  l,  h,     // (x0, y1, z1)
            -w,  l, -h,     // (x0, y1, z0)
             w, -l, -h,     // (x1, y0, z0)
             w, -l,  h,     // (x1, y0, z1)
             w,  l,  h,     // (x1, y1, z1)
             w,  l, -h);    // (x1, y1, z0)
        cam_corners = 0.5f * cam_corners;
        cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
        rotation_y.at<float>(0, 0) = cosf(angle);
        rotation_y.at<float>(0, 2) = sinf(angle);
        rotation_y.at<float>(2, 0) = -sinf(angle);
        rotation_y.at<float>(2, 2) = cosf(angle);
        // cos, 0, sin
        //   0, 1,   0
        //-sin, 0, cos
        cam_corners = cam_corners * rotation_y.t();
        for (int i = 0; i < 8; ++i)
        {
            cam_corners.at<float>(i, 0) += x;
            cam_corners.at<float>(i, 1) += y;
            cam_corners.at<float>(i, 2) += z;
        }
        cam_corners = cam_corners * intrinsic_.t();
        std::vector<cv::Point2f> img_corners(8);
        for (int i = 0; i < 8; ++i)
        {
            img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
            img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
            // img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2) / scale_x;
            // img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2) / scale_y;
        }
        for (int i = 0; i < 4; ++i)
        {
            const auto& p1 = img_corners[i];
            const auto& p2 = img_corners[(i + 1) % 4];
            const auto& p3 = img_corners[i + 4];
            const auto& p4 = img_corners[(i + 1) % 4 + 4];
            cv::line(input_image, p1, p2, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_image, p3, p4, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_image, p1, p3, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            // cv::line(img, p1, p2, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            // cv::line(img, p3, p4, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            // cv::line(img, p1, p3, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
        }
    }
    cv::imwrite("../result.png", input_image);
    // cv::imwrite("../result.png", img);
}