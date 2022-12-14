#include <iostream>
#include "../include/yolov5.h"
#include "../include/common.h"

#define INPUT_DEBUG
#define OUTPUT_DEBUG

static const int DEVICE  = 0;

Yolov5_Detector::Yolov5_Detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<"x"<<input_w<<"] constructed"<<std::endl;
    std::cout<<"Inference segmentation ["<<seg_height<<"x"<<seg_width<<"] constructed"<<std::endl;
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
    // TRTLogger logger;
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

    det_output_index = engine->getBindingIndex(detect_output_name.c_str());
    auto det_output_dims = engine->getBindingDimensions(det_output_index);
    num_bbox = det_output_dims.d[1];
    det_output_channels = det_output_dims.d[2];
    det_output_buffer_size = batchsize * num_bbox * det_output_channels * sizeof(float);
    std::cout<<"det output shape: "
            <<batchsize
            <<" x "<<num_bbox
            <<" x "<<det_output_channels
            <<", det_output_buffer_size: "<<det_output_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&det_output_cpu, det_output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[det_output_index], det_output_buffer_size));

    seg_output_index = engine->getBindingIndex(seg_output_name.c_str());
    auto seg_output_dims = engine->getBindingDimensions(seg_output_index);
    seg_channels = seg_output_dims.d[1];
    seg_height = seg_output_dims.d[2];
    seg_width = seg_output_dims.d[3];
    seg_output_buffer_size = batchsize * seg_channels * seg_height * seg_width * sizeof(float);
    std::cout<<"seg output shape: "
            <<batchsize
            <<" x "<<seg_channels
            <<" x "<<seg_height
            <<" x "<<seg_width
            <<", seg_output_buffer_size: "<<seg_output_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&seg_output_cpu, seg_output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[seg_output_index], seg_output_buffer_size));

    CHECK(cudaStreamCreate(&stream));
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
    for(int i = 0; i < 3; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }
    if(host_input)
        CHECK(cudaFreeHost(host_input));
    if(det_output_cpu)
        CHECK(cudaFreeHost(det_output_cpu));
    if(seg_output_cpu)
        CHECK(cudaFreeHost(seg_output_cpu));
    if(stream)
        cudaStreamDestroy(stream);
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

    std::string pytorch_preprocess_txt = "../pytorch_preprocess.txt";
    std::ifstream pytorch_preprocess_data(pytorch_preprocess_txt);
    std::vector<float>pytorch_preprocess_result;
    float data;
    while (pytorch_preprocess_data >> data)
    {
        pytorch_preprocess_result.push_back(data);
    }
#endif

    cv::Mat img = image.clone();
    int w, h, x, y;
	float r_w = input_w / (image.cols*1.0);
	float r_h = input_h / (image.rows*1.0);
	if (r_h > r_w) 
    {
		w = input_w;
		h = r_w * image.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else 
    {
		w = r_h * image.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
    padsize.push_back(h);
	padsize.push_back(w);
	padsize.push_back(y);
	padsize.push_back(x);
    std::cout<<"h: "<<h<<", w: "<<w<<std::endl;

    cv::Size new_shape = cv::Size(640, 640);
    float width = img.cols;
    float height = img.rows;
    float r = std::min(new_shape.width / width, new_shape.height / height);
    r = std::min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    std::cout<<"new_unpadW:"<<new_unpadW<<", new_unpadH:"<<new_unpadH<<std::endl;
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
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

    cv::imwrite("pre_img.jpg", dst);
    int image_area = dst.cols * dst.rows;
    unsigned char* pimage = dst.data;
    float* phost_b = host_input + image_area * 0;
    float* phost_g = host_input + image_area * 1;
    float* phost_r = host_input + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

#ifdef INPUT_DEBUG
    float max_diff = 0.0;
    for (int i = 0; i < input_buffer_size / sizeof(float); i++)
    {
        Tensorrt_preprocess<<host_input[i]<<"\n";
        float diff = std::abs(host_input[i] - pytorch_preprocess_result[i]);
        if (diff > max_diff)
        {
            std::cout<<i<<", "<<host_input[i]<<", "<<pytorch_preprocess_result[i]<<std::endl;
            max_diff = diff;
        }
        
    }
    std::cout<<"preprocess max_diff: "<<max_diff<<std::endl;
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
    std::cout<<"preprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    // context->enqueueV2(&device_buffers[input_index], stream, nullptr);
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    post_process(img);
    std::cout<<"Post-process done!"<<std::endl;
}

void Yolov5_Detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(det_output_cpu, device_buffers[det_output_index], \
                        det_output_buffer_size, \
                        cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(seg_output_cpu, device_buffers[seg_output_index], \
                        seg_output_buffer_size, \
                        cudaMemcpyDeviceToHost, stream));

#ifdef OUTPUT_DEBUG    
    std::string pytorch_result_txt = "../pred_result.txt";
    std::ifstream pytorch_result_data(pytorch_result_txt);
    std::vector<float>pred_result;
    float data;
    while (pytorch_result_data >> data)
    {
        pred_result.push_back(data);
    }
    float max_diff = 0.0;
    for (int i = 0; i < det_output_buffer_size / sizeof(float); i++)
    {
        float diff = std::abs(pred_result[i] - det_output_cpu[i]);
        if (diff > max_diff)
        {
            std::cout<<i<<", "<<pred_result[i]<<", "<<det_output_cpu[i]<<std::endl;
            max_diff = diff;
        }
    }
    std::cout<<"pred result max_diff: "<<max_diff<<std::endl;
#endif

    std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> picked_proposals;

	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	//printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
	float ratio_h = (float)img.rows / newh;
	float ratio_w = (float)img.cols / neww;

	int net_width = classes_num + 5 + seg_channels;
	float* pdata = det_output_cpu;
	for (int j = 0; j < num_bbox; ++j)
    {
		float box_score = pdata[4];
		if (box_score >= confThreshold)
        {
			cv::Mat scores(1, classes_num, CV_32FC1, pdata + 5);
			cv::Point classIdPoint;
			double max_class_socre;
			cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= confThreshold)
            {
				std::vector<float> temp_proto(pdata + 5 + classes_num, pdata + net_width);
				picked_proposals.push_back(temp_proto);

				float x = (pdata[0] - padw) * ratio_w;  //x
				float y = (pdata[1] - padh) * ratio_h;  //y
				float w = pdata[2] * ratio_w;  //w
				float h = pdata[3] * ratio_h;  //h

				int left = MAX((x - 0.5 * w) , 0);
				int top = MAX((y - 0.5 * h) , 0);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre * box_score);
				boxes.push_back(cv::Rect(left, top, int(w ), int(h )));
			}
		}
		pdata += net_width;
	}
    std::cout<<"boxes.size:"<<boxes.size()<<std::endl;
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nms_result);
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, img.cols, img.rows);
	std::vector<OutputSeg> output;
    std::cout<<"nms_result.size: "<<nms_result.size()<<std::endl;
	for (int i = 0; i < nms_result.size(); ++i)
    {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		output.push_back(result);

		temp_mask_proposals.push_back(picked_proposals[idx]);
	}
    std::cout<<"temp_mask_proposals.size:"<<temp_mask_proposals.size()<<std::endl;
	cv::Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		//std::cout<< Mat(temp_mask_proposals[i]).t().size();
		maskProposals.push_back(cv::Mat(temp_mask_proposals[i]).t());

	pdata = seg_output_cpu;
	std::vector<float> seg_mask(pdata, pdata + seg_channels * seg_width * seg_height);
	cv::Mat mask_protos = cv::Mat(seg_mask, true);
	cv::Mat protos = mask_protos.reshape(0, {seg_channels, seg_width * seg_height });

    std::cout<<"maskProposals.szie: "<<maskProposals.size()<<std::endl;
    std::cout<<"protos.size: "<<protos.size()<<std::endl;

	cv::Mat matmulRes = (maskProposals * protos).t();
	cv::Mat masks = matmulRes.reshape(output.size(), { seg_width, seg_height });

	std::vector<cv::Mat> maskChannels;
	cv::split(masks, maskChannels);
	std::cout << maskChannels.size();
	for (int i = 0; i < output.size(); ++i) 
    {
		cv::Mat dest, mask;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);//160*160
		
		cv::Rect roi(int((float)padw / input_w * seg_width), int((float)padh / input_h * seg_height), int(seg_width - padw / 2), int(seg_width - padh / 2));
		//std::cout << roi;
		dest = dest(roi);
		
		cv::resize(dest, mask, img.size(), cv::INTER_NEAREST);

		cv::Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > maskThreshold;

		output[i].boxMask = mask;
	}

	DrawPred(img, output);

	cv::imwrite("output.jpg", img);
}