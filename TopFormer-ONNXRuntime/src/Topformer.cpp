#include "../include/Topformer.h"

Topformer_detector::Topformer_detector(Net_config config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;

    std::string model_path = config.modelpath;
    std::string widestr = std::string(model_path.begin(), model_path.end());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    ort_session = new Ort::Session(env, widestr.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();

	std::cout<< "numInputNodes: " << numInputNodes
			<< ", numOutputNodes: "<< numOutputNodes
			<< std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->numChannels = input_node_dims[0][1];
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth  = input_node_dims[0][3];
	this->numInputElements = this->numChannels + this->inpHeight + this->inpWidth;
	std::cout<< "numChannels: " << this->numChannels
	        << ", inpHeight: " << this->inpHeight
			<< ", inpWidth: "<< this->inpWidth
			<< std::endl;

	this->numClasses = output_node_dims[0][1];
	this->outHeight = output_node_dims[0][2];
	this->outWidth  = output_node_dims[0][3];
	this->numOutputElements = this->numClasses + this->outHeight + this->outWidth;
	std::cout<< "numClasses: " << this->numClasses
			<< ", outHeight: " << this->outHeight
			<< ", outWidth: "<< this->outWidth
			<< std::endl;
}

Topformer_detector::~Topformer_detector()
{
    
}

void Topformer_detector::pre_process(cv::Mat img)
{
	cv::Mat dstimg;
	cv::cvtColor(img, dstimg, cv::COLOR_BGR2RGB);
	cv::resize(dstimg, dstimg, cv::Size(this->inpWidth, this->inpHeight));
	this->input_image_.resize(this->inpHeight * this->inpWidth * this->numChannels);
	int row = dstimg.rows;
	int col = dstimg.cols;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = \
						((pix / 255.0) - this->img_mean[c]) / this->img_std[c];
			}
		}
	}
}

void Topformer_detector::do_detection(cv::Mat& img)
{
    pre_process(img);
    std::array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = \
		Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), 
							input_shape_.data(), input_shape_.size());

    std::vector<Ort::Value> ort_outputs = \
		ort_session->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, 
					output_names.data(), output_names.size());

	Ort::Value &predictions = ort_outputs.at(0);
	output_tensor = predictions.GetTensorMutableData<float>();
	post_process(img);
}

void Topformer_detector::post_process(cv::Mat& img)
{
	int img_h = img.rows;
    int img_w = img.cols;
    float scale_x = this->outWidth / (float)img.cols;
    float scale_y = this->outHeight / (float)img.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 0;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 0;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat output_prob(this->outHeight, this->outWidth, CV_32F);
    cv::Mat output_index(this->outHeight, this->outWidth, CV_8U);
    float* pnet   = output_tensor;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);
    int wh = this->outHeight * this->outWidth;
    for(int index = 0; index < output_prob.cols * output_prob.rows; ++index, ++prob, ++pidx)
    {
        float max = -10000000;
        int max_ind = -1;
        float sum_c = 0.0;
        for(int k = 0; k < this->numClasses; k++)
        {
            float data = output_tensor[k * wh + index];
            if(data > max)
            {
                max = data;
                max_ind = k;
            }
            sum_c += expf(data);
        }
        // *prob  = 1. / (1. + expf(-pnet[max_ind]));
        *prob = expf(pnet[max_ind]) / sum_c;
        *pidx  = max_ind;
    }

    cv::warpAffine(output_prob, output_prob, m2x3_d2i, img.size(), cv::INTER_LINEAR);
    cv::warpAffine(output_index, output_index, m2x3_d2i, img.size(), cv::INTER_LINEAR);

	for (int i = 0; i < img_h; i++)
    {
        for (int j = 0; j < img_w; j++)
        {
            int max_ind = output_index.at<uint8_t>(i, j);
            img.at<cv::Vec3b>(i, j)[0] = (uint8_t)(img.at<cv::Vec3b>(i, j)[0] * 0.5 +  _classes_colors[max_ind][2] * 0.5);
            img.at<cv::Vec3b>(i, j)[1] = (uint8_t)(img.at<cv::Vec3b>(i, j)[1] * 0.5 +  _classes_colors[max_ind][1] * 0.5);
            img.at<cv::Vec3b>(i, j)[2] = (uint8_t)(img.at<cv::Vec3b>(i, j)[2] * 0.5 +  _classes_colors[max_ind][0] * 0.5);
        }
    }

    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("image-draw.jpg", img);
}