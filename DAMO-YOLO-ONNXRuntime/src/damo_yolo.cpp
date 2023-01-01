#include "../include/damo_yolo.h"

damo_yolo_detector::damo_yolo_detector(Net_config config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;

    std::string model_path = config.modelpath;
    std::string widestr = std::string(model_path.begin(), model_path.end());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    ort_session = new Ort::Session(env, widestr.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();

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
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

	std::string classesFile = "../include/coco.names";
	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void damo_yolo_detector::normalize_(cv::Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}

damo_yolo_detector::~damo_yolo_detector()
{
    destroy_context();
}

void damo_yolo_detector::destroy_context()
{
    if (ort_session)
    {
        free(ort_session);
    }
}

void damo_yolo_detector::pre_process(cv::Mat img)
{
    ratio = std::min(float(this->inpHeight) / float(img.rows), float(this->inpWidth) / float(img.cols));
	const int neww = int(img.cols * ratio);
	const int newh = int(img.rows * ratio);

	cv::Mat dstimg;
	cv::cvtColor(img, dstimg, cv::COLOR_BGR2RGB);
	cv::resize(dstimg, dstimg, cv::Size(neww, newh));
	cv::copyMakeBorder(dstimg, dstimg, 0, this->inpHeight - newh, 0, this->inpWidth - neww, cv::BORDER_CONSTANT, 1);

	this->normalize_(dstimg);
}

void damo_yolo_detector::do_detection(cv::Mat& img)
{
    pre_process(img);
    std::array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    std::vector<Ort::Value> ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
    std::vector<BoxInfo> generate_boxes;

	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);

	int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score
	const float* pscores = predictions.GetTensorMutableData<float>();
	const float* pbboxes = ort_outputs.at(1).GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)
	{
		int max_ind = 0;
		float class_socre = 0;
		for (k = 0; k < num_class; k++)
		{
			if (pscores[k] > class_socre)
			{
				class_socre = pscores[k];
				max_ind = k;
			}
		}

		if (class_socre > this->confThreshold)
		{
			float xmin = pbboxes[0] / ratio;
			float ymin = pbboxes[1] / ratio;
			float xmax = pbboxes[2] / ratio;
			float ymax = pbboxes[3] / ratio;

			generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_socre, max_ind });
		}
		pscores += nout;
		pbboxes += 4;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);

	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
		std::string label = cv::format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		cv::putText(img, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
	}
}

void damo_yolo_detector::nms(std::vector<BoxInfo>& input_boxes)
{
	std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	std::vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	std::vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (std::max)(float(0), xx2 - xx1 + 1);
			float h = (std::max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}