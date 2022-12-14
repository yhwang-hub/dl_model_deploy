#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class DAMO_YOLO
{
public:
	DAMO_YOLO(Net_config config);
	void detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "DAMO_YOLO");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

DAMO_YOLO::DAMO_YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;
	std::string widestr = std::string(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
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

	string classesFile = "../include/coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void DAMO_YOLO::normalize_(Mat img)
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

void DAMO_YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
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

void DAMO_YOLO::detect(Mat& frame)
{
	const float ratio = std::min(float(this->inpHeight) / float(frame.rows), float(this->inpWidth) / float(frame.cols));
	const int neww = int(frame.cols * ratio);
	const int newh = int(frame.rows * ratio);

	Mat dstimg;
	cvtColor(frame, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(neww, newh));
	copyMakeBorder(dstimg, dstimg, 0, this->inpHeight - newh, 0, this->inpWidth - neww, BORDER_CONSTANT, 1);

	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// ��ʼ����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	vector<BoxInfo> generate_boxes;
	
	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);

	int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score
	const float* pscores = predictions.GetTensorMutableData<float>();
	const float* pbboxes = ort_outputs.at(1).GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///����ͼ�߶�
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
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{
	Net_config cfg = { 0.5, 0.85, "../damoyolo_tinynasL20_T_192x320.onnx" }; ///choices=["weights/damoyolo_tinynasL20_T_192x320.onnx", "weights/damoyolo_tinynasL20_T_256x320.onnx", "weights/damoyolo_tinynasL20_T_256x416.onnx", "weights/damoyolo_tinynasL20_T_288x480.onnx", "weights/damoyolo_tinynasL20_T_384x640.onnx", "weights/damoyolo_tinynasL20_T_480x640.onnx", "weights/damoyolo_tinynasL20_T_480x800.onnx", "weights/damoyolo_tinynasL20_T_640x640.onnx", "weights/damoyolo_tinynasL20_T_736x1280.onnx", "weights/damoyolo_tinynasL25_S_192x320.onnx", "weights/damoyolo_tinynasL25_S_256x320.onnx", "weights/damoyolo_tinynasL25_S_256x416.onnx", "weights/damoyolo_tinynasL25_S_288x480.onnx", "weights/damoyolo_tinynasL25_S_384x640.onnx", "weights/damoyolo_tinynasL25_S_480x640.onnx", "weights/damoyolo_tinynasL25_S_480x800.onnx", "weights/damoyolo_tinynasL25_S_640x640.onnx", "weights/damoyolo_tinynasL25_S_736x1280.onnx", "weights/damoyolo_tinynasL35_M_192x320.onnx", "weights/damoyolo_tinynasL35_M_256x320.onnx", "weights/damoyolo_tinynasL35_M_256x416.onnx", "weights/damoyolo_tinynasL35_M_288x480.onnx", "weights/damoyolo_tinynasL35_M_384x640.onnx", "weights/damoyolo_tinynasL35_M_480x640.onnx", "weights/damoyolo_tinynasL35_M_480x800.onnx", "weights/damoyolo_tinynasL35_M_640x640.onnx", "weights/damoyolo_tinynasL35_M_736x1280.onnx"]
	DAMO_YOLO net(cfg);
	string imgpath = "../dog.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}