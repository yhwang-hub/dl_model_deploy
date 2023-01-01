#ifndef _DAMO_YOLO_H
#define _DAMO_YOLO_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "common.h"

class damo_yolo_detector
{    
public:
    damo_yolo_detector(Net_config config);
    virtual ~damo_yolo_detector();
    virtual void do_detection(cv::Mat& img);

private:
    int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	std::vector<std::string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	std::vector<float> input_image_;
	void normalize_(cv::Mat img);
	void nms(std::vector<BoxInfo>& input_boxes);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DAMO_YOLO");
	Ort::Session *ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

    float ratio;

    // void init_context();
    void destroy_context();
    void pre_process(cv::Mat img);
    void post_process(cv::Mat& img, std::vector<Ort::Value> ort_outputs);
};

#endif