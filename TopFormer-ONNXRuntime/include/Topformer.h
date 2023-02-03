#ifndef _TOPFORMER_H
#define _TOPFORMER_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "common.h"

class Topformer_detector
{
public:
    Topformer_detector(Net_config config);
    virtual ~Topformer_detector();
    virtual void do_detection(cv::Mat& img);

private:
	int numChannels;
    int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	int numClasses;
	int numInputElements;
	int numOutputElements;
	//-mean, /std
	float img_mean[3] = {0.485, 0.456, 0.406};
	float img_std[3]  = {0.229, 0.224, 0.225};
	float* output_tensor;

	float confThreshold;
	float nmsThreshold;
	std::vector<float> input_image_;
	float* host_input;
	void normalize_(cv::Mat img);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "TopFormer");
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
    void post_process(cv::Mat& img);
};

#endif