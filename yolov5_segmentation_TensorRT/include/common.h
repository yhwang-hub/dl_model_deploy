#ifndef COMMON_H
#define COMMON_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include <unistd.h>
#include <cmath>
#include <string>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;

static const int NUM_CLASSES = 80;

struct OutputSeg {
	int id;
	float confidence;
	cv::Rect box;
	cv::Mat boxMask;
};

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, std::vector<int>& padsize) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {//�����ڸ�
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	padsize.push_back(h);
	padsize.push_back(w);
	padsize.push_back(y);
	padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];

	return out;
}
cv::Rect get_rect(cv::Mat& img, float bbox[4], int INPUT_W, int INPUT_H) {
	int l, r, t, b;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w) {
		//l = bbox[0] - bbox[2] / 2.f;
		//r = bbox[0] + bbox[2] / 2.f;

		//t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		//b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		//l = l / r_w;
		//r = r / r_w;
		//t = t / r_w;
		//b = b / r_w;

		l = bbox[0];
		r = bbox[2];
		t = bbox[1]- (INPUT_H - r_w * img.rows) / 2;
		b = bbox[3] - (INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;

	}
	return cv::Rect(l, t, r - l, b - t);
}


void DrawPred(cv::Mat& img,std::vector<OutputSeg> result) {
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < NUM_CLASSES; i++) {
		int b = std::rand() % 256;
		int g = std::rand() % 256;
		int r = std::rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
	cv::Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);

		std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = std::max(top, labelSize.height);
		putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	cv::addWeighted(img, 0.5, mask, 0.5, 0, img);
}



#endif