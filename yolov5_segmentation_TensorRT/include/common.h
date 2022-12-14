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