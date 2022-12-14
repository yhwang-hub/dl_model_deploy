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
#include <algorithm>
#include <functional>


/* CHECK the state of CUDA */
#define PERCEPTION_CUDA_CHECK(status)                                   \
    {                                                                   \
        if (status != 0)                                                \
        {                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            abort();                                                    \
        }                                                               \
    }

static Logger gLogger;

static std::tuple<cv::Mat, cv::Mat> post_process(float* output, int output_width, int output_height, int num_class)
{

    cv::Mat output_prob(output_height, output_width, CV_32F);
    cv::Mat output_index(output_height, output_width, CV_8U);

    float* pnet   = output;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);

    // for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
    //     int ic = std::max_element(pnet, pnet + num_class) - pnet;
    //     *prob  = pnet[ic];
    //     *pidx  = ic;
    // }
    int wh = output_height * output_width;
    for(int index = 0; index < output_prob.cols * output_prob.rows; ++index, ++prob, ++pidx)
    {
        float max = -10000000;
            int max_ind = -1;
        for(int k = 0; k < num_class; k++)
        {
            float data = output[k * wh + index];
            if(data > max)
            {
                max = data;
                max_ind = k;
            }
        }
        *prob  = 1. / (1. + expf(-pnet[max_ind]));
        *pidx  = max_ind;
    }

    return std::make_tuple(output_prob, output_index);
}

static std::vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};

static void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass)
{

    auto pimage = image.ptr<cv::Vec3b>(0);
    auto pprob  = prob.ptr<float>(0);
    auto pclass = iclass.ptr<uint8_t>(0);

    for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

        int iclass        = *pclass;
        float probability = *pprob;
        auto& pixel       = *pimage;
        float foreground  = std::min(0.6f + probability * 0.2f, 0.8f);
        float background  = 1 - foreground;
        for(int c = 0; c < 3; ++c){
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
            pixel[c] = std::min((int)value, 255);
        }
    }
}

#endif