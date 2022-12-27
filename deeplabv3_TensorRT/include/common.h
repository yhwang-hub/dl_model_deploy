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

static std::vector<std::vector<int>> _classes_colors = 
            {{128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
            {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
            {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
            {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100},
            {0, 0, 230}, {119, 11, 32}};

void BilinearInterMethod(int channelNum, int width, int height, uchar* imageDataInput, int resized_width, int resized_height, uchar* resizedData)
{
    float r_scale = (float)height / resized_height;
    float c_scale = (float)width / resized_width;
 
    float r_delta = (height - resized_height * r_scale)*0.5f;
    float c_delta = (width - resized_width * c_scale)*0.5f;
 
    int r_w[2];
    int c_w[2];
 
    for (size_t r = 0; r < resized_height; r++)
    {
        float r_ori = r * r_scale + r_delta;
        int v_t = floor(r_ori);
        int v_b = ceil(r_ori);
        if (v_t > height-1 || v_b > height -1)
        {
            v_t = height - 1;
            v_b = height - 1;
        }
 
        r_w[0] = (v_b - r_ori)*256;
        r_w[1] = 256 - r_w[0];
 
        int ind = r * resized_width * channelNum;
 
        for (size_t c = 0; c < resized_width; c++)
        {
            float c_ori = c * c_scale + c_delta;
            int u_l = floor(c_ori);
            int u_r = ceil(c_ori);
            if (u_l > width - 1 || u_r > width - 1)
            {
                u_l = width - 1;
                u_r = width - 1;
            }
 
            c_w[0] = (u_r - c_ori)*256;
            c_w[1] = 256 - c_w[0];
 
            int index = ind + c* channelNum;
 
            for (size_t i = 0; i < channelNum; i++)
            {
                auto q1 = *(imageDataInput + v_t * width * channelNum + u_l * channelNum + i);
                auto q2 = *(imageDataInput + v_t * width * channelNum + u_r * channelNum + i);
                auto q3 = *(imageDataInput + v_b * width * channelNum + u_r * channelNum + i);
                auto q4 = *(imageDataInput + v_b * width * channelNum + u_l * channelNum + i);
                int value = (int)(r_w[0] * c_w[0] * q1 +
                    r_w[0] * c_w[1] * q2 +
                    r_w[1] * c_w[0] * q4 +
                    r_w[1] * c_w[1] * q3);
 
                *(resizedData + index + i) = value>>16;
            }
        }
    }
}

#endif