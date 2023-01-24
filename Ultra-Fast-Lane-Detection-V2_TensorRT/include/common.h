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

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

static std::vector<int32_t> argmax_1(const std::vector<float>& v, const std::vector<int32_t>& dims)
{
    std::vector<int32_t> ret;
    ret.resize(dims[0] * dims[2] * dims[3]);

    for (int32_t i = 0; i < dims[2]; i++)
    {
        for (int32_t j = 0; j < dims[3]; j++)
        {
            int32_t offset = dims[3] * i + j;
            float max_val = 0;
            int32_t max_index = 0;
            for (int32_t k = 0; k < dims[1]; k++)
            {
                size_t index = k * dims[2] * dims[3] + offset;
                if (v[index] > max_val)
                {
                    max_val = v[index];
                    max_index = k;
                }
            }
            ret[offset] = max_index;
        }
    }
    return ret;
}

static int32_t sum_valid(const std::vector<int32_t>& v, int32_t num, int32_t interval, int32_t offset)
{
    int32_t sum = 0;
    for (int32_t i = 0; i < num; i++)
    {
        sum += v[i * interval + offset];
    }
    return sum;
}

float Sigmoid(float x)
{
    if (x >= 0)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
    else
    {
        return std::exp(x) / (1.0f + std::exp(x));    /* to aovid overflow */
    }
}

float Logit(float x)
{
    if (x == 0)
    {
        return static_cast<float>(INT32_MIN);
    }
    else  if (x == 1)
    {
        return static_cast<float>(INT32_MAX);
    }
    else
    {
        return std::log(x / (1.0f - x));
    }
}


static inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

float SoftMaxFast(const float* src, float* dst, int32_t length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator{ 0 };

    for (int32_t i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int32_t i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

class NiceColorGenerator
{
public:
    NiceColorGenerator(int32_t num = 16);
    ~NiceColorGenerator();
    cv::Scalar Get(int32_t id);

private:
    int32_t num_;
    int32_t gap_;
    std::vector<int32_t> indices_;
    std::vector<cv::Scalar> color_list_;
};

NiceColorGenerator::NiceColorGenerator(int32_t num)
{
    num_ = num;
    gap_ = 256 / num_;

    std::vector<uint8_t> seq_num(256);
    std::iota(seq_num.begin(), seq_num.end(), 0);
    cv::Mat mat_seq = cv::Mat(256, 1, CV_8UC1, seq_num.data());
    cv::Mat mat_colormap;
    cv::applyColorMap(mat_seq, mat_colormap, cv::COLORMAP_JET);
    for (int32_t i = 0; i < 256; i++)
    {
        const auto& bgr = mat_colormap.at<cv::Vec3b>(i);
        color_list_.push_back(cv::Scalar(bgr[0], bgr[1], bgr[2]));
    }

    for (int32_t i = 0; i < 256; i++)
    {
        indices_.push_back((i % num_) * gap_ + i / gap_);
    }
}

NiceColorGenerator::~NiceColorGenerator() {}

cv::Scalar NiceColorGenerator::Get(int32_t id)
{
    return color_list_[indices_[id % 255]];
}

#endif