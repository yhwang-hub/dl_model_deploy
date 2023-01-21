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


/* (101,56,4), 计算 最大的 index  101_axis */
void argmax(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0,wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int max = -10000000;
            int max_ind = -1;
            for(int k = 0; k < chan; k++)
            {
                if(x[k * wh + i * cols + j] > max)
                {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
}


/* (101,56,4), add softmax on 101_axis and calculate Expect */
void softmax_mul(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0, wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float sum = 0.0;
            float expect = 0.0;
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] /= sum;
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}

std::vector<double> linspace(double start_in, double end_in, int num_in)
{
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input

    return linspaced;
}

std::vector<int> arrange(int num)
{
    std::vector<int> result;
    for (int i = 0; i < num; i++)
    {
        result.push_back(i);
    }
    return result;
}

#endif