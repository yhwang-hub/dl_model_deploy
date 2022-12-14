#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <string.h>
#include <float.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <unistd.h>

typedef float PDtype;

/* CUDA: PERCEPTION_CUDA_CHECK for error after kernel execution and exit loudly if there is one. */
#define CUDA_POST_KERNEL_CHECK PERCEPTION_CUDA_CHECK(cudaPeekAtLastError())

#define CUDA_NUM_THREADS        256
#define DIV_THEN_CEIL(x, y)     (((x) + (y) - 1) / (y))

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

//////////////////////////////////////////////////////////////////////////////
//////////////////// sigmoid & exp for multi-task model //////////////////////
//////////////////////////////////////////////////////////////////////////////
__global__ void get_mt_output_kernel(const int nthreads, const PDtype* mt_reg_data,
                                     const PDtype* mt_cls_data, const int reg_channels, const int cls_channels,
                                     const int height, const int width, PDtype* reg_buffer, PDtype* cls_buffer)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < nthreads; index += blockDim.x * gridDim.x)
    {
        int imgid = index / (width * height);
        int ptid = index % (width * height);
        int startloc_reg = imgid * height * width * reg_channels + ptid;
        int startloc_cls = imgid * height * width * cls_channels + ptid;

        for (int i = 0; i < reg_channels; i++)
        {
            int reg_data_index = startloc_reg + i * height * width;
            PDtype regval = mt_reg_data[reg_data_index];
            reg_buffer[reg_data_index] = expf(regval);
        }

        for (int i = 0; i < cls_channels; i++)
        {
            PDtype clsval = mt_cls_data[startloc_cls + i * height * width];
            cls_buffer[startloc_cls + i * height * width] = clsval;
        }
    }
}
// __global__ void get_mt_output_kernel(const int nthreads, const PDtype* mt_reg_data,
//                                      const PDtype* mt_cls_data, const int reg_channels, const int cls_channels,
//                                      const int height, const int width, PDtype* reg_buffer, PDtype* cls_buffer)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < nthreads)
//     {
//         int ptid = index % (width * height);
//         int startloc_reg = ptid;
//         int startloc_cls = ptid;

//         for (int i = 0; i < reg_channels; i++)
//         {
//             int reg_data_index = startloc_reg + i * height * width;
//             PDtype regval = mt_reg_data[reg_data_index];
//             reg_buffer[reg_data_index] = expf(regval);
//         }

//         for (int i = 0; i < cls_channels; i++)
//         {
//             PDtype clsval = mt_cls_data[startloc_cls + i * height * width];
//             cls_buffer[startloc_cls + i * height * width] = clsval;
//         }
//     }
// }

extern "C" void get_mt_output(const PDtype* mt_reg_data, const PDtype* mt_cls_data,
                   const int num, const int reg_channels, const int cls_channels,
                   const int height, const int width,
                   PDtype* reg_buffer, PDtype* cls_buffer) {
    const int numThreads = num * height * width;
    const int numBlocks = DIV_THEN_CEIL(numThreads, CUDA_NUM_THREADS);
    get_mt_output_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
            numThreads, mt_reg_data, mt_cls_data, reg_channels,
            cls_channels, height, width, reg_buffer, cls_buffer);
    CUDA_POST_KERNEL_CHECK;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////// input normalization and data copy ///////////////////////
//////////////////////////////////////////////////////////////////////////////
__global__ void write_img_data_to_buffer_kernel(const int nthreads,
                                                cv::cuda::PtrStepSz<uchar3> img, PDtype* buffer,
                                                PDtype mean_0, PDtype mean_1, PDtype mean_2, PDtype std_value, bool to_rgb)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < nthreads; index += blockDim.x * gridDim.x)
    {
        int area = img.rows * img.cols;
        int ptid = index % (area);
        int y = ptid / img.cols, x = ptid % img.cols;

        uchar3 values = img(y, x);
        if (true == to_rgb)
        {
            uchar tmp = values.x;
            values.x = values.z;
            values.z = tmp;
        }

        buffer[ptid]            = ((PDtype)(values.x) - mean_0) / std_value;
        buffer[ptid + area]     = ((PDtype)(values.y) - mean_1) / std_value;
        buffer[ptid + 2 * area] = ((PDtype)(values.z) - mean_2) / std_value;
    }
}

extern "C" void write_img_data_to_buffer(cv::cuda::PtrStepSz<uchar3> img, PDtype *buffer,
                              const std::vector<PDtype> mean_values, bool to_rgb)
{
    const int numThreads = img.rows * img.cols;
    const int numBlocks = DIV_THEN_CEIL(numThreads, CUDA_NUM_THREADS);

    write_img_data_to_buffer_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
            numThreads, img, buffer, mean_values[0], mean_values[1],
            mean_values[2], mean_values[3], to_rgb);
    CUDA_POST_KERNEL_CHECK;
}