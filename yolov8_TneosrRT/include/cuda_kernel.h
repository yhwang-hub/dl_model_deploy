#ifndef _CUDA_KERNEL
#define _CUDA_KERNEL

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

void decodeDevice(
        int batchsize, int num_class, int topK, float conf_thresh,
        float* src, int srcWidth, int srcHeight, int srcArea,
        float* dst, int dstWidth, int dstHeight);
void transposeDevice(
        int batchsize,
        float* src, int srcWidth, int srcHeight, int srcArea,
        float* dst, int dstWidth, int dstHeight);
void nmsDeviceV1(
        int batchsize, int topK, int iou_thresh,
        float* src, int srcWidth, int srcHeight, int srcArea);
void nmsDeviceV2(
    int batchsize, int topK, float iou_thresh,
    float* src, int srcWidth, int srcHeight, int srcArea, 
	int* idx, float* conf);

#endif