#include "../include/cuda_kernel.h"
#include <thrust/sort.h>

#define BLOCK_SIZE 8

/*
	src:
				8400 ->
	84	  x1 x2  ...... x8400
	 |    y1 y2  ...... y8400
	 V	  w1 w2  ...... w8400
		  h1 h2  ...... h8400
		  c0  .
		  c1  .
		  c2  .
		  .   .
		  .   .
		  .
		  c79

*/
__global__ void transpose_device_kernel(
        int batch_size,
        float* src, int srcWidth, int srcHeight, int srcArea,
        float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= dstHeight || dy >= batch_size)
	{
		return;
	}
	float* p_dst_row = dst + dy * dstArea + dx * dstWidth; // row = dx
	float* p_src_col = src + dy * srcArea + dx; // col = dx

	for (int i = 0; i < dstWidth; i++)
	{
		p_dst_row[i] = p_src_col[i * srcWidth];
	}
}

__global__ void decode_yolov8_device_kernel(
    int batch_size, int  num_class, int topK, float conf_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;

	// find max Pr(Classi/Object)
	//float* class_confidence = pitem + 5;  // Pr(Class0/Object)
	float* class_confidence = pitem + 4;    // Pr(Class0/Object)
	float confidence = *class_confidence++; // Pr(Class1/Object)
	int label = 0;
	for (int i = 1; i < num_class; ++i, ++class_confidence)
	{
		if (*class_confidence > confidence)
		{
			confidence = *class_confidence;
			label = i;
		}
	}
	if (confidence < conf_thresh)
	{
		return;
	}

	// parray:count, box1, box2, box3(count:)
	// parray[0]:count
	// atomicAdd -> count += 1
	// atomicAdd: return old_count
	//int index = atomicAdd(dst + dy * dstArea, 1);
	//assert(dy == 1);
	int index = atomicAdd(dst + dy * dstArea, 1);

	if (index >= topK)
	{
		return;
	}
	// xywh -> xyxy
	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;

	float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;

	/*float left = cx;
	float top = cy;
	float right = width;
	float bottom = height;*/
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left; // todo
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;


	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;// 1 = keep, 0 = ignore
}

// fast nms
static __device__ 
float box_iou(
	float aleft, float atop, float aright, float abottom,
	float bleft, float btop, float bright, float bbottom
) {
	float cleft = max(aleft, bleft);
	float ctop = max(atop, btop);
	float cright = min(aright, bright);
	float cbottom = min(abottom, bbottom);

	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if (c_area == 0.0f)
		return 0.0f;

	float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
	float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
	return c_area / (a_area + b_area - c_area);
}

__global__ void nms_fast_kernel(
    int topK, int batch_size, float iou_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea) // topK = srcHeigh
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;

	//int count = min((int)*(src + dy * srcArea), topK);
	if (dy >= batch_size) // prevent from p_temp out of range, eg: dy >= batch_size
	{
		return;
	}
	float* p_count = src + dy * srcArea;
	int count = min(int(p_count[0]), topK);
	
	if (dx >= count)
	{
		return;
	}

	// left, top, right, bottom, confidence, class, keepflag
	float* pcurrent = src + dy * srcArea + 1 + dx * srcWidth; // one row data
	for (int i = 0; i < count; ++i) 
	{
		float* pitem = src + dy * srcArea + 1 + i * srcWidth; 
		if (i == dx || pcurrent[5] != pitem[5]) 
			continue;

		if (pitem[4] >= pcurrent[4])
		{
			if (pitem[4] == pcurrent[4] && i < dx) 
				continue;

			float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
				pitem[0], pitem[1], pitem[2], pitem[3]);

			if (iou > iou_thresh)
			{
				pcurrent[6] = 0;  // 1=keep, 0=ignore
				return;
			}
		}
	}
}

__global__
void get_key_val_kernel(
    int batchSize,
    float* src, int srcWidth, int srcHeight, int srcArea,
	int* idx, float* conf)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dy >= batchSize || dx >= srcHeight) // srcHeight = topK
	{
		return;
	}
	int* p_idx_row    = idx  + dy * srcHeight + dx;
	float* p_conf_row = conf + dy * srcHeight + dx;

	p_idx_row[0] = dx;
	// left, top, right, bottom, confidence, class, keepflag
	float* p_src_row = src + dy * srcArea + 1 + dx * srcWidth; 
	p_conf_row[0] = p_src_row[4];
}

__global__
void nms_sort_kernel(int topK, int batch_size, float iou_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea,
	int* idx) // topK = srcHeigh,
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;

	//int count = min((int)*(src + dy * srcArea), topK);
	if (dy >= batch_size) // prevent from p_temp out of range, eg: dy >= batch_size
	{
		return;
	}
	float* p_count = src + dy * srcArea;
	int count = min(int(p_count[0]), topK);

	if (dx >= count)
	{
		return;
	}

	//
	int* p_idx1 = idx + dy * srcHeight + dx;
	float* pcurrent = src + dy * srcArea + 1 + p_idx1[0] * srcWidth;  // left, top, right, bottom, confidence, class, keepflag
	
	for (int i = (dx + 1); i < count; ++i) // 
	{
		int* p_idx2 = idx + dy * srcHeight + i;
		float* pitem = src + dy * srcArea + 1 + p_idx2[0] * srcWidth; //
		
		if (abs(pcurrent[5] - pitem[5]) > 1e-3) //
			continue;
		float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
			pitem[0], pitem[1], pitem[2], pitem[3]);

		if (iou > iou_thresh)
		{
			pitem[6] = 0;  // 1=keep, 0=ignore 
		}
		
	}
}

void transposeDevice(int batchsize, 
                float* src, int srcWidth, int srcHeight, int srcArea, 
                float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchsize + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = dstWidth * dstHeight;

	transpose_device_kernel << < grid_size, block_size, 0, nullptr >> > (batchsize,
		src, srcWidth, srcHeight, srcArea,
		dst, dstWidth, dstHeight, dstArea);
}

void decodeDevice(int batchsize, int num_class, int topK, float conf_thresh,
            float* src, int srcWidth, int srcHeight, int srcArea,
            float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchsize + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = 1 + dstWidth * dstHeight;

	decode_yolov8_device_kernel << < grid_size, block_size, 0, nullptr >> > (
                                    batchsize, num_class, topK, conf_thresh,
                                    src, srcWidth, srcHeight, srcArea,
                                    dst, dstWidth, dstHeight, dstArea);
}

void nmsDeviceV1(int batchsize, int topK, int iou_thresh,
                float* src, int srcWidth, int srcHeight, int srcArea)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((topK + BLOCK_SIZE - 1) / BLOCK_SIZE, // todo
		(batchsize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	nms_fast_kernel << < grid_size, block_size, 0, nullptr >> > (
                        topK, batchsize, iou_thresh,
                        src, srcWidth, srcHeight, srcArea);
}

// nms with sort
void nmsDeviceV2(
    int batchsize, int topK, float iou_thresh,
    float* src, int srcWidth, int srcHeight, int srcArea, 
	int* idx, float* conf)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((topK + BLOCK_SIZE - 1) / BLOCK_SIZE, // todo
		(batchsize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// get keys and vals(confs)
	get_key_val_kernel << < grid_size, block_size, 0, nullptr >> > (
                batchsize, src, srcWidth, srcHeight, srcArea, idx, conf);
	//checkRuntime(cudaDeviceSynchronize()); 

	// sort by conf
	for (size_t i = 0; i < batchsize; i++)
	{
		int* p_idx     = idx + i * srcHeight;
		float* p_conf = conf + i * srcHeight;
		thrust::sort_by_key(thrust::device, p_conf, p_conf + srcHeight, p_idx, thrust::greater<float>());
	}

	nms_sort_kernel << < grid_size, block_size, 0, nullptr >> > (
                topK, batchsize, iou_thresh,
                src, srcWidth, srcHeight, srcArea, idx);
}