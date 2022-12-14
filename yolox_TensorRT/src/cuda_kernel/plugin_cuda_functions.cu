#include "plugin_cuda_function.h"

__global__ void get_det_output_kernel(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int input_h, const int input_w, const int stage_w, const int stage_h,
                        const int origin_h, const int origin_w, const int max_out_obj,
                        const int stride, float scale, const int numThreads, const float confThreshold,
                        float* output)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numThreads)
        return;

    int row = (idx / stage_w) % stage_h;
    int col = idx % stage_w;
    int area = stage_w * stage_h;
    int cls_id = idx / stage_w / stage_h;

    float obj = obj_data[row * stage_w + col];
    obj = 1 / (1 + expf(-obj));
    // if (obj < confThreshold) return;
    
    float x_feat = bbox_data[row * stage_w + col];
    float y_feat = bbox_data[area + (row * stage_w + col)];
    float w_feat = bbox_data[area * 2 + (row * stage_w + col)];
    float h_feat = bbox_data[area * 3 + (row * stage_w + col)];

    float x_center = (x_feat + col) * stride;
    float y_center = (y_feat + row) * stride;
    float w = expf(w_feat) * stride;
    float h = expf(h_feat) * stride;

    float cls_feat = cls_data[idx];
    cls_feat = 1 / (1 + expf(-cls_feat));
    float score = cls_feat * obj;
    if(score < confThreshold)
        return;

    float left = (x_center - 0.5 * w) / scale;
    float top = (y_center - 0.5 * h ) / scale;
    float ww = (w / scale);
    float hh = (h / scale);

    float right = left + ww;
    float bottom = top + hh;

    /* clip */
    left = fmaxf(0, left);
    left = fminf(left, (float)origin_w);
    top = fmaxf(0, top);
    top = fminf(top, (float)origin_h);
    right = fmaxf(0, right);
    right = fminf(right, (float)origin_w);
    bottom = fmaxf(0, bottom);
    bottom = fminf(bottom, (float)origin_h);

    float* res_count = output;
    int count = (int)atomicAdd(res_count, 1);
    if(count >= max_out_obj)
        return;

    char* data = (char* )res_count + sizeof(float) + count * sizeof(Box);
    Box* det = (Box*)(data);

    det->left = left;
    det->top = top;
    det->right = right;
    det->bottom = bottom;
    det->confidence = score;
    det->label = cls_id;
}

__device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

extern "C" void get_det_output(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int origin_h, const int origin_w, const int max_out_obj,
                        const int input_h, const int input_w, const int stride, float scale,
                        const float confThreshold, const float nmsThreshold, const int NUM_BOX_ELEMENT,
                        float* output, cudaStream_t stream)
{
    int stage_h = (int) input_h / stride;
    int stage_w = (int) input_w / stride;
    const int numThreads = batchsize * stage_h * stage_w * det_cls_len;
    const int numBlocks = DIV_THEN_CEIL(numThreads, CUDA_NUM_THREADS);

    get_det_output_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(cls_data, obj_data, bbox_data,
                        batchsize,  det_obj_len, det_bbox_len, det_cls_len,
                        input_h, input_w, stage_w, stage_h,
                        origin_h, origin_w, max_out_obj,
                        stride, scale, numThreads, confThreshold,
                        output);
}


__global__ void gpu_decode_kernel(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int input_h, const int input_w, const int stage_w, const int stage_h,
                        const int origin_h, const int origin_w, const int max_out_obj, const int NUM_BOX_ELEMENT,
                        const int stride, float scale, const int numThreads, const float confThreshold,
                        float* output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numThreads)
        return;
    
    int row = (idx / stage_w) % stage_h;
    int col = idx % stage_w;
    int area = stage_w * stage_h;
    int cls_id = idx / stage_w / stage_h;
    float obj = obj_data[row * stage_w + col];
    obj = 1 / (1 + expf(-obj));

    float cls_feat = cls_data[idx];
    cls_feat = 1 / (1 + expf(-cls_feat));
    float confidence = cls_feat * obj;
    if(confidence < confThreshold)
        return;
    
    float x_feat = bbox_data[row * stage_w + col];
    float y_feat = bbox_data[area + (row * stage_w + col)];
    float w_feat = bbox_data[area * 2 + (row * stage_w + col)];
    float h_feat = bbox_data[area * 3 + (row * stage_w + col)];

    float x_center = (x_feat + col) * stride;
    float y_center = (y_feat + row) * stride;
    float w = expf(w_feat) * stride;
    float h = expf(h_feat) * stride;

    float left = (x_center - 0.5 * w) / scale;
    float top = (y_center - 0.5 * h ) / scale;
    float ww = (w / scale);
    float hh = (h / scale);

    float right = left + ww;
    float bottom = top + hh;

    /* clip */
    left = fmaxf(0, left);
    left = fminf(left, (float)origin_w);
    top = fmaxf(0, top);
    top = fminf(top, (float)origin_h);
    right = fmaxf(0, right);
    right = fminf(right, (float)origin_w);
    bottom = fmaxf(0, bottom);
    bottom = fminf(bottom, (float)origin_h);

    int count = (int)atomicAdd(output, 1);
    if(count >= max_out_obj)
        return;

    float* data = output + 1 + count * NUM_BOX_ELEMENT;
    *data++ = left;
    *data++ = top;
    *data++ = right;
    *data++ = bottom;
    *data++ = confidence;
    *data++ = cls_id;
    *data++ = 1;// 1 = keep, 0 = ignore
}

extern "C" void gpu_decode_nms(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int origin_h, const int origin_w, const int max_out_obj,
                        const int input_h, const int input_w, const int stride, float scale,
                        const float confThreshold, const float nmsThreshold, const int NUM_BOX_ELEMENT,
                        float* output, cudaStream_t stream)
{
    int stage_h = (int) input_h / stride;
    int stage_w = (int) input_w / stride;
    const int numThreads = batchsize * stage_h * stage_w * det_cls_len;
    const int numBlocks = DIV_THEN_CEIL(numThreads, CUDA_NUM_THREADS);

    gpu_decode_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(cls_data, obj_data, bbox_data,
                        batchsize,  det_obj_len, det_bbox_len, det_cls_len,
                        input_h, input_w, stage_w, stage_h,
                        origin_h, origin_w, max_out_obj, NUM_BOX_ELEMENT,
                        stride, scale, numThreads, confThreshold,
                        output);

    auto block = max_out_obj > 512 ? 512 : max_out_obj;
    auto grid = (max_out_obj + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(output, max_out_obj, nmsThreshold, NUM_BOX_ELEMENT);
}


// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix{
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */

    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const Size& from, const Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        // **
        float scale = min(scale_x, scale_y); // 缩放比例辅助视频讲解 https://v.douyin.com/NhrH8Gm/
        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
        scale,     0,      0
        0,     scale,      0
        0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
        1,        0,      -scale * from.width * 0.5
        0,        1,      -scale * from.height * 0.5
        0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
        T = [
        1,        0,      to.width * 0.5,
        0,        1,      to.height * 0.5,
        0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
        scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        /* 
            + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;
        i2d[1] = 0;
        // i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[2] = 0;
        i2d[3] = 0;
        i2d[4] = scale;
        // i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        i2d[5] = 0;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix
){
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr4vTF/
     */
   // 一个thread负责一个像素（3个通道）
    // cuda核函数是并行运行的，计算idx是为了指定某个线程，让某个线程执行以下代码
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 通过线程的idx来判断出执行的线程应该执行哪个图像中的像素点
    const int dx = idx % dst_width;
    const int dy = idx / dst_width;
    // 只有dx和dx在dst_width和dst_height的范围内，才需要往下执行，否者直接return
    if (dx >= dst_width || dy >= dst_height)  return;

    // 将像素点的数值默认设置都为fill_value
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    // 通过仿射变换矩阵的逆变换，可以知道dx和dy在原图中哪里取值
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - 双线性理论讲解：https://v.douyin.com/NhrH2tb/
        - 代码代码：https://v.douyin.com/NhrBqpc/ 
     */
    // 已知src_x和src_y,怎么考虑变换后的像素值----双线性差值
    // 仿射变换的逆变换的src_x和src_y超过范围了
    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }else{
        // 由于resize图中的像素点映射到原图上的，由于映射到原图，得到的像素点可能为非整数，如果求解这个在原图非整数对应的resize图上像素点的数值呢？通过原图非整数像素值周围的四个像素点来确定
        // 因此需要定义y_low、x_low、y_high、x_high
        int y_low = floorf(src_y);//  floorf：求最大的整数，但是不大于原数值
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // const_values[]常量数值，为啥是3个呢？因为一个像素点为3通道
        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        //双线性差值
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;

        // 对于原图上的四个点，如何计算中间非整数的点的像素值呢？---通过双线性插值
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; // 仿射变换的双线性插值的权重求解
        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // 该点的像素值，为啥要加0.5f，为了四舍五入
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    // // mean={0，0，0}，std={255.0，255.0，255.0}
    // c0 = (c0-0)/255.0;
    // c1 = (c1-0)/255.0;
    // c2 = (c2-0)/255.0;

    // // bgrbgrbgr->bbbgggrrr
    // int stride = dst_width*dst_height;
    // dst[dy*dst_width + dx] = c0;
    // dst[stride + dy*dst_width + dx] = c1;
    // dst[stride*2 + dy*dst_width + dx] = c2;

    // bgrbgrbgr->rrrgggbbb
    int stride = dst_width * dst_height;
    dst[dy*dst_width + dx] = c2;
    dst[stride + dy*dst_width + dx] = c1;
    dst[stride*2 + dy*dst_width + dx] = c0;
}

extern "C" void warp_affine_bilinear(
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhre7fV/
     */
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height, 
	float fill_value
){
    // 需要多少threads，启动dst_width*dst_height个线程，是为了让一个线程处理一个像素点
    const int n = dst_width*dst_height;
    // 设置一个块启动的线程个数
    int block_size = 1024;
    // 设置块的个数
    // 为啥要加上block_size-1，这是因为n/block_size有出现有小数的情况，为了向上取整，所以加上了block_size-1
    const int grid_size = (n + block_size - 1) / block_size;

    AffineMatrix affine;
    // 求解仿射变换矩阵---是为了得到原图和resize图的转换矩阵，通过该矩阵可以很方便根据原图来求出reize图中像素点的数值
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));
    // 下面的函数就是核函数,核函数的格式必须包含<<<...>>>
    // 在<<<...>>>中，第一个参数是指定块个数，第二个参数指定一个块中的线程个数，第三个参数是共享内存，第四个参数是stream
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine
    );
}