#include "../include/FoveaBox.h"
#include <chrono>
#include <iostream>
#include <fstream> 
#include "cuda_kernel/plugin_cuda_function.h"

static const int DEVICE  = 0;


FoveaBox::FoveaBox(const std::string& _engine_file):
    engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<mt_input_h<<"x"<<mt_input_w<<"] constructed"<<std::endl;
}

void FoveaBox::meshgrid(const int x_start, const int x_end,
                const int y_start, const int y_end,
                cv::Mat &X, cv::Mat &Y, const float grid_size)
{
    std::vector<float> t_x, t_y;
    for (int i = x_start; i < x_end; i++)
    {
        t_x.push_back(i + grid_size);
    }
    for (int j = y_start; j < y_end; j++)
    {
        t_y.push_back(j + grid_size);
    }
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

void FoveaBox::get_points()
{
    for(int i = 0; i < num_stages; ++i)
    {
        featmap_sizes[i][0] = int(mt_input_h / float(strides[i]) + 0.5);
        featmap_sizes[i][1] = int(mt_input_w / float(strides[i]) + 0.5);
        std::cout<<"featmap_sizes["<<i<<"][0]: "<<featmap_sizes[i][0]<<\
            " , featmap_sizes["<<i<<"][1]: "<<featmap_sizes[i][1]<<std::endl;
    }
    std::vector<cv::Mat> grid_x, grid_y;
    for (int i = 0; i < num_stages; i++)
    {
        int x_start, x_end, y_start, y_end;
        x_start = 0;
        x_end = featmap_sizes[i][1];
        y_start = 0;
        y_end = featmap_sizes[i][0];
        float grid_size = 0.5;

        meshgrid(x_start, x_end, y_start, y_end,
                 _grid_x[i], _grid_y[i], grid_size);
    }
}

FoveaBox::~FoveaBox()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<mt_input_h<<"x"<<mt_input_w<<"]"<<std::endl;
}

void FoveaBox::init_context()
{
    get_points();

    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    /* hard code: classes */
    _object_classes.reserve(8);
    _object_classes = {
            "car" , "person", "bicycle_move" , \
            "bicycle_stop", "motor_move", \
            "motor_stop", "tricycle", "cone"};
    input_buffer_size = mt_batch_size * mt_input_c * mt_input_w * mt_input_h * sizeof(float);
    PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));
    for(int i = 0;i < num_stages; ++i)
    {
        det_output_cls_buffer_size[i] = mt_batch_size *
                                         featmap_sizes[i][0] * featmap_sizes[i][1] * mt_det_cls * sizeof(float);
        det_output_reg_buffer_size[i] = mt_batch_size *
                                         featmap_sizes[i][0] * featmap_sizes[i][1] * mt_det_reg * sizeof(float);
        std::cout<<"_det_output_cls_buffer_size["<<i<<"]:"<<det_output_cls_buffer_size[i]<<std::endl;
        std::cout<<"_det_output_reg_buffer_size["<<i<<"]:"<<det_output_reg_buffer_size[i]<<std::endl;
        det_output_cls_index[i] = engine->getBindingIndex(mt_cls_output_name[i].c_str());
        det_output_reg_index[i] = engine->getBindingIndex(mt_reg_output_name[i].c_str());
        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_reg_cpu[i], \
                                            det_output_reg_buffer_size[i],\
                                            cudaHostAllocDefault));
        PERCEPTION_CUDA_CHECK(cudaHostAlloc((void**)&det_output_cls_cpu[i], \
                                            det_output_cls_buffer_size[i],\
                                            cudaHostAllocDefault));
        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_reg_index[i]],\
                                        det_output_reg_buffer_size[i]));
        PERCEPTION_CUDA_CHECK(cudaMalloc(&device_buffers[det_output_cls_index[i]],\
                                        det_output_cls_buffer_size[i]));
    }
    PERCEPTION_CUDA_CHECK(cudaStreamCreate(&stream));
}

void FoveaBox::destroy_context()
{
    bool cudart_ok = true;

    /* Release TensorRT */
    if(context)
    {
        context->destroy();
        context = nullptr;
    }

    if(engine)
    {
        engine->destroy();
        engine = nullptr;
    }

    if(stream) cudaStreamDestroy(stream);

    if(_det_bboxes_tmp) delete[] _det_bboxes_tmp;
    if(_det_scores_tmp) delete[] _det_scores_tmp;
    if(_det_labels_tmp) delete[] _det_labels_tmp;
    if(_det_bboxes)     delete[] _det_bboxes;
    if(_det_scores)     delete[] _det_scores;
    if(_det_labels)     delete[] _det_labels;
    if(_det_results)    delete[] _det_results;

    if(device_buffers[input_index])
        PERCEPTION_CUDA_CHECK(cudaFree(device_buffers[input_index]));
    for (int i = 0; i < num_stages; i++)
    {
        if(det_output_reg_cpu[i]) 
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_reg_cpu[i]));
        if(det_output_cls_cpu[i])
            PERCEPTION_CUDA_CHECK(cudaFreeHost(det_output_cls_cpu[i]));
    }
}

void FoveaBox::topk(int* topk_inds, const float* src, const int len, const unsigned int k)
{
    auto cmp = [](std::pair<float, int>& lhs, std::pair<float, int>& rhs) {
        return lhs.first > rhs.first ||
               (lhs.first == rhs.first && lhs.second < rhs.second);
    };

    /**
     * Build a min-heap, the heap element is pair of (value, idx)
     * the top of the heap is the smallest value
     */
    std::priority_queue<
            std::pair<float, int>,
            std::vector<std::pair<float, int>>,
            decltype(cmp)>
            p_queue(cmp);

    /**
     * Maintain the size of heap to be less or equal to k, so the
     * heap will hold the k largest values
     */
    for (int i = 0; i < len; ++i)
    {
        const auto value = src[i];
        if (p_queue.size() < k || value > p_queue.top().first)
        {
            p_queue.push(std::make_pair(value, i));
        }
        if (p_queue.size() > k)
        {
            p_queue.pop();
        }
    }

    /* store topk index */
    int j = 0;
    while (!p_queue.empty())
    {
        topk_inds[j++] = p_queue.top().second;
        p_queue.pop();
    }
}


void FoveaBox::pre_process(cv::Mat& img)
{
    cv::Size input_geometry(mt_input_w, mt_input_h);
    std::cout<<"img.size:"<<img.size()<<std::endl;
    cv::Mat img_resize;
    cv::resize(img, img_resize, input_geometry, 0, 0, cv::INTER_NEAREST);
    std::cout<<"img_resize.size:"<<img_resize.size()<<std::endl;
    // _img_gpu.upload(img, _gpu_stream);
    _img_gpu.upload(img_resize, _gpu_stream);
    std::cout<<"_img_gpu.size:"<<_img_gpu.size()<<std::endl;
}

void FoveaBox::nms(const int bbox_num, std::vector<int>& kept_idx)
{
    if (bbox_num <= 0) return;

    const float* scores = _det_scores_tmp;
    const float* bboxes = _det_bboxes_tmp;
    const int* labels   = _det_labels_tmp;

    auto overlap_1d = [](float x1_min, float x1_max, float x2_min, float x2_max) 
            -> float
    {
        if (x1_min > x2_min)
        {
            std::swap(x1_min, x2_min);
            std::swap(x1_max, x2_max);
        }
        return x1_max < x2_min ? 0 : std::min(x1_max, x2_max) - x2_min;
    };

    auto compute_IoU = [&overlap_1d](const float* bbox1, const float* bbox2) 
            -> float
    {
        float overlap_x = overlap_1d(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlap_y = overlap_1d(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap_2d = overlap_x * overlap_y;
        float u = area1 + area2 - overlap_2d;
        return u == 0 ? 0 : overlap_2d / u;
    };

    /* create a (score, index) pair */
    std::vector<std::pair<float, int>> score_idx(bbox_num);
    for (int i = 0; i < bbox_num; i++)
    {
        score_idx.at(i) = std::make_pair(scores[i], i);
    }

    std::stable_sort(score_idx.begin(), score_idx.end(),
                     [](const std::pair<float, int>& pair1,
                        const std::pair<float, int>& pair2){
                         return pair1.first > pair2.first; }); /* descending order */
    
    while (score_idx.size() > 0)
    {
        // std::cout<<"score_idx.size():"<<score_idx.size()<<std::endl;
        /* only store nms_max_out scores, labels and bboxes */
        if (kept_idx.size() >= static_cast<size_t>(mt_nms_max_out)) break;
        
        int base_idx = score_idx.cbegin()->second; /* index of max score */
        kept_idx.push_back(base_idx);
        /* store num of bboxes per class */
        // _kept_bbox_num_per_cls[std::min(7, labels[base_idx])] ++;

        /**
         * calculate IoU between proposal bboxes(including current bbox) and
         * current bbox
         */
        for (auto iter = score_idx.cbegin(); iter != score_idx.cend();)
        {
            int loop_idx = iter->second;
            float overlap = compute_IoU(bboxes + loop_idx * 4, bboxes + base_idx * 4);
            // std::cout << " test error: " << loop_idx << " d " << base_idx << "overlap " << overlap << std::endl;
            // std::cout << *(bboxes + base_idx * 4) << " " << *(bboxes + base_idx * 4 + 1) << " " << *(bboxes + base_idx * 4 +2) << " " << *(bboxes + base_idx * 4 +3) << std::endl;
            if (overlap > mt_iou_thr)
            {
                /* erase proposal bbox and return the next iterator */
                iter = score_idx.erase(iter);
            }
            else
            {
                iter ++;
            }
        }
    }
}

void FoveaBox::postprocess()
{
    for (int i = 0; i < num_stages; i++)
    {
        get_mt_output((const float*)device_buffers[det_output_reg_index[i]],
                     (const float*)device_buffers[det_output_cls_index[i]],
                     mt_batch_size,
                     mt_det_reg,
                     mt_det_cls,
                     featmap_sizes[i][0],
                     featmap_sizes[i][1],
                     (float*)device_buffers[det_output_reg_index[i]],
                     (float*)device_buffers[det_output_cls_index[i]]);
    }
    for (int i = 0; i < num_stages; i++)
    {
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_reg_cpu[i]), device_buffers[det_output_reg_index[i]],
                                              det_output_reg_buffer_size[i], cudaMemcpyDeviceToHost, stream));
        PERCEPTION_CUDA_CHECK(cudaMemcpyAsync((det_output_cls_cpu[i]), device_buffers[det_output_cls_index[i]],
                                              det_output_cls_buffer_size[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    _det_bboxes_tmp = new float[mt_nms_pre * num_stages * 4];
    _det_scores_tmp = new float[mt_nms_pre * num_stages];
    _det_labels_tmp = new int[mt_nms_pre * num_stages];
    _det_bboxes = new float[mt_nms_max_out * 4];
    _det_scores = new float[mt_nms_max_out];
    _det_labels = new int[mt_nms_max_out];
    /* FIXME: hard code: the det_cls is 7, we add 1 to 8 because tracking cls num is fixed to 8. */
    // _kept_bbox_num_per_cls = new int[mt_batch_size * 8];
    _det_results = new float[mt_batch_size *
                             (1 + (8) + mt_nms_max_out * (16 + 1))];

    int input_w = mt_input_w;
    int input_h = mt_input_h;
    int background_label = mt_background_label;
    int cls_channels = mt_det_cls;
    int nms_pre = mt_nms_pre;
    int next_idx = 0;/* record num of processed bboxes */
    int valid_bbox_num = 0;

    /* det out class - human, car*/
    /* track class = car human*/
    std::unordered_map<int, int> mappings = {{0, 1}, {1, 0}};
    for(int level = 0; level < num_stages; ++level)
    {
        float* cls_score = det_output_cls_cpu[level];
        float* bbox_pred = det_output_reg_cpu[level];
        int point_num = featmap_sizes[level][0] * featmap_sizes[level][1];
        int stride = strides[level];
        int base_len = mt_base_len[level];

        /* to be recorded bboxes, scores, labels, HWC format */
        float* next_det_bboxes = _det_bboxes_tmp + next_idx * 4;
        float* next_det_scores = _det_scores_tmp + next_idx;
        int*   next_det_labels = _det_labels_tmp + next_idx;
        float* x = reinterpret_cast<float*>(_grid_x[level].data);
        float* y = reinterpret_cast<float*>(_grid_y[level].data);

        /* find max score cross channel-wise and keep valid results */
        float* max_scores = new float[point_num];
        float* labels     = new float[point_num];
        std::vector<int> valid_inds;

        // std::cout<<"point_num:"<<point_num<<std::endl;

        for(int p = 0; p < point_num; ++p)
        {
            float max_value = -100;
            for(int c = 0; c < cls_channels; ++c)
            {
                /* for sigmoid classifier, background_label = -1, no need to skip */
                /* for softmax classifier, only skip cls_score of background */
                if(c != mt_background_label && cls_score[p + c * point_num] > max_value)
                {
                    max_value = cls_score[p + c * point_num];
                    int real_label = (c > background_label) ? c - 1 : c;
                    labels[p] = mappings[real_label];
                }
            }
            if(max_value > mt_score_thr)
            {
                valid_inds.push_back(p);
            }
            max_scores[p] = max_value;
        }
        // std::cout<<"valid_inds.size:"<<valid_inds.size()<<std::endl;
        int num_valid = static_cast<int>(valid_inds.size());
        if(nms_pre > 0 && nms_pre < num_valid)
        {
            /* get top k index where k = nms_pre */
            int* topk_inds = new int[nms_pre];
            topk(topk_inds, max_scores, num_valid, nms_pre);

            /* store bboxes[x1,y1,x2,y2,...] and scores */
            for (int i = 0; i < nms_pre; i++)
            {
                next_det_bboxes[i * 4]     = clamp(stride * x[topk_inds[i]] - base_len * bbox_pred[topk_inds[i]], 0.f, static_cast<float>(input_w - 1));
                next_det_bboxes[i * 4 + 1] = clamp(stride * y[topk_inds[i]] - base_len * bbox_pred[topk_inds[i] + point_num], 0.f, static_cast<float>(input_h - 1));
                next_det_bboxes[i * 4 + 2] = clamp(stride * x[topk_inds[i]] + base_len * bbox_pred[topk_inds[i] + point_num * 2], 0.f, static_cast<float>(input_w - 1));
                next_det_bboxes[i * 4 + 3] = clamp(stride * y[topk_inds[i]] + base_len * bbox_pred[topk_inds[i] + point_num * 3], 0.f, static_cast<float>(input_h - 1));
                /* scores are stored in ascending order */
                next_det_scores[i] = max_scores[topk_inds[i]];
                next_det_labels[i] = labels[topk_inds[i]];
            }
            next_idx += nms_pre; /* update num of recorded bboxes */
            delete [] topk_inds;
        }
        else if (nms_pre >= num_valid)
        {
            for (int i = 0; i < num_valid; i++)
            {
                next_det_bboxes[i * 4]     = clamp(stride * x[valid_inds[i]] - base_len * bbox_pred[valid_inds[i]], 0.f, static_cast<float>(input_w - 1));
                next_det_bboxes[i * 4 + 1] = clamp(stride * y[valid_inds[i]] - base_len * bbox_pred[valid_inds[i] + point_num], 0.f, static_cast<float>(input_h - 1));
                next_det_bboxes[i * 4 + 2] = clamp(stride * x[valid_inds[i]] + base_len * bbox_pred[valid_inds[i] + point_num * 2], 0.f, static_cast<float>(input_w - 1));
                next_det_bboxes[i * 4 + 3] = clamp(stride * y[valid_inds[i]] + base_len * bbox_pred[valid_inds[i] + point_num * 3], 0.f, static_cast<float>(input_h - 1));
                next_det_scores[i] = max_scores[valid_inds[i]];
                next_det_labels[i] = labels[valid_inds[i]];
            }
            next_idx += num_valid;
        }
        else
        {
            std::cerr<< "nms_pre parameter is illegal!" << std::endl;
            std::abort();
        }

        delete [] max_scores;
        delete [] labels;
        valid_bbox_num += valid_inds.size();
        valid_inds.clear();
    }
    /* nms */
    std::vector<int> kept_idx;
    /* FIXME: hard code: the det_cls is 7, we add 1 to 8 because tracking cls num is fixed to 8. */
    // memset(_kept_bbox_num_per_cls, 0, 8 * sizeof(int));
    // std::cout<<"valid_bbox_num: "<<valid_bbox_num<<std::endl;
    nms(valid_bbox_num, kept_idx);

    std::vector<std::pair<int, int>> label_idx;
    // memset(_kept_bbox_num_per_cls, 0, 8 * sizeof(int));

    _kept_num = kept_idx.size(); /* store num of total bboxes */
    // cout<<"_kept_num:"<<_kept_num<<endl;
    // for(int kept_num=0;kept_num<_kept_num;++kept_num){
    //     std::cout<<"kept_idx["<<kept_num<<"]: "<<kept_idx[kept_num]<<std::endl;
    // }
    /* get index of car */
    std::vector<int> car_idx;
    for (int i = 0; i < _kept_num; i++)
    {
        int j = kept_idx[i];
        int label = _det_labels_tmp[j]; // label in tracker
        // std::cout<<"label: "<<label<<std::endl;
        if (label == 0)
        {
            car_idx.push_back(j);
        }
    }
    for (int i = 0; i < _kept_num; i++)
    {
        int j = kept_idx[i];
        int label = _det_labels_tmp[j];
        /* filter out bboxes of person inside car */
        bool record = true;
        if (label == 1)
        {
            float x = (_det_bboxes_tmp[j * 4] + _det_bboxes_tmp[j * 4 + 2]) / 2;
            float y = _det_bboxes_tmp[j * 4 + 3];
            for (size_t k = 0; k < car_idx.size(); k++)
            {
                int idx = car_idx[k];
                float x1 = _det_bboxes_tmp[idx * 4];
                float y1 = _det_bboxes_tmp[idx * 4 + 1];
                float x2 = _det_bboxes_tmp[idx * 4 + 2];
                float y2 = _det_bboxes_tmp[idx * 4 + 3];
                /* hard code, filter out bboxes of person_in_car */
 
                float person_in_car_bottom_border = y2;
                if ((x1 < x && x < x2) && (y1 < y && y < person_in_car_bottom_border))
                {
                    record = false;
                    break;
                }
            }
        }

        if (record == true)
        {
            label_idx.push_back(std::make_pair(label, j));
            /* store num of bboxes per class */
            // _kept_bbox_num_per_cls[std::min(7, label)]++;
        }
    }
    _kept_num = label_idx.size(); // update _kept_num 

    std::stable_sort(label_idx.begin(), label_idx.end(),
                     [](const std::pair<int, int>& pair1,
                        const std::pair<int, int>& pair2){
                         return pair1.first <= pair2.first; }); /* ascending order */
    // cout<<"_kept_num:"<<_kept_num<<endl;
    /* store results */
    for (int i = 0; i < _kept_num; i++)
    {
        int j = label_idx[i].second;
        _det_scores[i] = _det_scores_tmp[j];
        _det_labels[i] = _det_labels_tmp[j];
        _det_bboxes[i * 4] = _det_bboxes_tmp[j * 4];
        _det_bboxes[i * 4 + 1] = _det_bboxes_tmp[j * 4 + 1];
        _det_bboxes[i * 4 + 2] = _det_bboxes_tmp[j * 4 + 2];
        _det_bboxes[i * 4 + 3] = _det_bboxes_tmp[j * 4 + 3];
    }
}

void FoveaBox::printResultsDet(cv::Mat& img)
{
    int x1, y1, x2, y2;
    float score;
    int label;
    // cout<<"_kept_num:"<<_kept_num<<endl;
    for (int i = 0; i < _kept_num; i++)
    {
        x1 = _det_bboxes[i * 4];
        y1 = _det_bboxes[i * 4 + 1];
        x2 = _det_bboxes[i * 4 + 2];
        y2 = _det_bboxes[i * 4 + 3];
        score = _det_scores[i];
        label = std::min(7, _det_labels[i]);

        /* useless with transfer_label function, but keep for now */
        if (img.size()!=cv::Size(1280, 720))
        {
            cv::rectangle(img, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)),
                          cv::Scalar(0,0,255), 1.7, 1, 0);
            cv::putText(img,
                        _object_classes[label] + ": " + std::to_string(score).substr(0,5),
                        cv::Point(int((x1-2)), int((y1-2))), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0x00,0xD7,0xFF), 1, 8, false);
        }
        else
        {
            cv::rectangle(img, cv::Point(int(x1*1280.0/672), int(y1*720.0/384)), cv::Point(int(x2*1280.0/672.0), int(y2*720.0/384.0)),
                          cv::Scalar(0,0,255), 1.7, 1, 0);
            cv::putText(img,
                        _object_classes[label] + ": " + std::to_string(score).substr(0,5),
                        cv::Point(int((x1-2)*1280.0/672), int((y1-2)*720.0/384.0)), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0x00,0xD7,0xFF), 1, 8, false);
        }
        std::cout<<_object_classes[label]<<" "<<x1<<" "<<x2<<" "<<y1<<" "<<y2<<" "<<score<<std::endl;
    }
}

void FoveaBox::do_inference(cv::Mat& image, cv::Mat& dst)
{
    assert(context != nullptr);

    pre_process(image);
    write_img_data_to_buffer(_img_gpu, (float*)device_buffers[input_index], mean_std, false);
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    postprocess();
    printResultsDet(dst);
}