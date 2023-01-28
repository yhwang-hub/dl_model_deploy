#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include <vector>
#include "kernel.h"
#include "common.h"

/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
*/
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

class PostProcessCuda {
  private:
    Params params_;
    float *anchors_;
    float *anchor_bottom_heights_;
    int *object_counter_;
    cudaStream_t stream_ = 0;
  public:
    PostProcessCuda(cudaStream_t stream = 0);
    ~PostProcessCuda();

    int doPostprocessCuda(const float *cls_input, float *box_input, const float *dir_cls_input, float *bndbox_output);
};

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred);

#endif
