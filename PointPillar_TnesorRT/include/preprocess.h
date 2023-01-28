#include "kernel.h"
#include "params.h"

class PreProcessCuda {
  private:
    Params params_;
    unsigned int *mask_;
    float *voxels_;
    int *voxelsList_;
    float *params_cuda_;
    cudaStream_t stream_ = 0;

  public:
    PreProcessCuda(cudaStream_t stream);
    ~PreProcessCuda();

    //points cloud -> voxels (BEV) -> feature*4 
    int generateVoxels(float *points, size_t points_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs);

    //feature*4 -> feature * 10 
    int generateFeatures(float* voxel_features,
          unsigned int *voxel_num,
          unsigned int* voxel_idxs,
          unsigned int *params,
          float* features);
};

