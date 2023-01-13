#include "NvInfer.h"
#include "common.hpp"
#include <string>

int main()
{
    // std::string onnx_file = "../../yolox_s_sim_modify.onnx";
    std::string onnx_file = "../../yolox_s_0914_sim_modify.onnx";
    //std::string engine_file = "../yolox_s.trt";
    // std::string engine_file = "../../yolox_s_sim_modify.engine";
    std::string engine_file = "../../yolox_s_0914_sim_modify.engine";
    nvinfer1::ICudaEngine *engine;
    
    onnxToTRTModel(onnx_file, engine_file, engine, 1);
    assert(engine != nullptr);
    
    return 1;
}
