#include "../include/smoke.h"
#include "./plugins/modulated_deform_conv/trt_modulated_deform_conv.hpp"

// #define ONXX2TRT

int main(int argc, char **argv)
{
    const std::string img_path = "../kitti_000008.png";
    // const std::string img_path = "../000002.png";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }

    cv::Mat intrinsic = (cv::Mat_<float>(3, 3) << 
        721.5377, 0.0, 609.5593, 0.0, 721.5377, 172.854, 0.0, 0.0, 1.0);
    std::string onnx_path = "../smoke_dla34.onnx";
    std::string engine_path = "../smoke_dla34.engine";

    std::ifstream f(engine_path.c_str());
    bool engine_file_exist = f.good();
    std::cout << "engine_file_exist: " << engine_file_exist << std::endl;
    Smoke_detector smoke_detector_instance;

#ifdef ONNX2TRT
    smoke_detector_instance.LoadOnnx(onnx_path);
#else
    smoke_detector_instance.init_context(engine_path);
    for (int i = 0; i < 100; i++)
    {
        smoke_detector_instance.do_detection(image, intrinsic);
    }
#endif

    return 0;
}