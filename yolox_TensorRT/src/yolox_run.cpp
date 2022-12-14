#include <chrono>
#include <gflags/gflags.h>

#include "../include/yolox.h"

int main(int argc, char **argv)
{
    /* Read image */
    const int FLAGS_dump_res = 1;
    const std::string img_path = "/home/uisee/mmdetection/demo/demo.jpg";
    // const std::string img_path = "../20220809-113739.png";
    // const std::string engine_file_path = "/sdb/wyh/mmdetection/yolox_tiny2onnx/yolox_tiny_sim.engine";
    // const std::string engine_file_path = "../yolox_s_sim_modify.engine";
    const std::string engine_file_path = "../yolox_s_sim_modify.engine";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    Yolox_Detector* yolox_instance = new Yolox_Detector(engine_file_path);
    
    /* End-to-end infer */
    yolox_instance->do_detection(image);
    
    if(FLAGS_dump_res)
    {
        std::string save_path = "./yolox.jpg";
        cv::imwrite(save_path, image);
    }
    
    if(yolox_instance)
    {
        delete yolox_instance;
        yolox_instance = nullptr;
    }
    return 0;
}
