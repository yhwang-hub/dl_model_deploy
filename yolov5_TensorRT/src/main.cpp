#include <chrono>
#include <gflags/gflags.h>

/* The Only header file for yolox inference */
#include "../include/yolov5.h"

const int INPUT_H = 640;
// const int INPUT_W = 640;
const int INPUT_W = 480;

int main(int argc, char** argv){
    const std::string img_path = "../bus.jpg";
    // const std::string img_path = "../1.jpg";
    const std::string model_path = "../yolov5s.engine";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    YOLOV5* yolov5_instance = new YOLOV5(INPUT_H,
                                        INPUT_W,
                                        model_path);
    yolov5_instance->do_inference(image);

    if(yolov5_instance)
    {
        delete yolov5_instance;
        yolov5_instance = nullptr;
    }
    return 0;
}