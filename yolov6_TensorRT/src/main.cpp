#include <chrono>
#include <gflags/gflags.h>

/* The Only header file for yolox inference */
#include "../include/yolov6.h"

const int INPUT_H = 640;
const int INPUT_W = 640;

int main(int argc, char** argv){
    const std::string img_path = "../image1.jpg";
    const std::string model_path = "../yolov6s_simplify.engine";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    YOLOV6* yolov6_instance = new YOLOV6(INPUT_H,
                                        INPUT_W,
                                        model_path);
    cv::Mat dst = image.clone();
    yolov6_instance->do_inference(image, dst);

    std::string save_path = "./yolov6.jpg";
    cv::imwrite(save_path, dst);

    if(yolov6_instance)
    {
        delete yolov6_instance;
        yolov6_instance = nullptr;
    }
    return 0;
}