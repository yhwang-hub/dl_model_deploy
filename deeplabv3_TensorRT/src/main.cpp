#include<iostream>
#include<string>
#include <chrono>
#include <gflags/gflags.h>
#include "../include/deeplabv3.h"

int main(int argc, char **argv)
{
    const std::string img_path = "../demo.png";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../deeplabv3.engine";
    deeplabv3_Detector* deeplabv3_instance = new deeplabv3_Detector(model_path);
    deeplabv3_instance->do_detection(image);

    if(deeplabv3_instance)
    {
        delete deeplabv3_instance;
        deeplabv3_instance = nullptr;
    }
    return 0;
}