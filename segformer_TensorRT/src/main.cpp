#include<iostream>
#include<string>
#include <chrono>
#include <gflags/gflags.h>
#include "../include/segformer.h"

int main(int argc, char **argv)
{
    const std::string img_path = "../demo.png";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../segformer.engine";
    segformer_Detector* segformer_instance = new segformer_Detector(model_path);
    segformer_instance->do_detection(image);

    if(segformer_instance)
    {
        delete segformer_instance;
        segformer_instance = nullptr;
    }
    return 0;
}