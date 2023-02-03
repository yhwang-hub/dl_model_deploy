#include<iostream>
#include<string>
#include <chrono>
#include <gflags/gflags.h>
#include "../include/seaformer.h"

int main(int argc, char **argv)
{
    const std::string img_path = "../demo.png";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../SeaFormer-S_1024x1024_1x8_160k_Cityscapes_sim_fp16.engine";
    seaformer_Detector* seaformer_instance = new seaformer_Detector(model_path);
    seaformer_instance->do_detection(image);

    if(seaformer_instance)
    {
        delete seaformer_instance;
        seaformer_instance = nullptr;
    }
    return 0;
}