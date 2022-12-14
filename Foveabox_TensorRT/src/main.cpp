#include<iostream>
#include<string>
#include <chrono>
#include <gflags/gflags.h>
#include "../include/FoveaBox.h"

int main(int argc, char** argv)
{
    const std::string input_img_path = "../1.jpg";
    const std::string model_path = "../fovea_dfnet_yolofpn_foveaplus.engine";
    cv::Mat image = cv::imread(input_img_path);

    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    FoveaBox* FoveaBox_instance = new FoveaBox(model_path);
    
    /* End-to-end infer */
    cv::Mat dst = image.clone();
    FoveaBox_instance->do_inference(image, dst);

    std::string save_path = "./result.jpg";
    cv::imwrite(save_path, dst);


    if(FoveaBox_instance)
    {
        delete FoveaBox_instance;
        FoveaBox_instance = nullptr;
    }

    return 0;
}