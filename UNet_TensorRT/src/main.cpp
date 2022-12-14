#include <chrono>
#include <gflags/gflags.h>

#include "../include/unet.h"

int main(int argc, char** argv)
{
    /* Read image */
    const std::string img_path = "../street.jpg";

    const std::string engine_file_path = "../unet_vgg_voc.engine";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    UNet_Detector* unet_instance = new UNet_Detector(engine_file_path);
    
    /* End-to-end infer */
    unet_instance->do_detection(image);
    
    if(unet_instance)
    {
        delete unet_instance;
        unet_instance = nullptr;
    }
    return 0;
}