#include "../include/ldrn.h"

int main(int argc, char** argv)
{
    const std::string img_path = "../dashcam_00.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../ldrn_kitti_resnext101_pretrained_data_grad_256x512.engine";
    Ldrn_detector* ldrn_instance = new Ldrn_detector(model_path);
    ldrn_instance->do_detection(image);

    std::string save_path = "./ufld.jpg";
    cv::imwrite(save_path, image);

    if(ldrn_instance)
    {
        delete ldrn_instance;
        ldrn_instance = nullptr;
    }
    return 0;
}