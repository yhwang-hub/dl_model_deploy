#include "../include/lane-detector.h"

int main(int argc, char** argv)
{
    const std::string img_path = "../dashcam_00.jpg";
    // const std::string img_path = "../1.jpg";
    // const std::string img_path = "../04500.jpg";
    // const std::string img_path = "../00660.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../ufldv2_culane_res18_320x1600.engine";
    Lane_detector* lane_det_instance = new Lane_detector(model_path);
    lane_det_instance->do_detection(image);

    std::string save_path = "./ufldv2.jpg";
    cv::imwrite(save_path, image);

    if(lane_det_instance)
    {
        delete lane_det_instance;
        lane_det_instance = nullptr;
    }
    return 0;
}