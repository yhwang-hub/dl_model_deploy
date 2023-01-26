#include "../include/road-seg.h"

int main(int argc, char** argv)
{
    // const std::string img_path = "../000001.png";
    const std::string img_path = "../dashcam_00.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    // const std::string model_path = "../road-segmentation-adas_float32_sim.engine";
    const std::string model_path = "../road-segmentation-adas_sim.engine";
    RoadSeg_detector* roadseg_instance = new RoadSeg_detector(model_path);
    roadseg_instance->do_detection(image);

    std::string save_path = "./roadSeg.jpg";
    cv::imwrite(save_path, image);

    if(roadseg_instance)
    {
        delete roadseg_instance;
        roadseg_instance = nullptr;
    }
    return 0;
}