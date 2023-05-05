#include "../include/rt_detr_detector.h"

int test_image()
{
    const std::string img_path = "../dog.jpg";
    // const std::string img_path = "../bus.jpg";
    // const std::string img_path = "../demo.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../rtdetr_r50vd_6x_coco.trt";
    // const std::string model_path = "../yolov8s_640x480.engine";
    rt_detr_detector* rt_detr_instance = new rt_detr_detector(model_path);
    rt_detr_instance->do_detection(image);

    std::string save_path = "./rt_detr_result.jpg";
    cv::imwrite(save_path, image);

    if(rt_detr_instance)
    {
        delete rt_detr_instance;
        rt_detr_instance = nullptr;
    }
    return 0;
}

int main(int argc, char **argv)
{
    // test_coco();
    test_image();
    return 0;
}