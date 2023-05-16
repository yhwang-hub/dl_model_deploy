#include "../include/yolo_nas_detector.h"

int test_image()
{
    // const std::string img_path = "../dog.jpg";
    const std::string img_path = "../bus.jpg";
    // const std::string img_path = "../demo.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../yolo-nas-s_fp16.trt";
    // const std::string model_path = "../yolo-nas-s_fp32.trt";
    yolo_nas_detector* yolo_nas_instance = new yolo_nas_detector(model_path);
    yolo_nas_instance->do_detection(image);

    std::string save_path = "./yolo_nas_result.jpg";
    cv::imwrite(save_path, image);

    if(yolo_nas_instance)
    {
        delete yolo_nas_instance;
        yolo_nas_instance = nullptr;
    }
    return 0;
}

int main(int argc, char **argv)
{
    test_image();
    return 0;
}