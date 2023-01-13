#include "../include/yolov8.h"

int main(int argc, char** argv)
{
    const std::string img_path = "../bus.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../yolov8s_640x480.engine";
    // const std::string model_path = "../yolov8s_640x480.engine";
    Yolov8_detector* yolov8_instance = new Yolov8_detector(model_path);
    yolov8_instance->do_detection(image);

    std::string save_path = "./yolov8.jpg";
    cv::imwrite(save_path, image);

    if(yolov8_instance)
    {
        delete yolov8_instance;
        yolov8_instance = nullptr;
    }
    return 0;
}

// void GetFileNames(std::string path, std::vector<std::string>& filenames)
// {
//     DIR *pDir;
//     struct dirent* ptr;
//     if(!(pDir = opendir(path.c_str())))
//         return;
//     while((ptr = readdir(pDir))!=0) {
//         if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
//             filenames.push_back(path + "/" + ptr->d_name);
//     }
//     closedir(pDir);
// }

// int main(int argc, char **argv)
// {
//     const std::string model_path = "../yolov8s_640x480.engine";
//     Yolov8_detector* yolov8_instance = new Yolov8_detector(model_path);
//     const std::string cocodataset_path = "/home/uisee/disk/coco";
//     const std::string cocoval_path = cocodataset_path + "/images/val2017";
//     const std::string result_dir = "../results";
//     std::vector<std::string> files;
//     GetFileNames(cocoval_path, files);
//     std::cout<<files.size()<<std::endl;
//     auto totaltime = 0;
//     int count = 0;
//     for (int i = 0; i < files.size(); ++i)
//     {
//         const std::string input_image_path = files[i];
//         std::cout<<i<< ":" <<files[i].c_str()<<std::endl;
//         cv::Mat img = cv::imread(input_image_path);
//         if(img.empty())
//         {
//             std::cout<<"Input image path wrong!!"<<std::endl;
//             return -1;
//         }
//         cv::Mat dst = img.clone();
//         auto start = std::chrono::system_clock::now();
//         yolov8_instance->do_detection(img);
//         auto end = std::chrono::system_clock::now();
//         totaltime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//         std::string dst_img_path = result_dir + "/" + std::to_string(count) + ".jpg";
//         count++;
//         cv::imwrite(dst_img_path, dst);
//     }
//     std::cout<<"Inference Time:"<<totaltime / 5000.<<" ms"<<std::endl;
//     std::cout<<"Done!"<<std::endl;
//     if(yolov8_instance)
//     {
//         delete yolov8_instance;
//         yolov8_instance = nullptr;
//     }
//     return 0;
// }