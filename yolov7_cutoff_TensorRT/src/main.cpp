#include<iostream>
#include<string>
#include <chrono>
#include <gflags/gflags.h>

/* The Only header file for yolov7 inference */
#include "../include/yolov7.h"

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
//     const std::string model_path = "../yolov7_cutoff.engine";
//     Yolov7_Detector* yolov7_instance = new Yolov7_Detector(model_path);
//     const std::string cocodataset_path = "/home/uisee/disk/coco";
//     const std::string cocoval_path = cocodataset_path + "/val2017";
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
//         yolov7_instance->do_detection(img);
//         auto end = std::chrono::system_clock::now();
//         totaltime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//         std::string dst_img_path = result_dir + "/" + std::to_string(count) + ".jpg";
//         count++;
//         // cv::imwrite(dst_img_path, dst);
//     }
//     std::cout<<"Inference Time:"<<totaltime / 5000.<<" ms"<<std::endl;
//     std::cout<<"Done!"<<std::endl;
//     if(yolov7_instance)
//     {
//         delete yolov7_instance;
//         yolov7_instance = nullptr;
//     }
//     return 0;
// }

int main(int argc, char **argv)
{
    const std::string img_path = "../bus.jpg";
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    const std::string model_path = "../yolov7_cutoff.engine";
    Yolov7_Detector* yolov7_instance = new Yolov7_Detector(model_path);
    yolov7_instance->do_detection(image);
    
    if(yolov7_instance)
    {
        delete yolov7_instance;
        yolov7_instance = nullptr;
    }
    return 0;
}