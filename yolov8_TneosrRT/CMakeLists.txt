cmake_minimum_required(VERSION 2.6)

project(yolov8)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# cudnn
#include_directories(/usr/local/cuda/cudnn/v8.0.4/include)
#link_directories(/usr/local/cuda/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/home/uisee/disk/TensorRT-7.2.3.4/include)
link_directories(/home/uisee/disk/TensorRT-7.2.3.4/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fsanitize=address -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA_ADD_EXECUTABLE(yolov7 ${PROJECT_SOURCE_DIR}/yolox.cpp ${PROJECT_SOURCE_DIR}/plugin_cuda_functions.cu)
# CUDA_ADD_EXECUTABLE(yolov8 ${PROJECT_SOURCE_DIR}/src/main.cpp  ${PROJECT_SOURCE_DIR}/src/yolov8.cpp ${PROJECT_SOURCE_DIR}/src/yolov8_decode.cu)
CUDA_ADD_EXECUTABLE(yolov8 ${PROJECT_SOURCE_DIR}/src/main.cpp  ${PROJECT_SOURCE_DIR}/src/yolov8.cpp)
target_link_libraries(yolov8 nvinfer)
target_link_libraries(yolov8 cudart)
target_link_libraries(yolov8 ${OpenCV_LIBS} )

add_definitions(-O2 -pthread)