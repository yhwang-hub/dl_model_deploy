cmake_minimum_required(VERSION 2.6)

project(rtdetr)

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
include_directories(/home/uisee/disk/TensorRT-8.5.1.7/include)
link_directories(/home/uisee/disk/TensorRT-8.5.1.7/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fsanitize=address -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(rtdetr
                    ${PROJECT_SOURCE_DIR}/src/main.cpp
                    ${PROJECT_SOURCE_DIR}/src/rt_detr_detector.cpp
                    )
target_link_libraries(rtdetr nvinfer)
target_link_libraries(rtdetr cudart)
target_link_libraries(rtdetr ${OpenCV_LIBS} )

add_definitions(-O2 -pthread)