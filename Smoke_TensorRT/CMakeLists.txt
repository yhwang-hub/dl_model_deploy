cmake_minimum_required(VERSION 2.6)

project(smoke)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
#include_directories(/usr/local/cuda/cudnn/v8.0.4/include)
#link_directories(/usr/local/cuda/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/home/uisee/disk/TensorRT-8.2.1.8/include)
link_directories(/home/uisee/disk/TensorRT-8.2.1.8/lib)

include_directories(src/plugins/modulated_deform_conv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(trt_modulated_deform_conv_plugin SHARED
  src/plugins/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu
  src/plugins/modulated_deform_conv/trt_modulated_deform_conv.cpp
)

add_executable(${PROJECT_NAME}
  src/smoke.cpp
  src/smoke_utils.cpp
  src/main.cpp
)
target_link_libraries(smoke nvinfer)
target_link_libraries(smoke nvinfer_plugin)
target_link_libraries(smoke nvonnxparser)
target_link_libraries(smoke cudart)
target_link_libraries(smoke trt_modulated_deform_conv_plugin)
target_link_libraries(smoke ${OpenCV_LIBS})

add_definitions(-O2 -pthread)