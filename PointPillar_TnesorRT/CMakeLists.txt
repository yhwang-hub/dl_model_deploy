cmake_minimum_required(VERSION 2.6)

project(pointpillar)

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
include_directories(/home/uisee/disk/TensorRT-8.2.1.8/include)
link_directories(/home/uisee/disk/TensorRT-8.2.1.8/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fsanitize=address -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(pointpillar
                ${PROJECT_SOURCE_DIR}/src/main.cpp
                ${PROJECT_SOURCE_DIR}/src/pointpillar.cpp
                ${PROJECT_SOURCE_DIR}/src/preprocess.cpp
                ${PROJECT_SOURCE_DIR}/src/postprocess.cpp
                ${PROJECT_SOURCE_DIR}/src/pillarScatter.cpp
                ${PROJECT_SOURCE_DIR}/src/pillarScatterKernels.cu
                ${PROJECT_SOURCE_DIR}/src/postprocess_kernels.cu
                ${PROJECT_SOURCE_DIR}/src/preprocess_kernels.cu
                )

target_link_libraries(pointpillar nvinfer)
target_link_libraries(pointpillar cudart)
target_link_libraries(pointpillar ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
)

add_definitions(-O2 -pthread)