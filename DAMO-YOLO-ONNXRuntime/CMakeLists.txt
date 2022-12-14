cmake_minimum_required(VERSION 3.13)
project(damo_yolo)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++0x -fexceptions -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")


# onnxruntime (居然还得自己找路径？ 不能find_package? 这可太不方便了……)
find_package(CUDA REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

# set(ONNXRUNTIME_LIB "/home/uisee/disk/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0")
set(ONNXRUNTIME_LIB "/usr/local/lib/libonnxruntime.so.1.8.2")

# include_directories("/home/uisee/disk/onnxruntime-linux-x64-1.10.0/include")
include_directories("/usr/local//include/onnxruntime/core/session")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# C++ branch
add_executable(damo_yolo src/main.cpp )
target_link_libraries(damo_yolo ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})