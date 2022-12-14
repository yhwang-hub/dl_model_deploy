cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(yolox_openvino)

find_package(OpenCV REQUIRED)

set(openvino_LIBRARIES "${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/lib/intel64/libopenvino.so")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/include
    ${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/include/ie
    ${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/include/ngraph
    ${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/include/openvino
    ${OpenCV_INCLUDE_DIR}
)

link_directories("${PROJECT_SOURCE_DIR}/openvino_toolkit_ubuntu18_2022.2.0.7713/runtime/lib")
add_executable(yolox_openvino ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/yolox.cpp)
target_link_libraries(yolox_openvino PRIVATE ${openvino_LIBRARIES} ${OpenCV_LIBS})