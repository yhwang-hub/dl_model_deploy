# ******************************************************************************
# Copyright 2017-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
#
#
# FindNGraph
# ------
#
# This script defines the following variables and imported targets:
#
#   ngraph::ngraph                         - nGraph core target
#   ngraph_FOUND                           - True if the system has the nGraph library
#   NGRAPH_LIBRARIES                       - nGraph libraries
#
# Frontends:
#
#   ngraph_onnx_frontend_FOUND             - True if the system has ngraph::onnx_frontend library
#   ngraph::onnx_frontend                  - ONNX FrontEnd target (optional)
#
#   ngraph_paddle_frontend_FOUND           - True if the system has Paddle frontend
#   ngraph::paddle_frontend                - nGraph Paddle frontend (optional)
#
#   ngraph_ir_frontend_FOUND               - True if the system has OpenVINO IR frontend
#
#   ngraph_tensorflow_frontend_FOUND       - True if the system has TensorFlow frontend
#   ngraph::tensorflow_frontend       - nGraph TensorFlow frontend (optional)
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ngraphConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

find_dependency(OpenVINO
                PATHS "${CMAKE_CURRENT_LIST_DIR}"
                      "${CMAKE_CURRENT_LIST_DIR}/ngraph"
                NO_CMAKE_FIND_ROOT_PATH
                NO_DEFAULT_PATH)

# create targets with old names for compatibility
if(TARGET openvino::runtime AND NOT TARGET ngraph::ngraph)
    add_library(ngraph::ngraph INTERFACE IMPORTED)
    set_target_properties(ngraph::ngraph PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::runtime)
endif()

if(TARGET openvino::frontend::onnx AND NOT TARGET ngraph::onnx_frontend)
    add_library(ngraph::onnx_frontend INTERFACE IMPORTED)
    set_target_properties(ngraph::onnx_frontend PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::frontend::onnx)
endif()

if(TARGET openvino::frontend::paddle AND NOT TARGET ngraph::paddle_frontend)
    add_library(ngraph::paddle_frontend INTERFACE IMPORTED)
    set_target_properties(ngraph::paddle_frontend PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::frontend::paddle)
endif()

if(TARGET openvino::frontend::tensorflow AND NOT TARGET ngraph::tensorflow_frontend)
    add_library(ngraph::tensorflow_frontend INTERFACE IMPORTED)
    set_target_properties(ngraph::tensorflow_frontend PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::frontend::tensorflow)
endif()

set(ngraph_ngraph_FOUND ON)
set(NGRAPH_LIBRARIES ngraph::ngraph)

set(ngraph_onnx_frontend_FOUND ${OpenVINO_Frontend_ONNX_FOUND})
set(ngraph_tensorflow_frontend_FOUND ${OpenVINO_Frontend_TensorFlow_FOUND})
set(ngraph_paddle_frontend_FOUND ${OpenVINO_Frontend_Paddle_FOUND})
set(ngraph_onnx_importer_FOUND ${OpenVINO_Frontend_ONNX_FOUND})

if(ngraph_onnx_importer_FOUND)
    set(ONNX_IMPORTER_LIBRARIES ngraph::onnx_frontend)
    # ngraph::onnx_importer target and variables are deprecated
    # but need to create a dummy target for BW compatibility
    if(NOT TARGET ngraph::onnx_importer)
        add_library(ngraph::onnx_importer INTERFACE IMPORTED)
        set_target_properties(ngraph::onnx_importer PROPERTIES
            INTERFACE_LINK_LIBRARIES ngraph::onnx_frontend)
    endif()
endif()

set(ngraph_paddle_frontend_FOUND ${OpenVINO_Frontend_Paddle_FOUND})
set(ngraph_tensorflow_frontend_FOUND ${OpenVINO_Frontend_TensorFlow_FOUND})
set(ngraph_onnx_frontend_FOUND ${OpenVINO_Frontend_ONNX_FOUND})
set(ngraph_ir_frontend_FOUND ${OpenVINO_Frontend_IR_FOUND})

check_required_components(ngraph)
