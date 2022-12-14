# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
# Inference Engine cmake config
# ------
#
# This config defines the following variables:
#
#   InferenceEngine_FOUND        - True if the system has the Inference Engine library
#   InferenceEngine_INCLUDE_DIRS - Inference Engine include directories
#   InferenceEngine_LIBRARIES    - Inference Engine libraries
#
# and the following imported targets:
#
#   IE::inference_engine            - The Inference Engine library
#   IE::inference_engine_c_api      - The Inference Engine C API library
#
# Inference Engine version variables:
#
#   InferenceEngine_VERSION_MAJOR - major version component
#   InferenceEngine_VERSION_MINOR - minor version component
#   InferenceEngine_VERSION_PATCH - patch version component
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was InferenceEngineConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

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

if(NOT DEFINED CMAKE_FIND_PACKAGE_NAME)
    set(CMAKE_FIND_PACKAGE_NAME InferenceEngine)
    set(_ie_need_package_name_reset ON)
endif()

# need to store current PACKAGE_PREFIX_DIR, because it's overwritten by sub-package one
set(_ie_package_prefix_dir "${PACKAGE_PREFIX_DIR}")

include(CMakeFindDependencyMacro)

find_dependency(OpenVINO
                PATHS "${CMAKE_CURRENT_LIST_DIR}"
                NO_CMAKE_FIND_ROOT_PATH
                NO_DEFAULT_PATH)

# create targets with old names for compatibility
if(TARGET openvino::runtime AND NOT TARGET IE::inference_engine)
    add_library(IE::inference_engine INTERFACE IMPORTED)
    set_target_properties(IE::inference_engine PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::runtime)
endif()

if(TARGET openvino::runtime::c AND NOT TARGET IE::inference_engine_c_api)
    add_library(IE::inference_engine_c_api INTERFACE IMPORTED)
    set_target_properties(IE::inference_engine_c_api PROPERTIES
        INTERFACE_LINK_LIBRARIES openvino::runtime::c)
endif()

# mark components as available
foreach(comp inference_engine inference_engine_c_api)
    set(${CMAKE_FIND_PACKAGE_NAME}_${comp}_FOUND ON)
endforeach()

if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS inference_engine inference_engine_c_api)
endif()

unset(InferenceEngine_LIBRARIES)
foreach(comp IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    # check if the component is available
    if(${CMAKE_FIND_PACKAGE_NAME}_${comp}_FOUND)
        set(pcomp IE::${comp})

        list(APPEND InferenceEngine_LIBRARIES ${pcomp})
    endif()
endforeach()

# restore PACKAGE_PREFIX_DIR
set(PACKAGE_PREFIX_DIR ${_ie_package_prefix_dir})
unset(_ie_package_prefix_dir)

set_and_check(InferenceEngine_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/runtime/include/ie")

check_required_components(${CMAKE_FIND_PACKAGE_NAME})

if(_ie_need_package_name_reset)
    unset(CMAKE_FIND_PACKAGE_NAME)
    unset(_ie_need_package_name_reset)
endif()
