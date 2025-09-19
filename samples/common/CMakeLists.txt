# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
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

cmake_minimum_required(VERSION 3.5.1)

project(INFER_UTILS_PROJECT)

# Compile options
add_compile_options(
    -std=c++14
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# eg:
# export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
# export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub
# cmake .. -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc
# 
# export DDK_PATH=$HOME/Ascend/ascend-toolkit/svp_latest
# export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub
# cmake .. -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc -DSOC_VERSION=SS928V100

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    message(FATAL_ERROR "need set DDK_PATH")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    message(FATAL_ERROR "need set NPU_HOST_LIB")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(ACL_UTILS_DIR ./log ./infer ./infer/platform ./infer/preprocess ./infer/post_process)

if(SOC_VERSION STREQUAL "SS928V100")
    add_definitions(-DSVP_ACL_PLATFORM)
    set(ACL_LIB_LIST svp_acl pthread securec dl protobuf-c)
    list(APPEND ACL_UTILS_DIR ./infer/platform/svp_nnn)
    file(GLOB PLATFORM_SRC_LIST "./infer/platform/svp_nnn/*cpp")
    message(STATUS "${PLATFORM_SRC_LIST}") 
else()
    set(ACL_LIB_LIST ascendcl pthread)
    list(APPEND ACL_UTILS_DIR ./infer/platform/nnn)
    file(GLOB PLATFORM_SRC_LIST "./infer/platform/nnn/*cpp")
    message(STATUS "${PLATFORM_SRC_LIST}") 
endif ()

include_directories(
    ${ACL_UTILS_DIR}
    ${INC_PATH}/acllib/include/
    ../opensource/json/single_include/
    ../opensource/opencv/include/
)

find_library(OpenCV_WORLD_SHARED_LIB
    NAMES libopencv_world.so.412
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(OpenCV_WORLD_LIB
    NAMES libopencv_world.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ADE_LIB
    NAMES libade.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ITTNOTIFY_LIB
    NAMES libittnotify.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(JPEG_TURBO_LIB
    NAMES liblibjpeg-turbo.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(OPENJP2_LIB
    NAMES liblibopenjp2.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ZLIB_LIB
    NAMES libzlib.a
    PATHS "../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

link_directories(
    ${LIB_PATH}
)

file(GLOB INFER_SRC_LIST "./infer/*cpp")
file(GLOB PREPROCESS_SRC_LIST "./infer/preprocess/*cpp")
file(GLOB POSTPROCESS_SRC_LIST "./infer/post_process/*cpp")

# STATIC
add_library(infer_common STATIC ${PLATFORM_SRC_LIST} ${INFER_SRC_LIST} ${PREPROCESS_SRC_LIST} ${POSTPROCESS_SRC_LIST} log/log.cpp)
target_link_libraries(infer_common ${ACL_LIB_LIST} stdc++)

add_executable(test_infer sample/test.cpp)
target_link_libraries(test_infer ${ACL_LIB_LIST} stdc++ infer_common ${OpenCV_WORLD_SHARED_LIB})

# 依赖静态库
# target_link_libraries(test_infer 
#     ${ACL_LIB_LIST}  
#     infer_common 
#     ${OpenCV_WORLD_LIB}
#     ${ADE_LIB}
#     ${ITTNOTIFY_LIB}
#     ${JPEG_TURBO_LIB}
#     ${OPENJP2_LIB}
#     ${ZLIB_LIB}
#     m
#     stdc++)

# STATIC
add_library(infer_utils_static STATIC ${PLATFORM_SRC_LIST} log/log.cpp)
target_link_libraries(infer_utils_static ${ACL_LIB_LIST} stdc++)

add_library(infer_utils SHARED ${PLATFORM_SRC_LIST} log/log.cpp)
target_link_libraries(infer_utils ${ACL_LIB_LIST} stdc++)
