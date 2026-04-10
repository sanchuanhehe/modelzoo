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

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# 公共编译选项
add_compile_options(
    -std=c++14
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2
    -O2
)

# 公共编译Flag
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -DNDEBUG -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# 环境变量检查
if (NOT DEFINED ENV{NPU_INCLUDE_PATH})
    message(FATAL_ERROR "need set NPU_INCLUDE_PATH")
endif()

if (NOT DEFINED ENV{NPU_LIB_PATH})
    message(FATAL_ERROR "need set NPU_LIB_PATH")
endif()

# 平台相关ACL配置
set(ACL_UTILS_DIR ./infer/platform ./infer/preprocess ./infer/post_process)
if(SOC_VERSION STREQUAL "SS928V100")
    add_definitions(-DSVP_ACL_PLATFORM)
    list(APPEND ACL_UTILS_DIR ./infer/platform/svp_nnn)
    file(GLOB PLATFORM_SRC_LIST "./infer/platform/svp_nnn/*cpp")

    if(LINUX_PLATFORM)
        set(ACL_LIB_LIST svp_acl pthread securec dl protobuf-c)
    elseif(OHOS_PLATFORM)
        set(ACL_LIB_LIST svp_acl pthread securec dl protobuf-c ot_osal ot_base ot_irq ot_sys ss_mpi ss_voice_engine ss_upvqe ss_dnvqe)
    endif()
    if(ENABLE_SVP_ACCELERATE)
        add_definitions(-DENABLE_SVP_ACCELERATE)
        message(STATUS "SVP NPU accelerate is enabled")
    else()
        message(STATUS "SVP NPU accelerate is disabled (use CPU fallback)")
    endif()
elseif(SOC_VERSION STREQUAL "Hi3516CV610")
    add_definitions(-DSVP_ACL_PLATFORM)
    list(APPEND ACL_UTILS_DIR ./infer/platform/svp_nnn)
    file(GLOB PLATFORM_SRC_LIST "./infer/platform/svp_nnn/*cpp")
    set(ACL_LIB_LIST svp_acl pthread securec dl protobuf-c ss_mpi_sysmem)
else()
    set(ACL_LIB_LIST ascendcl)
    list(APPEND ACL_UTILS_DIR ./infer/platform/nnn)
    file(GLOB PLATFORM_SRC_LIST "./infer/platform/nnn/*cpp")
endif()
message(STATUS "PLATFORM_SRC_LIST: ${PLATFORM_SRC_LIST}")

# 公共头文件目录
include_directories(
    ${ACL_UTILS_DIR}
    ${INC_PATH}
    ./include
    ../opensource/json/single_include/
    ../opensource/opencv/include/
    ../opensource/pillow-resize/include/
    ../opensource/sentencepiece/include/
)

find_library(OpenCV_WORLD_SHARED_LIB
    NAMES libopencv_world.so.412
    PATHS ${OPENCV_LIB_PATH}
    NO_DEFAULT_PATH
)
message(STATUS "OPENCV_LIB_PATH: ${OPENCV_LIB_PATH}")

link_directories(
    ${LIB_PATH}
)

# 源文件列表
file(GLOB INFER_SRC_LIST "./infer/*cpp")
file(GLOB PREPROCESS_SRC_LIST "./infer/preprocess/*cpp")
file(GLOB POSTPROCESS_SRC_LIST "./infer/post_process/*cpp")
file(GLOB UTILS_SRC_LIST "./utils/*cpp")
set(PILLOW_RESIZE_SRC "../opensource/pillow-resize/src/PillowResize/PillowResize.cc")

# 静态库：infer_common
add_library(infer_common STATIC 
    ${PLATFORM_SRC_LIST} ${INFER_SRC_LIST} ${PREPROCESS_SRC_LIST}
    ${POSTPROCESS_SRC_LIST} ${UTILS_SRC_LIST} ${PILLOW_RESIZE_SRC} log/log.cpp
)

target_link_libraries(infer_common ${ACL_LIB_LIST} ${OpenCV_WORLD_SHARED_LIB} m ${CXX_STDLIB})

# 静态库：infer_utils_static
add_library(infer_utils_static STATIC ${PLATFORM_SRC_LIST} log/log.cpp)
target_link_libraries(infer_utils_static ${ACL_LIB_LIST} ${CXX_STDLIB})

# 动态库：infer_utils
add_library(infer_utils SHARED ${PLATFORM_SRC_LIST} log/log.cpp)
target_link_libraries(infer_utils ${ACL_LIB_LIST} ${CXX_STDLIB})