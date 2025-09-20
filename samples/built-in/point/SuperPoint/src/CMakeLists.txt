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

project(ACL_INFER_SAMPLE)

add_subdirectory(../../../../common/ ./infer_utils_build)

# Compile options
add_compile_options(
    -std=c++11
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall -lm")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -lm")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

set(INC_PATH $ENV{DDK_PATH})
set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    message(FATAL_ERROR "need set NPU_HOST_LIB")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

include_directories(
    ../../../../common/infer/platform
    ../../../../common/infer/platform/svp_nnn
    ../../../../common/log
    ${INC_PATH}/acllib/include/
    ../../../../opensource/json/single_include/
    ../../../../opensource/opencv/include/   
)


find_library(OpenCV_WORLD_SHARED_LIB
    NAMES libopencv_world.so.412
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(OpenCV_WORLD_LIB
    NAMES libopencv_world.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ADE_LIB
    NAMES libade.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ITTNOTIFY_LIB
    NAMES libittnotify.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(JPEG_TURBO_LIB
    NAMES liblibjpeg-turbo.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(OPENJP2_LIB
    NAMES liblibopenjp2.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

find_library(ZLIB_LIB
    NAMES libzlib.a
    PATHS "../../../../opensource/opencv/lib"
    NO_DEFAULT_PATH
)

link_directories(
    ${LIB_PATH}
)

add_executable(main main.cpp  superpoint_utils.cpp)

target_link_libraries(main infer_utils_static stdc++ ${OpenCV_WORLD_SHARED_LIB})

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})