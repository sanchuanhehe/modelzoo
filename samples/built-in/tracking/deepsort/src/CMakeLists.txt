# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

project(INFER_DEEPSORT_PROJECT)

set(COMMON_PATH "../../../../common/")

add_subdirectory(${COMMON_PATH} ./common_build)

# Compile options
add_compile_options(
    -std=c++14
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# NPU paths
set(LIB_PATH $ENV{NPU_LIB_PATH} "$ENV{NPU_LIB_PATH}/svp_npu")

# OpenCV paths
set(OPENCV_INC_PATH "../../../../opensource/opencv/include/")
set(JSON_INC_PATH "../../../../opensource/json/single_include/")

include_directories(
    ${COMMON_PATH}/infer/platform
    ${COMMON_PATH}/infer/preprocess
    ${COMMON_PATH}/infer/post_process
    ${COMMON_PATH}/include
    ${OPENCV_INC_PATH}
    ${JSON_INC_PATH}
    .
)

find_library(OpenCV_WORLD_SHARED_LIB
    NAMES libopencv_world.so.412
    PATHS ${OPENCV_LIB_PATH}
    NO_DEFAULT_PATH
)

link_directories(
    ${LIB_PATH}
)

set(SRC_FILES
    main.cpp
    deepsort.cpp
    tracker.cpp
    track.cpp
    kalmanfilter.cpp
    nn_matching.cpp
    hungarian.cpp
)

add_executable(main ${SRC_FILES})

if(SOC_VERSION STREQUAL "SS928V100")
    target_compile_definitions(main PRIVATE MODEL_CUST_CONIFG)
endif ()

# Link against inference utils natively from framework
# Make sure to link opencv_world which is commonly provided in modelzoo-dev OpenCV
target_link_libraries(main infer_common ${OpenCV_WORLD_SHARED_LIB} m stdc++)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
