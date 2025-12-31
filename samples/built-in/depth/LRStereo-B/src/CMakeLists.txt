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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(COMMON_PATH "../../../../common/")

add_subdirectory(${COMMON_PATH} ./common_build)

include_directories(
    ${COMMON_PATH}/infer/platform
    ${COMMON_PATH}/include
    ../../../../opensource/json/single_include/
    ../../../../opensource/opencv/include/   
)

link_directories(
    ${LIB_PATH}
)

add_executable(main main.cpp)

target_link_libraries(main infer_common)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})