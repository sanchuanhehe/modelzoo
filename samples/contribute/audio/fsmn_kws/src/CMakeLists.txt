cmake_minimum_required(VERSION 3.5.1)

project(ACL_INFER_SAMPLE)

set(COMMON_PATH "../../../../common/")



add_subdirectory(${COMMON_PATH} ./common_build)

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

set(KNF_SRC_PATH ../../../../opensource/kaldi-native-fbank/kaldi-native-fbank)

include_directories(
    ${COMMON_PATH}/infer/platform
    ${COMMON_PATH}/include
    ../process
    ${KNF_SRC_PATH}
)

link_directories(
    ${LIB_PATH}
)

set(KNF_SRC_PATH ../../../../opensource/kaldi-native-fbank/kaldi-native-fbank/kaldi-native-fbank/csrc)

add_executable(main
    main.cpp
    ../process/xiaoyun_preprocess.cpp
    ../process/xiaoyun_postprocess.cpp
    ${KNF_SRC_PATH}/feature-fbank.cc
    ${KNF_SRC_PATH}/feature-functions.cc
    ${KNF_SRC_PATH}/feature-window.cc
    ${KNF_SRC_PATH}/log.cc
    ${KNF_SRC_PATH}/mel-computations.cc
    ${KNF_SRC_PATH}/rfft.cc
    ${KNF_SRC_PATH}/fftsg.c
)

target_link_libraries(main infer_common)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})