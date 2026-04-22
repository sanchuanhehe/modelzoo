# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_CLASSIFICATION_RESNET50)

# Compile options
add_compile_options(
    -std=c++11
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")  # 开启调试符号
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_SKIP_RPATH TRUE)

# Locate DDK/Ascend toolkit paths across old and new directory layouts.
set(_ddk_root_candidates
    "$ENV{DDK_PATH}"
    "$ENV{ASCEND_TOOLKIT_HOME}"
    "$ENV{ASCEND_HOME_PATH}"
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit/6.10.t01spc030b600"
)

set(DDK_ROOT "")
foreach(_candidate IN LISTS _ddk_root_candidates)
    if(_candidate AND EXISTS "${_candidate}")
        set(DDK_ROOT "${_candidate}")
        break()
    endif()
endforeach()

if(NOT DDK_ROOT)
    message(FATAL_ERROR "Cannot find Ascend toolkit root. Please set DDK_PATH or ASCEND_TOOLKIT_HOME.")
endif()

set(_include_candidates
    "${DDK_ROOT}/acllib/include"
    "${DDK_ROOT}/x86_64-linux/acllib/include"
)
set(DDK_INCLUDE_DIR "")
foreach(_dir IN LISTS _include_candidates)
    if(EXISTS "${_dir}/acl/svp_acl.h")
        set(DDK_INCLUDE_DIR "${_dir}")
        break()
    endif()
endforeach()

if(NOT DDK_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find svp_acl headers under ${DDK_ROOT}.")
endif()

set(LIB_PATH "")
set(LIB_PATHS "")
if (target MATCHES "Simulator_Function" OR target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    set(_sim_candidates
        "${DDK_ROOT}/toolkit/tools/sim/lib"
        "${DDK_ROOT}/x86_64-linux/toolkit/tools/sim/lib"
    )
    foreach(_dir IN LISTS _sim_candidates)
        if(EXISTS "${_dir}")
            set(LIB_PATH "${_dir}")
            list(APPEND LIB_PATHS "${_dir}")
            break()
        endif()
    endforeach()
else()
    set(_stub_dir_candidates "${DDK_ROOT}/acllib/lib64/stub")
    file(GLOB _all_stub_dirs "${DDK_ROOT}/x86_64-linux/acllib/lib64_*/stub")

    if(CMAKE_CXX_COMPILER MATCHES "musl")
        foreach(_dir IN LISTS _all_stub_dirs)
            if(_dir MATCHES "linux-musl/stub$" OR _dir MATCHES "ohos-musl-clang/stub$")
                list(APPEND _stub_dir_candidates "${_dir}")
            endif()
        endforeach()
    else()
        foreach(_dir IN LISTS _all_stub_dirs)
            if(_dir MATCHES "linux-gnu/stub$")
                list(APPEND _stub_dir_candidates "${_dir}")
            endif()
        endforeach()
    endif()

    list(APPEND _stub_dir_candidates ${_all_stub_dirs})
    list(REMOVE_DUPLICATES _stub_dir_candidates)

    foreach(_dir IN LISTS _stub_dir_candidates)
        if(EXISTS "${_dir}/libsvp_acl.so"
           AND EXISTS "${_dir}/libot_osal.so"
           AND EXISTS "${_dir}/libot_base.so"
           AND EXISTS "${_dir}/libot_sys.so"
           AND EXISTS "${_dir}/libot_irq.so"
           AND EXISTS "${_dir}/libss_mpi.so"
           AND EXISTS "${_dir}/libss_voice_engine.so"
           AND EXISTS "${_dir}/libss_upvqe.so"
           AND EXISTS "${_dir}/libss_dnvqe.so")
            set(LIB_PATH "${_dir}")
            break()
        endif()
    endforeach()

    if(LIB_PATH)
        list(APPEND LIB_PATHS "${LIB_PATH}")
    endif()

    set(_common_stub_candidates
        "${DDK_ROOT}/acllib/lib64/stub"
        "${DDK_ROOT}/x86_64-linux/acllib/lib64/stub"
    )
    foreach(_dir IN LISTS _common_stub_candidates)
        if(EXISTS "${_dir}/libsecurec.so" OR EXISTS "${_dir}/libprotobuf-c.so.1")
            list(APPEND LIB_PATHS "${_dir}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES LIB_PATHS)
endif()

if(NOT LIB_PATHS)
    message(FATAL_ERROR "Cannot find required stub libraries under ${DDK_ROOT}.")
endif()

include_directories(
    ${DDK_INCLUDE_DIR}
    ../inc/
)

message(STATUS "Using Ascend toolkit root: ${DDK_ROOT}")
message(STATUS "Using Ascend include dir: ${DDK_INCLUDE_DIR}")
message(STATUS "Using Ascend lib dirs: ${LIB_PATHS}")
link_directories(${LIB_PATHS})
FILE(GLOB SAMPLE_SRC_FILE *.cpp)

add_executable(main ${SAMPLE_SRC_FILE})

if (target MATCHES "Simulator_Function")
    target_link_libraries(main funcsim)
elseif (target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    target_link_libraries(main instsim)
else()
    target_link_libraries(main svp_acl ot_osal ot_base ot_sys ot_irq ss_mpi ss_voice_engine ss_upvqe ss_dnvqe pthread securec dl protobuf-c stdc++)
    get_target_property(LINK_LIBS main LINK_LIBRARIES)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
