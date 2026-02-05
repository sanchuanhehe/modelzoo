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
set(CMAKE_BUILD_TYPE Debug)  # 确保使用调试模式

set(CMAKE_SKIP_RPATH TRUE)

# Header path
include_directories(
$ENV{DDK_PATH}/acllib/include/
    ../inc/
)

# add host lib path
if (target MATCHES "Simulator_Function" OR target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/toolkit/tools/sim/lib)
else()
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/acllib/lib64/stub)
endif()
set(LIB_PATH $ENV{NPU_HOST_LIB})
MESSAGE(NPU_HOST_LIB=$ENV{NPU_HOST_LIB})
link_directories(${LIB_PATH})
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

