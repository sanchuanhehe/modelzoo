cmake_minimum_required(VERSION 3.5.1)
project(ACT)
# 编译选项
add_compile_options(
    -std=c++11
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2
    -O2
)
# 可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_SKIP_RPATH TRUE)

# 交叉编译编译器配置
set(CMAKE_CXX_COMPILER aarch64-mix210-linux-g++)
set(CMAKE_C_COMPILER /usr/bin/cc)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

# 头文件路径
include_directories(
    ../inc/
    /home/Ascend/ascend-toolkit/5.20.t6.2.b060/x86_64-linux/x86_64-linux/include
)

# 库文件路径
link_directories(
    /home/Ascend/ascend-toolkit/5.20.t6.2.b060/x86_64-linux/runtime/lib64
)

# 源文件收集
FILE(GLOB SAMPLE_SRC_FILE ../src/*.cpp)
add_executable(main ${SAMPLE_SRC_FILE})

target_link_libraries(main
    ascendcl  # dlite核标准AscendCL库
    pthread dl stdc++  # 仅保留系统必存在的基础库
)

# 安装配置（原始配置保留，将可执行文件安装到输出目录）
install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
