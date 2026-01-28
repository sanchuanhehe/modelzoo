file(GLOB OP_HOST_SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.h
)

file(GLOB OP_KERNEL_SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.h
)

# 添加到父作用域的
set(ASCEND_HOST_SRC 
    ${ASCEND_HOST_SRC}  # 包含已有的文件
    ${OP_HOST_SRC}        # 添加新文件
    PARENT_SCOPE        # 关键：传递回父目录
)

set(ASCEND_KERNEL_SRC 
    ${ASCEND_KERNEL_SRC}  # 包含已有的文件
    ${OP_KERNEL_SRC}        # 添加新文件
    PARENT_SCOPE        # 关键：传递回父目录
)

