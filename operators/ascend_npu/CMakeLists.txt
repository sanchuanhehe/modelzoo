cmake_minimum_required(VERSION 3.19.0)
project(modelzoo_ops)

set(OPS_VERSION "0.1" CACHE STRING "ModelZoo ops version")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type Release or Debug")
set(ASCEND_PYTHON_EXECUTABLE "python3" CACHE STRING "Python command")
set(vendor_name "customize" CACHE STRING "Vender name")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/latest" CACHE PATH "CANN install path")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/output CACHE PATH "Cmake install path" FORCE)
set(ENABLE_BINARY_PACKAGE "True" CACHE BOOL "Enable binary package")
set(ENABLE_ONNX "True" CACHE BOOL "Enable onnx transformation")

# Soc that need to be compiled
set(ASCEND_COMPUTE_UNIT $ENV{BUILD_ASCEND_COMPUTE_UNIT})
if(NOT ASCEND_COMPUTE_UNIT)
    message(WARNING "ASCEND_COMPUTE_UNIT is not set. Using default: ascend310p")
    set(ASCEND_COMPUTE_UNIT "ascend310p")
endif()
message(STATUS "Soc that need to be compiled: ${ASCEND_COMPUTE_UNIT}")

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/modelzoo_ops/kernel)
add_subdirectory(${PROJECT_SOURCE_DIR}/modelzoo_ops/framework/onnx_plugin)
add_subdirectory(${PROJECT_SOURCE_DIR}/modelzoo_ops/_ext_src)

if (CMAKE_STEP EQUAL 3)
    add_custom_target(modify_vendor ALL
        DEPENDS ${MODELZOO_OPS_PATH}/scripts/install.sh ${MODELZOO_OPS_PATH}/scripts/upgrade.sh
    )

    add_custom_command(OUTPUT ${MODELZOO_OPS_PATH}/scripts/install.sh
                              ${MODELZOO_OPS_PATH}/scripts/upgrade.sh
        COMMAND mkdir -p ${MODELZOO_OPS_PATH}/scripts
        COMMAND cp -r ${ASCEND_PROJECT_DIR}/scripts/* ${MODELZOO_OPS_PATH}/scripts/
        COMMAND chmod +w ${MODELZOO_OPS_PATH}/scripts/*
    )

    install(DIRECTORY ${MODELZOO_OPS_PATH}/scripts/
        DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
    )

    install(DIRECTORY ${MODELZOO_OPS_PATH}/packages
        DESTINATION .
    )

    install(DIRECTORY ${MODELZOO_OPS_PATH}/ops
        DESTINATION packages/vendors/customize/framework/modelzoo_ops
    )

    install(FILES ${MODELZOO_OPS_PATH}/_ext.so
        DESTINATION packages/vendors/customize/framework/modelzoo_ops
    )

    install(FILES ${MODELZOO_OPS_PATH}/__init__.py
        DESTINATION packages/vendors/customize/framework/modelzoo_ops
    )

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack ops project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ops project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CPACK_PACKAGE_FILE_NAME "Hispark-ModelZoo-ops-${OPS_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_CMAKE_DIR}/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()