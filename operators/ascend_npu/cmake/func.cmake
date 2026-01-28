function(install_target)
    cmake_parse_arguments(INSTALL_TARGET "" "DST;TRG" "" ${ARGN})
    set_target_properties(
        ${INSTALL_TARGET_TRG} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${MODELZOO_OPS_PATH}/${INSTALL_TARGET_DST})
    install(TARGETS ${INSTALL_TARGET_TRG}
        LIBRARY DESTINATION ${INSTALL_TARGET_DST})
endfunction()

function(install_file)
    cmake_parse_arguments(INSTALL_TARGET "" "DST;TRG" "SRC" ${ARGN})
    file(MAKE_DIRECTORY ${MODELZOO_OPS_PATH}/${INSTALL_TARGET_DST})
    foreach(SOURCE_FILE ${INSTALL_TARGET_SRC})
        add_custom_command(
            TARGET ${INSTALL_TARGET_TRG}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE}
                ${MODELZOO_OPS_PATH}/${INSTALL_TARGET_DST})
    endforeach()
    install(FILES ${INSTALL_TARGET_SRC} DESTINATION ${INSTALL_TARGET_DST})
endfunction()

function(get_system_info SYSTEM_INFO)
    if(UNIX)
        execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
        string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
        set(${SYSTEM_INFO}
                ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
                PARENT_SCOPE)
    elseif(WIN32)
        message(STATUS "System is Windows. Only for pre-build.")
    else()
        message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
    endif()
endfunction()

function(add_ops_info_target)
    cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
    get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
    add_custom_command(
        OUTPUT ${OPINFO_OUTPUT}
        COMMAND mkdir -p ${opinfo_file_path}
        COMMAND
            ${ASCEND_PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/parse_ini_to_json.py ${OPINFO_OPS_INFO}
            ${OPINFO_OUTPUT})
    add_custom_target(${OPINFO_TARGET} ALL DEPENDS ${OPINFO_OUTPUT})
    install(FILES ${OPINFO_OUTPUT} DESTINATION ${OPINFO_INSTALL_DIR})
endfunction()

function(add_ops_compile_options OP_TYPE)
    cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
    file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
        "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

function(add_ops_impl_target)
    cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR"
        "OPS_BATCH;OPS_ITERATE" ${ARGN})
    add_custom_command(
        OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
        COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
        COMMAND
            ${ASCEND_PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/ascendc_impl_build.py ${OPIMPL_OPS_INFO}
            \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\" ${OPIMPL_IMPL_DIR}
            ${OPIMPL_OUT_DIR}/dynamic ${ASCEND_AUTOGEN_PATH}
        COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
        COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
        DEPENDS ${OPIMPL_OPS_INFO} ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/ascendc_impl_build.py)
    add_custom_target(${OPIMPL_TARGET} ALL DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)
endfunction()

function(add_npu_support_target)
    cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
    get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
    add_custom_command(
        OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
        COMMAND mkdir -p ${NPUSUP_OUT_DIR}
        COMMAND bash ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR} ${NPUSUP_OUT_DIR})
    add_custom_target(npu_supported_ops ALL
        DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json)
    install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
        DESTINATION ${NPUSUP_INSTALL_DIR})
endfunction()

function(add_bin_compile_target)
    cmake_parse_arguments(
        BINCMP
        ""
        "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;ADP_DIR;OUT_DIR;INSTALL_DIR;KERNEL_DIR"
        ""
        ${ARGN})
    file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src)
    file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/gen)
    file(MAKE_DIRECTORY ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT})
    execute_process(
        COMMAND
            ${ASCEND_PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/ascendc_bin_param_build.py
            ${BINCMP_OPS_INFO} ${BINCMP_OUT_DIR}/gen ${BINCMP_COMPUTE_UNIT}
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE EXEC_INFO
        ERROR_VARIABLE EXEC_ERROR)
    if(${EXEC_RESULT})
        message("ops binary compile scripts gen info: ${EXEC_INFO}")
        message("ops binary compile scripts gen error: ${EXEC_ERROR}")
        message(FATAL_ERROR "ops binary compile scripts gen failed!")
    endif()
    add_custom_target(${BINCMP_TARGET})
    set(KERNELS "${KERNEL_NAME}")
    set(bin_scripts)
    foreach(KERNEL ${KERNELS})
        file(GLOB scripts ${BINCMP_OUT_DIR}/gen/*${KERNEL}*.sh)
        list(APPEND bin_scripts ${scripts})
    endforeach()

    # if bin_scripts not empty
    if(bin_scripts)
        add_custom_target(
            ${BINCMP_TARGET}_gen_ops_config ALL
            COMMAND
                ${ASCEND_PYTHON_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/insert_simplified_keys.py -p
                ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}
            COMMAND
                ${ASCEND_PYTHON_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/opensource/ascendsamples/cmake/util/ascendc_ops_config.py -p
                ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT} -s ${BINCMP_COMPUTE_UNIT})

        foreach(bin_script ${bin_scripts})
            get_filename_component(bin_file ${bin_script} NAME_WE)
            string(REPLACE "-" ";" bin_sep ${bin_file})
            list(GET bin_sep 0 op_type)
            list(GET bin_sep 1 op_file)
            list(GET bin_sep 2 op_index)
            if(NOT TARGET ${BINCMP_TARGET}_${op_file}_copy)
                add_custom_target(
                    ${BINCMP_TARGET}_${op_file}_copy
                    COMMAND cp -r ${BINCMP_IMPL_DIR}/${op_file}/op_kernel/*.*
                        ${BINCMP_OUT_DIR}/src
                    COMMAND cp ${BINCMP_ADP_DIR}/${op_file}.py
                        ${BINCMP_OUT_DIR}/src/${op_type}.py
                    DEPENDS ascendc_impl_gen)
                install(
                    DIRECTORY ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/${op_file}
                    DESTINATION ${BINCMP_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT}
                    OPTIONAL)
                install(
                    FILES ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT}/${op_file}.json
                    DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}
                    OPTIONAL)
            endif()
            add_custom_target(
                ${BINCMP_TARGET}_${op_file}_${op_index}
                COMMAND
                    export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && export
                    ASCEND_CUSTOM_OPP_PATH=${MODELZOO_OPS_PATH}/packages/vendors/${vendor_name}
                    && bash ${PROJECT_SOURCE_DIR}/cmake/retry.sh \"bash ${bin_script}
                    ${BINCMP_OUT_DIR}/src/${op_type}.py
                    ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/${op_file}\"
                WORKING_DIRECTORY ${BINCMP_OUT_DIR})
            add_dependencies(${BINCMP_TARGET}_${op_file}_${op_index} ${BINCMP_TARGET}
                ${BINCMP_TARGET}_${op_file}_copy)
            add_dependencies(${BINCMP_TARGET}_gen_ops_config ${BINCMP_TARGET}_${op_file}_${op_index})
        endforeach()
        add_custom_command(
            TARGET ${BINCMP_TARGET}_gen_ops_config
            POST_BUILD
            COMMAND mv ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/*.json
                ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT})
        install(
            FILES ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT}/binary_info_config.json
            DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}
            OPTIONAL)
    endif()
endfunction()

function(protobuf_generate)
    cmake_parse_arguments(PROTOBUF_GEN "" "PROTO_FILE;OUT_DIR" "" ${ARGN})
    set(OUT_DIR ${PROTOBUF_GEN_OUT_DIR}/proto/onnx)
    file(MAKE_DIRECTORY ${OUT_DIR})
    get_filename_component(file_name ${PROTOBUF_GEN_PROTO_FILE} NAME_WE)
    get_filename_component(file_dir ${PROTOBUF_GEN_PROTO_FILE} PATH)
    execute_process(
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND protoc -I${file_dir} --cpp_out=${OUT_DIR} ${PROTOBUF_GEN_PROTO_FILE}
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE EXEC_INFO
        ERROR_VARIABLE EXEC_ERROR)
    if(${EXEC_RESULT})
        message("protobuf gen info: ${EXEC_INFO}")
        message("protobuf gen error: ${EXEC_ERROR}")
        message(FATAL_ERROR "protobuf gen failed!")
    endif()
endfunction()
