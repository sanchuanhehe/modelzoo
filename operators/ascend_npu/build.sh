CUR_DIR=$(dirname $(readlink -f $0))
BUILD_PACKAGES_DIR=${CUR_DIR}/output
PYTHON_VER=$(python3 --version 2>/dev/null | awk '{print $2}' | cut -d. -f1-2)
BUILD_TYPE='Release'
SOC_VERSION='Hi3591P'

function check_soc_version() {
    case "${SOC_VERSION}" in
        "Hi3591P")
            ASCEND_COMPUTE_UNIT='ascend310p'
            return 0
            ;;
        "Hi3591B")
            ASCEND_COMPUTE_UNIT='ascend310b'
            return 0
            ;;
        *)
            echo "invalid soc version, or soc version is not support yet"
            exit 1
            ;;
    esac
}

function check_python_version() {
    if [ -z "$PYTHON_VER" ]; then
        PYTHON_VER=$(python --version 2>/dev/null | awk '{print $2}' | cut -d. -f1-2)
    fi

    if [ -z "$PYTHON_VER" ]; then
        echo "Warning: Could not detect Python version. Using default: 3.7"
        PYTHON_VER='3.7'
    fi

    # 提取主版本和次版本号
    MAJOR=$(echo "${PYTHON_VER}" | cut -d. -f1)
    MINOR=$(echo "${PYTHON_VER}" | cut -d. -f2)
    
    # 检查版本格式
    if [[ -z "${MAJOR}" || -z "${MINOR}" ]]; then
        echo "Invalid Python version format: ${PYTHON_VER}"
        exit 1
    fi
    
    # 检查是否 >= 3.7
    if [[ "${MAJOR}" -lt 3 ]] || ([[ "${MAJOR}" -eq 3 ]] && [[ "${MINOR}" -lt 7 ]]); then
        echo "Error: Python ${PYTHON_VER} is too old. Minimum required version is 3.7"
        exit 1
    fi
    
    echo "Python ${PYTHON_VER} meets minimum requirement (>=3.7)"
}

function usage() {
    echo "Usage: $0 [--debug]" 1>&2
}
function parse_input_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --soc_version=*)
                SOC_VERSION="${1#*=}"
                shift 1
                ;;
            --debug)
                BUILD_TYPE='Debug'
                shift 1
                ;;
            *)
            usage
            return 1
            ;;
        esac
    done
    return 0
}

function main()
{
    if [ -z "$ASCEND_OPP_PATH" ]; then
        echo "ImportError: cannot open cann shared object file. Please check that the CANN package is installed."
        echo "If cann package is installed, please run 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' in the CANN installation path."
        exit 1
    else
        echo "ASCEND_OPP_PATH = $ASCEND_OPP_PATH"
    fi

    if ! parse_input_args "$@"; then
        echo "Failed to parse input args. Please check input args."
        exit 1
    fi

    rm -rf ${BUILD_PACKAGES_DIR}
    check_soc_version
    export BUILD_ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}
    check_python_version
    export BUILD_PYTHON_VERSION=${PYTHON_VER}
    python"${PYTHON_VER}" setup.py build
    if [ $? != 0 ]; then
        echo "Failed to build package. Please check source code."
        exit 1
    fi

    exit 0
}

main "$@"
