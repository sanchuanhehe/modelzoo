#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ci/build_sample.sh --sample <path> --build-def <SS928V100|OPTG>
EOF
}

SAMPLE=""
BUILD_DEF=""
while (($#)); do
    case "$1" in
        --sample) SAMPLE=${2:?}; shift 2 ;;
        --build-def) BUILD_DEF=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$SAMPLE" || -z "$BUILD_DEF" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "$SAMPLE/CMakeLists.txt" ]]; then
    printf 'Sample is missing CMakeLists.txt: %s\n' "$SAMPLE" >&2
    exit 1
fi
: "${NPU_INCLUDE_PATH:?NPU_INCLUDE_PATH is required}"
: "${NPU_LIB_PATH:?NPU_LIB_PATH is required}"

TOOLCHAIN_FILE=samples/common/cmake/toolchain_aarch64_linux.cmake
BUILD_DIR="$SAMPLE/build"
rm -rf "$BUILD_DIR" "$SAMPLE/out"

cmake -S "$SAMPLE" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$(pwd)/$TOOLCHAIN_FILE" \
    -DSOC_VERSION="$BUILD_DEF"
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

if [[ ! -x "$SAMPLE/out/main" ]]; then
    printf 'Build completed without the expected executable: %s/out/main\n' "$SAMPLE" >&2
    exit 1
fi

file "$SAMPLE/out/main"

