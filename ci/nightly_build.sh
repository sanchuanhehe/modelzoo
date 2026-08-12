#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:?engine required}
TARGETS_JSON=${2:?targets JSON required}
SUMMARY=${3:?summary path required}
mkdir -p "$(dirname "$SUMMARY")" nightly-artifacts
success=0; failed=0
while IFS=$'\t' read -r id sample build_def; do
    log="nightly-artifacts/$id.log"
    if bash ci/build_sample.sh --sample "$sample" --build-def "$build_def" >"$log" 2>&1; then
        ((success+=1)); result=success
    else
        ((failed+=1)); result=failed
    fi
    printf '%s\t%s\t%s\t%s\n' "$id" "$sample" "$build_def" "$result" >> "$SUMMARY"
done < <(python3 -c 'import json,sys; [print(x["id"],x["sample"],x["buildDef"],sep="\t") for x in json.loads(sys.argv[1])]' "$TARGETS_JSON")
printf 'engine=%s success=%d failed=%d\n' "$ENGINE" "$success" "$failed" | tee -a "$SUMMARY"
((failed == 0))
