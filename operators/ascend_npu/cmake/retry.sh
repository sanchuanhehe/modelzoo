#!/bin/bash

MAX_RETRY_COUNT=3
DELAY_TIME=2
CNT=0

while [ $CNT -lt $MAX_RETRY_COUNT ]; do
  eval "$@" && break
  CNT=$((CNT+1))
  if [ $CNT -lt $MAX_RETRY_COUNT ]; then
    echo "Compile fail, retry again."
    sleep $DELAY_TIME
  else
    echo "Compile fail after retrying $CNT times."
    exit 1
  fi
done

