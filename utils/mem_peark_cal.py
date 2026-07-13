# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re
import sys

DEFAULT_MEM_TOTAL_KB = 469736
LINES_PER_TIME_POINT = 2

def parse_values(file_path, output_file, mem_total):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]

    time_points = []
    for i in range(0, len(lines), LINES_PER_TIME_POINT):
        time_point = (i // LINES_PER_TIME_POINT) + 1
        os_used = None
        mmz_used = None

        for line in lines[i : i + LINES_PER_TIME_POINT]:
            mmz_match = re.search(r"used=(\d+)KB", line)
            if mmz_match:
                mmz_used = int(mmz_match.group(1))

            mem_avail_match = re.search(r"MemAvailable:\s*(\d+) kB", line)
            if mem_avail_match:
                mem_available = int(mem_avail_match.group(1))
                os_used = mem_total - mem_available

        if os_used is not None and mmz_used is not None:
            current_total = os_used + mmz_used
        else:
            current_total = None

        time_points.append({
            'time_point': time_point,
            'os_used_kb': os_used,
            'mmz_used_kb': mmz_used,
            'total_used_kb': current_total
        })

    if not time_points:
        print("No valid memory data found in the file.")
        return  # 修复逻辑漏洞：防止后续访问空列表

    baseline = time_points[0]
    print(f"Baseline values (time point 1):")
    print(f"os baseline : {baseline['os_used_kb']} kB")
    print(f"mmz baseline: {baseline['mmz_used_kb']} kB")
    print(f"total baseline: {baseline['total_used_kb']} kB")

    change_data = []
    total_changes = []
    
    for tp in time_points:
        os_change = (tp['os_used_kb'] - baseline['os_used_kb']) if tp['os_used_kb'] is not None else None
        mmz_change = (tp['mmz_used_kb'] - baseline['mmz_used_kb']) if tp['mmz_used_kb'] is not None else None
        
        if os_change is not None and mmz_change is not None:
            total_change = os_change + mmz_change
            total_changes.append(total_change)
        else:
            total_change = None

        change_data.append({
            'time_point': tp['time_point'],
            'os_change_kb': os_change,
            'mmz_change_kb': mmz_change,
            'total_change_kb': total_change
        })

    max_total_change = max(total_changes) if total_changes else None

    with open(output_file, 'w') as f:
        f.write("time_point os_change_kB mmz_change_kB total_change_kB\n")
        for data in change_data:
            os_val = data['os_change_kb'] if data['os_change_kb'] is not None else 'N/A'
            mmz_val = data['mmz_change_kb'] if data['mmz_change_kb'] is not None else 'N/A'
            total_val = data['total_change_kb'] if data['total_change_kb'] is not None else 'N/A'
            f.write(f"{data['time_point']} {os_val} {mmz_val} {total_val}\n")

    # 过滤 None 值计算最大变化
    valid_os_changes = [d['os_change_kb'] for d in change_data if d['os_change_kb'] is not None]
    valid_mmz_changes = [d['mmz_change_kb'] for d in change_data if d['mmz_change_kb'] is not None]

    print(f"\nSuccessfully processed {len(time_points)} time points")
    print(f"Memory change results saved to {output_file}")
    print(f"Maximum os memory change (relative to baseline): {max(valid_os_changes, default=0)} kB")
    print(f"Maximum mmz memory change (relative to baseline): {max(valid_mmz_changes, default=0)} kB")
    if max_total_change is not None:
        print(f"Maximum total memory change (relative to baseline): {max_total_change} kB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate memory peak changes.")
    parser.add_argument("input_file", help="Path to the input memory log file")
    parser.add_argument("output_file", nargs='?', default="memory_change_results.txt", help="Path to the output results file")
    parser.add_argument("--mem-total", type=int, default=DEFAULT_MEM_TOTAL_KB, help=f"Total memory in KB (default: {DEFAULT_MEM_TOTAL_KB})")
    
    args = parser.parse_args()
    parse_values(args.input_file, args.output_file, args.mem_total)