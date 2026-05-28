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

import re
import os
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

def load_groundtruth(groundtruth_file):
    groundtruth = {}
    with open(groundtruth_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                label = int(parts[1])
                groundtruth[filename] = label
    return groundtruth

def main():
    parser = argparse.ArgumentParser(description="Evaluate board KWS inference results")
    parser.add_argument("--result_dir", type=str, help="Directory containing result txt files")
    parser.add_argument("--result_file", type=str, help="Single result txt file (optional)")
    parser.add_argument("--groundtruth", type=str, help="Groundtruth file")
    parser.add_argument("--output_file", type=str, help="Output result file")
    parser.add_argument("--threshold", type=float, default=0.0, help="Score threshold for detection")
    args = parser.parse_args()
    
    result_dir = args.result_dir if args.result_dir else os.path.join(PROJECT_DIR, "out/result/txt")
    groundtruth_file = args.groundtruth if args.groundtruth else os.path.join(PROJECT_DIR, "data/groundtruth.txt")
    output_file = args.output_file if args.output_file else os.path.join(PROJECT_DIR, "data/board_eval_result.txt")
    
    groundtruth_map = load_groundtruth(groundtruth_file)
    
    pattern_wav = re.compile(r'([^/\s]+\.wav)')
    pattern_detected = re.compile(r'DETECTED keyword=xiaoyunxiaoyun score=([\d.]+)')
    pattern_rejected = re.compile(r'REJECTED.*no keyword')
    
    results = []
    
    result_files = []
    if args.result_file:
        result_files = [args.result_file]
    else:
        result_files = glob.glob(os.path.join(result_dir, "*.txt"))
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        if len(lines) < 2:
            continue
        
        wav_match = pattern_wav.search(lines[0])
        if wav_match:
            filename = wav_match.group(1)
            
            score = 0.0
            predict = 0
            
            detected_match = pattern_detected.search(lines[1])
            rejected_match = pattern_rejected.search(lines[1])
            
            if detected_match:
                score = float(detected_match.group(1))
                predict = 1 if score > args.threshold else 0
            elif rejected_match:
                score = 0.0
                predict = 0
            
            groundtruth = groundtruth_map.get(filename, -1)
            if groundtruth == -1:
                continue
            
            correct = 1 if predict == groundtruth else 0
            
            results.append({
                'filename': filename,
                'score': score,
                'groundtruth': groundtruth,
                'predict': predict,
                'correct': correct
            })
    
    total = len(results)
    correct_count = sum(r['correct'] for r in results)
    
    tp = sum(1 for r in results if r['groundtruth'] == 1 and r['predict'] == 1)
    fp = sum(1 for r in results if r['groundtruth'] == 0 and r['predict'] == 1)
    fn = sum(1 for r in results if r['groundtruth'] == 1 and r['predict'] == 0)
    tn = sum(1 for r in results if r['groundtruth'] == 0 and r['predict'] == 0)
    
    accuracy = correct_count / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines = ["filename score groundtruth predict correct\n"]
    lines.extend(f"{r['filename']} {r['score']:.6f} {r['groundtruth']} {r['predict']} {r['correct']}\n" for r in results)
    lines.extend([
        "\n",
        f"Total: {total}\n",
        f"Correct: {correct_count}\n",
        f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n",
        f"\nConfusion Matrix:\n",
        f"  TP (keyword detected correctly): {tp}\n",
        f"  FP (false alarm): {fp}\n",
        f"  FN (missed detection): {fn}\n",
        f"  TN (non-keyword correctly rejected): {tn}\n",
        f"\nMetrics:\n",
        f"  Precision: {precision:.4f}\n",
        f"  Recall: {recall:.4f}\n",
        f"  F1 Score: {f1:.4f}\n"
    ])
    with open(output_file, 'w') as f:
        f.write(''.join(lines))
    
    print(f"=== Evaluation Result ===")
    print(f"Total: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()