# Copyimg_right (c) ModelZoo. 2025-2025. All img_rights reserved.
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

import os
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import json
import math
from collections import namedtuple
import argparse
from pathlib import Path


"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""
class DetectionDetEvalEvaluator(object):
    def __init__(
        self,
        area_recall_constraint=0.7, area_precision_constraint=0.3,
        ev_param_ind_center_diff_thr=1,
        mtype_oo_o=1.0, mtype_om_o=0.7, mtype_om_m=1.0
    ):


        self.area_recall_constraint = area_recall_constraint
        self.area_precision_constraint = area_precision_constraint
        self.ev_param_ind_center_diff_thr = ev_param_ind_center_diff_thr
        self.mtype_oo_o = mtype_oo_o
        self.mtype_om_o = mtype_om_o
        self.mtype_om_m = mtype_om_m

    def evaluate_image(self, gt, pred):

        def get_union(pD,pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD,pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD,pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def one_to_one_match(row, col):
            cont = 0
            for j in range(len(recallMat[0])):    
                if recallMat[row,j] >= self.area_recall_constraint and precisionMat[row,j] >= self.area_precision_constraint:
                    cont = cont +1
            if (cont != 1):
                return False
            cont = 0
            for i in range(len(recallMat)):    
                if recallMat[i,col] >= self.area_recall_constraint and precisionMat[i,col] >= self.area_precision_constraint:
                    cont = cont +1
            if (cont != 1):
                return False
            
            if recallMat[row,col] >= self.area_recall_constraint and precisionMat[row,col] >= self.area_precision_constraint:
                return True
            return False
        
        def num_overlaps_gt(gtNum):
            cont = 0
            for detNum in range(len(detRects)):
                if detNum not in detDontCareRectsNum:
                    if recallMat[gtNum,detNum] > 0 :
                        cont = cont +1
            return cont

        def num_overlaps_det(detNum):
            cont = 0
            for gtNum in range(len(recallMat)):    
                if gtNum not in gtDontCareRectsNum:
                    if recallMat[gtNum,detNum] > 0 :
                        cont = cont +1
            return cont
        
        def is_single_overlap(row, col):
            if num_overlaps_gt(row)==1 and num_overlaps_det(col)==1:
                return True
            else:
                return False
        
        def one_to_many_match(gtNum):
            many_sum = 0
            detRects = []
            for detNum in range(len(recallMat[0])):    
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                    if precisionMat[gtNum,detNum] >= self.area_precision_constraint:
                        many_sum += recallMat[gtNum,detNum]
                        detRects.append(detNum)
            if round(many_sum,4) >= self.area_recall_constraint:
                return True,detRects
            else:
                return False,[]         
        
        def many_to_one_match(detNum):
            many_sum = 0
            gtRects = []
            for gtNum in range(len(recallMat)):    
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                    if recallMat[gtNum,detNum] >= self.area_recall_constraint:
                        many_sum += precisionMat[gtNum,detNum]
                        gtRects.append(gtNum)
            if round(many_sum,4) >= self.area_precision_constraint:
                return True,gtRects
            else:
                return False,[]
        
        def center_distance(r1, r2):
            return ((np.mean(r1, axis=0) - np.mean(r2, axis=0)) ** 2).sum() ** 0.5
        
        def diag(r):
            r = np.array(r)
            return ((r[:, 0].max() - r[:, 0].min()) ** 2 + (r[:, 1].max() - r[:, 1].min()) ** 2) ** 0.5
        
        perSampleMetrics = {}
        
        recall = 0
        precision = 0
        hmean = 0        
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCareRectsNum = []#Array of Ground Truth Rectangles' keys marked as don't Care
        detDontCareRectsNum = []#Array of Detected Rectangles' matched with a don't Care GT
        pairs = []
        evaluationLog = ""
        
        recallMat = np.empty([1,1])
        precisionMat = np.empty([1,1])              
        
        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtRects.append(points)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCareRectsNum.append( len(gtRects)-1 )                 
        
        evaluationLog += "GT rectangles: " + str(len(gtRects)) + (" (" + str(len(gtDontCareRectsNum)) + " don't care)\n" if len(gtDontCareRectsNum)>0 else "\n")
        
        for n in range(len(pred)):
            points = pred[n]['points']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detRect = points
            detRects.append(detRect)
            detPolPoints.append(points)
            if len(gtDontCareRectsNum)>0 :
                for dontCareRectNum in gtDontCareRectsNum:
                    dontCareRect = gtRects[dontCareRectNum]
                    intersected_area = get_intersection(dontCareRect,detRect)
                    rdDimensions = Polygon(detRect).area
                    if (rdDimensions==0) :
                        precision = 0
                    else:
                        precision= intersected_area / rdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCareRectsNum.append( len(detRects)-1 )
                        break

        evaluationLog += "DET rectangles: " + str(len(detRects)) + (" (" + str(len(detDontCareRectsNum)) + " don't care)\n" if len(detDontCareRectsNum)>0 else "\n")

        if len(gtRects)==0:
            recall = 1
            precision = 0 if len(detRects)>0 else 1

        if len(detRects)>0:
            #Calculate recall and precision matrixs
            outputShape=[len(gtRects),len(detRects)]
            recallMat = np.empty(outputShape)
            precisionMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtRects),np.int8)
            detRectMat = np.zeros(len(detRects),np.int8)
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    rG = gtRects[gtNum]
                    rD = detRects[detNum]
                    intersected_area = get_intersection(rG,rD)
                    rgDimensions = Polygon(rG).area
                    rdDimensions = Polygon(rD).area
                    recallMat[gtNum,detNum] = 0 if rgDimensions==0 else  intersected_area / rgDimensions
                    precisionMat[gtNum,detNum] = 0 if rdDimensions==0 else intersected_area / rdDimensions

            # Find one-to-one matches
            evaluationLog += "Find one-to-one matches\n"
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum :
                        match = one_to_one_match(gtNum, detNum)
                        if match is True :
                            #in deteval we have to make other validation before mark as one-to-one
                            if is_single_overlap(gtNum, detNum) is True :
                                rG = gtRects[gtNum]
                                rD = detRects[detNum]
                                normDist = center_distance(rG, rD);
                                normDist /= diag(rG) + diag(rD);
                                normDist *= 2.0;
                                if normDist < self.ev_param_ind_center_diff_thr:
                                    gtRectMat[gtNum] = 1
                                    detRectMat[detNum] = 1
                                    recallAccum += self.mtype_oo_o
                                    precisionAccum += self.mtype_oo_o
                                    pairs.append({'gt':gtNum,'det':detNum,'type':'OO'})
                                    evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " normDist: " + str(normDist) + " \n"
                            else:
                                evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"
            # Find one-to-many matches
            evaluationLog += "Find one-to-many matches\n"
            for gtNum in range(len(gtRects)):
                if gtNum not in gtDontCareRectsNum:
                    match,matchesDet = one_to_many_match(gtNum)
                    if match is True :
                        evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                        #in deteval we have to make other validation before mark as one-to-one
                        if num_overlaps_gt(gtNum)>=2 :
                            gtRectMat[gtNum] = 1
                            recallAccum += (self.mtype_oo_o if len(matchesDet)==1 else self.mtype_om_o)
                            precisionAccum += (self.mtype_oo_o if len(matchesDet)==1 else self.mtype_om_o*len(matchesDet))
                            pairs.append({'gt':gtNum,'det':matchesDet,'type': 'OO' if len(matchesDet)==1 else 'OM'})
                            for detNum in matchesDet :
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"    

            # Find many-to-one matches
            evaluationLog += "Find many-to-one matches\n"
            for detNum in range(len(detRects)):
                if detNum not in detDontCareRectsNum:
                    match,matchesGt = many_to_one_match(detNum)
                    if match is True :
                        #in deteval we have to make other validation before mark as one-to-one
                        if num_overlaps_det(detNum)>=2 :                          
                            detRectMat[detNum] = 1
                            recallAccum += (self.mtype_oo_o if len(matchesGt)==1 else self.mtype_om_m*len(matchesGt))
                            precisionAccum += (self.mtype_oo_o if len(matchesGt)==1 else self.mtype_om_m)
                            pairs.append({'gt':matchesGt,'det':detNum,'type': 'OO' if len(matchesGt)==1 else 'MO'})
                            for gtNum in matchesGt :
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"                                    

            numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if len(detRects)>0 else float(1)
            else:
                recall = float(recallAccum) / numGtCare
                precision =  float(0) if (len(detRects) - len(detDontCareRectsNum))==0 else float(precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
            hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)  

        numGtCare = len(gtRects) - len(gtDontCareRectsNum)
        numDetCare = len(detRects) - len(detDontCareRectsNum)

        perSampleMetrics = {
            'precision':precision,
            'recall':recall,
            'hmean':hmean,
            'pairs':pairs,
            'recallMat':[] if len(detRects)>100 else recallMat.tolist(),
            'precisionMat':[] if len(detRects)>100 else precisionMat.tolist(),
            'gtPolPoints':gtPolPoints,
            'detPolPoints':detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare':gtDontCareRectsNum,
            'detDontCare':detDontCareRectsNum,
            'recallAccum':recallAccum,
            'precisionAccum':precisionAccum,
            'evaluationLog': evaluationLog
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGt = 0
        numDet = 0
        methodRecallSum = 0
        methodPrecisionSum = 0

        for result in results:
            numGt += result['gtCare']
            numDet += result['detCare']
            methodRecallSum += result['recallAccum']
            methodPrecisionSum += result['precisionAccum']

        methodRecall = 0 if numGt==0 else methodRecallSum/numGt
        methodPrecision = 0 if numDet==0 else methodPrecisionSum/numDet
        methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)
        
        methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean  }

        return methodMetrics

def convert_annotations(pre_data):
    """
    将预标注数据转换为gt格式
    这里需要根据你的实际数据格式进行调整
    
    Args:
        pre_data: 预标注数据
    
    Returns:
        list: 转换后的gt标注列表
    """
    gt_list = []
    
    # 假设pre_data包含标注信息
    # 这里只是一个示例，需要根据你的实际数据结构调整
    for annotation in pre_data:
        gt_item = {
            "points": annotation.get("points", []),
            "text": annotation.get("text", ""),
        }
        gt_list.append(gt_item)
    
    return gt_list

def convert_single_file(input_path, converted_data, 
                       image_base_path="../datasets/xfund_ppocr/images/"):
    """
    转换单个文件的格式
    
    Args:
        input_path: 输入文件路径
        converted_data: 输出文件路径
        image_base_path: 图像基础路径
    """
    try:
        # 读取源文件
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        

        
        for item in source_data:
            # 获取图像名
            img_name = item.get("img_name", "")
            
            # 构建转换后的数据
            converted_item = {
                "img": os.path.join(image_base_path, img_name),
                "pre": []  # 初始化为空，后续可以填充
            }
            
            # 如果有预标注数据，可以转换
            if "pre" in item and item["pre"]:
                # 根据你的实际标注格式转换
                converted_item["pre"] = convert_annotations(item["pre"])
            
            converted_data.append(converted_item)
        print(f"已转换: {os.path.basename(input_path)} ")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败 {input_path}: {e}")
        return False

def batch_convert_folder(input_folder, converted_data, 
                        image_base_path="../datasets/xfund_ppocr/images/"):
    """
    批量转换文件夹中的所有JSON文件
    
    Args:
        input_folder: 输入文件夹路径
        converted_data: json数据
        image_base_path: 图像基础路径
    """
    # 获取所有JSON文件
    json_files = list(Path(input_folder).glob("*.json"))
    
    if not json_files:
        print(f"在 {input_folder} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    success_count = 0
    fail_count = 0
    
    for json_file in json_files:
    
        # 转换文件
        if convert_single_file(str(json_file), converted_data, image_base_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n转换完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")

"""
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': True,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model inference')
    parser.add_argument('--gt_path', type=str, default="./end_gt_xfund.json")
    parser.add_argument('--out_path', type=str, default="./out/result/txt/")
    opt = parser.parse_args()
    evaluator = DetectionDetEvalEvaluator()
    
    gt = []
    pre = []
    with open(opt.gt_path, "r") as f:
        loaded_data = json.load(f)
        gt = loaded_data
    # 使用示例
    input_folder = opt.out_path
    # 转换数据
    converted_data = []
    batch_convert_folder(input_folder, converted_data)
    pre = converted_data
    results = []
    for x in pre:
        file_name = x["img"]
        print(" file_name: " , file_name)
        img_name = Path(file_name).stem
        gt_one = []
        for item in gt:
           # print(" item: " , item)
            print(" item img: " , item.get("img"))
            if img_name in item.get("img"):
                gt_one = item["gt"]
                #print(item["img"])
                break
        #print(" gt: " , gt_one)
        pre_one = x["pre"]
        #print(" p[re0]: " , pre_one)

        
        for gt_1, pred_1 in zip([gt_one], [pre_one]):
            results.append(evaluator.evaluate_image(gt_1, pred_1))
    metrics = evaluator.combine_results(results)
    print(metrics)
