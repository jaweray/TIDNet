#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import editdistance
import pdb
from difflib import SequenceMatcher

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

####计算重叠iou
def area(a, b):
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx>=0) and (dy>=0):
                return dx*dy
        else:
                return 0.

def rectLimits(points):
    x_arr = [points[0],points[2],points[4],points[6]]
    y_arr = [points[1],points[3],points[5],points[7]]
    return [min(x_arr), min(y_arr), max(x_arr), max(y_arr)]


def load_txt_boxes(txt_file):
    final_bboxes_label=[]
    with open(txt_file, 'r',encoding='utf-8') as file_gt:
        context = file_gt.readlines()
        for num,co in enumerate(context):
            points_array=co.split('\n')[0].split('\t')[0].split(',')
            label=co.split('\n')[0].split('\t')[1]

            print(points_array)
            points=[int(float(points_array[0])),int(float(points_array[1])),int(float(points_array[2])),int(float(points_array[3])),int(float(points_array[4])),int(float(points_array[5])),int(float(points_array[6])),int(float(points_array[7]))]
            gtRect = Rectangle(*rectLimits(points))
            final_bboxes_label.append([points,gtRect,label])

    return final_bboxes_label

def load_txt_boxes_pred(txt_file):
    final_bboxes_label=[]
    with open(txt_file, 'r',encoding='utf-8') as file_gt:
        context = file_gt.readlines()
        for num,co in enumerate(context):
            points_array=co.split('\n')[0].split('\t')[0].split(',')
            label=co.split('\n')[0].split('\t')[1]
            # label=points_array[8]

            points=[int(float(points_array[0])),int(float(points_array[1])),int(float(points_array[2])),int(float(points_array[3])),int(float(points_array[4])),int(float(points_array[5])),int(float(points_array[6])),int(float(points_array[7]))]
            gtRect = Rectangle(*rectLimits(points))
            final_bboxes_label.append([points,gtRect,label])

    return final_bboxes_label

def relace_word(label):
    label=label.rstrip().replace(' ','').upper()
    char_array=',:[]()'
    char_repalce='，：【】（）'
    for i in range(len(char_array)):
        label=label.replace(char_array[i],char_repalce[i])

    return label


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def compare_txt_and_gt():

    pred_txt_dirs=r'C:\Users\Administrator\Desktop\tt1\txt_moire_pp'
    gt_txt_dirs=r'C:\Users\Administrator\Desktop\tt1\label'


    gt_files=os.listdir(gt_txt_dirs)

    ###初始化所有图像的总和统计的值
    sum_gt_boxes_num=0
    sum_pred_boxes_num=0

    sum_pred_label_right=0
    sum_gt_label_right=0

    sum_pred_box_right=0
    sum_gt_box_right=0

    levenshtein_sum=0
    similarity_sum = 0
    similarity_score = 0

    count_biaodian_false=0
    len_all_gtlabel=0

    character_right = 0
    character_sum = 0
    character_pre_sum = 0
    for gt_num_,gt_file in enumerate(gt_files):
        print(gt_num_, gt_file)
        gt_boxes_label=load_txt_boxes(os.path.join(gt_txt_dirs,gt_file))  #读入gt
        pred_boxes_label=load_txt_boxes_pred(os.path.join(pred_txt_dirs,gt_file))  #读入pred

        if(len(gt_boxes_label)==0):
            continue

        sum_every_pred_label_right=0
        sum_every_gt_label_right=0

        sum_every_pred_box=0
        sum_every_gt_box=0

        for gt_num,gt_boxes in enumerate(gt_boxes_label):
            gt_label = relace_word(gt_boxes[2])
            character_sum += len(gt_label)

        for pred_num,pred_boxes in enumerate(pred_boxes_label):
            pred_label=relace_word(pred_boxes[2])
            character_pre_sum += len(pred_label)


        ##### 多对一 #####
        for pred_num, pred_boxes in enumerate(pred_boxes_label):  ##pred 逐个检测框与gt对比
            pred_points = pred_boxes[0]
            pred_rect = pred_boxes[1]
            pred_label = relace_word(pred_boxes[2])

            sum_area = 0
            preDimensions = ((pred_rect.xmax - pred_rect.xmin + 1) * (pred_rect.ymax - pred_rect.ymin + 1))
            for gt_num, gt_boxes in enumerate(gt_boxes_label):
                gt_points = gt_boxes[0]
                gt_rect = gt_boxes[1]
                gt_label = relace_word(gt_boxes[2])

                ####计算gt内所有的框与pred框的iou,计算重叠面积占gt的比例值
                intersected_area = area(pred_rect, gt_rect)
                gtDimensions = ((gt_rect.xmax - gt_rect.xmin + 1) * (gt_rect.ymax - gt_rect.ymin + 1))
                recall_iou_ratios = float(intersected_area) / gtDimensions

                if recall_iou_ratios > 0.5:
                    sum_every_gt_box += 1
                    sum_area += intersected_area

            if sum_area / preDimensions > 0.5:
                sum_every_pred_box += 1

        ##### 一对多 #####
        for gt_num, gt_boxes in enumerate(gt_boxes_label):
            gt_points = gt_boxes[0]
            gt_rect = gt_boxes[1]
            gt_label = relace_word(gt_boxes[2])

            sum_area = 0
            gtDimensions = ((gt_rect.xmax - gt_rect.xmin + 1) * (gt_rect.ymax - gt_rect.ymin + 1))
            for pred_num, pred_boxes in enumerate(pred_boxes_label):  ##pred 逐个检测框与gt对比
                pred_points = pred_boxes[0]
                pred_rect = pred_boxes[1]
                pred_label = relace_word(pred_boxes[2])

                ####计算gt内所有的框与pred框的iou,计算重叠面积占gt的比例值
                intersected_area = area(pred_rect, gt_rect)
                preDimensions = ((pred_rect.xmax - pred_rect.xmin + 1) * (pred_rect.ymax - pred_rect.ymin + 1))
                recall_iou_ratios = float(intersected_area) / preDimensions

                if recall_iou_ratios > 0.5:
                    sum_every_pred_box += 1
                    sum_area += intersected_area

            if sum_area / gtDimensions > 0.5:
                sum_every_gt_box += 1


        for pred_num,pred_boxes in enumerate(pred_boxes_label):   ##pred 逐个检测框与gt对比
            pred_points=pred_boxes[0]
            pred_rect=pred_boxes[1]
            pred_label=relace_word(pred_boxes[2])

            for gt_num,gt_boxes in enumerate(gt_boxes_label):
                gt_points=gt_boxes[0]
                gt_rect=gt_boxes[1]
                gt_label=relace_word(gt_boxes[2])

                ####计算gt内所有的框与pred框的iou,计算重叠面积占gt的比例值
                intersected_area = area(pred_rect, gt_rect)
                gtDimensions = ((gt_rect.xmax - gt_rect.xmin + 1) * (gt_rect.ymax - gt_rect.ymin + 1))
                preDimensions = ((pred_rect.xmax - pred_rect.xmin + 1) * (pred_rect.ymax - pred_rect.ymin + 1))
                recall_iou_ratios = float(intersected_area) / gtDimensions
                pred_iou_ratios = float(intersected_area) / preDimensions

                # 一对一
                if recall_iou_ratios > 0.5 and pred_iou_ratios > 0.5:
                    sum_every_gt_box -= 1
                    sum_every_pred_box -= 1

                # #判断iou 如果大于0.5，则认为是同一个框，只能处理一对一的场景，后期需添加一对多，和多对一的场景
                # if(recall_iou_ratios>0.5):
                #
                #     sum_every_pred_box=sum_every_pred_box+1
                #     ##判断当前的标签与预测的是否一致
                #
                #     # import re
                #     # # 过滤掉除了中文以外的字符
                #     # gt_label = re.sub("[\!\%\[\]\,\。，：【】（）.、．;]", "", gt_label)
                #     # pred_label = re.sub("[\!\%\[\]\,\。，：【】（）.、．;]", "", pred_label)
                #
                #     edit_count=editdistance.eval(pred_label, gt_label)
                #     match_similarity=SequenceMatcher(None, pred_label, gt_label).ratio()
                #
                #     levenshtein_sum = levenshtein_sum + edit_count
                #     similarity = 1 - float(edit_count) / float(max(len(pred_label), len(gt_label)))
                #     similarity_sum = similarity_sum + similarity
                #     similarity_score = similarity_score + match_similarity
                #     len_all_gtlabel=len_all_gtlabel+len(gt_label)
                #
                #     if(pred_label==gt_label):
                #         sum_every_pred_label_right=sum_every_pred_label_right+1
                #     else:
                #         ###判断是否是标点字符的错误
                #         print(gt_label + '    ' + pred_label + '   ' + str(match_similarity) + '   ' + str(edit_count))
                #         import re
                #         # 过滤掉除了中文以外的字符
                #         gt_label = re.sub("[\!\%\[\]\,\。，：【】（）.、．;]", "", gt_label)
                #         pred_label = re.sub("[\!\%\[\]\,\。，：【】（）.、．;]", "", pred_label)
                #         #pdb.set_trace()
                #         if(gt_label==pred_label):
                #             count_biaodian_false=count_biaodian_false+1
                #             #print(gt_label + '    ' + pred_label)
                #     break
                if float(intersected_area) / gtDimensions > 0.5 or float(intersected_area) / preDimensions > 0.5:
                    character_right += lcs(pred_label, gt_label)

        # every_recall_label=float(sum_every_pred_label_right)/len(gt_boxes_label)
        # #pdb.set_trace()
        # every_precision_label=float(sum_every_pred_label_right)/len(pred_boxes_label)
        #
        # every_recall_box=float(sum_every_pred_box)/len(gt_boxes_label)
        # every_precision_box=float(sum_every_pred_box)/len(pred_boxes_label)

        ###累加每张图像的结果
        sum_gt_boxes_num=sum_gt_boxes_num+len(gt_boxes_label)
        sum_pred_boxes_num=sum_pred_boxes_num+len(pred_boxes_label)

        #label
        sum_pred_label_right=sum_pred_label_right+sum_every_pred_label_right
        sum_gt_label_right=sum_gt_label_right+sum_every_pred_label_right

        #detect
        sum_pred_box_right=sum_pred_box_right+sum_every_pred_box
        sum_gt_box_right=sum_gt_box_right+sum_every_gt_box

        #print(str(gt_num_)+'  \t'+gt_file+'   \t'+"Detection  recall:   "+str(every_recall_box)+"\tprecision:   \t"+str(every_precision_box)+"\tRecognition  recall:   "+str(every_recall_label)+"\tprecision:   \t"+str(every_precision_label))
    ##最终评估结果输出#################################
    #检测的结果
    det_recall=float(sum_gt_box_right)/sum_gt_boxes_num
    det_precision=float(sum_pred_box_right)/sum_pred_boxes_num
    print('\n' * 4)
    print("\tDetection  sum_gt_box_right:   " + str(sum_gt_box_right) + "\tsum_gt_boxes_num:   \t" + str(sum_gt_boxes_num)+ "\tsum_pred_box_right:   \t" + str(
        sum_pred_box_right) + "\tsum_pred_boxes_num:   \t" + str(sum_pred_boxes_num))
    # #识别的结果
    # recogn_recall=float(sum_gt_label_right)/sum_gt_boxes_num
    # recogn_precision=float(sum_pred_label_right)/sum_pred_boxes_num

    det_hmean=2* det_recall * det_precision / (det_recall + det_precision)
    print("All images result: \nDetection  recall:   "+str(det_recall)+"\tprecision:   \t"+str(det_precision)+"\tf-score:   \t"+str(det_hmean))
    # recogn_heman=2* recogn_recall * recogn_precision / (recogn_recall + recogn_precision)
    # print("\tRecognition  sum_gt_label_right:   " + str(sum_gt_label_right) + "\tsum_gt_boxes_num:   \t" + str(sum_gt_boxes_num)+ "\tsum_pred_label_right:   \t" + str(
    #     sum_pred_label_right) + "\tsum_pred_boxes_num:   \t" + str(sum_pred_boxes_num))
    # print("\tRecognition  recall:   "+str(recogn_recall)+"\tprecision:   \t"+str(recogn_precision)+"\thmean:   \t"+str(recogn_heman))
    #
    ###编辑距离的结果
    #print(str(len_all_gtlabel)+"      "+str(levenshtein_sum)+"    "+str())
    #print("similarity_score: "+str(similarity_score)+"\t sum_gt_boxes_num: "+str(sum_gt_boxes_num))
    # print("\tlevenshtein_average:   " + str(levenshtein_sum/sum_gt_boxes_num) + "\tsimilarity_sequenceMathcher:   \t" + str(
    #     similarity_score/sum_gt_boxes_num) + "\tedit smilarity:   \t" + str(similarity_sum/sum_gt_boxes_num))
    #
    #
    # print("count_biaodian_false:   "+str(count_biaodian_false))

    rec_hmean = 2*(character_right / character_sum)*(character_right / character_pre_sum) / ((character_right / character_sum)+(character_right / character_pre_sum))
    print('character_right: ' + str(character_right) + '\tcharacter_sum: ' + str(character_sum) + '\tcharacter_pre_sum: ' + str(character_pre_sum))
    print('recall: ' + str(character_right / character_sum) + '\t' + 'precision: ' + str(character_right / character_pre_sum)+"\tf-score:   \t"+str(rec_hmean))


if __name__ == '__main__':
    compare_txt_and_gt()