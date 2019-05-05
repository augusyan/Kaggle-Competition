# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : load_data.py
@time : 2019/4/30 21:12
@function : 
"""

import csv
import os
import time
import numpy as np
import pandas as pd

# global config
start_time = time.time()
loacl_path = os.getcwd()
#  path
test_path = "./test.csv"
# test_path = "./sample_submission.csv"

def npLoad():
    list = np.loadtxt(open(test_path, "rb"), delimiter=",", skiprows=1)  # 0.25052809715270996s
    pass

def pdLoadCsv():
    # data=pd.read_csv(test_path,header=None)
    # #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
    # list=data[1:-1].values.tolist()  # 0.18649768829345703s
    pass

def fileLoadCsv():
    # readin = csv.reader(open(test_path,'r'))
    # for line in readin:
    #     # 0.16954636573791504s
    #     print(line)
    # # result file
    # stu1 = ['marry',26]
    # stu2 = ['bob',23]
    # #打开文件，追加a
    # out = open('Stu_csv.csv','a', newline='')
    # #设定写入模式
    # csv_write = csv.writer(out,dialect='excel')
    # #写入具体内容
    # csv_write.writerow(stu1)
    pass


def readCSVFile(file):
    rawData = []
    trainFile = open(path + file, 'rb')
    reader = csv.reader(trainFile)
    for line in reader:
        rawData.append(line)  # 42001 lines,the first line is header
    rawData.pop(0)  # remove header
    intData = np.array(rawData).astype(np.int32)
    return intData


def loadTrainingData():
    intData = readCSVFile("train.csv")
    label = intData[:, 0]
    data = intData[:, 1:]
    data = np.where(data > 0, 1, 0)  # replace positive in feature vector to 1
    return data, label


def loadTestData():
    intData = readCSVFile("test.csv")
    data = np.where(intData > 0, 1, 0)
    return data


if __name__ == '__main__':
    print(list[3])
    print("cost time:|",time.time()-start_time)