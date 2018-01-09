import xlrd
import numpy as np
import time
# data = xlrd.open_workbook('G:\dataMining\TrafficFlowData_20171229.xlsx')
# sheet1 = data.sheets()[0]
# row1 = sheet1.row_values(0)
# print(row1)

def getData(filename):
    '''

    :param filename: xlsx文件地址
    :return:所有数据组成的列表
    '''
    dataLine = 321
    excel =  xlrd.open_workbook(filename)
    sheet1 = excel.sheets()[0]
    allData = []
    for i in range(dataLine):
        data = sheet1.row_values(i)
        allData.append(data)
    return allData

def euclDistance(vector1, vector2):
    '''

    :param vector1: 向量1
    :param vector2: 向量2
    :return: 两向量之间的欧氏距离
    '''
    return np.sqrt(np.sum((vector1-vector2)**2))

def setCentroids(dataMat, k):
    '''

    :param data: 数据集数组
    :param k: 选取的k值
    :return: 选取初始质心点的数据集的数组
    '''
    row = dataMat.shape[0]
    centroids = np.zeros((k,dataMat.shape[1]))
    for i in range(k):
        index = int(np.random.uniform(0,row))
        centroids[i, :] = dataMat[index, :]
    return centroids

def kmeans(data, k):
    dataMat = np.array(data)
    dataNum = dataMat.shape[0]
    centroids = setCentroids(dataMat, k)
    centroidsChange = True
    classDict = []
    classfiyMat = np.zeros((dataNum, 2))

    while centroidsChange:
        centroidsChange = False
        for i in range(dataNum):
            minDistance = euclDistance(dataMat[i],centroids[0])
            minIndex = 0
            for j in range(k):
                distance = euclDistance(dataMat[i], centroids[j])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j

            if classfiyMat[i, 0] != minIndex:
                centroidsChange = True
                classfiyMat[i, :] = minIndex, minDistance**2

        for u in range(k):
            centroidsAvg = dataMat[np.nonzero(classfiyMat[:, 0] == u)]
            centroids[u, :] = np.mean(centroidsAvg,axis=0)

    return centroids, classfiyMat;


#计算轮廓系数，值处于-1~1之间，越大表示聚类效果越好
def getSC(classfiyList,dataMat, k):
    '''

    :param classfiyList: 每行数据对应的分类
    :param dataMat: 数据集
    :return:轮廓系数
    '''
    classDict = {}
    for p in range(k):
        classDict[p] = [x for x in range(dataMat.shape[0]) if classfiyList[x]==p]
        if len(classDict[p])==0 or len(classDict[p])==1:
            return -1
    SC = 0
    for i in range(dataMat.shape[0]):
        avgADistance = 0
        for j in classDict[classfiyList[i]]:
            avgADistance += euclDistance(dataMat[i], dataMat[j])
        avgADistance /= len(classDict[classfiyList[i]])

        otherClusters = [x for x in range(k) if x!=classDict[classfiyList[i]]]
        minAvgBDistance = 1000000000000000
        for o in otherClusters:
            avgBDistance = 0
            for u in classDict[o]:
                avgBDistance += euclDistance(dataMat[i], dataMat[u])
            avgBDistance /= len(classDict[o])
            if avgBDistance < minAvgBDistance:
                minAvgBDistance = avgBDistance
        SC = SC + (minAvgBDistance - avgADistance) / max(avgADistance,minAvgBDistance)
    SC /= dataMat.shape[0]
    return SC



data = getData('G:\dataMining\TrafficFlowData_20171229.xlsx')
centroids, classfitMat = kmeans(data,10)
maxSC = -1
# for k in range(5,200):
#     centroids, classfitMat = kmeans(data,k)
#     a = getSC(classfitMat[:,0], np.array(data),k)
#     if a > maxSC:
#         maxSC = a
#         optimalK = k
#     print(k)
# print(maxSC,optimalK)

for k in range(11,21):
    a = 0
    for i in range(10):
        centroids, classfitMat = kmeans(data,k)
        b = getSC(classfitMat[:,0], np.array(data),k)
        while b==-1:
            centroids, classfitMat = kmeans(data,k)
            b = getSC(classfitMat[:,0], np.array(data),k)
        a = a + b
    a /= 10
    print(k,a)
