import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws



def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("该矩阵没有逆矩阵")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTets(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


abX, abY = loadDataSet('abalone.txt')
yHat01 = lwlrTets(abX[0:99], abX[0:99], abY[0:99], k=0.1)
yHat1 = lwlrTets(abX[0:99], abX[0:99], abY[0:99], k=1)
yHat10 = lwlrTets(abX[0:99], abX[0:99], abY[0:99], k=10)

ws = standRegres(abX[0:99], abY[0:99])
yHat = abX[0:99] * ws
print(type(abX))
print(ws.shape)
print(yHat.shape)
print(rssError(abY[0:99], yHat01.T))
print(rssError(abY[0:99], yHat1.T))
print(rssError(abY[0:99], yHat10.T))
print(rssError(abY[0:99], yHat.T.A))




