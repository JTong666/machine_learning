import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(numFeat - 1):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
        labelMat.append(curLine[-1])
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):#将数据进行阈值分类
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMatrix)
    numSetps = 10
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSzie = (rangeMax - rangeMin) / numSetps
        for j in range(-1, int(stepSzie + 1)):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSzie)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIter = 40):
    weakClassArr = []
    m, n = np.shape(dataArr)
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5*np.log((1-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)               #????????????????
        D = np.multiply(D, np.exp(expon))
        D = D/sum(D)
        aggClassEst += alpha*classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != classLabels.T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToclass, classifierArr):
    dataMatrix = np.mat(datToclass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m ,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return np.sign(aggClassEst)






