import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLables):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLables).T
    m , n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights +alpha*dataMatrix.T*error
    return weights


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



def stocGradAscent0(dataMatrix, classLabels):#随机梯度上升算法
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            t = alpha * error
            weights = weights + np.multiply(t, dataMatrix[randIndex])
            #del(dataIndex[randIndex])
    return weights



def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingset = []
    traininglabels = []
    for word in frTrain.readlines():
        pp = word.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(pp[i]))
        trainingset.append(lineArr)
        traininglabels.append(float(pp[21]))
    weights = stocGradAscent1(np.array(trainingset), traininglabels)
    error = 0
    numtest = 0
    for curr in frTest.readlines():
        numtest += 1
        currword = curr.strip().split('\t')
        linecurr = []
        for i in range(21):
            linecurr.append(float(currword[i]))
        h = classifyVector(np.array(linecurr), weights)
        if (h-int(currword[21])) != 0:
            error += 1
    errorRate = float(error/numtest)

    return error, errorRate


error, errorRate = colicTest()
print(error)
print(errorRate)

#dataArr, labelMat = loadDataSet()

#weights = stocGradAscent1(dataArr, labelMat)
#plotBestFit(weights)
